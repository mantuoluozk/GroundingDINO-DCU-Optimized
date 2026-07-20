
import os
import os.path as osp
import sys

import cv2
import torch

import migraphx as mgx
import numpy as np
import onnxruntime as ort
import prettytable
from transformers import AutoTokenizer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SPECIAL_TOKENS = [101, 102, 1012]  # [CLS], [SEP], .


class MIGraphXInferencer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model(model_path)

        input_indexes = sorted(self.program.get_input_indexes().items(), key=lambda x: x[1])
        self.input_names = [x[0] for x in input_indexes]
        self.input_shapes = {k: s.lens() for k, s in self.program.get_inputs().items()}

        self.model_data = {}
        for key, shape in self.program.get_parameter_shapes().items():
            if key not in self.input_names:
                self.model_data[key] = mgx.allocate_gpu(s=shape)

    def load_model(self, model_path):
        if model_path.endswith(".onnx"):
            self.program = mgx.parse_onnx(model_path)
            mgx.quantize_fp16(self.program)
            self.program.compile(t=mgx.get_target("gpu"), offload_copy=False, device_id=0)
        elif model_path.endswith(".mxr"):
            self.program = mgx.load(model_path)
        else:
            raise ValueError("Unknown model type:", model_path)

    def run(self, input_list):
        print(f"Model running: {self.model_path}")
        for i, d in enumerate(input_list):
            input_name = self.input_names[i]
            self.model_data[input_name] = mgx.to_gpu(mgx.argument(d))
        results = self.program.run(self.model_data)
        results = [mgx.from_gpu(result) for result in results]
        results = [np.array(result) for result in results]
        return results


class ONNXInferencer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.input_names = [x.name for x in self.session.get_inputs()]
        self.input_shapes = {x.name: x.shape for x in self.session.get_inputs()}

    def run(self, input_list):
        print(f"Model running: {self.model_path}")
        input_dict = {}
        for i, d in enumerate(input_list):
            input_name = self.input_names[i]
            input_dict[input_name] = d
        results = self.session.run(None, input_dict)
        return results


def get_inferencer(model_path):
    if model_path.endswith(".onnx"):
        return ONNXInferencer(model_path)
    elif model_path.endswith(".mxr"):
        return MIGraphXInferencer(model_path)
    else:
        raise ValueError("Unknown model type:", model_path)


def generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens_list, tokenizer):
    """Generate attention mask between each pair of special tokens
    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.
    Returns:
        torch.Tensor: attention mask between each special tokens.
    """
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token), device=input_ids.device).bool()
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = (
        torch.eye(num_token, device=input_ids.device).bool().unsqueeze(0).repeat(bs, 1, 1)
    )
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    cate_to_token_mask_list = [[] for _ in range(bs)]
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device
            )
            c2t_maski = torch.zeros((num_token), device=input_ids.device).bool()
            c2t_maski[previous_col + 1 : col] = True
            cate_to_token_mask_list[row].append(c2t_maski)
        previous_col = col

    cate_to_token_mask_list = [
        torch.stack(cate_to_token_mask_listi, dim=0)
        for cate_to_token_mask_listi in cate_to_token_mask_list
    ]

    # # padding mask
    # padding_mask = tokenized['attention_mask']
    # attention_mask = attention_mask & padding_mask.unsqueeze(1).bool() & padding_mask.unsqueeze(2).bool()

    return attention_mask, position_ids.to(torch.long), cate_to_token_mask_list


def image_preprocess(image_path: str, size=(800, 1200)):
    """Load image and preprocess to NCHW float32 for ONNX."""
    image_bgr = cv2.imread(image_path)
    assert image_bgr is not None, f"failed to read image: {image_path}"
    h, w = size
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (w, h))
    img = resized.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)[None]  # [1, 3, H, W]
    img = np.ascontiguousarray(img)
    return image_bgr, img.astype(np.float32)


def text_preprocess(caption: str, tokenizer_path: str = "../bert-base-uncased"):
    """Tokenize caption and build ONNX text inputs."""
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenized = tokenizer([caption], padding="longest", return_tensors="pt")
    text_self_attention_masks, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, SPECIAL_TOKENS, tokenizer
    )

    return {
        "input_ids": tokenized["input_ids"].numpy().astype(np.int64),
        "attention_mask": tokenized["attention_mask"].numpy().astype(bool),
        "position_ids": position_ids.numpy().astype(np.int64),
        "token_type_ids": tokenized["token_type_ids"].numpy().astype(np.int64),
        "text_token_mask": text_self_attention_masks.numpy().astype(bool),
        "tokenizer": tokenizer,
        "caption": caption,
        "tokenized": tokenized,
    }


def postprocess(
    image_bgr: np.ndarray,
    logits: np.ndarray,
    boxes: np.ndarray,
    text_inputs: dict,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    save_path: str = "result.jpg",
):
    """Filter predictions and draw boxes on the original image."""
    scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    conf = scores.max(axis=-1)
    keep = conf > box_threshold
    scores, boxes, conf = scores[keep], boxes[keep], conf[keep]

    tokenizer = text_inputs["tokenizer"]
    tokenized = text_inputs["tokenized"]
    input_ids = tokenized["input_ids"][0].tolist()

    h, w = image_bgr.shape[:2]
    vis = image_bgr.copy()
    phrases = []
    for score, box, c in zip(scores, boxes, conf):
        # phrase from tokens above text_threshold
        token_ids = [tid for tid, s in zip(input_ids, score) if s > text_threshold and tid not in SPECIAL_TOKENS]
        phrase = tokenizer.decode(token_ids).replace(".", "").strip() or "object"
        phrases.append(phrase)

        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{phrase} {c:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(save_path, vis)
    print(f"saved {save_path}, detections={list(zip(phrases, conf.round(3).tolist()))}")
    return vis


def compare_data(a, b):
    show_decimal = 8
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    print(">" * 80)

    pt1 = prettytable.PrettyTable()
    pt1.field_names = ['framework', 'dtype', 'shape', 'min', 'max', 'mean',
                        'std', 'has-nan', 'has-inf']
    pt1.add_row(['data-A', a.dtype, a.shape,
                    round(a.min().item(), show_decimal),
                    round(a.max().item(), show_decimal),
                    round(a.astype('float64').mean().item(), show_decimal),
                    round(a.astype('float64').std().item(), show_decimal),
                    np.isnan(a).any().item(), np.isinf(a).any().item()])
    pt1.add_row(['data-B', b.dtype, b.shape,
                    round(b.min().item(), show_decimal),
                    round(b.max().item(), show_decimal),
                    round(b.astype('float64').mean().item(), show_decimal),
                    round(b.astype('float64').std().item(), show_decimal),
                    np.isnan(b).any().item(), np.isinf(b).any().item()])
    print(pt1)

    a_ = a.flatten().astype(np.float64)
    b_ = b.flatten().astype(np.float64)
    cos = np.dot(a_, b_) / (np.linalg.norm(a_) * np.linalg.norm(b_))

    diff = np.abs(a_ - b_)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)
    sum_diff = np.sum(diff)
    dis_match = np.sum(diff > 0.01) / diff.size
    print("max_diff:", max_diff)

    all_close = np.allclose(a.flatten(), b.flatten(), rtol=1e-03, atol=1e-06, equal_nan=True)

    pt2 = prettytable.PrettyTable()
    pt2.field_names = ['cosine-similarity', 'all-close', 'max-diff',
                        'mean-diff', 'sum-diff', 'dis-match']
    pt2.add_row([round(cos.item(), show_decimal),
                    all_close,
                    round(max_diff.item(), show_decimal),
                    round(avg_diff.item(), show_decimal),
                    round(sum_diff.item(), show_decimal),
                    round(dis_match.item(), 3)])
    print(pt2)

    num = 5
    print(f"前{num}元素(data-A):", a.flatten()[:num])
    print(f"前{num}元素(data-B):", b.flatten()[:num])
    print(f"后{num}元素(data-A):", a.flatten()[-num:])
    print(f"后{num}元素(data-B):", b.flatten()[-num:])
    idx = diff.argmax()
    print(f"最大差值: {a.flatten()[idx].item()}-{b.flatten()[idx].item()}={diff[idx].item()}")
    start = max(0, idx - num//2)
    end = min(len(a.flatten()), start + num)
    print(f"最大误差元素(data-A):", a.flatten()[start:end])
    print(f"最大误差元素(data-B):", b.flatten()[start:end])
    print("<" * 80)


def step_infer(onnx_dir, mxr_dir, image_path, caption, save_path):

    input_dict = text_preprocess(caption)
    image_bgr, img = image_preprocess(image_path)
    input_dict["img"] = img

    model_paths = {
        "backbone": {
            "onnx": osp.join(onnx_dir, "backbone_.onnx"),
            "mxr": osp.join(mxr_dir, "backbone.mxr"),
        },
        "bert": {
            "onnx": osp.join(onnx_dir, "bert_.onnx"),
            "mxr": osp.join(mxr_dir, "bert.mxr"),
        },
        "encoder": {
            "onnx": osp.join(onnx_dir, "encoder_.onnx"),
            "mxr": osp.join(mxr_dir, "encoder.mxr"),
        },
        "decoder": {
            "onnx": osp.join(onnx_dir, "decoder_.onnx"),
            "mxr": osp.join(mxr_dir, "decoder.mxr"),
        },
    }
    model_select = {
        "backbone": "onnx",
        "bert": "onnx",
        "encoder": "onnx",
        "decoder": "onnx",
    }
    # model_select = {
    #     "backbone": "mxr",
    #     "bert": "mxr",
    #     "encoder": "mxr",
    #     "decoder": "mxr",
    # }

    backbone_inferencer = get_inferencer(model_paths["backbone"][model_select["backbone"]])
    backbone_result = backbone_inferencer.run([input_dict["img"]])[0]
    bert_inferencer = get_inferencer(model_paths["bert"][model_select["bert"]])
    bert_result = bert_inferencer.run([input_dict["input_ids"],
                                       input_dict["token_type_ids"],
                                       input_dict["position_ids"],
                                       input_dict["text_token_mask"]])[0]
    encoder_inferencer = get_inferencer(model_paths["encoder"][model_select["encoder"]])
    encoder_result1, encoder_result2 = encoder_inferencer.run([backbone_result,
                                                               bert_result,
                                                               input_dict["position_ids"],
                                                               input_dict["text_token_mask"],
                                                               input_dict["attention_mask"]])
    decoder_inferencer = get_inferencer(model_paths["decoder"][model_select["decoder"]])
    logits, boxes = decoder_inferencer.run([encoder_result1,
                                            encoder_result2,
                                            input_dict["attention_mask"]])
    postprocess(image_bgr, logits, boxes, input_dict, save_path=save_path)


def model_infer(model_path, image_path, caption, save_path):

    inferencer = get_inferencer(model_path)
    h, w = inferencer.input_shapes['img'][2:]
    print(f"input image size (WxH): {w}x{h}")

    input_dict = text_preprocess(caption)
    image_bgr, img = image_preprocess(image_path, size=(h, w))
    input_dict["img"] = img

    logits, boxes = inferencer.run([input_dict[name] for name in inferencer.input_names])
    postprocess(image_bgr, logits, boxes, input_dict, save_path=save_path)


def main():
    # caption = "cat ."
    # image_path = "./cat_dog.jpeg"
    # step_infer("./", "./", image_path, caption, "ort_step_result.jpg")
    # model_infer("../ground_sim.onnx", image_path, caption, "ort_infer_result.jpg")
    # model_infer("../ground_bs1_opt_0703.mxr", image_path, caption, "mgx_infer_result.jpg")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="onnx or mxr model path")
    parser.add_argument("--image_path", type=str, default="../.asset/cat_dog.jpeg")
    parser.add_argument("--caption", type=str, default="cat .")
    # parser.add_argument("--onnx_dir", type=str, default=None, help="onnx model directory")
    # parser.add_argument("--mxr_dir", type=str, default=None, help="mxr model directory")
    parser.add_argument("--save_path", type=str, default="./result.jpg")
    args = parser.parse_args()
    model_infer(args.model_path, args.image_path, args.caption, args.save_path)


if __name__ == "__main__":
    main()
