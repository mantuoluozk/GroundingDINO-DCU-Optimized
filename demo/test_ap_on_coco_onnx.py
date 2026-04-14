import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import onnxruntime as ort

import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import collate_fn
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        w, h = img.size
        boxes = [obj["bbox"] for obj in target]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        target_new = {}
        image_id = self.ids[idx]
        target_new["image_id"] = image_id
        target_new["boxes"] = boxes
        target_new["orig_size"] = torch.as_tensor([int(h), int(w)])

        if self._transforms is not None:
            img, target = self._transforms(img, target_new)

        return img, target


class PostProcessCocoGrounding(nn.Module):
    """保持和原代码一致的后处理逻辑"""
    def __init__(self, num_select=300, coco_api=None, tokenlizer=None) -> None:
        super().__init__()
        self.num_select = num_select

        assert coco_api is not None
        category_dict = coco_api.dataset['categories']
        cat_list = [item['name'] for item in category_dict]
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
        positive_map = create_positive_map_from_span(
            tokenlizer(captions), tokenspanlist)  # 80, 256. normed

        id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                  41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

        new_pos_map = torch.zeros((91, 256))
        for k, v in id_map.items():
            new_pos_map[v] = positive_map[k]
        self.positive_map = new_pos_map

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False):
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        prob_to_token = torch.from_numpy(out_logits).sigmoid()  # 适配numpy输入
        pos_maps = self.positive_map.to(prob_to_token.device)
        prob_to_label = prob_to_token @ pos_maps.T

        assert prob_to_label.shape[0] == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(
            prob.view(prob_to_label.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]

        if not_to_xyxy:
            boxes = torch.from_numpy(out_bbox)
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(torch.from_numpy(out_bbox))

        boxes = torch.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b}
                   for s, l, b in zip(scores, labels, boxes)]

        return results


def load_onnx_model(onnx_path, device="cuda"):
    """加载ONNX模型并创建推理session"""
    providers = ['CPUExecutionProvider']
    if device == "cuda" and ort.get_device() == "GPU":
        providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
    
    session = ort.InferenceSession(
        onnx_path,
        providers=providers,
        provider_options=[{'device_id': 0}] if "CUDAExecutionProvider" in providers else []
    )
    return session


def onnx_inference(session, images, captions):
    """ONNX模型推理（需匹配模型输入格式）"""
    # 转换为numpy（ONNX Runtime不支持torch tensor）
    images_np = images.cpu().numpy().astype(np.float32)
    
    # 注意：此处需根据你的ONNX模型输入名调整（可通过netron查看）
    # 假设模型输入为 "images" 和 "captions"（需根据实际情况修改）
    input_feed = {
        session.get_inputs()[0].name: images_np,
        # 如果caption是文本token，需补充token化逻辑，此处假设已处理
        # session.get_inputs()[1].name: captions_np
    }
    
    # 执行推理
    outputs = session.run(None, input_feed)
    
    # 解析输出（需匹配模型输出格式，假设输出为logits和bbox）
    # 需根据你的ONNX模型输出调整维度和顺序
    pred_logits = outputs[0]  # 形状: [bs, 100, 256]
    pred_boxes = outputs[1]   # 形状: [bs, 100, 4]
    
    return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


def main(args):
    # 配置（主要用于tokenizer和后处理）
    cfg = SLConfig.fromfile(args.config_file)

    # 加载ONNX模型
    onnx_session = load_onnx_model(args.onnx_path, args.device)

    # 构建数据加载器（和原代码一致）
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = CocoDetection(
        args.image_dir, args.anno_path, transforms=transform)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # 构建后处理器（和原代码一致）
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessCocoGrounding(
        coco_api=dataset.coco, tokenlizer=tokenlizer)

    # 构建评估器（和原代码一致）
    evaluator = CocoGroundingEvaluator(
        dataset.coco, iou_types=("bbox",), useCats=True)

    # 构建文本提示（和原代码一致）
    category_dict = dataset.coco.dataset['categories']
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    # 运行推理
    start = time.time()
    for i, (images, targets) in enumerate(data_loader):
        # 预处理图像（和原代码一致）
        images = images.tensors.to(args.device)
        bs = images.shape[0]
        input_captions = [caption] * bs

        # ONNX推理（替换原PyTorch模型推理）
        outputs = onnx_inference(onnx_session, images, input_captions)

        # 后处理（适配ONNX输出格式）
        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0).to(args.device)
        results = postprocessor(outputs, orig_target_sizes)
        cocogrounding_res = {
            target["image_id"]: output for target, output in zip(targets, results)}
        evaluator.update(cocogrounding_res)

        # 打印进度
        if (i+1) % 30 == 0:
            used_time = time.time() - start
            eta = len(data_loader) / (i+1e-5) * used_time - used_time
            print(
                f"processed {i}/{len(data_loader)} images. time: {used_time:.2f}s, ETA: {eta:.2f}s")

    # 评估指标汇总
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    print("Final results:", evaluator.coco_eval["bbox"].stats.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Grounding DINO ONNX eval on COCO", add_help=True)
    # 新增ONNX模型路径参数
    parser.add_argument("--onnx_path", type=str, required=True, help="path to onnx model file")
    # 保留原配置文件参数（用于tokenizer和后处理）
    parser.add_argument("--config_file", "-c", type=str,
                        required=True, help="path to config file")
    parser.add_argument("--device", type=str, default="cuda",
                        help="running device (default: cuda)")
    # 后处理参数
    parser.add_argument("--num_select", type=int, default=300,
                        help="number of topk to select")
    # COCO数据集参数
    parser.add_argument("--anno_path", type=str,
                        required=True, help="coco annotation path")
    parser.add_argument("--image_dir", type=str,
                        required=True, help="coco image dir")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for dataloader")
    args = parser.parse_args()

    main(args)