import argparse
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util import get_tokenlizer
from groundingdino.util.slconfig import SLConfig
from groundingdino.models.GroundingDINO.bertwarper import (
    generate_masks_with_special_tokens_and_transfer_map,
)


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    for box, label in zip(boxes, labels):
        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3,h,w
    return image_pil, image


def preprocess_caption(caption: str) -> str:
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    return caption


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_text_tensors(
    config_file: str,
    caption: str,
    device: str,
):
    cfg = SLConfig.fromfile(config_file)
    tokenizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    special_token_ids = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

    caption = preprocess_caption(caption)
    tokenized = tokenizer([caption], padding="longest", return_tensors="pt")
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    text_self_attention_masks, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, special_token_ids, tokenizer
    )

    max_text_len = getattr(cfg, "max_text_len", 256)
    if text_self_attention_masks.shape[1] > max_text_len:
        s = max_text_len
        text_self_attention_masks = text_self_attention_masks[:, :s, :s]
        position_ids = position_ids[:, :s]
        tokenized["input_ids"] = tokenized["input_ids"][:, :s]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, :s]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :s]

    # 同时返回 tokenizer 和“单句 tokenize”（用于 get_phrases_from_posmap 行为对齐）
    tokenized_single = tokenizer(caption)

    return (
        cfg,
        tokenizer,
        tokenized_single,
        tokenized["input_ids"].to(torch.int64),
        tokenized["token_type_ids"].to(torch.int64),
        tokenized["attention_mask"].to(torch.int64),
        position_ids.to(torch.int64),
        text_self_attention_masks,
    )


def ort_create_session(onnx_path: str, device: str, num_threads: int = 0):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if num_threads and num_threads > 0:
        so.intra_op_num_threads = int(num_threads)
        so.inter_op_num_threads = int(num_threads)

    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        # 若环境支持 onnxruntime-gpu，会自动启用 CUDA provider
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


def onnx_infer_once(
    sess,
    image: torch.Tensor,
    input_ids: torch.Tensor,
    token_type_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    text_self_attention_masks: torch.Tensor,
    use_cuda_sync: bool,
) -> Tuple[np.ndarray, np.ndarray, float]:
    # ORT 输入必须是 numpy
    feeds = {
        "image": image[None].detach().cpu().numpy().astype(np.float32),
        "input_ids": input_ids.detach().cpu().numpy().astype(np.int64),
        "token_type_ids": token_type_ids.detach().cpu().numpy().astype(np.int64),
        "attention_mask": attention_mask.detach().cpu().numpy().astype(np.int64),
        "position_ids": position_ids.detach().cpu().numpy().astype(np.int64),
        "text_self_attention_masks": text_self_attention_masks.detach().cpu().numpy(),
    }

    if use_cuda_sync:
        torch.cuda.synchronize()
    start = time.perf_counter()
    pred_logits, pred_boxes = sess.run(["pred_logits", "pred_boxes"], feeds)
    if use_cuda_sync:
        torch.cuda.synchronize()
    infer_time = time.perf_counter() - start
    return pred_logits, pred_boxes, infer_time


def postprocess_and_phrases(
    pred_logits: np.ndarray,  # [B,NQ,S]
    pred_boxes: np.ndarray,  # [B,NQ,4]
    tokenized_single,
    tokenizer,
    box_threshold: float,
    text_threshold: float,
    with_logits: bool = True,
):
    # 对齐 torch 版：取 batch=0
    logits = sigmoid(pred_logits[0])  # [NQ,S]
    boxes = pred_boxes[0]  # [NQ,4]

    max_per_query = logits.max(axis=1)
    mask = max_per_query > box_threshold
    logits_filt = logits[mask]
    boxes_filt = boxes[mask]

    pred_phrases: List[str] = []
    for logit in logits_filt:
        posmap = torch.from_numpy(logit) > text_threshold
        phrase = get_phrases_from_posmap(posmap, tokenized_single, tokenizer)
        phrase = phrase.replace(".", "")
        if with_logits:
            pred_phrases.append(phrase + f"({str(float(logit.max()))[:4]})")
        else:
            pred_phrases.append(phrase)

    return torch.from_numpy(boxes_filt), pred_phrases


def benchmark_performance_onnx(
    sess,
    image: torch.Tensor,
    input_ids: torch.Tensor,
    token_type_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    text_self_attention_masks: torch.Tensor,
    warmup_runs: int = 5,
    test_runs: int = 10,
    use_cuda_sync: bool = False,
):
    print(f"\n=== 预热阶段 ({warmup_runs} 次) ===")
    for i in range(warmup_runs):
        _ = onnx_infer_once(
            sess,
            image,
            input_ids,
            token_type_ids,
            attention_mask,
            position_ids,
            text_self_attention_masks,
            use_cuda_sync=use_cuda_sync,
        )
        print(f"预热完成 {i+1}/{warmup_runs}")

    print(f"\n=== 正式测试阶段 ({test_runs} 次) ===")
    total_time = 0.0
    infer_times = []
    for i in range(test_runs):
        _, _, infer_time = onnx_infer_once(
            sess,
            image,
            input_ids,
            token_type_ids,
            attention_mask,
            position_ids,
            text_self_attention_masks,
            use_cuda_sync=use_cuda_sync,
        )
        infer_times.append(infer_time)
        total_time += infer_time
        print(f"测试 {i+1}/{test_runs} - 单次推理时延: {infer_time*1000:.2f} ms")

    avg_infer_time = total_time / test_runs
    fps = test_runs / total_time
    std_infer_time = float(np.std(infer_times))

    print("\n" + "=" * 50)
    print("📊 ONNX 性能测试报告")
    print("=" * 50)
    print(f"测试环境: {'GPU (CUDAExecutionProvider)' if use_cuda_sync else 'CPU/Unknown'}")
    print(f"测试次数: {test_runs} 次 (预热 {warmup_runs} 次)")
    print(f"平均推理时延: {avg_infer_time*1000:.2f} ms (±{std_infer_time*1000:.2f} ms)")
    print(f"最大推理时延: {max(infer_times)*1000:.2f} ms")
    print(f"最小推理时延: {min(infer_times)*1000:.2f} ms")
    print(f"平均FPS: {fps:.2f} 帧/秒")
    print("=" * 50 + "\n")

    return avg_infer_time, fps, infer_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO ONNX 推理与性能测试", add_help=True)
    parser.add_argument("--onnx_path", type=str, required=True, help="onnx 模型路径")
    parser.add_argument("--config_file", "-c", type=str, required=True, help="用于加载 tokenizer 等配置")
    parser.add_argument("--image_path", "-i", type=str, required=True)
    parser.add_argument("--text_prompt", "-t", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True)
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--test-runs", type=int, default=10)
    parser.add_argument("--ort-threads", type=int, default=0, help="onnxruntime 线程数(0=默认)")
    args = parser.parse_args()

    device = "cpu" if args.cpu_only else ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda_sync = device == "cuda"

    print(f"📌 ORT 设备偏好: {device}")
    if use_cuda_sync:
        print(f"📌 GPU型号: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.output_dir, exist_ok=True)

    image_pil, image = load_image(args.image_path)
    image_pil.save(os.path.join(args.output_dir, "raw_image.jpg"))

    (
        _cfg,
        tokenizer,
        tokenized_single,
        input_ids,
        token_type_ids,
        attention_mask,
        position_ids,
        text_self_attention_masks,
    ) = build_text_tensors(args.config_file, args.text_prompt, device="cpu")

    # image 在 GPU 上计时同步更准确，但 feeds 最终还是走 numpy(cpu)；这里只保持与 torch 版一致：
    # 计时逻辑保留 + 可视化保留；模型本体推理走 ORT
    sess = ort_create_session(args.onnx_path, device=device, num_threads=args.ort_threads)

    avg_infer_time, fps, infer_times = benchmark_performance_onnx(
        sess,
        image,
        input_ids,
        token_type_ids,
        attention_mask,
        position_ids,
        text_self_attention_masks,
        warmup_runs=args.warmup_runs,
        test_runs=args.test_runs,
        use_cuda_sync=use_cuda_sync,
    )

    print("\n=== 生成推理结果图片 ===")
    pred_logits, pred_boxes, single_infer_time = onnx_infer_once(
        sess,
        image,
        input_ids,
        token_type_ids,
        attention_mask,
        position_ids,
        text_self_attention_masks,
        use_cuda_sync=use_cuda_sync,
    )

    boxes_filt, pred_phrases = postprocess_and_phrases(
        pred_logits=pred_logits,
        pred_boxes=pred_boxes,
        tokenized_single=tokenized_single,
        tokenizer=tokenizer,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        with_logits=True,
    )

    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    image_with_box.save(os.path.join(args.output_dir, "pred.jpg"))

    performance_file = os.path.join(args.output_dir, "performance_report_onnx.txt")
    with open(performance_file, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("Grounding DINO ONNX 性能测试报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"推理后端: onnxruntime\n")
        f.write(f"设备偏好: {device}\n")
        if use_cuda_sync:
            f.write(f"GPU型号: {torch.cuda.get_device_name(0)}\n")
        f.write(f"预热次数: {args.warmup_runs}\n")
        f.write(f"测试次数: {args.test_runs}\n")
        f.write(f"平均推理时延: {avg_infer_time*1000:.2f} ms\n")
        f.write(f"时延标准差: {np.std(infer_times)*1000:.2f} ms\n")
        f.write(f"最大时延: {max(infer_times)*1000:.2f} ms\n")
        f.write(f"最小时延: {min(infer_times)*1000:.2f} ms\n")
        f.write(f"平均FPS: {fps:.2f} 帧/秒\n")
        f.write(f"单次推理时延（最后一次）: {single_infer_time*1000:.2f} ms\n")

    print(f"\n✅ 性能报告已保存至: {performance_file}")
    print(f"✅ 推理结果图片已保存至: {os.path.join(args.output_dir, 'pred.jpg')}")

