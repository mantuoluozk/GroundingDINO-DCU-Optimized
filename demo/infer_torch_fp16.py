import argparse
import os
import sys
import time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None, use_fp16=False):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    
    # 核心推理计时
    torch.cuda.synchronize() if device == "cuda" else None  # 等待GPU操作完成，确保计时准确
    start_time = time.perf_counter()  # 高精度计时
    
    with torch.no_grad():
        if use_fp16 and device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(image[None], captions=[caption])
        else:
            outputs = model(image[None], captions=[caption])
    
    torch.cuda.synchronize() if device == "cuda" else None  # 等待GPU推理完成
    infer_time = time.perf_counter() - start_time  # 推理耗时（秒）
    
    # 输出转回 FP32，避免后续 CPU 操作出问题
    logits = outputs["pred_logits"].float().sigmoid()[0]
    boxes = outputs["pred_boxes"].float()[0]

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases, infer_time


def benchmark_performance(model, image, text_prompt, box_threshold, text_threshold, cpu_only, token_spans, warmup_runs=5, test_runs=10, use_fp16=False):
    """
    性能基准测试：预热 + 多次推理计算平均FPS和时延
    Args:
        warmup_runs: 预热次数（排除初始加载的影响）
        test_runs: 正式测试次数
    """
    # 1. 预热阶段（忽略耗时）
    print(f"\n=== 预热阶段 ({warmup_runs} 次) ===")
    for i in range(warmup_runs):
        _, _, _ = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, cpu_only, token_spans, use_fp16=use_fp16
        )
        print(f"预热完成 {i+1}/{warmup_runs}")
    
    # 2. 正式测试阶段
    print(f"\n=== 正式测试阶段 ({test_runs} 次) ===")
    total_time = 0.0
    infer_times = []  # 记录每次推理的时延
    
    for i in range(test_runs):
        _, _, infer_time = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, cpu_only, token_spans, use_fp16=use_fp16
        )
        infer_times.append(infer_time)
        total_time += infer_time
        print(f"测试 {i+1}/{test_runs} - 单次推理时延: {infer_time*1000:.2f} ms")
    
    # 3. 计算性能指标
    avg_infer_time = total_time / test_runs  # 平均时延（秒）
    fps = test_runs / total_time            # 平均FPS
    std_infer_time = np.std(infer_times)    # 时延标准差（稳定性）
    
    # 4. 输出性能报告
    print("\n" + "="*50)
    print("📊 性能测试报告")
    print("="*50)
    print(f"测试环境: {'GPU (CUDA)' if not cpu_only else 'CPU'}")
    print(f"测试次数: {test_runs} 次 (预热 {warmup_runs} 次)")
    print(f"平均推理时延: {avg_infer_time*1000:.2f} ms (±{std_infer_time*1000:.2f} ms)")
    print(f"最大推理时延: {max(infer_times)*1000:.2f} ms")
    print(f"最小推理时延: {min(infer_times)*1000:.2f} ms")
    print(f"平均FPS: {fps:.2f} 帧/秒")
    print("="*50 + "\n")
    
    return avg_infer_time, fps, infer_times


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO 性能测试", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--box_threshold", type=float, default=0.35, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    # 新增性能测试参数
    parser.add_argument("--warmup-runs", type=int, default=5, help="预热运行次数，默认5")
    parser.add_argument("--test-runs", type=int, default=10, help="正式测试运行次数，默认10")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 inference")

    args = parser.parse_args()

    # cfg
    config_file = args.config_file
    checkpoint_path = args.checkpoint_path
    image_path = args.image_path
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans
    warmup_runs = args.warmup_runs
    test_runs = args.test_runs

    # 打印基础信息
    print(f"📌 使用设备: {'GPU' if not args.cpu_only else 'CPU'}")
    if not args.cpu_only and torch.cuda.is_available():
        print(f"📌 GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"📌 GPU编号: {torch.cuda.current_device()}")

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    
    # load image
    image_pil, image = load_image(image_path)
    
    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")

    # 运行性能基准测试
    avg_infer_time, fps, infer_times = benchmark_performance(
        model, image, text_prompt, box_threshold, text_threshold, 
        args.cpu_only, eval(f"{token_spans}") if token_spans else None,
        warmup_runs, test_runs, use_fp16=args.fp16
    )

    # 单次推理并保存结果（保留原有功能）
    print("\n=== 生成推理结果图片 ===")
    boxes_filt, pred_phrases, single_infer_time = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, args.cpu_only, 
        eval(f"{token_spans}") if token_spans else None, use_fp16=args.fp16
    )

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    image_with_box.save(os.path.join(output_dir, "pred.jpg"))
    
    # 保存性能结果到文件
    performance_file = os.path.join(output_dir, "performance_report.txt")
    with open(performance_file, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write("Grounding DINO 性能测试报告\n")
        f.write("="*50 + "\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试设备: {'GPU' if not args.cpu_only else 'CPU'}\n")
        if not args.cpu_only and torch.cuda.is_available():
            f.write(f"GPU型号: {torch.cuda.get_device_name(0)}\n")
        f.write(f"预热次数: {warmup_runs}\n")
        f.write(f"测试次数: {test_runs}\n")
        f.write(f"平均推理时延: {avg_infer_time*1000:.2f} ms\n")
        f.write(f"时延标准差: {np.std(infer_times)*1000:.2f} ms\n")
        f.write(f"最大时延: {max(infer_times)*1000:.2f} ms\n")
        f.write(f"最小时延: {min(infer_times)*1000:.2f} ms\n")
        f.write(f"平均FPS: {fps:.2f} 帧/秒\n")
        f.write(f"单次推理时延（最后一次）: {single_infer_time*1000:.2f} ms\n")
    
    print(f"\n✅ 性能报告已保存至: {performance_file}")
    print(f"✅ 推理结果图片已保存至: {os.path.join(output_dir, 'pred.jpg')}")