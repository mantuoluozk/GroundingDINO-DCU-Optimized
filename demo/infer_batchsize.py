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

# ====================== 核心配置 - 只需要改这里就能调整batch size ======================
INFERENCE_BATCH_SIZE = 8  # 推理批次大小，修改这个值即可改变batch size
# ====================================================================================

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


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    
    # ========== 修改1: 构建batch数据 ==========
    # 复制图像数据以构建batch (batch_size, 3, H, W)
    image_batch = image.unsqueeze(0).repeat(INFERENCE_BATCH_SIZE, 1, 1, 1).to(device)
    # 复制文本prompt以构建batch (batch_size,)
    caption_batch = [caption] * INFERENCE_BATCH_SIZE
    
    # 核心推理计时
    torch.cuda.synchronize() if device == "cuda" else None  # 等待GPU操作完成，确保计时准确
    start_time = time.perf_counter()  # 高精度计时
    
    with torch.no_grad():
        outputs = model(image_batch, captions=caption_batch)  # 传入batch数据
    
    torch.cuda.synchronize() if device == "cuda" else None  # 等待GPU推理完成
    infer_time = time.perf_counter() - start_time  # 推理耗时（秒）
    # 计算单张图片的平均推理时间
    avg_single_infer_time = infer_time / INFERENCE_BATCH_SIZE
    
    # ========== 修改2: 处理batch输出 ==========
    # 取第一个样本的输出作为结果（所有样本结果相同）
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

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
            model.tokenizer(caption),
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

    # 返回batch推理总时间，方便性能计算
    return boxes_filt, pred_phrases, infer_time, avg_single_infer_time


def benchmark_performance(model, image, text_prompt, box_threshold, text_threshold, cpu_only, token_spans, warmup_runs=5, test_runs=10):
    """
    性能基准测试：预热 + 多次推理计算平均FPS和时延
    适配batch推理，计算正确的吞吐量
    Args:
        warmup_runs: 预热次数（排除初始加载的影响）
        test_runs: 正式测试次数
    """
    # 1. 预热阶段（忽略耗时）
    print(f"\n=== 预热阶段 ({warmup_runs} 次) ===")
    for i in range(warmup_runs):
        _, _, _, _ = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, cpu_only, token_spans
        )
        print(f"预热完成 {i+1}/{warmup_runs}")
    
    # 2. 正式测试阶段
    print(f"\n=== 正式测试阶段 ({test_runs} 次) ===")
    total_batch_time = 0.0  # 总batch推理时间
    total_single_time = 0.0  # 总单张推理时间
    batch_times = []        # 记录每次batch推理的时延
    single_times = []       # 记录每次单张推理的平均时延
    
    for i in range(test_runs):
        _, _, batch_infer_time, avg_single_infer_time = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, cpu_only, token_spans
        )
        batch_times.append(batch_infer_time)
        single_times.append(avg_single_infer_time)
        total_batch_time += batch_infer_time
        total_single_time += avg_single_infer_time
        print(f"测试 {i+1}/{test_runs} - Batch推理时延: {batch_infer_time*1000:.2f} ms | 单张平均时延: {avg_single_infer_time*1000:.2f} ms")
    
    # 3. 计算性能指标
    avg_batch_time = total_batch_time / test_runs                # 平均batch推理时延（秒）
    avg_single_time = total_single_time / test_runs              # 平均单张推理时延（秒）
    batch_throughput = (test_runs * INFERENCE_BATCH_SIZE) / total_batch_time  # 总吞吐量 (张/秒)
    batch_std_time = np.std(batch_times)                         # batch时延标准差
    single_std_time = np.std(single_times)                       # 单张时延标准差
    
    # 4. 输出性能报告
    print("\n" + "="*60)
    print("📊 性能测试报告 (Batch Size = {})".format(INFERENCE_BATCH_SIZE))
    print("="*60)
    print(f"测试环境: {'GPU (CUDA)' if not cpu_only else 'CPU'}")
    print(f"测试次数: {test_runs} 次 (预热 {warmup_runs} 次)")
    print(f"Batch Size: {INFERENCE_BATCH_SIZE}")
    print(f"平均Batch推理时延: {avg_batch_time*1000:.2f} ms (±{batch_std_time*1000:.2f} ms)")
    print(f"平均单张推理时延: {avg_single_time*1000:.2f} ms (±{single_std_time*1000:.2f} ms)")
    print(f"最大Batch时延: {max(batch_times)*1000:.2f} ms | 最大单张时延: {max(single_times)*1000:.2f} ms")
    print(f"最小Batch时延: {min(batch_times)*1000:.2f} ms | 最小单张时延: {min(single_times)*1000:.2f} ms")
    print(f"总吞吐量: {batch_throughput:.2f} 张/秒")
    print("="*60 + "\n")
    
    return avg_batch_time, avg_single_time, batch_throughput, batch_times, single_times


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO 性能测试 (Batch推理)", add_help=True)
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
    print(f"📌 推理Batch Size: {INFERENCE_BATCH_SIZE}")
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
    avg_batch_time, avg_single_time, throughput, batch_times, single_times = benchmark_performance(
        model, image, text_prompt, box_threshold, text_threshold, 
        args.cpu_only, eval(f"{token_spans}") if token_spans else None,
        warmup_runs, test_runs
    )

    # 单次推理并保存结果（保留原有功能）
    print("\n=== 生成推理结果图片 ===")
    boxes_filt, pred_phrases, batch_infer_time, single_infer_time = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, args.cpu_only, 
        eval(f"{token_spans}") if token_spans else None
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
        f.write("="*60 + "\n")
        f.write(f"Grounding DINO 性能测试报告 (Batch Size = {INFERENCE_BATCH_SIZE})\n")
        f.write("="*60 + "\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试设备: {'GPU' if not args.cpu_only else 'CPU'}\n")
        if not args.cpu_only and torch.cuda.is_available():
            f.write(f"GPU型号: {torch.cuda.get_device_name(0)}\n")
        f.write(f"Batch Size: {INFERENCE_BATCH_SIZE}\n")
        f.write(f"预热次数: {warmup_runs}\n")
        f.write(f"测试次数: {test_runs}\n")
        f.write(f"平均Batch推理时延: {avg_batch_time*1000:.2f} ms\n")
        f.write(f"Batch时延标准差: {np.std(batch_times)*1000:.2f} ms\n")
        f.write(f"平均单张推理时延: {avg_single_time*1000:.2f} ms\n")
        f.write(f"单张时延标准差: {np.std(single_times)*1000:.2f} ms\n")
        f.write(f"最大Batch时延: {max(batch_times)*1000:.2f} ms\n")
        f.write(f"最小Batch时延: {min(batch_times)*1000:.2f} ms\n")
        f.write(f"总吞吐量: {throughput:.2f} 张/秒\n")
        f.write(f"最后一次Batch推理时延: {batch_infer_time*1000:.2f} ms\n")
        f.write(f"最后一次单张推理时延: {single_infer_time*1000:.2f} ms\n")
    
    print(f"\n✅ 性能报告已保存至: {performance_file}")
    print(f"✅ 推理结果图片已保存至: {os.path.join(output_dir, 'pred.jpg')}")