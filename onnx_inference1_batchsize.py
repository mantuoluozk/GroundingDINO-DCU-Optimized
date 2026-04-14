from typing import Tuple, List, Dict

import cv2
import numpy as np
import torch
import onnxruntime as ort
from transformers import BertTokenizer, AutoTokenizer
import bisect
import time

from groundingdino.util.inference import load_image
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map

# 加入推理延迟等指标

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_phrases_from_posmap(
    posmap: np.ndarray, tokenized: Dict, tokenizer: AutoTokenizer, left_idx: int = 0, right_idx: int = 255
):
    assert isinstance(posmap, np.ndarray), "posmap must be np.ndarray"
    if posmap.ndim == 1:
        # 将指定范围内的元素设为 False
        posmap[:left_idx + 1] = False
        posmap[right_idx:] = False

        # 获取非零元素的索引
        non_zero_idx = np.nonzero(posmap)[0]
        token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
        return tokenizer.decode(token_ids)
    else:
        raise NotImplementedError("posmap must be 1-dim")

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

# 核心优化：增加tokenizer参数，从外部传入 + 适配batch_size=4
def predict_batch(
        ort_session,
        tokenizer: AutoTokenizer,  # 外部预加载的tokenizer
        images: np.array,  # 修改：接收批量图像 (batch_size, 3, H, W)
        captions: List[str],  # 修改：接收批量文本
        box_threshold: float,
        text_threshold: float,
        device: str = "cpu",
        remove_combined: bool = False,
        is_benchmark: bool = False  # 新增：标记是否为基准测试（控制日志输出）
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
    """
    批量推理函数（batch_size=8）
    返回：每个样本的boxes、confs、phrases列表
    """
    BATCH_SIZE = images.shape[0]
    if not is_benchmark:
        print(f"\n开始批量推理 - batch_size: {BATCH_SIZE}")

    # 1. 文本预处理
    t0 = time.time()
    captions = [preprocess_caption(caption=c) for c in captions]
    if not is_benchmark:
        print(f"Caption processing took {(time.time() - t0):.3f}s")

    # 3. 编码文本（批量）
    t0 = time.time()
    # 移除重复加载tokenizer的性能黑洞
    tokenized = tokenizer(captions, padding="longest", return_tensors="pt").to(device)
    specical_tokens = tokenizer.convert_tokens_to_ids (["[CLS]", "[SEP]", ".", "?"])
    if not is_benchmark:
        print(f"Word embedding took {(time.time() - t0):.3f}s")

    # 4. 生成注意力掩码和位置信息
    t0 = time.time()
    (
        text_self_attention_masks,
        position_ids,
        cate_to_token_mask_list,
    ) = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, specical_tokens, tokenizer)
    if not is_benchmark:
        print(f"Generate attention masks took {(time.time() - t0):.3f}s")

    # 5. 处理超长文本
    max_text_len = 256
    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[
                                    :, : max_text_len, : max_text_len]

        position_ids = position_ids[:, : max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, : max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, : max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : max_text_len]

    # 6. 执行模型推理
    attention_mask = np.asarray(tokenized["attention_mask"]).astype(bool)

    input_dict = {
        "img": images,  # 批量图像 (8, 3, H, W)
        "input_ids": np.asarray(tokenized["input_ids"]),
        "attention_mask": attention_mask,
        "position_ids": np.asarray(position_ids),
        "token_type_ids": np.asarray(tokenized["token_type_ids"]),
        "text_token_mask": np.asarray(text_self_attention_masks)
    }
    
    t0 = time.time()
    outputs = ort_session.run(['logits', 'boxes'], input_dict)
    infer_time = time.time() - t0
    if not is_benchmark:
        print(f"Inference time (batch): {infer_time:.3f}s")
        print(f"Single sample avg infer time: {infer_time/BATCH_SIZE:.3f}s")

    # 7. 获取预测结果（批量）
    prediction_logits = np.apply_along_axis(sigmoid, -1, outputs[0])  # (4, N, L)
    prediction_boxes = outputs[1]  # (4, N, 4)

    if not is_benchmark:
        print(f"\n=== Debug Info (Batch) ===")
        print(f"Prediction logits shape: {prediction_logits.shape}")
        print(f"Prediction boxes shape: {prediction_boxes.shape}")

    # 存储每个样本的结果
    all_boxes = []
    all_confs = []
    all_phrases = []

    # 逐个样本处理
    for idx in range(BATCH_SIZE):
        logits = prediction_logits[idx]
        boxes = prediction_boxes[idx]

        # 8. 应用过滤条件
        max_values = np.max(logits, axis=1)
        mask = max_values > box_threshold
        filtered_logits = logits[mask]
        filtered_boxes = boxes[mask]

        # 9. 处理文本匹配
        single_tokenized = tokenizer(captions[idx])

        # 10. 处理特殊标记
        if remove_combined:
            sep_idx = [i for i in range(len(single_tokenized['input_ids'])) if single_tokenized['input_ids'][i] in [101, 102, 1012]]

            phrases = []
            for logit in filtered_logits:
                max_idx = logit.argmax()
                insert_idx = bisect.bisect_left(sep_idx, max_idx)
                right_idx = sep_idx[insert_idx]
                left_idx = sep_idx[insert_idx - 1]
                phrases.append(
                    get_phrases_from_posmap(logit > text_threshold, single_tokenized, tokenizer, left_idx, right_idx).replace('.', '')
                )
        else:
            phrases = [
                get_phrases_from_posmap(logit > text_threshold, single_tokenized, tokenizer).replace('.', '')
                for logit in filtered_logits
            ]

        all_boxes.append(filtered_boxes)
        all_confs.append(np.max(filtered_logits, axis=1))
        all_phrases.append(phrases)

    return all_boxes, all_confs, all_phrases

# 新增：完整的批量性能测试函数（包含预热+实际推理）
def benchmark_performance_batch(
    ort_session, tokenizer, batch_images, batch_captions, box_threshold, text_threshold,
    warmup_runs=5, test_runs=10, device="cpu"
):
    """
    批量性能测试函数（batch_size=8）
    """
    BATCH_SIZE = batch_images.shape[0]
    print("="*60)
    print(f"📊 开始批量性能测试（batch_size={BATCH_SIZE}）")
    print("="*60)

    # 1. 预热阶段
    print(f"\n🔥 预热阶段（{warmup_runs} 次）- 不计入性能统计")
    warmup_start = time.time()
    for i in range(warmup_runs):
        t0 = time.time()
        predict_batch(ort_session, tokenizer, batch_images, batch_captions, 
                     box_threshold, text_threshold, device, is_benchmark=True)
        warmup_time = time.time() - t0
        print(f"预热 {i+1}/{warmup_runs} - 批次耗时: {warmup_time*1000:.2f} ms, 单样本平均: {warmup_time/BATCH_SIZE*1000:.2f} ms")
    total_warmup_time = time.time() - warmup_start
    print(f"\n预热完成 - 总耗时: {total_warmup_time:.3f} s, 批次平均: {total_warmup_time/warmup_runs*1000:.2f} ms")

    # 2. 实际推理测试阶段
    print(f"\n🚀 实际推理测试阶段（{test_runs} 次）- 统计性能指标")
    test_start = time.time()
    batch_infer_times = []  # 记录每次批次推理耗时

    for i in range(test_runs):
        t0 = time.time()
        predict_batch(ort_session, tokenizer, batch_images, batch_captions,
                     box_threshold, text_threshold, device, is_benchmark=True)
        infer_time = time.time() - t0
        batch_infer_times.append(infer_time)
        print(f"实际推理 {i+1}/{test_runs} - 批次耗时: {infer_time*1000:.2f} ms, 单样本平均: {infer_time/BATCH_SIZE*1000:.2f} ms")

    # 3. 计算性能指标
    total_test_time = time.time() - test_start
    total_samples = test_runs * BATCH_SIZE
    avg_batch_time = np.mean(batch_infer_times)
    std_batch_time = np.std(batch_infer_times)
    avg_sample_time = avg_batch_time / BATCH_SIZE
    
    # 关键：计算FPS（单样本）
    fps = total_samples / total_test_time  # 总样本数 / 总耗时
    batch_fps = test_runs / total_test_time  # 批次FPS（参考）

    # 4. 输出性能报告
    print("\n" + "="*60)
    print(f"📈 批量性能测试报告（batch_size={BATCH_SIZE}）")
    print("="*60)
    print(f"测试批次: {test_runs} 次, 总样本数: {total_samples}")
    print(f"总推理耗时: {total_test_time:.3f} s")
    print(f"平均批次耗时: {avg_batch_time*1000:.2f} ms (±{std_batch_time*1000:.2f} ms)")
    print(f"平均单样本耗时: {avg_sample_time*1000:.2f} ms")
    print(f"批次FPS: {batch_fps:.2f} 批次/秒")
    print(f"单样本FPS: {fps:.2f} 帧/秒 (核心指标)")
    print("="*60)

    return {
        "batch_size": BATCH_SIZE,
        "warmup_runs": warmup_runs,
        "test_runs": test_runs,
        "total_samples": total_samples,
        "avg_batch_time_ms": avg_batch_time*1000,
        "avg_sample_time_ms": avg_sample_time*1000,
        "batch_fps": batch_fps,
        "sample_fps": fps
    }

if __name__ == '__main__':
    # 配置参数
    model_path = 'weights/ground_bs8.onnx'  # 修改：使用batch_size=8的模型
    img_paths = [
        'images/in/car_1.jpg',
        'images/in/car_1.jpg', 
        'images/in/car_1.jpg',
        'images/in/car_1.jpg',  # 8张图片对应batch_size=8
        'images/in/car_1.jpg',
        'images/in/car_1.jpg', 
        'images/in/car_1.jpg',
        'images/in/car_1.jpg' 
    ]
    TEXT_PROMPTS = ["car .", "car .", "car .", "car .","car .", "car .", "car .", "car ."]  # 批量文本（8个）
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    DEVICE = "cpu"
    WARMUP_RUNS = 5    # 预热次数
    TEST_RUNS = 10     # 实际测试次数
    BATCH_SIZE = 8

    # ===================== 加载批量图像 =====================
    print("🔍 加载批量图像（batch_size=8）")
    batch_images = []
    batch_image_sources = []
    for img_path in img_paths:
        image_source, image = load_image(img_path)
        batch_image_sources.append(image_source)
        batch_images.append(image)
    # 转换为numpy数组 (8, 3, H, W)
    batch_images_np = np.stack(batch_images, axis=0)
    print(f"✅ 批量图像加载完成 - 形状: {batch_images_np.shape}")

    # ===================== 加载ONNX模型 =====================
    print("\n🔍 加载ONNX模型（batch_size=8）")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # 启用所有图优化
    sess_options.log_severity_level = 3  # 减少日志输出
    sess_options.enable_profiling = True # 启用性能分析
    ort_session = ort.InferenceSession(model_path, 
                                       sess_options=sess_options, 
                                       providers=['ROCMExecutionProvider'])

    # 查看当前执行引擎
    current_provider = ort_session.get_providers()
    print(f"✅ 模型加载完成 - 当前执行引擎: {current_provider}")

    # ===================== 预加载tokenizer =====================
    print("\n📝 预加载BERT Tokenizer（仅加载一次）")
    t0 = time.time()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(f"✅ Tokenizer加载完成 - 耗时: {(time.time() - t0):.3f} s")

    # ===================== 第一步：批量性能测试 =====================
    performance_result = benchmark_performance_batch(
        ort_session, tokenizer, batch_images_np, TEXT_PROMPTS,
        BOX_TRESHOLD, TEXT_TRESHOLD,
        WARMUP_RUNS, TEST_RUNS, DEVICE
    )

    # ===================== 第二步：执行一次完整批量推理 =====================
    print("\n" + "="*60)
    print("🎯 执行最终批量推理（带详细日志+保存结果）")
    print("="*60)
    all_boxes, all_confs, all_phrases = predict_batch(
        ort_session, tokenizer, batch_images_np, TEXT_PROMPTS,
        BOX_TRESHOLD, TEXT_TRESHOLD, DEVICE
    )

    # ===================== 保存批量推理结果 =====================
    for idx in range(BATCH_SIZE):
        # 读取原始图像
        ori_img = cv2.imread(img_paths[idx])
        img_h = ori_img.shape[0]
        img_w = ori_img.shape[1]
        
        # 绘制检测框
        boxes = all_boxes[idx]
        confs = all_confs[idx]
        phrases = all_phrases[idx]
        
        for i in range(len(boxes)):
            one_box = boxes[i]
            one_conf = confs[i]
            one_cls = phrases[i]
            x1 = int((one_box[0] - one_box[2] / 2) * img_w)
            y1 = int((one_box[1] - one_box[3] / 2) * img_h)
            x2 = int((one_box[0] + one_box[2] / 2) * img_w)
            y2 = int((one_box[1] + one_box[3] / 2) * img_h)
            cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                ori_img, f'{one_cls} {one_conf:.2f}', 
                (x1-15, y1-15), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                color=(255, 255, 255), 
                fontScale=1.5, 
                thickness=3
            )

        # 保存结果
        output_path = f'./images/out/result_{idx+1}.jpg'
        cv2.imwrite(output_path, ori_img)
        print(f"✅ 样本 {idx+1} 结果已保存至: {output_path}")
        print(f"   检测到目标: {phrases} (共 {len(boxes)} 个)")
    
    profile_file = ort_session.end_profiling()
    print(f"\n📊 Profiling 文件已生成: {profile_file}")