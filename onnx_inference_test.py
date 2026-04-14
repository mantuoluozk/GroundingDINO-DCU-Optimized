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

# 核心优化：增加tokenizer参数，从外部传入
def predict(
        ort_session,
        tokenizer: AutoTokenizer,  # 外部预加载的tokenizer
        image: np.array,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cpu",
        remove_combined: bool = False,
        is_benchmark: bool = False  # 新增：标记是否为基准测试（控制日志输出）
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    # 1. 文本预处理
    t0 = time.time()
    caption = preprocess_caption(caption=caption)
    if not is_benchmark:
        print(f"Caption processing took {(time.time() - t0):.3f}s")

    captions = [caption]
    # 3. 编码文本
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

    #  6. 执行模型推理


    # attention_mask: True=可见 → False=mask
    attention_mask = tokenized["attention_mask"].float()
    attention_mask = (1 - attention_mask) * -1e9  # 关键！
    attention_mask = np.asarray(attention_mask)

    # text_token_mask 同理（如果参与 attention）
    text_self_attention_masks = text_self_attention_masks.float()
    text_self_attention_masks = (1 - text_self_attention_masks) * -1e9


    input_dict = {
        "img": np.expand_dims(np.asarray(image), axis=0),
        "input_ids": np.asarray(tokenized["input_ids"]),
        "attention_mask": attention_mask,
        "position_ids": np.asarray(position_ids),
        "token_type_ids": np.asarray(tokenized["token_type_ids"]),
        "text_token_mask": np.asarray(text_self_attention_masks)
    }
    # input_dict = {
    #     "img": np.expand_dims(np.asarray(image), axis=0),
    #     "input_ids": np.asarray(tokenized["input_ids"]),
    #     "attention_mask": attention_mask,
    #     "position_ids": np.asarray(position_ids),
    #     "token_type_ids": np.asarray(tokenized["token_type_ids"]),
    #     "text_token_mask": np.asarray(text_self_attention_masks)
    # }
    
    t0 = time.time()
    outputs = ort_session.run(['logits', 'boxes'], input_dict)
    infer_time = time.time() - t0
    if not is_benchmark:
        print(f"Inference time: {infer_time:.3f}s")

    # 7. 获取预测结果
    prediction_logits = np.apply_along_axis(sigmoid, -1, outputs[0][0])
    prediction_boxes = outputs[1][0]

    if not is_benchmark:
        print(f"\n=== Debug Info ===")
        print(f"Prediction logits shape: {prediction_logits.shape}")
        print(f"Prediction boxes shape: {prediction_boxes.shape}")
        print(f"Max logit value: {np.max(prediction_logits):.4f}")
        print(f"Mean logit value: {np.mean(prediction_logits):.4f}")

    # 8. 应用过滤条件
    max_values = np.max(prediction_logits, axis=1)
    mask = max_values > box_threshold
    logits = prediction_logits[mask]
    boxes = prediction_boxes[mask]

    # 9. 处理文本匹配
    tokenized = tokenizer(caption)

    # 10. 处理特殊标记
    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]

        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(
                get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', '')
            )
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit in logits
        ]

    return boxes, np.max(logits, axis=1), phrases

# 新增：完整的性能测试函数（包含预热+实际推理）
def benchmark_performance(
    ort_session, tokenizer, image, caption, box_threshold, text_threshold,
    warmup_runs=5, test_runs=10, device="cpu"
):
    """
    性能测试函数：包含预热和实际推理
    :param warmup_runs: 预热次数
    :param test_runs: 实际测试次数
    """
    print("="*60)
    print("📊 开始性能测试（包含预热+实际推理）")
    print("="*60)

    # 1. 预热阶段
    print(f"\n🔥 预热阶段（{warmup_runs} 次）- 不计入性能统计")
    warmup_start = time.time()
    for i in range(warmup_runs):
        t0 = time.time()
        predict(ort_session, tokenizer, image, caption, box_threshold, text_threshold, device, is_benchmark=True)
        warmup_time = time.time() - t0
        print(f"预热 {i+1}/{warmup_runs} - 耗时: {warmup_time*1000:.2f} ms")
    total_warmup_time = time.time() - warmup_start
    print(f"\n预热完成 - 总耗时: {total_warmup_time:.3f} s, 平均每次: {total_warmup_time/warmup_runs*1000:.2f} ms")

    # 2. 实际推理测试阶段
    print(f"\n🚀 实际推理测试阶段（{test_runs} 次）- 统计性能指标")
    test_start = time.time()
    infer_times = []  # 记录每次推理耗时

    for i in range(test_runs):
        t0 = time.time()
        predict(ort_session, tokenizer, image, caption, box_threshold, text_threshold, device, is_benchmark=True)
        infer_time = time.time() - t0
        infer_times.append(infer_time)
        print(f"实际推理 {i+1}/{test_runs} - 耗时: {infer_time*1000:.2f} ms")

    # 3. 计算性能指标
    total_test_time = time.time() - test_start
    avg_infer_time = np.mean(infer_times)
    std_infer_time = np.std(infer_times)
    max_infer_time = np.max(infer_times)
    min_infer_time = np.min(infer_times)
    fps = test_runs / total_test_time

    # 4. 输出性能报告
    print("\n" + "="*60)
    print("📈 性能测试报告（仅实际推理阶段）")
    print("="*60)
    print(f"测试次数: {test_runs} 次")
    print(f"总推理耗时: {total_test_time:.3f} s")
    print(f"平均推理耗时: {avg_infer_time*1000:.2f} ms (±{std_infer_time*1000:.2f} ms)")
    print(f"最大推理耗时: {max_infer_time*1000:.2f} ms")
    print(f"最小推理耗时: {min_infer_time*1000:.2f} ms")
    print(f"平均FPS: {fps:.2f} 帧/秒")
    print("="*60)

    return {
        "warmup_runs": warmup_runs,
        "test_runs": test_runs,
        "avg_infer_time_ms": avg_infer_time*1000,
        "std_infer_time_ms": std_infer_time*1000,
        "max_infer_time_ms": max_infer_time*1000,
        "min_infer_time_ms": min_infer_time*1000,
        "fps": fps
    }

if __name__ == '__main__':
    # 配置参数
    model_path = 'weights/ground_test.onnx'
    img_path = 'images/in/car_1.jpg'
    TEXT_PROMPT = "car ."
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    DEVICE = "cpu"
    WARMUP_RUNS = 5    # 预热次数
    TEST_RUNS = 10     # 实际测试次数

    # 加载图片
    image_source, image = load_image(img_path)

    # 加载ONNX模型（启用优化）
    print("🔍 加载ONNX模型")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # 启用所有图优化
    sess_options.log_severity_level = 3  # 减少日志输出
    # sess_options.enable_profiling = True # 启用性能分析

    ort_session = ort.InferenceSession(model_path, 
                                       sess_options=sess_options, 
                                       providers=['ROCMExecutionProvider']
                                        #    provider_options=[{
                                        #         "device_id": 0,
                                        #         "migraphx_fp16_enable": "False",
                                        #         "migraphx_int8_enable": "False",
                                        #         # 尝试禁用 MIGraphX 内部优化
                                        #         "migraphx_save_compiled_model": "False",
                                        #     }]
                                       )

    # 查看当前执行引擎
    current_provider = ort_session.get_providers()
    print(f"✅ 模型加载完成 - 当前执行引擎: {current_provider}")

    # 预加载tokenizer（只加载一次，核心优化）
    print("\n📝 预加载BERT Tokenizer（仅加载一次）")
    t0 = time.time()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(f"✅ Tokenizer加载完成 - 耗时: {(time.time() - t0):.3f} s")

    # 第一步：运行完整的性能测试（预热+实际推理）
    performance_result = benchmark_performance(
        ort_session, tokenizer, image, TEXT_PROMPT,
        BOX_TRESHOLD, TEXT_TRESHOLD,
        WARMUP_RUNS, TEST_RUNS, DEVICE
    )

    # 第二步：执行一次完整推理（带详细日志，保存结果图片）
    print("\n" + "="*60)
    print("🎯 执行最终推理（带详细日志+保存结果）")
    print("="*60)
    boxes, confs, phrases = predict(
        ort_session, tokenizer, image, TEXT_PROMPT,
        BOX_TRESHOLD, TEXT_TRESHOLD, DEVICE
    )

    # 绘制并保存结果图片
    ori_img = cv2.imread(img_path)
    img_h = ori_img.shape[0]
    img_w = ori_img.shape[1]
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
    cv2.imwrite('./images/out/result.jpg', ori_img)
    print(f"\n✅ 结果已保存至: ./images/out/result.jpg")
    print(f"✅ 检测到目标: {phrases} (共 {len(boxes)} 个)")

    # profile_file = ort_session.end_profiling()
    # print(f"\n📊 Profiling 文件已生成: {profile_file}")