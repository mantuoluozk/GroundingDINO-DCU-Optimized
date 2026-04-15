from typing import Tuple, List, Dict

import cv2
import numpy as np
import torch
import onnxruntime as ort
import bisect
import time
import os
"""
    针对模型前后处理和代码结构进行优化
        1.预测结果获取优化prediction_logits = sigmoid(outputs[0][0])
        2.输入数据提前获取直接传入，移除了对tokenizer的依赖
"""

from groundingdino.util.inference import load_image

so_options = ort.SessionOptions()
custom_op_lib_path = "../ort_plugin/build/libms_deform_attn_ort.so" 
so_options.register_custom_ops_library(custom_op_lib_path)
# 开启ort优化
so_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_phrases_from_posmap(
    posmap: np.ndarray, tokens: List[str], left_idx: int = 0, right_idx: int = 255
):
    """
    【核心优化】直接用字符串列表映射，抛弃沉重的 Tokenizer
    """
    assert isinstance(posmap, np.ndarray), "posmap must be np.ndarray"
    if posmap.ndim == 1:
        # 将指定范围内的元素设为 False
        posmap[:left_idx + 1] = False
        posmap[right_idx:] = False

        # 获取非零元素的索引
        non_zero_idx = np.nonzero(posmap)[0]
        # 提取被激活的单词，并自动过滤掉特殊占位符
        words = [tokens[i] for i in non_zero_idx if tokens[i] not in ["[CLS]", "[SEP]", "."]]
        return " ".join(words).strip()
    else:
        raise NotImplementedError("posmap must be 1-dim")

def predict(
        ort_session,
        image: np.array,
        text_cache: dict,
        box_threshold: float,
        text_threshold: float,
        remove_combined: bool = False,
        is_benchmark: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:

    input_dict = {
        "img": np.expand_dims(np.asarray(image), axis=0),
        "input_ids": text_cache['input_ids'],
        "attention_mask": text_cache['attention_mask'],
        "position_ids": text_cache['position_ids'],
        "token_type_ids": text_cache['token_type_ids'],
        "text_token_mask": text_cache['text_token_mask']
    }

    t0 = time.time()

    outputs = ort_session.run(['logits', 'boxes'], input_dict)
    infer_time = time.time() - t0
    if not is_benchmark:
        print(f"Inference time: {infer_time:.3f}s")

    t0 = time.time()
    prediction_logits = sigmoid(outputs[0][0])
    prediction_boxes = outputs[1][0]
    post_time = time.time() - t0
    if not is_benchmark:
        print(f"post time: {post_time:.3f}s")

    if not is_benchmark:
        print(f"\n=== Debug Info ===")
        print(f"Prediction logits shape: {prediction_logits.shape}")
        print(f"Prediction boxes shape: {prediction_boxes.shape}")
        print(f"Max logit value: {np.max(prediction_logits):.4f}")
        print(f"Mean logit value: {np.mean(prediction_logits):.4f}")

    # 应用过滤条件
    max_values = np.max(prediction_logits, axis=1)
    mask = max_values > box_threshold
    logits = prediction_logits[mask]
    boxes = prediction_boxes[mask]

    # 处理文本匹配
    tokens = text_cache['tokens']
    input_ids = text_cache['input_ids'][0].tolist()

    # 处理特殊标记
    if remove_combined:
        sep_idx = [i for i in range(len(input_ids)) if input_ids[i] in [101, 102, 1012]]
        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(
                get_phrases_from_posmap(logit > text_threshold, tokens, left_idx, right_idx)
            )
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokens)
            for logit in logits
        ]

    return boxes, np.max(logits, axis=1), phrases

def benchmark_performance(
    ort_session, image, text_cache, box_threshold, text_threshold,
    warmup_runs=5, test_runs=10
):
    """
    性能测试函数：包含预热和实际推理
    :param warmup_runs: 预热次数
    :param test_runs: 实际测试次数
    """
    print("="*60)
    print("📊 开始性能测试（包含预热+实际推理）")
    print("="*60)

    print(f"\n🔥 预热阶段（{warmup_runs} 次）- 不计入性能统计")
    warmup_start = time.time()
    for i in range(warmup_runs):
        t0 = time.time()
        predict(ort_session, image, text_cache, box_threshold, text_threshold, is_benchmark=True)
        warmup_time = time.time() - t0
        print(f"预热 {i+1}/{warmup_runs} - 耗时: {warmup_time*1000:.2f} ms")
    total_warmup_time = time.time() - warmup_start
    print(f"\n预热完成 - 总耗时: {total_warmup_time:.3f} s, 平均每次: {total_warmup_time/warmup_runs*1000:.2f} ms")

    print(f"\n🚀 实际推理测试阶段（{test_runs} 次）- 统计性能指标")
    test_start = time.time()
    infer_times = []  # 记录每次推理耗时

    for i in range(test_runs):
        t0 = time.time()
        predict(ort_session, image, text_cache, box_threshold, text_threshold, is_benchmark=True)
        infer_time = time.time() - t0
        infer_times.append(infer_time)
        print(f"实际推理 {i+1}/{test_runs} - 耗时: {infer_time*1000:.2f} ms")

    # 计算性能指标
    total_test_time = time.time() - test_start
    avg_infer_time = np.mean(infer_times)
    std_infer_time = np.std(infer_times)
    max_infer_time = np.max(infer_times)
    min_infer_time = np.min(infer_times)
    fps = test_runs / total_test_time

    # 输出性能报告
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
    model_path = '../weights_400x600/ground_deform.onnx'
    """
        ../weights/ground_deform.onnx 普通版本
        ../weights/ground_deform_sim.onnx 简化版本
        ../weights/ground_deform_fp16.onnx FP16版本(其中自定义算子fp32)
        ../weights/ground_deform_fp16_all.onnx 纯FP16版本
        ../weights/ground_deform_400x600.onnx 图像尺度400x600版本
    """
    img_path = '../images/in/car_1.jpg'
    TEXT_PROMPT = "car ."
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    DEVICE = "cpu"
    WARMUP_RUNS = 5    # 预热次数
    TEST_RUNS = 10     # 实际测试次数

    image_source, image = load_image(img_path)

    providers = [
        'ROCMExecutionProvider', 
        'CPUExecutionProvider'
    ]

    print("🔍 加载ONNX模型")
    ort_session = ort.InferenceSession(model_path, 
                                       sess_options=so_options, 
                                       providers=providers
                                       )
    print(f"✅ 模型加载成功！自定义算子已就绪！当前执行引擎:{ort_session.get_providers()}")

    # 提前通过get_caption_mask.py计算得到
    TEXT_CACHE = {
        'input_ids': np.array([[ 101, 2482, 1012,  102]], dtype=np.int64),
        'attention_mask': np.array([[ True,  True,  True,  True]], dtype=np.bool_),
        'position_ids': np.array([[0, 0, 1, 0]], dtype=np.int64),
        'token_type_ids': np.array([[0, 0, 0, 0]], dtype=np.int64),
        'text_token_mask': np.array([[[ True, False, False, False],
                                      [False,  True,  True, False],
                                      [False,  True,  True, False],
                                      [False, False, False,  True]]], dtype=np.bool_),
        # 存放 ID 对应的单词，用于快速 decode
        'tokens': ["[CLS]", "car", ".", "[SEP]"]
    }

    # 第一步：运行完整的性能测试（预热+实际推理）
    performance_result = benchmark_performance(
        ort_session, image, TEXT_CACHE,
        BOX_TRESHOLD, TEXT_TRESHOLD,
        WARMUP_RUNS, TEST_RUNS
    )

    # 第二步：执行一次完整推理（带详细日志，保存结果图片）
    print("\n" + "="*60)
    print("🎯 执行最终推理（带详细日志+保存结果）")
    print("="*60)
    boxes, confs, phrases = predict(
        ort_session, image, TEXT_CACHE,
        BOX_TRESHOLD, TEXT_TRESHOLD
    )

    # 绘制并保存结果图片
    print("\n🎯 执行最终推理并保存结果图")
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
    cv2.imwrite('./result.jpg', ori_img)
    print(f"\n✅ 结果已保存至: ./result.jpg")
    print(f"✅ 检测到目标: {phrases} (共 {len(boxes)} 个)")
