import cv2
import numpy as np
import torch
import time
import os
import bisect
import migraphx
from typing import Tuple, List, Dict
import groundingdino.datasets.transforms as T
from PIL import Image

# =========================
# 预处理
# =========================
def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ]
    )

    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)

    return image, image_transformed


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# =========================
# 文本标签还原逻辑 (移除 Tokenizer 依赖)
# =========================
def get_phrases_from_posmap(
    posmap: np.ndarray, tokens: List[str], left_idx: int = 0, right_idx: int = 255
):
    """
    直接用字符串列表映射，抛弃沉重的 Tokenizer
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


# =========================
# 分配输出 GPU 内存 (offload_copy=False 必须)
# =========================
def allocate_output_memory(model):
    output_data = {}
    for key in model.get_outputs().keys():
        output_data[key] = migraphx.allocate_gpu(
            s=model.get_outputs()[key]
        )
    return output_data


# =========================
# MIGraphX 模型类 
# =========================
class MIGraphXModel:
    def __init__(self,
                 onnx_path,
                 cache_path="../weights/ground_opt_0430.mxr",
                 device_id=3,
                 force_recompile=False):

        self.cache_path = cache_path

        if os.path.exists(cache_path) and not force_recompile:
            print(f"⚡ 直接加载缓存模型: {cache_path}")
            self.model = migraphx.load(cache_path)
        else:
            print("🔍 从 ONNX 构建模型")
            self.model = migraphx.parse_onnx(onnx_path)

            print("\n=== 输入信息 ===")
            for k, v in self.model.get_inputs().items():
                print(f"{k}: {v}")

            print("\n=== 输出信息 ===")
            for k, v in self.model.get_outputs().items():
                print(f"{k}: {v}")

            print("\n⚙️ 编译模型（GPU + offload=false）")
            self.model.compile(
                t=migraphx.get_target("gpu"),
                offload_copy=False,
                device_id=device_id
            )

            print(f"💾 保存 mxr: {cache_path}")
            migraphx.save(self.model, cache_path)

        self.inputs = self.model.get_inputs()
        self.outputs = self.model.get_outputs()
        self.param_names = self.model.get_parameter_names()
        
        print("✅ param_names:", self.param_names)
        print("✅ input_shape:", self.inputs)
        print("✅ output_shapes keys:", list(self.outputs.keys()))

        self.output_gpu = allocate_output_memory(self.model)
        print("✅ 模型初始化完成")

    def infer(self, input_dict):
        mgx_data = self.output_gpu.copy()

        for name in self.inputs.keys():
            data = input_dict[name]
            if data.dtype == np.float64:
                data = data.astype(np.float32)
            mgx_data[name] = migraphx.to_gpu(migraphx.argument(data))

        start = time.time()
        results = self.model.run(mgx_data)
        infer_time = time.time() - start

        outputs = [
            np.array(migraphx.from_gpu(r))
            for r in results
        ]

        return outputs, infer_time


# =========================
# 推理逻辑 (引入真正的后处理还原)
# =========================
def predict(
        model,
        image,
        text_cache,
        box_threshold,
        text_threshold,
        remove_combined=False,
        is_benchmark=False
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    
    # 使用传入的 text_cache 替代硬编码
    input_dict = {
        "img": np.expand_dims(np.asarray(image), axis=0).astype(np.float32),
        "input_ids": text_cache['input_ids'],
        "attention_mask": text_cache['attention_mask'],
        "position_ids": text_cache['position_ids'],
        "token_type_ids": text_cache['token_type_ids'],
        "text_token_mask": text_cache['text_token_mask']
    }

    outputs, infer_time = model.infer(input_dict)

    if not is_benchmark:
        print(f"Inference time: {infer_time:.3f}s")

    t0 = time.time()
    prediction_logits = sigmoid(outputs[0][0])
    prediction_boxes = outputs[1][0]
    post_time = time.time() - t0

    if not is_benchmark:
        print(f"post time: {post_time:.3f}s")
        print(f"\n=== Debug Info ===")
        print(f"Prediction logits shape: {prediction_logits.shape}")
        print(f"Prediction boxes shape: {prediction_boxes.shape}")
        print(f"Max logit value: {np.max(prediction_logits):.4f}")
        print(f"Mean logit value: {np.mean(prediction_logits):.4f}")

    # 1. 框过滤
    max_values = np.max(prediction_logits, axis=1)
    mask = max_values > box_threshold

    logits = prediction_logits[mask]
    boxes = prediction_boxes[mask]

    tokens = text_cache['tokens']
    input_ids = text_cache['input_ids'][0].tolist()

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


# =========================
# Benchmark
# =========================
def benchmark_performance(
    model, image, text_cache, box_threshold, text_threshold,
    warmup_runs=5, test_runs=10
):
    print("="*60)
    print("📊 开始性能测试（包含预热+实际推理）")
    print("="*60)

    print(f"\n🔥 预热阶段（{warmup_runs} 次）- 不计入性能统计")
    warmup_start = time.time()
    for i in range(warmup_runs):
        t0 = time.time()
        predict(model, image, text_cache, box_threshold, text_threshold, is_benchmark=True)
        warmup_time = time.time() - t0
        print(f"预热 {i+1}/{warmup_runs} - 耗时: {warmup_time*1000:.2f} ms")
    total_warmup_time = time.time() - warmup_start
    print(f"\n预热完成 - 总耗时: {total_warmup_time:.3f} s, 平均每次: {total_warmup_time/warmup_runs*1000:.2f} ms")

    print(f"\n🚀 实际推理测试阶段（{test_runs} 次）- 统计性能指标")
    test_start = time.time()
    infer_times = []  

    for i in range(test_runs):
        t0 = time.time()
        predict(model, image, text_cache, box_threshold, text_threshold, is_benchmark=True)
        infer_time = time.time() - t0
        infer_times.append(infer_time)
        print(f"实际推理 {i+1}/{test_runs} - 耗时: {infer_time*1000:.2f} ms")

    total_test_time = time.time() - test_start
    avg_infer_time = np.mean(infer_times)
    std_infer_time = np.std(infer_times)
    max_infer_time = np.max(infer_times)
    min_infer_time = np.min(infer_times)
    fps = test_runs / total_test_time

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


# =========================
# 主函数
# =========================
if __name__ == "__main__":

    model_path = "../weights/ground_opt_0601.onnx"
    cache_path = "../weights/ground_opt_0601.mxr"
    img_path = "../images/in/car_1.jpg"

    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    
    WARMUP_RUNS = 5
    TEST_RUNS = 10

    model = MIGraphXModel(
        model_path,
        cache_path=cache_path,
        device_id=2,
        force_recompile=False 
    )

    image_source, image = load_image(img_path)

    # =========================
    # 提前计算得到的 Text Cache
    # =========================
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

    benchmark_performance(
        model, image, TEXT_CACHE, 
        BOX_TRESHOLD, TEXT_TRESHOLD,
        WARMUP_RUNS, TEST_RUNS
    )

    print("\n" + "="*60)
    print("🎯 执行最终推理（带详细日志+保存结果）")
    print("="*60)
    
    # 传入 TEXT_CACHE
    boxes, confs, phrases = predict(
        model, image, TEXT_CACHE,
        BOX_TRESHOLD, TEXT_TRESHOLD
    )

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
        
        # 此时打印的 one_cls 将是真实的类别名称（如 "car"）
        cv2.putText(
            ori_img, f'{one_cls} {one_conf:.2f}',  
            (x1-15, y1-15), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            color=(255, 255, 255), 
            fontScale=1.5, 
            thickness=3
        )

    cv2.imwrite('../weights/result_migraphx.jpg', ori_img)
    print(f"\n✅ 结果已保存至: ../weights/result_migraphx.jpg")
    print(f"✅ 检测到目标: {phrases} (共 {len(boxes)} 个)")