import cv2
import numpy as np
import torch
import time
import os
import migraphx
from typing import Tuple
import groundingdino.datasets.transforms as T
from PIL import Image

"""
使用cpu数据做推理
"""

def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            # T.RandomResize([400], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

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

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def to_mgx(x):
    if x.dtype == np.int64:
        return migraphx.argument(x.astype(np.int64))
    elif x.dtype == np.bool_:
        return migraphx.argument(x.astype(np.bool_))
    else:
        return migraphx.argument(x.astype(np.float32))


def _mgx_shape_to_numpy(shape):
    # 将 migraphx input shape 映射到 numpy dtype + lens 以生成零填充张量
    shape_str = str(shape)
    if "int64_type" in shape_str:
        dtype = np.int64
    elif "bool_type" in shape_str:
        dtype = np.bool_
    elif "half_type" in shape_str:
        dtype = np.float16
    else:
        dtype = np.float32
    try:
        dims = list(shape.dims())
    except Exception:
        dims = []
    try:
        lens = list(shape.lens())
    except Exception:
        lens = []
    # 优先用 dims，dims 为空时才退化到 lens
    return dtype, (dims if len(dims) > 0 else lens)
    

# =========================
# 🚀 MIGraphX 推理类（带缓存）
# =========================
class MIGraphXModel:
    def __init__(self, onnx_path, cache_path="../weights/ground_opt_0506.mxr", force_recompile=False):
        self.cache_path = cache_path

        # ====== 优先加载缓存 ======
        if os.path.exists(cache_path) and not force_recompile:
            print(f"⚡ 直接加载已编译模型: {cache_path}")
            self.model = migraphx.load(cache_path)
        else:
            print("🔍 从 ONNX 构建 MIGraphX")
            self.model = migraphx.parse_onnx(onnx_path) 
            # print(self.model)

             # ====================== 2. 打印模型输入输出信息 ======================
            print("=== 模型输入信息 ===")
            inputs = self.model.get_inputs()
            for key, value in inputs.items():
                print(f"{key}: {value}")
            
            print("\n=== 模型输出信息 ===")
            outputs = self.model.get_outputs()
            for key, value in outputs.items():
                print(f"{key}: {value}")

            print("⚙️ 编译 MIGraphX（GPU）")
            self.model.compile(
                t=migraphx.get_target("gpu"), device_id=3, offload_copy=True
            )

            # ====== 保存缓存 ======
            print(f"💾 保存编译模型到: {cache_path}")
            migraphx.save(self.model, cache_path)

        self.param_names = self.model.get_parameter_names()
        self.input_shapes = self.model.get_inputs()
        print("✅ param_names:", self.param_names)
        print("✅ input_shape:", self.input_shapes)
        try:
            self.output_shapes = self.model.get_outputs()
            print("✅ output_shapes keys:", list(self.output_shapes.keys()))
        except Exception:
            self.output_shapes = None

    def infer(self, input_dict):
        # 只按模型 get_inputs() 定义的输入签名来组装
        mgx_inputs = {}
        provided_names = set(input_dict.keys())
        # 某些 mxr 会把内部输出别名也暴露到 get_parameter_names/get_inputs 里，
        # 这里显式排除 main:#output_*，避免把内部输出当成输入填充。
        required_names = {
            k for k in self.input_shapes.keys()
            if not str(k).startswith("main:#output")
        }

        missing = required_names - provided_names
        if missing:
            print("⚠️ 缺失模型输入，准备按 shape 自动补齐:")
            for name in sorted(missing):
                shape = self.input_shapes[name]
                dtype, lens = _mgx_shape_to_numpy(shape)
                mgx_inputs[name] = to_mgx(np.zeros(lens, dtype=dtype))
                print(f"   - {name}: shape={lens}, dtype={dtype.__name__}")

        for name in (required_names & provided_names):
            mgx_inputs[name] = to_mgx(input_dict[name])

        # 额外的 key 不喂给模型，避免和内部签名冲突
        extra = provided_names - required_names
        if extra:
            print("ℹ️ 有多余输入参数将被忽略:")
            for name in sorted(extra):
                print(f"   - {name}")

        start = time.time()
        result = self.model.run(mgx_inputs)
        infer_time = time.time() - start

        outputs = [np.array(r) for r in result]
        return outputs, infer_time


# =========================
# 推理函数
# =========================
def predict(
        model,
        image,
        caption,
        box_threshold,
        text_threshold,
        is_benchmark=False
):

    # 提前针对car .生成对应输入
    input_dict = {
        "img": np.expand_dims(np.asarray(image), axis=0).astype(np.float32),
        "position_ids": np.array([[0, 0, 1, 0]], dtype=np.int64),
        "input_ids": np.array([[101, 2482, 1012, 102]], dtype=np.int64),
        "token_type_ids": np.array([[0, 0, 0, 0]], dtype=np.int64),
        "text_token_mask": np.array([[
            [True, False, False, False],
            [False, True, True, False],
            [False, True, True, False],
            [False, False, False, True]
        ]], dtype=np.bool_),
        "attention_mask": np.array([[True, True, True, True]], dtype=np.bool_)
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
# Benchmark (完全移植 ORT 格式)
# =========================
def benchmark_performance(
    model, image, caption, box_threshold, text_threshold,
    warmup_runs=5, test_runs=10
):
    """
    性能测试函数：包含预热和实际推理
    """
    print("="*60)
    print("📊 开始性能测试（包含预热+实际推理）")
    print("="*60)

    print(f"\n🔥 预热阶段（{warmup_runs} 次）- 不计入性能统计")
    warmup_start = time.time()
    for i in range(warmup_runs):
        t0 = time.time()
        predict(model, image, caption, box_threshold, text_threshold, is_benchmark=True)
        warmup_time = time.time() - t0
        print(f"预热 {i+1}/{warmup_runs} - 耗时: {warmup_time*1000:.2f} ms")
    total_warmup_time = time.time() - warmup_start
    print(f"\n预热完成 - 总耗时: {total_warmup_time:.3f} s, 平均每次: {total_warmup_time/warmup_runs*1000:.2f} ms")

    print(f"\n🚀 实际推理测试阶段（{test_runs} 次）- 统计性能指标")
    test_start = time.time()
    infer_times = []  

    for i in range(test_runs):
        t0 = time.time()
        predict(model, image, caption, box_threshold, text_threshold, is_benchmark=True)
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


# =========================
# 主函数
# =========================
if __name__ == "__main__":

    model_path = "../weights/ground_opt_0430.onnx"
    cache_path = "../weights/ground_opt_0506.mxr"   # ⭐ 缓存文件

    img_path = "../images/in/car_1.jpg"

    TEXT_PROMPT = "car ."
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    
    WARMUP_RUNS = 5
    TEST_RUNS = 10

    # 🚀 加载模型（自动缓存）
    model = MIGraphXModel(
        model_path,
        cache_path=cache_path,
        force_recompile=False  # 改成 True 可强制重编译
    )

    image_source, image = load_image(img_path)

    # 第一步：运行完整的性能测试（预热+实际推理）
    benchmark_performance(
        model, image, TEXT_PROMPT, 
        BOX_TRESHOLD, TEXT_TRESHOLD,
        WARMUP_RUNS, TEST_RUNS
    )

    # 第二步：执行最终推理并画图保存
    print("\n" + "="*60)
    print("🎯 执行最终推理（带详细日志+保存结果）")
    print("="*60)
    
    boxes, confs, phrases = predict(
        model, image,
        TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD
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
    cv2.imwrite('../weights/result_migraphx.jpg', ori_img)
    print(f"\n✅ 结果已保存至: ../weights/result_migraphx.jpg")
    print(f"✅ 检测到目标: {phrases} (共 {len(boxes)} 个)")