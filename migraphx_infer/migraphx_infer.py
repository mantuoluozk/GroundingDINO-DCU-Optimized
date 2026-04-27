import cv2
import numpy as np
import torch
import time
import os
import migraphx

from transformers import BertTokenizer
from groundingdino.util.inference import load_image
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map

# =========================
# 工具函数
# =========================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
    def __init__(self, onnx_path, cache_path="weights/ground_opt.mxr", force_recompile=False):
        self.cache_path = cache_path

        # ====== 优先加载缓存 ======
        if os.path.exists(cache_path) and not force_recompile:
            print(f"⚡ 直接加载已编译模型: {cache_path}")
            self.model = migraphx.load(cache_path)
        else:
            print("🔍 从 ONNX 构建 MIGraphX")
            self.model = migraphx.parse_onnx(onnx_path) 
            print(self.model)

             # ====================== 2. 打印模型输入输出信息 ======================
            print("=== 模型输入信息 ===")
            inputs = self.model.get_inputs()
            for key, value in inputs.items():
                print(f"{key}: {value}")
            
            print("\n=== 模型输出信息 ===")
            outputs = self.model.get_outputs()
            for key, value in outputs.items():
                print(f"{key}: {value}")

            """
            === 模型输入信息 ===
            text_token_mask: bool_type, {1, 4, 4}, {16, 4, 1}
            token_type_ids: int64_type, {1, 4}, {4, 1}
            position_ids: int64_type, {1, 4}, {4, 1}
            attention_mask: bool_type, {1, 4}, {4, 1}
            input_ids: int64_type, {1, 4}, {4, 1}
            img: float_type, {1, 3, 800, 1200}, {2880000, 960000, 1200, 1}

            === 模型输出信息 ===
            boxes: float_type, {1, 900, 4}, {3600, 4, 1}
            logits: float_type, {1, 900, 256}, {230400, 256, 1}

            输入节点名称: text_token_mask
            输入形状 (N, C, H, W): [1, 4, 4]
            """
            # print("\n⚡ 量化模型（FP16）")
            # migraphx.quantize_fp16(self.model)

            print("⚙️ 编译 MIGraphX（GPU）")
            self.model.compile(
                t=migraphx.get_target("gpu"),device_id=5
            )
            # offload_copy=False, fast_math=False, exhaustive_tune=False

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
        tokenizer,
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
        print(f"Inference time: {infer_time*1000:.2f} ms")

    logits = sigmoid(outputs[0][0])
    boxes = outputs[1][0]

    max_values = np.max(logits, axis=1)
    mask = max_values > box_threshold

    logits = logits[mask]
    boxes = boxes[mask]

    phrases = ["object"] * len(boxes)

    return boxes, np.max(logits, axis=1), phrases


# =========================
# Benchmark
# =========================
def benchmark(model, tokenizer, image, caption, box_th, text_th, warmup=5, runs=10):
    print("\n🔥 预热")
    for _ in range(warmup):
        predict(model, tokenizer, image, caption, box_th, text_th, True)

    print("\n🚀 测试")
    times = []
    for i in range(runs):
        start = time.time()
        predict(model, tokenizer, image, caption, box_th, text_th, True)
        times.append(time.time() - start)

    print(f"\n平均耗时: {np.mean(times)*1000:.2f} ms")
    print(f"FPS: {1/np.mean(times):.2f}")


# =========================
# 主函数
# =========================
if __name__ == "__main__":

    model_path = "../weights/ground_opt.onnx"
    cache_path = "../weights/ground_opt.mxr"   # ⭐ 缓存文件

    img_path = "../images/in/car_1.jpg"

    TEXT_PROMPT = "car ."
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    # 🚀 加载模型（自动缓存）
    model = MIGraphXModel(
        model_path,
        cache_path=cache_path,
        force_recompile=False  # 改成 True 可强制重编译
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    image_source, image = load_image(img_path)

    benchmark(model, tokenizer, image, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD)

    boxes, confs, phrases = predict(
        model, tokenizer, image,
        TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD
    )

    print("检测结果:", phrases)