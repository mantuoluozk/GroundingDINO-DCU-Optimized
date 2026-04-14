import cv2
import numpy as np
import torch
import time
import os
os.environ["MIGRAPHX_SAVE_TEMPS"] = "1"
os.environ["MIGRAPHX_TRACE"] = "1"
os.environ["MIGRAPHX_LOG_LEVEL"] = "DEBUG" 
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
    """将 migraphx shape 转为 numpy dtype 和 lens。"""
    shape_str = str(shape)
    if "int64_type" in shape_str:
        dtype = np.int64
    elif "bool_type" in shape_str:
        dtype = np.bool_
    elif "half_type" in shape_str:
        dtype = np.float16
    else:
        dtype = np.float32
    return dtype, list(shape.lens())

# =========================
# 🚀 MIGraphX 推理类（带缓存）
# =========================
class MIGraphXModel:
    def __init__(self, onnx_path, cache_path="weights/ground_xiongke.mxr", force_recompile=False):
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
            
            # 获取输入节点名称和输入形状
            inputName = list(self.model.get_inputs().keys())[0]
            inputShape = inputs[inputName].lens()
            print(f"\n输入节点名称: {inputName}")
            print(f"输入形状 (N, C, H, W): {inputShape}")
            inputName1 = list(self.model.get_inputs().keys())[1]
            inputShape1 = inputs[inputName].lens()
            print(f"\n输入节点名称: {inputName1}")
            print(f"输入形状 (N, C, H, W): {inputShape1}")
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

            # passes = [
            #     migraphx.pass_dead_code_elimination(),               # 删除未使用的节点/常量
            #     migraphx.pass_eliminate_contiguous(),                # 合并相邻的 contiguous 操作
            #     migraphx.pass_simplify_reshapes(),                   # 合并/简化 reshape
            #     migraphx.pass_simplify_algebra(),                    # 简化代数表达式 (add/mul/..)
            #     migraphx.pass_eliminate_identity(),                  # 删除 Identity ops
            #     migraphx.pass_common_subexpression_elimination(),    # CSE
            # ]
            # self.model.apply_passes(passes)


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
        print("✅ 输入节点:", self.param_names)

    def infer(self, input_dict):
        mgx_inputs = {k: to_mgx(v) for k, v in input_dict.items()}

        # 某些通过 disable passes 生成的 mxr 会多出内部别名参数（如 main:#output_*）。
        # 若缺失，运行期可能触发 VMFault，这里按 shape 自动补零缓冲区。
        auto_filled = []
        for name in self.param_names:
            if name in mgx_inputs:
                continue
            if name not in self.input_shapes:
                continue
            dtype, lens = _mgx_shape_to_numpy(self.input_shapes[name])
            mgx_inputs[name] = to_mgx(np.zeros(lens, dtype=dtype))
            auto_filled.append((name, lens, dtype.__name__))
        if auto_filled:
            print("⚠️ 自动补齐内部输入参数:")
            for item in auto_filled:
                print(f"   - {item[0]} shape={item[1]} dtype={item[2]}")

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
    caption = preprocess_caption(caption)
    captions = [caption]

    tokenized = tokenizer(captions, padding="longest", return_tensors="pt")
    specical_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

    (
        text_self_attention_masks,
        position_ids,
        _
    ) = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, specical_tokens, tokenizer
    )

    max_text_len = 256
    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
        position_ids = position_ids[:, :max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, :max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, :max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :max_text_len]

    input_dict = {
        "img": np.expand_dims(np.asarray(image), axis=0).astype(np.float32),
        "input_ids": np.asarray(tokenized["input_ids"]).astype(np.int64),
        "attention_mask": np.asarray(tokenized["attention_mask"]).astype(np.bool_),
        "position_ids": np.asarray(position_ids).astype(np.int64),
        "token_type_ids": np.asarray(tokenized["token_type_ids"]).astype(np.int64),
        "text_token_mask": np.asarray(text_self_attention_masks).astype(np.bool_)
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

    #model_path = "weights/ground.onnx"
    model_path = "weights/ground_fixed.onnx"
    cache_path = "weights/ground_xiongke.mxr"   # ⭐ 缓存文件

    img_path = "images/in/car_1.jpg"

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
