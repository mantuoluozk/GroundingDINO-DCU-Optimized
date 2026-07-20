import cv2
import numpy as np
import torch
import time
import os
import migraphx
import argparse
from typing import Tuple

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

"""
使用device数据进行推理
"""

def load_image(image_path: str, image_height: int, image_width: int) -> Tuple[np.ndarray, np.ndarray]:
    """使用 OpenCV/NumPy 完成预处理，不依赖 groundingdino 包。"""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (image_width, image_height))
    image = resized.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    image = np.ascontiguousarray(image.transpose(2, 0, 1))

    return image_rgb, image


def sigmoid(x):
    """Numerically stable sigmoid for large positive/negative logits."""
    x = np.asarray(x)
    result = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive].astype(np.float32)))
    exp_x = np.exp(x[~positive].astype(np.float32))
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


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
# MIGraphX 模型类 (保留 offload_copy=False 逻辑，增加调试输出)
# =========================
class MIGraphXModel:
    def __init__(self,
                 onnx_path,
                 cache_path="../weights_0720/ground_bs1_opt.mxr",
                 device_id=0,
                 force_recompile=False):

        self.cache_path = cache_path

        # =========================
        # 优先加载 mxr
        # =========================
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

        # =========================
        # 记录输入输出信息与添加调试输出
        # =========================
        self.inputs = self.model.get_inputs()
        self.outputs = self.model.get_outputs()
        self.param_names = self.model.get_parameter_names()
        self.batch_size, _, self.image_height, self.image_width = self.inputs["img"].lens()
        
        print("✅ param_names:", self.param_names)
        print("✅ input_shape:", self.inputs)
        print("✅ output_shapes keys:", list(self.outputs.keys()))
        print(f"✅ batch size: {self.batch_size}")
        print(f"✅ model image size (HxW): {self.image_height}x{self.image_width}")

        # =========================
        # 分配输出 GPU buffer
        # =========================
        self.output_gpu = allocate_output_memory(self.model)

        print("✅ 模型初始化完成")

    def infer(self, input_dict):
        # 1. 复制预先分配好的输出缓冲区，避免直接修改原始字典引发后续推理问题
        mgx_data = self.output_gpu.copy()

        # 2. CPU → GPU 将输入数据搬运到显存，并填入到执行字典中
        for name in self.inputs.keys():
            data = input_dict[name]

            # 强制类型转换，ONNX 很多时候不支持 float64
            if data.dtype == np.float64:
                data = data.astype(np.float32)

            # 将输入数据转化为 migraphx 参数并放入 GPU
            mgx_data[name] = migraphx.to_gpu(migraphx.argument(data))

        # 3. 推理 (传入的 mgx_data 同时包含了 input buffer 和 output buffer)
        start = time.perf_counter()
        results = self.model.run(mgx_data)
        infer_time = time.perf_counter() - start

        # 4. GPU → CPU 将结果从显存取回
        outputs = [
            np.array(migraphx.from_gpu(r))
            for r in results
        ]

        return outputs, infer_time


# =========================
# 推理逻辑 (引入 Debug 输出与处理)
# =========================
def predict(
        model,
        image,
        caption,
        box_threshold,
        text_threshold,
        is_benchmark=False
):
    batch_size = model.batch_size
    input_dict = {
        "img": np.repeat(np.expand_dims(np.asarray(image), axis=0), batch_size, axis=0).astype(np.float32),
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

    for name in ("position_ids", "input_ids", "token_type_ids", "text_token_mask", "attention_mask"):
        input_dict[name] = np.repeat(input_dict[name], batch_size, axis=0)

    outputs, infer_time = model.infer(input_dict)

    if not is_benchmark:
        print(f"Inference time: {infer_time:.3f}s")

    t0 = time.perf_counter()
    prediction_logits = sigmoid(outputs[0][0])
    prediction_boxes = outputs[1][0]
    post_time = time.perf_counter() - t0

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

    # 这里为了适配 GroundingDINO 的结构，简单占位，若有完整分词器可替换为真实文本解码
    phrases = ["object"] * len(boxes)

    return boxes, np.max(logits, axis=1), phrases


# =========================
# Benchmark
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
    warmup_start = time.perf_counter()
    for i in range(warmup_runs):
        predict(model, image, caption, box_threshold, text_threshold, is_benchmark=True)
    total_warmup_time = time.perf_counter() - warmup_start
    print(f"\n预热完成 - 总耗时: {total_warmup_time:.3f} s, 平均每次: {total_warmup_time/warmup_runs*1000:.2f} ms")

    print(f"\n🚀 实际推理测试阶段（{test_runs} 次）- 统计性能指标")
    e2e_times = []

    for i in range(test_runs):
        t0 = time.perf_counter()
        predict(model, image, caption, box_threshold, text_threshold, is_benchmark=True)
        e2e_times.append(time.perf_counter() - t0)

    # 计算性能指标
    total_test_time = float(np.sum(e2e_times, dtype=np.float64))
    avg_infer_time = np.mean(e2e_times)
    std_infer_time = np.std(e2e_times)
    max_infer_time = np.max(e2e_times)
    min_infer_time = np.min(e2e_times)
    p50_infer_time, p90_infer_time, p95_infer_time = np.percentile(
        e2e_times, [50, 90, 95]
    )
    fps = (test_runs * model.batch_size) / total_test_time

    # 输出性能报告
    print("\n" + "="*60)
    print("📈 端到端推理性能报告")
    print("="*60)
    print("统计范围: 输入构造 + H2D + MIGraphX + D2H + 后处理")
    print("不包含: 模型加载、图片读取/缩放归一化、画框和保存")
    print(f"Batch size: {model.batch_size}")
    print(f"测试次数: {test_runs} 次")
    print(f"总推理耗时: {total_test_time:.3f} s")
    print(f"平均推理耗时: {avg_infer_time*1000:.2f} ms (±{std_infer_time*1000:.2f} ms)")
    print(f"P50/P90/P95: {p50_infer_time*1000:.2f} / {p90_infer_time*1000:.2f} / {p95_infer_time*1000:.2f} ms")
    print(f"最大推理耗时: {max_infer_time*1000:.2f} ms")
    print(f"最小推理耗时: {min_infer_time*1000:.2f} ms")
    print(f"吞吐量: {fps:.2f} images/s")
    print("="*60)

    return {
        "warmup_runs": warmup_runs,
        "test_runs": test_runs,
        "avg_infer_time_ms": avg_infer_time*1000,
        "std_infer_time_ms": std_infer_time*1000,
        "p50_infer_time_ms": p50_infer_time*1000,
        "p90_infer_time_ms": p90_infer_time*1000,
        "p95_infer_time_ms": p95_infer_time*1000,
        "max_infer_time_ms": max_infer_time*1000,
        "min_infer_time_ms": min_infer_time*1000,
        "fps": fps
    }


# =========================
# 主函数
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GroundingDINO MIGraphX inference and benchmark")
    parser.add_argument("--model_path", required=True, help="Path to the compiled .mxr model")
    parser.add_argument("--image_path", default="../images/in/car_1.jpg", help="Input image path")
    parser.add_argument("--image_size", type=int, nargs=2, default=(800, 1200),
                        metavar=("HEIGHT", "WIDTH"), help="Preprocessing image size: height width")
    parser.add_argument("--save_path", default="../weights/result_migraphx.jpg", help="Output image path")
    parser.add_argument("--caption", default="car .", help="Text prompt (current token IDs support 'car .')")
    parser.add_argument("--device_id", type=int, default=0, help="MIGraphX device id")
    parser.add_argument("--warmup_runs", type=int, default=5)
    parser.add_argument("--test_runs", type=int, default=10)
    args = parser.parse_args()
    image_height, image_width = args.image_size

    if not os.path.isfile(args.model_path):
        parser.error(f"MXR model does not exist: {args.model_path}")
    if not args.model_path.endswith(".mxr"):
        parser.error("--model_path must point to a compiled .mxr model")
    if args.warmup_runs <= 0 or args.test_runs <= 0:
        parser.error("--warmup_runs and --test_runs must be greater than 0")

    model_path = "../weights/ground_bs1_opt.onnx"
    cache_path = args.model_path
    img_path = args.image_path

    TEXT_PROMPT = args.caption
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    
    WARMUP_RUNS = args.warmup_runs
    TEST_RUNS = args.test_runs

    model = MIGraphXModel(
        model_path,
        cache_path=cache_path,
        device_id=args.device_id,
        force_recompile=False  # 改成 True 可强制重编译
    )

    if (image_height, image_width) != (model.image_height, model.image_width):
        raise ValueError(
            f"Requested image size {image_height}x{image_width} does not match "
            f"the static MXR input size {model.image_height}x{model.image_width} (HxW)"
        )

    print("\n=== Inference configuration ===")
    print(f"Model: {args.model_path}")
    print(f"Batch size: {model.batch_size}")
    print(f"Image size (HxW): {image_height}x{image_width}")
    print(f"Input image: {img_path}")

    image_source, image = load_image(img_path, image_height, image_width)

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
    cv2.imwrite(args.save_path, ori_img)
    print(f"\n✅ 结果已保存至: {args.save_path}")
    print(f"✅ 检测到目标: {phrases} (共 {len(boxes)} 个)")
