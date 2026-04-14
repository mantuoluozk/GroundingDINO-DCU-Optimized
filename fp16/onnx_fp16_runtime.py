import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

# ===== 1. 路径 =====
input_model = "weights/ground.onnx"
output_model = "weights/ground_fp16.onnx"

# ===== 2. 加载 =====
model = onnx.load(input_model)

# ===== 3. 转换 =====
model_fp16 = convert_float_to_float16(
    model,
    keep_io_types=True,  # ⭐ 强烈建议
)

# ===== 4. 保存 =====
onnx.save(model_fp16, output_model)

print("✅ ONNXRuntime FP16 转换完成")