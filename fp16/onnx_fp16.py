import onnx
from onnxconverter_common import float16

# 1. 加载模型
model = onnx.load("weights/ground.onnx")

# 2. 转换为 FP16
model_fp16 = float16.convert_float_to_float16(
    model,    
    keep_io_types=True,
    # op_block_list=["Cast"]
)

# 3. 验证模型
onnx.checker.check_model(model_fp16)

# 4. 保存
onnx.save(model_fp16, "weights/ground_fp16.onnx")
print("FP16 model saved!")
