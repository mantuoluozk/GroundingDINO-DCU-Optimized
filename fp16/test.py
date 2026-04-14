import onnx
from onnx import TensorProto

# 加载你报错的FP16模型
model = onnx.load("weights/ground_fp16.onnx")

# 🔥 精准修复：找到报错的中间张量，强制修改类型为FP16
target_arg = "/backbone/backbone.0/Cast_output_0"

# 遍历模型所有张量类型声明，修复冲突项
for vi in model.graph.value_info:
    if vi.name == target_arg:
        vi.type.tensor_type.elem_type = TensorProto.FLOAT16
        print(f"✅ 已修复：{target_arg} 类型 → FP16")

# 额外校验+保存修复后的模型
onnx.checker.check_model(model)
onnx.save(model, "weights/ground_fp16_fixed.onnx")
print("\n🎉 模型修复完成！加载：ground_fp16_fixed.onnx")