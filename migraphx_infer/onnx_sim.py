import onnx
from onnxsim import simplify
from onnxconverter_common import float16

onnx_model_path = "../weights/ground.onnx"
sim_model_path = "../weights/ground_sim.onnx"

print("1️⃣ 正在进行 ONNX Simplify...")
model = onnx.load(onnx_model_path)
model_simp, check = simplify(model)

if check:
    onnx.save(model_simp, sim_model_path)
    print(f"✅ Simplify 完成！已保存至 {sim_model_path}")
else:
    print("❌ Simplify 验证失败！")
    exit()