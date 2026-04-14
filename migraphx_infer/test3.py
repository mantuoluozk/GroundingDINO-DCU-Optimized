import onnx, onnx.external_data_helper as ex
import sys, os
print("🟢 python :", sys.executable)
print("🟢 onnx   :", onnx.__file__)          # 路径，确认是哪个包
print("🟢 version:", onnx.__version__)       # 必须是 >= 1.12
src = "weights/ground.onnx"
dst = "weights/ground_external.onnx"

m = onnx.load(src)
# 把每个 Tensor 都做 external data，统一写入 ground_weights.bin
ex.convert_model_to_external_data(m,
                                  all_tensors_to_one_file=True,
                                  location="ground_weights.bin")
onnx.save_model(m, dst)
print("[✅] external ONNX →", dst)