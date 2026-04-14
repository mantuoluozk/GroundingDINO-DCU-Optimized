import onnx
from onnx import numpy_helper

model = onnx.load("weights/ground.onnx")

for init in model.graph.initializer:
    if "Constant" in init.name:
        arr = numpy_helper.to_array(init)
        if arr.dtype in [np.int32, np.int64]:
            if (arr < 0).any() or (arr > 10000).any():
                print("🚨 可疑 index:", init.name, arr)