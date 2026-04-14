import onnx
import numpy as np
from onnx import numpy_helper

model = onnx.load("weights/ground.onnx")

# 找所有 initializer
init_map = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

for node in model.graph.node:
    if node.op_type == "Gather":
        index_name = node.input[1]

        if index_name in init_map:
            idx = init_map[index_name]

            print("\n🚨 Gather index:", index_name)
            print("dtype:", idx.dtype)
            print("min:", idx.min())
            print("max:", idx.max())
            print("shape:", idx.shape)

            if (idx < 0).any():
                print("❌ NEGATIVE index")

            if (idx > 10000).any():
                print("❌ SUSPICIOUS LARGE index")