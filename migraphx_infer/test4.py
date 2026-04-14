import onnx
from onnx import shape_inference

model = onnx.load("weights/ground.onnx")
model = shape_inference.infer_shapes(model)
onnx.save(model, "weights/ground_shape.onnx")