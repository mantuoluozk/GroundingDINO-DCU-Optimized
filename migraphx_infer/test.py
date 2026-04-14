import onnx

model = onnx.load("weights/ground.onnx")

for vi in model.graph.value_info:
    dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    if any(d == 0 for d in dims):
        print("⚠️ zero dim in value_info:", vi.name, dims)

for vi in model.graph.output:
    dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    if any(d == 0 for d in dims):
        print("⚠️ zero dim in output:", vi.name, dims)