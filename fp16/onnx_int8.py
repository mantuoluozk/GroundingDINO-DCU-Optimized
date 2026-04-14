from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="weights/ground.onnx",
    model_output="weights/ground_int8.onnx",
    weight_type=QuantType.QInt8,
)
print("int8 quantization done!")
