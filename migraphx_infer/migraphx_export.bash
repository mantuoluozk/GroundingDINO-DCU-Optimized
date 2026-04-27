export MIGRAPHX_ENABLE_MIOPEN_CONCAT=1
migraphx-driver perf --onnx \
    ../weights/ground_opt.onnx \
    --fp16 \
    --output \
    ../weights/ground_opt.mxr