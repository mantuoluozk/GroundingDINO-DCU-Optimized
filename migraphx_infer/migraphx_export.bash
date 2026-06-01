# export MIGRAPHX_TRACE_COMPILE=1
MIGRAPHX_ENABLE_GRAPHAPI_REDUCTION=1
MIGRAPHX_ENABLE_LAYERNORM_FUSION=1

migraphx-driver perf --onnx \
    ../weights/ground_opt_0601.onnx \
    --fp16 \
    --output \
    ../weights/ground_opt_0601.mxr

    # ../weights/ground_opt_0430.mxr > migraphx_log.log 2>&1