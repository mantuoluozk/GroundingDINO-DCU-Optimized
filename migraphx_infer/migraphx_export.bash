# export MIGRAPHX_TRACE_COMPILE=1
MIGRAPHX_ENABLE_GRAPHAPI_REDUCTION=1
MIGRAPHX_ENABLE_LAYERNORM_FUSION=1

migraphx-driver perf --onnx \
    ../test0525/ground_opt_0509.onnx \
    --fp16 \
    --output \
    ../test0525/ground_opt_0515.mxr

    # ../weights/ground_opt_0430.mxr > migraphx_log.log 2>&1