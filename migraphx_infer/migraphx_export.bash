export MIGRAPHX_TRACE_COMPILE=1

migraphx-driver perf --onnx \
    ../weights/ground_opt_0430.onnx \
    --fp16 \
    --output \
    ../weights/ground_opt_0430.mxr

    # ../weights/ground_opt_0430.mxr > migraphx_log.log 2>&1