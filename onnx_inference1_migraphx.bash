MIGRAPHX_LOG=debug migraphx-driver compile \
    --onnx weights/ground_external.onnx \
    --gpu \
    -p dead_code_elimination \
    --output weights/ground.mgx

    # -p eliminate_contiguous \
    # -p simplify_reshapes \
    # -p simplify_algebra \
    # -p eliminate_identity \
    # -p common_subexpression_elimination \