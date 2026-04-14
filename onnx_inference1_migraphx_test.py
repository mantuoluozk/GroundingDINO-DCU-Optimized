import migraphx as mgx

p = mgx.parse_onnx("weights/ground_external.onnx")   # 只读取，不优化
passes = [
    mgx.pass_dead_code_elimination(),               # 删除未使用的节点/常量
    mgx.pass_eliminate_contiguous(),                # 合并相邻的 contiguous 操作
    mgx.pass_simplify_reshapes(),                   # 合并/简化 reshape
    mgx.pass_simplify_algebra(),                    # 简化代数表达式 (add/mul/..)
    mgx.pass_eliminate_identity(),                  # 删除 Identity ops
    mgx.pass_common_subexpression_elimination(),    # CSE
]
p.apply_passes(passes)            # 手动执行
p.compile(mgx.target("gpu"))
p.save("weights/ground.mgx")