import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np


def convert_fp16_manual(input_path, output_path, keep_io_types=True):
    model = onnx.load(input_path)
    graph = model.graph

    fp32 = TensorProto.FLOAT
    fp16 = TensorProto.FLOAT16

    # ========== 1. 收集所有 name -> type ==========
    type_map = {}

    for init in graph.initializer:
        type_map[init.name] = init.data_type

    for inp in graph.input:
        type_map[inp.name] = inp.type.tensor_type.elem_type

    for out in graph.output:
        type_map[out.name] = out.type.tensor_type.elem_type

    # ========== 2. Initializer: FP32 -> FP16 ==========
    for i, init in enumerate(graph.initializer):
        if init.data_type == fp32:
            arr = numpy_helper.to_array(init)
            # 处理 inf / -inf / 超大值
            arr = np.clip(arr, -65504, 65504)
            arr = arr.astype(np.float16)
            new_init = numpy_helper.from_array(arr, init.name)
            graph.initializer[i].CopyFrom(new_init)
            type_map[init.name] = fp16

    # ========== 3. Constant 节点: FP32 -> FP16 ==========
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.t.data_type == fp32:
                arr = numpy_helper.to_array(attr.t)
                arr = np.clip(arr, -65504, 65504).astype(np.float16)
                attr.t.CopyFrom(numpy_helper.from_array(arr))
                type_map[node.output[0]] = fp16

    # ========== 4. 遍历节点，插入 Cast ==========
    new_nodes = []
    cast_id = [0]

    # 需要保持 FP32 的 op（不转其输出）
    fp32_ops = {"Shape", "NonMaxSuppression", "Range",
                "TopK", "SequenceConstruct", "SequenceEmpty"}

    for node in graph.node:
        if node.op_type == "Constant":
            new_nodes.append(node)
            continue

        # 这些 op 输出本身就是整数或索引，跳过
        if node.op_type in fp32_ops:
            new_nodes.append(node)
            for o in node.output:
                type_map[o] = fp32  # 标记为 FP32（实际是 int64 等）
            continue

        # ---- 找目标类型：用第一个已知输入的类型 ----
        target = None
        for inp_name in node.input:
            if inp_name and inp_name in type_map:
                t = type_map[inp_name]
                if t in (fp32, fp16):
                    target = t
                    break

        # 默认目标类型 = FP16
        if target is None:
            target = fp16

        # ---- 对每个输入做类型检查 ----
        for idx, inp_name in enumerate(node.input):
            if not inp_name or inp_name not in type_map:
                continue
            inp_type = type_map[inp_name]

            # 输入是 FP32，目标是 FP16 -> 插 Cast to FP16
            if inp_type == fp32 and target == fp16:
                cast_out = f"_cast_{cast_id[0]}"
                cast_id[0] += 1
                cast_node = helper.make_node(
                    "Cast", inputs=[inp_name], outputs=[cast_out], to=fp16
                )
                new_nodes.append(cast_node)
                node.input[idx] = cast_out
                type_map[cast_out] = fp16

            # 输入是 FP16，目标是 FP32 -> 插 Cast to FP32
            elif inp_type == fp16 and target == fp32:
                cast_out = f"_cast_{cast_id[0]}"
                cast_id[0] += 1
                cast_node = helper.make_node(
                    "Cast", inputs=[inp_name], outputs=[cast_out], to=fp32
                )
                new_nodes.append(cast_node)
                node.input[idx] = cast_out
                type_map[cast_out] = fp32

        new_nodes.append(node)

        # ---- 更新输出类型 ----
        for o in node.output:
            type_map[o] = target

    # ========== 5. 替换节点 ==========
    del graph.node[:]
    graph.node.extend(new_nodes)

    # ========== 6. 修复 graph output 类型声明 ==========
    if keep_io_types:
        # 保持原始 IO 类型为 FP32
        # 输出需要 Cast 回 FP32
        for out in graph.output:
            if out.name in type_map and type_map[out.name] == fp16:
                cast_out = f"_cast_out_{out.name}"
                cast_node = helper.make_node(
                    "Cast", inputs=[cast_out], outputs=[out.name], to=fp32
                )
                # 重命名原始输出
                # 先找到最后产生这个输出的节点，改其输出名
                for node in graph.node:
                    for i, o in enumerate(node.output):
                        if o == out.name:
                            node.output[i] = cast_out
                            break
                graph.node.append(cast_node)
                type_map[out.name] = fp32
    else:
        # 输出也改为 FP16
        for out in graph.output:
            if out.name in type_map:
                out.type.tensor_type.elem_type = type_map[out.name]

    # ========== 7. 验证 ==========
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"✅ 转换完成 -> {output_path}")
    print(f"   节点数: {len(graph.node)}")
    print(f"   Cast 插入数: {cast_id[0]}")


# ========== 运行 ==========
convert_fp16_manual(
    "weights/ground.onnx",
    "weights/ground_fp16.onnx",
    keep_io_types=True,
)
