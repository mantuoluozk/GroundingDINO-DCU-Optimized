import onnx
from onnx import helper

INPUT_MODEL = "weights/ground_simplified.onnx"
OUTPUT_MODEL = "weights/ground_fix.onnx"


def add_identity(graph, input_name, suffix, new_nodes, processed):
    if input_name in processed:
        return input_name + suffix

    new_name = input_name + suffix

    identity_node = helper.make_node(
        "Identity",
        inputs=[input_name],
        outputs=[new_name],
        name=input_name + suffix + "_identity"
    )

    new_nodes.append(identity_node)
    processed.add(input_name)

    return new_name


def patch_model(model):
    graph = model.graph

    new_nodes = []
    processed = set()

    for node in graph.node:

        # ✅ 1. 处理 Gather（你之前做的）
        if node.op_type == "Gather":
            idx = node.input[1]
            node.input[1] = add_identity(graph, idx, "_block", new_nodes, processed)

        # ✅ 2. 🔥 关键：处理 ScatterND
        if node.op_type.lower().startswith("scatter"):
            # scatternd(data, indices, updates)
            data = node.input[0]
            indices = node.input[1]
            updates = node.input[2]

            node.input[0] = add_identity(graph, data, "_block", new_nodes, processed)
            node.input[1] = add_identity(graph, indices, "_block", new_nodes, processed)
            node.input[2] = add_identity(graph, updates, "_block", new_nodes, processed)

        # ✅ 3. where（也可能触发 constant folding）
        if node.op_type == "Where":
            for i in range(3):
                node.input[i] = add_identity(graph, node.input[i], "_block", new_nodes, processed)

    # 插入到最前面
    for i, n in enumerate(new_nodes):
        graph.node.insert(i, n)

    return model


def main():
    print("🔍 加载模型...")
    model = onnx.load(INPUT_MODEL)

    print("⚙️ 全面阻断 constant folding（Gather + ScatterND + Where）...")
    model = patch_model(model)

    print("💾 保存模型...")
    onnx.save(model, OUTPUT_MODEL)

    print("✅ 完成:", OUTPUT_MODEL)


if __name__ == "__main__":
    main()