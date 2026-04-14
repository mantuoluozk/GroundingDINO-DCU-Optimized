import onnx

model = onnx.load("weights/ground_simplified.onnx")

# 基本信息
print(f"模型名称: {model.graph.name}")
print(f"opset 版本: {model.opset_import[0].version}")

# 输入
print("\n=== 输入 ===")
for inp in model.graph.input:
    shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {inp.type.tensor_type.elem_type}, shape={shape}")

# 输出
print("\n=== 输出 ===")
for out in model.graph.output:
    shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: {out.type.tensor_type.elem_type}, shape={shape}")

# 统计算子类型
from collections import Counter
op_counts = Counter(node.op_type for node in model.graph.node)
print("\n=== 算子统计 (前20) ===")
for op, count in op_counts.most_common(20):
    print(f"  {op}: {count}")

# 检查是否有控制流算子
control_ops = [op for op in op_counts if op in ["If", "Loop", "Scan", "SequenceMap"]]
if control_ops:
    print(f"\n⚠️  包含控制流算子: {control_ops}")

'''
模型名称: main_graph
opset 版本: 17

=== 输入 ===
  img: 1, shape=[1, 3, 800, 1200]
  input_ids: 7, shape=[1, 4]
  attention_mask: 9, shape=[1, 4]
  position_ids: 7, shape=[1, 4]
  token_type_ids: 7, shape=[1, 4]
  text_token_mask: 9, shape=[1, 4, 4]

=== 输出 ===
  logits: 1, shape=['Gatherlogits_dim_0', 'Gatherlogits_dim_1', 'Gatherlogits_dim_2']
  boxes: 1, shape=['Gatherboxes_dim_0', 'Gatherboxes_dim_1', 4]

=== 算子统计 (前20) ===
  Constant: 7315
  Unsqueeze: 1919
  Concat: 1051
  Reshape: 916
  Shape: 843
  Gather: 762
  Add: 716
  Slice: 603
  MatMul: 528
  Mul: 513
  Transpose: 507
  Cast: 459
  Div: 265
  Where: 230
  Expand: 223
  ConstantOfShape: 218
  Equal: 183
  LayerNormalization: 147
  Sub: 79
  Softmax: 78

  # 经过简化后：
  === 输入 ===
  img: 1, shape=[1, 3, 800, 1200]
  input_ids: 7, shape=[1, 4]
  attention_mask: 9, shape=[1, 4]
  position_ids: 7, shape=[1, 4]
  token_type_ids: 7, shape=[1, 4]
  text_token_mask: 9, shape=[1, 4, 4]

=== 输出 ===
  logits: 1, shape=[1, 900, 256]
  boxes: 1, shape=[1, 900, 4]

=== 算子统计 (前20) ===
  Reshape: 703
  Add: 679
  MatMul: 527
  Transpose: 459
  Mul: 204
  Slice: 194
  Gather: 155
  Unsqueeze: 152
  LayerNormalization: 147
  Concat: 97
  Div: 96
  Softmax: 78
  Clip: 57
  Relu: 48
  GridSample: 48
  Sub: 36
  Erf: 36
  Where: 35
  Pad: 25
  Sin: 25
  
  
  '''