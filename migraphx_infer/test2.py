import onnx
from onnx import numpy_helper

# 加载你的模型
model = onnx.load("weights/ground_sim_fp16.onnx")

print("=== 检查所有常量张量大小 ===")
for init in model.graph.initializer:
    name = init.name
    shape = tuple(init.dims)
    # 计算元素个数
    elem_count = 1
    for d in shape:
        elem_count *= d
    # 计算大小（MB）
    dtype_size = onnx.helper.tensor_dtype_to_np_dtype(init.data_type).itemsize
    size_mb = (elem_count * dtype_size) / (1024 * 1024)
    
    # 只打印 >10MB 的常量（你可以改阈值）
    if size_mb > 10:
        print(f"⚠️  超大常量：{name}")
        print(f"   形状：{shape}")
        print(f"   大小：{size_mb:.2f} MB\n")
"""
=== ground.onnx检查所有常量张量大小 ===
⚠️  超大常量：bert.embeddings.word_embeddings.weight
   形状：(30522, 768)
   大小：89.42 MB

⚠️  超大常量：onnx::MatMul_25479
   形状：(1024, 3072)
   大小：12.00 MB

⚠️  超大常量：onnx::MatMul_25503
   形状：(1024, 4096)
   大小：16.00 MB

⚠️  超大常量：onnx::MatMul_25504
   形状：(4096, 1024)
   大小：16.00 MB

⚠️  超大常量：onnx::MatMul_25513
   形状：(1024, 3072)
   大小：12.00 MB

⚠️  超大常量：onnx::MatMul_25541
   形状：(1024, 4096)
   大小：16.00 MB

⚠️  超大常量：onnx::MatMul_25542
   形状：(4096, 1024)
   大小：16.00 MB

   ground_simplified.onnx
=== 检查所有常量张量大小 ===
⚠️  超大常量：bert.embeddings.word_embeddings.weight
   形状：(30522, 768)
   大小：89.42 MB

⚠️  超大常量：onnx::MatMul_25479
   形状：(1024, 3072)
   大小：12.00 MB

⚠️  超大常量：onnx::MatMul_25503
   形状：(1024, 4096)
   大小：16.00 MB

⚠️  超大常量：onnx::MatMul_25504
   形状：(4096, 1024)
   大小：16.00 MB

⚠️  超大常量：onnx::MatMul_25513
   形状：(1024, 3072)
   大小：12.00 MB

⚠️  超大常量：onnx::MatMul_25541
   形状：(1024, 4096)
   大小：16.00 MB

⚠️  超大常量：onnx::MatMul_25542
   形状：(4096, 1024)
   大小：16.00 MB

⚠️  超大常量：/backbone/backbone.0/layers.0/blocks.1/attn/Unsqueeze_7_output_0
   形状：(1, 425, 1, 144, 144)
   大小：33.62 MB

⚠️  超大常量：/transformer/Concat_10_output_0
   形状：(1, 19947, 256)
   大小：19.48 MB

⚠️  超大常量：/transformer/enc_out_class_embed/ConstantOfShape_output_0
   形状：(1, 19947, 256)
   大小：19.48 MB   
   
"""

"""=== ground_fp16.onnx检查所有常量张量大小 ===
⚠️  超大常量：bert.embeddings.word_embeddings.weight
   形状：(30522, 768)
   大小：44.71 MB
"""

""" ground_sim_fp16.onnx
=== 检查所有常量张量大小 ===
⚠️  超大常量：bert.embeddings.word_embeddings.weight
   形状：(30522, 768)
   大小：44.71 MB

⚠️  超大常量：/backbone/backbone.0/layers.0/blocks.1/attn/Unsqueeze_7_output_0
   形状：(1, 425, 1, 144, 144)
   大小：16.81 MB
"""