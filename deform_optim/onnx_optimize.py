import onnx
from onnxsim import simplify
from onnxconverter_common import float16

onnx_model_path = "../weights/ground_deform.onnx"
sim_model_path = "../weights/ground_deform_sim.onnx"
fp16_model_path = "../weights/ground_deform_fp16_all.onnx"
custom_op_lib_path = "../ort_plugin_fp16/build/libms_deform_attn_ort.so" 

# ==========================================
# 第一步：ONNX Simplify (附带自定义算子库)
# ==========================================
# print("1️⃣ 正在进行 ONNX Simplify...")
# model = onnx.load(onnx_model_path)
# model_simp, check = simplify(model, custom_lib=custom_op_lib_path)

# if check:
#     onnx.save(model_simp, sim_model_path)
#     print(f"✅ Simplify 完成！已保存至 {sim_model_path}")
# else:
#     print("❌ Simplify 验证失败！")
#     exit()



# ==========================================
# 第二步：FP16 混合精度转换 (避开自定义算子)
# ==========================================
print("\n2️⃣ 正在进行 FP16 混合精度转换...")

# 重新加载 sim 后的模型
model_to_fp16 = onnx.load(sim_model_path)

original_cast_nodes = [node.name for node in model_to_fp16.graph.node if node.op_type == "Cast"]
print(f"🔍 查找到 {len(original_cast_nodes)} 个原生 Cast 节点，已全部加入保护名单。")


model_fp16 = float16.convert_float_to_float16(
    model_to_fp16,
    # op_block_list=["ms_deform_attn"],     # 屏蔽自定义的注意力算子, 如果是fp32版本自定义算子
    node_block_list=original_cast_nodes,  # 保护所有原生的 Cast 节点
    keep_io_types=True                # 保持整个模型的总输入/输出还是 FP32
)

onnx.save(model_fp16, fp16_model_path)
print(f"✅ FP16 转换完成！已保存至 {fp16_model_path}")

