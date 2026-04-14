import onnxruntime as ort
from onnxruntime.transformers.optimizer import optimize_model

# 1. 路径配置（指向你刚刚生成的纯血 FP16 模型）
model_path = "../weights/ground_deform_fp16_all.onnx"
out_path = "../weights/ground_deform_fused_final.onnx"

# 【关键】确保这个路径指向你刚刚编译好的最新 FP16 .so 文件
custom_op_lib = "../ort_plugin_fp16/build/libms_deform_attn_ort.so" 

print(f"🚀 准备黑入底层并注入自定义算子: {custom_op_lib}")

# =====================================================================
# 💀 核心魔法：Monkey Patching (猴子补丁)
# 拦截 ORT 优化器内部的 Session 创建，强行注入 .so 库，并精准踢出 MIGraphX
# =====================================================================
original_init = ort.InferenceSession.__init__

def patched_init(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()
    
    # 强行注入自定义算子
    sess_options.register_custom_ops_library(custom_op_lib)
    
    # 【极其关键】：只允许 ROCm 和 CPU，强行踢掉 MIGraphX 防止离线推导时崩溃！
    providers = ['ROCMExecutionProvider', 'CPUExecutionProvider'] 
    
    original_init(self, path_or_bytes, sess_options, providers, provider_options, **kwargs)

ort.InferenceSession.__init__ = patched_init
print("✅ 拦截器注入成功，已精确屏蔽 MIGraphX 干扰，保留纯净 ROCm...")
# =====================================================================

try:
    # 2. 召唤官方 Transformer 优化引擎
    optimized_model = optimize_model(
        input=model_path,
        model_type='bert',  # BERT 拓扑图匹配 (匹配 Attention 和 LayerNorm)
        use_gpu=True        # 保持按 GPU 的标准进行大算子融合
    )
    
    # 3. 保存优化后的“超级计算图”
    optimized_model.save_model_to_file(out_path)
    print(f"\n🎉 大功告成！融合后的超级模型已保存至: {out_path}")
    print("👉 现在把测速脚本里的模型改成这个，去测最终的极限 FPS 吧！")

except Exception as e:
    print(f"\n❌ 优化失败: {e}")