import onnxruntime as ort
from onnxruntime.transformers.optimizer import optimize_model

model_path = "../weights/ground_deform.onnx"
out_path = "../weights/ground_fused.onnx"
custom_op_lib = "../ort_plugin/build/libms_deform_attn_ort.so" 

print(f"🚀 准备黑入底层并注入自定义算子: {custom_op_lib}")

# =====================================================================
original_init = ort.InferenceSession.__init__

def patched_init(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()
    
    # 注入自定义算子
    sess_options.register_custom_ops_library(custom_op_lib)
    
    providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
    
    original_init(self, path_or_bytes, sess_options, providers, provider_options, **kwargs)

ort.InferenceSession.__init__ = patched_init
print("✅ 拦截器注入成功，已强行屏蔽 MIGraphX 干扰...")
# =====================================================================

try:
    # 正常调用优化器
    optimized_model = optimize_model(
        input=model_path,
        model_type='bert',  
        use_gpu=True  # 保持为 True，意思是让优化器“按 GPU 的胃口”去融合大算子
    )
    
    optimized_model.save_model_to_file(out_path)
    print(f"\n🎉 融合后的模型已保存至: {out_path}")

except Exception as e:
    print(f"\n❌ 优化失败: {e}")