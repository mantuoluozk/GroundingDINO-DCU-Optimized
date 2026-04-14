import onnx
import onnxoptimizer as optimizer

def optimize_onnx_model(input_onnx_path, output_onnx_path, fp16=False):
    """
    优化ONNX模型，支持基础优化+可选FP16量化
    :param input_onnx_path: 原始ONNX文件路径
    :param output_onnx_path: 优化后保存路径
    :param fp16: 是否开启FP16量化（需GPU支持，速度提升更明显）
    """
    # 1. 加载原始模型
    model = onnx.load(input_onnx_path)
    
    # 2. 定义优化策略（按需增减，下面是通用高效的组合）
    optimize_passes = [
        "eliminate_deadend",          # 删除无输出的节点
        "eliminate_identity",         # 移除Identity算子
        "eliminate_unused_initializer",# 删除未使用的初始化参数
        "fuse_bn_into_conv",          # 将BN融合到Conv中（核心优化）
        "fuse_consecutive_concats"   # 融合连续的Concat算子
    ]
    
    # 3. 执行基础优化
    optimized_model = optimizer.optimize(model, optimize_passes)
    
    # 4. 可选：FP16量化（GPU推理提速显著，精度损失可控）
    # if fp16:
    #     from onnxconverter_common import float16
    #     optimized_model = float16.convert_float_to_float16(
    #         optimized_model,
    #         keep_io_types=True,  # 保持输入输出为FP32，避免数据类型兼容问题
    #         disable_shape_infer=False
    #     )
    
    # 5. 检查优化后模型合法性并保存
    onnx.checker.check_model(optimized_model)
    onnx.save(optimized_model, output_onnx_path)
    print(f"优化后的模型已保存至：{output_onnx_path}")

# 调用示例
if __name__ == "__main__":
    # 替换为你的ONNX文件路径
    input_onnx = "weights/ground_simplified.onnx"
    output_onnx = "weights/ground_simoptim.onnx"
    # 建议先试False（FP32），GPU环境再开True（FP16）
    optimize_onnx_model(input_onnx, output_onnx, fp16=False)