# -*- coding: utf-8 -*-
import cv2
import numpy as np
import migraphx


def ReadImage(pathOfImage, inputShape):
    """
    读取并预处理图像，转换为模型输入要求的NCHW格式
    
    Args:
        pathOfImage: 图像文件路径
        inputShape: 模型输入形状 (N, C, H, W)
    
    Returns:
        预处理后的图像数据，NCHW格式，float32类型
    """
    # 读取彩色图像
    srcImage = cv2.imread(pathOfImage, cv2.IMREAD_COLOR)
    if srcImage is None:
        raise ValueError(f"无法读取图像文件: {pathOfImage}")
    
    # resize到模型要求的尺寸 (W, H)
    resizedImage = cv2.resize(srcImage, (inputShape[3], inputShape[2]))
    
    # 转换为float32类型
    resizedImage_Float = resizedImage.astype("float32")
    
    # HWC -> CHW
    srcImage_CHW = np.transpose(resizedImage_Float, (2, 0, 1))
    
    # 预处理：减均值，乘缩放因子
    mean = np.array([127.5, 127.5, 127.5])
    scale = np.array([0.0078125, 0.0078125, 0.0078125])
    
    # 创建NCHW格式的输入数据
    inputData = np.zeros(inputShape).astype("float32")
    
    # 对每个通道进行预处理
    for i in range(srcImage_CHW.shape[0]):
        inputData[0, i, :, :] = (srcImage_CHW[i, :, :] - mean[i]) * scale[i]
    
    # 如果batch维度大于1，复制第一份数据填充（仅用于示例）
    for i in range(inputData.shape[0]):
        if i != 0:
            inputData[i, :, :, :] = inputData[0, :, :, :]
    
    return inputData


if __name__ == '__main__':
    # ====================== 1. 加载ONNX模型 ======================
    # try:
    #     model = migraphx.parse_onnx("ResNet50.onnx")
    # except Exception as e:
    #     raise RuntimeError(f"加载模型失败: {e}")
    # 加载ONNX模型（启用优化）
    print("🔍 加载ONNX模型")
    import onnxruntime as ort
    model_path = 'ResNet50.onnx'
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # 启用所有图优化
    sess_options.log_severity_level = 3  # 减少日志输出
    # sess_options.enable_profiling = True # 启用性能分析
    ort_session = ort.InferenceSession(model_path, 
                                       sess_options=sess_options, 
                                       providers=['CPUExecutionProvider, ROCMExecutionProvider'])

    # 查看当前执行引擎
    current_provider = ort_session.get_providers()
    print(f"✅ 模型加载完成 - 当前执行引擎: {current_provider}")

    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    output_name = ort_session.get_outputs()[0].name
    print(f"模型输入名称：{input_name}, 输入形状：{input_shape}")
    print(f"模型输出名称：{output_name}")
    # ====================== 4. 图像预处理 ======================
    pathOfImage = "../images/in/ImageNet_01.jpg"
    try:
        image = ReadImage(pathOfImage, input_shape)
    except Exception as e:
        raise RuntimeError(f"图像预处理失败: {e}")
    
    # outputs = ort_session.run({input_name: image})
    for i in range(3):  # 示例：运行10次
        outputs = ort_session.run([output_name], {input_name: image})

    import time
    for i in range(10):  # 示例：运行10次
        start_time = time.time()
        outputs = ort_session.run([output_name], {input_name: image})
        end_time = time.time()
        print(f"推理时间: {(end_time - start_time) * 1000:.2f} ms")