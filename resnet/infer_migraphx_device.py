# -*- coding: utf-8 -*-
import cv2
import numpy as np
import migraphx
import time
import os


def ReadImage(pathOfImage, inputShape):
    """
    读取并预处理图像，转换为模型输入要求的NCHW格式
    """
    srcImage = cv2.imread(pathOfImage, cv2.IMREAD_COLOR)
    if srcImage is None:
        raise ValueError(f"无法读取图像文件: {pathOfImage}")
    
    resizedImage = cv2.resize(srcImage, (inputShape[3], inputShape[2]))
    resizedImage_Float = resizedImage.astype("float32")
    srcImage_CHW = np.transpose(resizedImage_Float, (2, 0, 1))
    
    mean = np.array([127.5, 127.5, 127.5])
    scale = np.array([0.0078125, 0.0078125, 0.0078125])
    
    inputData = np.zeros(inputShape).astype("float32")
    for i in range(srcImage_CHW.shape[0]):
        inputData[0, i, :, :] = (srcImage_CHW[i, :, :] - mean[i]) * scale[i]
        
    for i in range(inputData.shape[0]):
        if i != 0:
            inputData[i, :, :, :] = inputData[0, :, :, :]
    
    return inputData

def AllocateOutputMemory(model):
    outputData={}
    for key in model.get_outputs().keys():
        outputData[key] = migraphx.allocate_gpu(s=model.get_outputs()[key])
    return outputData


if __name__ == '__main__':
    # ====================== 模型路径配置 ======================
    onnx_path = "ResNet50.onnx"
    mxr_path = "ResNet50_gpu.mxr"

    # ====================== MIGraphX 加载 / 编译 / 保存 mxr ======================
    print("🔍 加载ONNX模型")
    
    if os.path.exists(mxr_path):
        # ✅ 正确加载 mxr
        model = migraphx.load(mxr_path)
        print("✅ 从缓存加载编译好的模型 (mxr)")
    else:
        # ✅ 正确编译 + 保存 mxr（修复了 save 报错）
        model = migraphx.parse_onnx(onnx_path)
        model.compile(migraphx.get_target("gpu"), offload_copy=False, device_id=0)
        # 👇 这里是正确写法！不是 model.save()
        migraphx.save(model, mxr_path)
        print("✅ 模型编译完成，并已保存为 mxr")

    print(f"✅ 模型加载完成 - 当前执行引擎: ['ROCMExecutionProvider', 'CPUExecutionProvider'] (MIGraphX GPU)")

    # ====================== 获取输入输出信息 ======================
    input_name = list(model.get_inputs().keys())[0]
    input_shape = model.get_inputs()[input_name].lens()
    output_name = list(model.get_outputs().keys())[0]

    print(f"模型输入名称：{input_name}, 输入形状：{input_shape}")
    print(f"模型输出名称：{output_name}")

    # 为输出节点分配device内存，用于保存输出数据
    modelData=AllocateOutputMemory(model)

    # ====================== 图像预处理 ======================
    pathOfImage = "../images/in/ImageNet_01.jpg"
    image = ReadImage(pathOfImage, input_shape)

    # 将输入数据转换为device数据作为输入数据
    modelData[input_name]=migraphx.to_gpu(migraphx.argument(image))

    # ====================== 预热 3 次 ======================
    for i in range(3):
        outputs = model.run({input_name: image})

    # ====================== 推理测速 10 次 ======================
    for i in range(10):
        start_time = time.time()
        outputs = model.run({input_name: image})
        end_time = time.time()
        print(f"推理时间: {(end_time - start_time) * 1000:.2f} ms")