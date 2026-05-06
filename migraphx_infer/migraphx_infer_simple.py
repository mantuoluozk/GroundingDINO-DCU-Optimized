import cv2
import numpy as np
import migraphx

"""
本示例演示了如何使用migraphx进行推理，主要步骤如下：
1. 加载模型
2. 获取模型输入输出节点信息
3. 编译模型
4. 为输出节点分配device内存，用于保存输出数据
5. 预处理并转换为NCHW
6. 将输入数据转换为device数据作为输入数据
7. 推理
"""

def ReadImage(pathOfImage,inputShape):
    srcImage = cv2.imread(pathOfImage, cv2.IMREAD_COLOR)
    # resize并转换为CHW
    resizedImage = cv2.resize(srcImage,(inputShape[3], inputShape[2]))
    resizedImage_Float = resizedImage.astype("float32") # 转换为float32
    srcImage_CHW = np.transpose(resizedImage_Float, (2, 0, 1)) # 转换为CHW
    # 预处理
    mean = np.array([127.5, 127.5, 127.5])
    scale = np.array([0.0078125, 0.0078125, 0.0078125])
    inputData = np.zeros(inputShape).astype("float32") # NCHW
    for i in range(srcImage_CHW.shape[0]):
        inputData[0,i, :, :] = (srcImage_CHW[i, :, :] - mean[i]) * scale[i]

    for i in range(inputData.shape[0]):
        if i!=0:
            inputData[i,:, :, :]=inputData[0,:, :, :]

    return inputData

def AllocateOutputMemory(model):
    outputData={}
    for key in model.get_outputs().keys():
        outputData[key] = migraphx.allocate_gpu(s=model.get_outputs()[key])
    return outputData

if __name__ == '__main__':

    # 加载模型
    model = migraphx.parse_onnx("ResNet50.onnx")
    # 获取模型输入输出节点信息
    print("inputs:")
    inputs=model.get_inputs()
    for key,value in inputs.items():
        print("{}:{}".format(key,value))
    print("outputs:")
    outputs=model.get_outputs()

    for key,value in outputs.items():
        print("{}:{}".format(key,value))

    inputName=list(model.get_inputs().keys())[0]
    inputShape=inputs[inputName].lens()

    # 编译
    model.compile(t=migraphx.get_target("gpu"),offload_copy=False,device_id=0)
    # 为输出节点分配device内存，用于保存输出数据
    modelData=AllocateOutputMemory(model)
    # 预处理并转换为NCHW
    pathOfImage ="Test.jpg"
    image = ReadImage(pathOfImage,inputShape)
    # 将输入数据转换为device数据作为输入数据
    modelData[inputName]=migraphx.to_gpu(migraphx.argument(image))
    # 推理
    results = model.run(modelData)
    # 获取输出节点属性
    result=migraphx.from_gpu(results[0]) # 将第一个输出节点的数据拷贝到host端,migraphx.argument类型
    outputShape=result.get_shape() # 输出节点的shape,migraphx.shape类型
    outputSize=outputShape.lens() # 每一维大小，维度顺序为(N,C,H,W),list类型
    numberOfOutput=outputShape.elements() # 输出节点元素的个数

    # 转换为numpy
    result = np.array(result)
    print(result)
