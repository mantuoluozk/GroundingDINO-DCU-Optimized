import torch
import torchvision

# Pytorch模型文件
pathOfPytorchModel = "resnet50-19c8e357.pth"
# 创建ResNet50模型
net = torchvision.models.resnet50(pretrained=False)
# 定义输入
input = torch.randn(32,3,224,224)
# 生成的ONNX模型的路径
pathOfONNX = "ResNet50.onnx"
net.load_state_dict(torch.load(pathOfPytorchModel))
net.eval()
# 导出ONNX模型
torch.onnx.export(net,input,pathOfONNX,input_names = ["input"])
