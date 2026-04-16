# 该文件夹下均为自定义算子版本相关脚本

## 1.默认图像大小800x1200

1. 导出onnx
```bash
python export_onnx_deform.py
```
2. 优化并量化onnx
```bash
python onnx_optimize.py
```
3. 基础版本ort推理
```bash
python onnx_inference_deform.py
```
4. 优化版本ort推理
```bash
python onnx_inference_deform_optim.py
```
5. 优化版本ort推理+iobinding(该项目下无提升)
```bash
python onnx_inference_deform_optim_iobinding.py 
```

## 2.当模型图像输入改为400x600

1. 先修改export_onnx_deform.py中
```bash
# img = torch.randn(1, 3, 800, 1200).to(device)
img = torch.randn(1, 3, 400, 600).to(device)

# 导出 ONNX
# onnx_output_path = "../weights/ground_deform.onnx"
onnx_output_path = "../weights_400x600/ground_deform.onnx"
```

2. 正常导出并优化量化

3. 修改groundingdino/util/inference.py中load_image函数，800改为400

```bash
T.RandomResize([400], max_size=1333),
```
4. 正常ort推理，修改调用onnx路径