# GroundingDINO Inference & Deployment

本项目主要包含 GroundingDINO 的原生 PyTorch 推理以及基于 ONNX Runtime (ORT) 的推理部署流程，并针对 `ms_deform_attn` 算子提供了自定义的加速与量化方案。

## 1. 环境准备

本环境配置参考 [官方 GroundingDINO 仓库](https://github.com/IDEA-Research/GroundingDINO)。

### 1.1 基础环境变量设置
在进行编译和运行前，请先激活相关的计算栈环境并配置 HuggingFace 镜像：
```bash
source /opt/dtk/cuda/env.sh
export HF_ENDPOINT=https://hf-mirror.com
```

### 1.2 编译 GroundingDINO 库
进入项目目录并执行编译安装：
```bash
cd GroundingDINO/
pip3 install -e . --no-build-isolation
```

> **💡 注意：Numpy 版本兼容性问题**
> 如果在运行过程中遇到因 `numpy` 版本过高导致的报错，请强制重装指定版本：
> ```bash
> python3 -m pip install numpy==1.26.4 --force-reinstall
> ```

### 1.3 模型下载
新建weights文件夹并下载模型权重：
```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```



---

## 2. 推理模式前置配置

由于导出 ONNX 模型会改变模型的输入结构，在进行具体的推理操作前，**必须手动替换底层文件**以适配不同的推理后端。

目标文件路径：`groundingdino/models/GroundingDINO/groundingdino.py`

* **当使用 PyTorch 推理时**：
    将 `groundingdino_torch.py` 的内容复制并覆盖到上述目标文件中。
* **当使用 ONNX Runtime 导出与推理时**：
    将 `groundingdino_onnx.py` 的内容复制并覆盖到上述目标文件中。

---

## 3. PyTorch 原生推理

确保已按上述说明将代码切换为 Torch 模式后，直接运行测试脚本：
```bash
bash infer_test.sh
```

---

## 4. 标准 ONNX Runtime 推理

请先确保代码已切换为 ONNX 模式。本测试包含 5 轮预热（Warmup）和 10 轮正式测试。

**Step 1: 导出 ONNX 模型**
```bash
python export_onnx.py
```

**Step 2: 执行 ORT 推理**
```bash
python onnx_inference1.py
```
> **📝 参数说明**：
> 运行前请在 `onnx_inference1.py` 中填入对应的 `.onnx` 模型路径和图像路径。图片外的其他 Text/Mask 输入已自动生成并填入（可通过 `get_caption_mask.py` 生成）。

---

## 5. 进阶：带自定义算子 (ms_deform_attn) 的 ORT 推理

为了进一步优化性能，我们实现了 `ms_deform_attn` 的自定义算子，并提供了多种优化方案（含 FP16）。

### 5.1 编译自定义算子
根据需求选择对应的算子实现目录（支持的方案包括：`ort_plugin`, `ort_plugin_fp16`, `ort_plugin_fp16_B`, `ort_plugin_fp16_C`）。

以 `ort_plugin` 为例进行编译：
```bash
cd ort_plugin
mkdir build && cd build
cmake ..
make
```
编译成功后，将在该目录下生成动态链接库 `libms_deform_attn_ort.so`。

### 5.2 导出与优化模型
进入 deform 推理工作区：
```bash
cd deform_ort
```

**1. 导出带自定义算子的 ONNX：**
```bash
python export_onnx_deform.py
```

**2. 模型简化与量化：**
```bash
python onnx_optimize.py
```
> 该脚本会输出两种量化后的 ONNX 模型：
> - 一种**跳过了**自定义算子。
> - 另一种**保留了**自定义算子（需搭配不带 FP16 的 `.so` 库使用）。

### 5.3 运行自定义算子推理
最后，执行优化后的推理脚本（包含 5 轮预热，10 轮测试）：
```bash
python onnx_inference_deform_optim.py
```
> **📝 参数说明**：
> 运行前，请务必在代码中正确填入**ONNX 模型位置**以及**步骤 5.1 生成的自定义算子 `.so` 库位置**。

---

## 6. 测试结果对比
单张BW150，
图像输入800x1200，
batchsize=1

| 推理模式 | 优化方案 / 精度 | 预热次数 | 测试次数 | 平均推理延迟 (ms) | FPS |
| :--- | :--- | :---: | :---: | :---: | :--- |
| PyTorch | FP32 (Base) | 5 | 10 | 144.25 | 6.93 |
| ORT | 标准 ONNX | 5 | 10 | 173.66 | 5.76 |
| ORT + Deform Plugin | 动态库（ort_plugin） | 5 | 10 | 121.67 | 8.22 |
| ORT + Deform Plugin | FP16 混合精度量化（ort_plugin） | 5 | 10 | 95.17 | 10.5 |
| ORT + Deform Plugin | 纯FP16 方案 B（ort_plugin_fp16_B） | 5 | 10 | 87.34| 11.44 |
| ORT + Deform Plugin | 纯FP16 方案 C（ort_plugin_fp16_C） | 5 | 10 | 84.52 | 11.82 |