# GroundingDINO Inference & Deployment

本项目主要包含 GroundingDINO 的原生 PyTorch 推理以及基于 ONNX Runtime (ORT) 的推理部署流程，并针对 `ms_deform_attn` 算子提供了自定义的加速与量化方案。

## 1\. 环境准备

本环境配置参考 [官方 GroundingDINO 仓库](https://github.com/IDEA-Research/GroundingDINO)，并针对海光 DCU 硬件环境进行了底层适配。

### 1.1 拉取 DCU 基础计算镜像

为确保底层 DTK 算子库与 PyTorch 版本完美匹配，推荐下面的容器内进行后续的代码编译与推理测试：

```bash
docker pull image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk25.04.2-py3.10
```

*(注：请基于该镜像启动开发容器后，再执行后续操作。)*

### 1.2 基础环境变量设置

在进行编译和运行前，请先激活相关的计算栈环境并配置 HuggingFace 镜像：

```bash
source /opt/dtk/cuda/cuda-12/env.sh
cp -r /opt/dtk/cuda/cuda-12/include /opt/dtk/cuda/cuda-12/include-bak
rm -rf /opt/dtk/cuda/cuda-12/include
export HF_ENDPOINT=https://hf-mirror.com
```

### 1.3 编译 GroundingDINO 库

进入项目目录并执行编译安装：

```bash
cd GroundingDINO/
pip3 install -e . --no-build-isolation
```

> **💡 注意：Numpy 版本兼容性问题**
> 如果在运行过程中遇到因 `numpy` 版本过高导致的报错，请强制重装指定版本：
>
> ```bash
> python3 -m pip install numpy==1.26.4 --force-reinstall
> ```

### 1.4 模型下载

新建 weights 文件夹并下载模型权重：

```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

-----

## 2\. 推理模式前置配置

由于导出 ONNX 模型会改变模型的输入结构，在进行具体的推理操作前，**必须手动替换底层文件**以适配不同的推理后端。

目标文件路径：`groundingdino/models/GroundingDINO/groundingdino.py`

  * **当使用 PyTorch 推理时**：
    将 `groundingdino_torch.py` 的内容复制并覆盖到上述目标文件中。
  * **当使用 ONNX Runtime 导出与推理时**：
    将 `groundingdino_onnx.py` 的内容复制并覆盖到上述目标文件中。

-----

## 3\. PyTorch 原生推理

确保已按上述说明将代码切换为 Torch 模式后，直接运行测试脚本：

```bash
bash infer_test.sh
```

-----

## 4\. 标准 ONNX Runtime 推理

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

-----

## 5\. 进阶：带自定义算子 (ms\_deform\_attn) 的 ORT 推理

为了进一步优化性能，实现了 `ms_deform_attn` 的自定义算子，并提供了多种优化方案（含 FP16）。

### 5.1 编译自定义算子

根据需求选择对应的算子实现目录（支持的方案包括：`ort_plugin`, `ort_plugin_fp16`, `ort_plugin_fp16_B`, `ort_plugin_fp16_C`），推荐使用 `ort_plugin`（fp32 算子实现）和 `ort_plugin_fp16_C`（fp16 算子实现）。

以 `ort_plugin` 为例进行编译：

```bash
cd ort_plugin
mkdir build && cd build
cmake ..
make
```

编译成功后，将在该目录下生成动态链接库 `libms_deform_attn_ort.so`，后续只需要在推理脚本中更改 so 文件位置。

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
>
>   - 一种**跳过了**自定义算子。
>   - 另一种**保留了**自定义算子（需搭配不带 FP16 的 `.so` 库使用）。


### 5.3 运行自定义算子推理

最后，执行优化后的推理脚本（包含 5 轮预热，10 轮测试）：

```bash
python onnx_inference_deform_optim.py
```

> **📝 参数说明**：
> 运行前，请务必在代码中正确填入**ONNX 模型位置**以及**步骤 5.1 生成的自定义算子 `.so` 库位置**。

## 6\. 低分辨率输入版本ORT推理（400x800）

如需使用更低分辨率的图像输入（如 400x800）以进一步加速推理，可按以下步骤操作：

1. 修改导出脚本

编辑 `deform_ort/export_onnx_deform.py`，修改图像尺寸与导出路径：

```python
# img = torch.randn(1, 3, 800, 1200).to(device)
img = torch.randn(1, 3, 400, 800).to(device)

# onnx_output_path = "../weights/ground_deform.onnx"
onnx_output_path = "../weights_400x800/ground_deform.onnx"
```

2. 正常导出并量化

```bash
cd deform_ort
python export_onnx_deform.py
python onnx_optimize.py
```

3. 修改推理预处理分辨率

编辑 `groundingdino/util/inference.py` 中的 `load_image` 函数，将 `RandomResize` 的参数从 800 改为 400：

```python
# T.RandomResize([800], max_size=1333),
T.RandomResize([400], max_size=1333),
```

4. 执行 ORT 推理

运行推理脚本，并确保代码中的 ONNX 模型路径指向 `weights_400x800/` 下对应的模型文件：

```bash
python onnx_inference_deform_optim.py
```

> **💡 提示**：低分辨率输入会显著减少推理耗时，但可能对检测精度（尤其小目标）产生影响，请根据实际场景权衡速度与精度。

-----

## 7\. MIGraphX 推理

### 7.1 环境准备

MIGraphX 推理环境使用 DTK 26.04.2、MIGraphX 5.1.2，并需要正确配置
hipDNN、hipBLASLt 及 PyTorch 动态库路径。完整安装包名称和替换步骤以交付环境中的
MIGraphX 环境配置说明为准。

进入容器后激活 DTK，并配置运行时动态库：

```bash
source /opt/dtk/env.sh
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}
```

运行模型编译或推理前，启用以下优化开关：

```bash
export MIGRAPHX_ENABLE_GRAPHAPI_REDUCTION=1
export MIGRAPHX_ENABLE_LAYERNORM_FUSION=1
export MIGRAPHX_ENABLE_GRAPHAPI_SOFTMAX=1
export MIGRAPHX_ENABLE_GEMM_ACT_FUSION=1
```

如需指定空闲物理 DCU，例如设备 5：

```bash
export HIP_VISIBLE_DEVICES=5
export ROCR_VISIBLE_DEVICES=5
```

### 7.2 生成优化 ONNX

以下命令以 batch size 2 为例。batch size 1、4、8 只需替换文件名中的数字：

```bash
cd /data/zk/GroundingDINO

onnxsim \
  weights/ground_bs2.onnx \
  weights/ground_bs2_sim.onnx

python migraphx_infer/modify_onnx_0720.py \
  weights/ground_bs2_sim.onnx \
  weights_0720/ground_bs2_opt.onnx
```

`modify_onnx_0720.py` 会从静态 ONNX 输入中自动解析 batch size。对于 batch size
大于 1 的模型，脚本会正确还原 attention head 数量，并跳过仅适用于 batch size 1
的自定义 GemmAct 图融合。

### 7.3 编译 MXR

```bash
cd /data/zk/GroundingDINO/migraphx_infer

migraphx-driver perf \
  --onnx ../weights_0720/ground_bs2_opt.onnx \
  --fp16 \
  --output ../weights_0720/ground_bs2_opt.mxr
```

MXR 是与输入 shape、DCU 架构及 MIGraphX/DTK 环境相关的静态编译产物。交付时应
记录 batch size、输入分辨率、目标 DCU 架构和软件版本，避免跨环境直接复用。

### 7.4 Python 推理与性能测试

`migraphx_infer/migraphx_infer.py` 不依赖已安装的 `groundingdino` Python 包，
图像预处理由 OpenCV 和 NumPy 完成。下面以 batch size 2、输入分辨率 800x1200
为例：

```bash
cd /data/zk/GroundingDINO/migraphx_infer

python migraphx_infer.py \
  --model_path ../weights_0720/ground_bs2_opt.mxr \
  --image_path ../images/in/car_1.jpg \
  --image_size 800 1200 \
  --caption "car ." \
  --save_path ../weights_0720/result_bs2.jpg \
  --warmup_runs 20 \
  --test_runs 100
```

主要参数：

| 参数 | 说明 |
| --- | --- |
| `--model_path` | 已编译的静态 `.mxr` 模型路径，必填 |
| `--image_path` | 输入图片路径 |
| `--image_size HEIGHT WIDTH` | 图像预处理尺度，顺序为高度、宽度 |
| `--caption` | 文本提示词；当前静态文本输入仅支持 `car .` |
| `--save_path` | 检测结果图保存路径 |
| `--warmup_runs` | 不计入性能结果的预热次数 |
| `--test_runs` | 正式性能测试次数 |

脚本会从 MXR 的 `img` 输入自动读取并打印 batch size 和静态输入分辨率。
`--image_size` 必须与模型编译时的输入分辨率一致，否则脚本将在推理前报错。

#### 推理链路端到端性能口径

图片读取、resize 和归一化在正式 benchmark 之前完成，不计入性能数据。正式统计
范围为：

```text
batch 图像张量和固定文本输入构造
→ CPU 到 GPU
→ MIGraphX 模型执行
→ GPU 到 CPU
→ sigmoid 和阈值后处理
```

模型加载、图片读取/解码、resize、归一化、预热、检测框绘制和结果图片写盘均不计入
该指标。该结果应在交付报告中称为“推理链路端到端延迟与吞吐量”，不能简称为包含
图片 I/O 和预处理的应用端到端性能。

bs>1 时，脚本会把同一张预处理图片和同一组固定文本输入复制到整个 batch；结果图
和检测后处理使用 batch 中第一项。

吞吐量计算方式为：

```text
images/s = batch_size * test_runs / 正式测试推理链路总耗时
```

报告同时输出平均延迟、标准差、P50、P90、P95、最小/最大延迟和 images/s。
交付测试报告中应同时记录测试次数、设备编号、batch size 和输入分辨率。


-----
## 8\. 测试结果对比

*以下测试均包含 5 轮预热（Warmup）和 10 轮正式测试。*

> **📌 路径简写说明：**
> 为保证表格排版简洁，下方表格中的路径已作简写：
>
>   * **模型文件**：默认存放于 `../weights/` 目录下。
>   * **自定义算子目录**：对应的完整动态库路径均为 `../[目录名]/build/libms_deform_attn_ort.so`。

### 8.1 ORT BW150 测试结果

单张 BW150 卡，图像输入 800x1200，Batch Size = 1

| 推理模型 | 优化方案 / 精度 | 模型文件 | 自定义算子目录 | 推理时间 (ms) | FPS |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **PyTorch** | FP32 (Base) 原生推理 | - | - | 144.25 | 6.93 |
| **ORT** | 标准 ONNX 推理 (原始模型) | `ground.onnx` | - | 173.66 | 5.76 |
| **ORT + Plugin** | +自定义算子<br>+前后处理、模型简化 | `ground_deform.onnx` | `ort_plugin` | 121.67 | 8.22 |
| **ORT + Plugin** | +自定义算子<br>+FP16 混合精度量化 | `ground_deform_fp16.onnx` | `ort_plugin` | 95.17 | 10.50 |
| **ORT + Plugin** | +自定义算子<br>+FP16 纯量化方案 B | `ground_deform_fp16_all.onnx` | `ort_plugin_fp16_B` | 87.34 | 11.44 |
| **ORT + Plugin** | +自定义算子<br>+FP16 极致优化方案 C | `ground_deform_fp16_all.onnx` | `ort_plugin_fp16_C` | 84.52 | 11.82 |

### 8.2 ORT BW100 测试结果

单张 BW100 卡，图像输入 800x1200，Batch Size = 1

| 推理模型 | 优化方案 / 精度 | 模型文件 | 自定义算子目录 | 推理时间 (ms) | FPS |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **ORT** | 标准 ONNX 推理 (原始模型) | `ground.onnx` | - | 204.49 | 4.89 |
| **ORT + Plugin** | +自定义算子<br>+前后处理、模型简化 | `ground_deform.onnx` | `ort_plugin` | 136.25 | 7.34 |
| **ORT + Plugin** | +自定义算子<br>+FP16 混合精度量化 | `ground_deform_fp16.onnx` | `ort_plugin` | 105.46 | 9.48 |
| **ORT + Plugin** | +自定义算子<br>+FP16 纯量化方案 B | `ground_deform_fp16_all.onnx` | `ort_plugin_fp16_B` | 105.35 | 9.49 |
| **ORT + Plugin** | +自定义算子<br>+FP16 极致优化方案 C | `ground_deform_fp16_all.onnx` | `ort_plugin_fp16_C` | 100.91 | 9.90 |

### 8.3 migraphx BW150、BW100、K100_AI 测试结果


汇总结果
| 设备 | 推理方式 | FPS | 
| :--- | :--- | :--- | 
| BW150 | migraphx-driver | 26.51 | 
| BW150 | Python + MIGraphX（device） | 23.02 | 
| BW100 | migraphx-driver | 22.93 | 
| BW100 | Python + MIGraphX（device） | 19.39 | 
| K100_AI | migraphx-driver | 19.26 | 
| K100_AI | Python + MIGraphX（device） | 15.90 | 


-----

## 参考项目

本项目在开发过程中参考了以下开源项目：

  - [**GroundingDINO**](https://github.com/IDEA-Research/GroundingDINO) - GroundingDINO 官方仓库，提供基础模型与算法实现。
  - [**GroundingDINO-TensorRT-and-ONNX-Inference**](https://github.com/wingdzero/GroundingDINO-TensorRT-and-ONNX-Inference) - 提供了 GroundingDINO 的 TensorRT 及 ONNX 推理部署参考实现。
