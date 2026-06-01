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

## 7\. migraphx推理

1. 环境准备(http://42.228.13.241:10068/wangwf/groundingdino)

* 下载 DTK-26.04-txpl-temp-0312-ubuntu20.04-x86_64(http://112.11.77.146:65182/jenkins/rocm/26.04/intel/ubuntu20.04/DTK-26.04-txpl-temp-0312-ubuntu20.04-x86_64.tar.gz) ，解压后替换掉原 /opt/dtk 目录。
```bash
tar -zxvf DTK-26.04-txpl-temp-0312-ubuntu20.04-x86_64.tar.gz -C /opt/
rm /opt/dtk # 删除原来的软链接
ln -s /opt/dtk-26.04-txpl-temp-0312 /opt/dtk # 创建新的软链接
```

* 替换hipdnn：
```bash
tar -zxvf package_resize.tar.gz
cd package_resize
cp -r install/lib/hipdnn_plugins /opt/dtk/lib/
cp -r install/lib/libhipdnn_backend.so /opt/dtk/lib/
rm -rf /opt/dtk/include/hipdnn
cp -r install/include/hipdnn /opt/dtk/include/hipdnn
rm -rf /opt/dtk/lib/cmake/hipdnn*
cp -r install/lib/cmake/* /opt/dtk/lib/cmake/
rm -f /opt/dtk/lib/hipdnn_plugins/engines/libmiopen_legacy_plugin.so
```

* 激活dtk
```bash
source /opt/dtk/env.sh
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}
```


* 安装migraphx
```bash
chmod +x ./migraphx-5.1.2+das.opt1.ab9210b.dtk2604-cp310-cp310-manylinux_2_35_x86_64.run
./migraphx-5.1.2+das.opt1.ab9210b.dtk2604-cp310-cp310-manylinux_2_35_x86_64.run
```


* 模型优化

```bash
cd weights
onnxsim ground.onnx ground_sim.onnx
cd migraphx_infer
python modify_onnx_0601.py ../weights/ground_sim.onnx ../weights/ground_opt.onnx
```



2. 性能测试(编译加运行)
```bash
export MIGRAPHX_ENABLE_GRAPHAPI_REDUCTION=1
export MIGRAPHX_ENABLE_LAYERNORM_FUSION=1
migraphx-driver perf --onnx ground_opt.onnx --fp16 --output ground_opt.mxr
```

或者进入migraphx_infer文件夹，运行
```bash
cd migraphx_infer
bash migraphx_export.bash
```

3. 如果已经得到了mxr文件，也可以直接测试
```bash
bash migraphx_perf.bash
```

4. 使用python脚本测试
```bash
python migraphx_infer.py  
# offload=False推理，提前开辟gpu空间，数据放在device推理

python migraphx_infer1.py 
# offload=True推理，会慢一些
```


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

### 8.3 migraphx BW150和BW100 测试结果
BW100示例结果：
```
Batch size: 1
Rate: 6.05197 inferences/sec
Total time: 165.235ms (Min: 165.115ms, Max: 165.535ms, 
Mean: 165.258ms, Median: 165.225ms)
Percentiles (90%, 95%, 99%): (165.358ms, 165.358ms, 165.358ms)
Total instructions time: 205.275ms
Overhead time: 2.32812ms, -40.0399ms
Overhead: 1%, -24%
```
汇总结果
| 设备 | 推理方式 | FPS | 平均推理时间 (ms) |
| :--- | :--- | :--- | :--- |
| BW150 | migraphx-driver | 14.93 | 66.97 |
| BW150 | Python + MIGraphX（device） | 13.65 | 73.20（包含前后处理） |
| BW100 | migraphx-driver | 13.54 | 73.87 |
| BW100 | Python + MIGraphX（device） | 12.12 | 82.44（包含前后处理） |


-----

## 参考项目

本项目在开发过程中参考了以下开源项目：

  - [**GroundingDINO**](https://github.com/IDEA-Research/GroundingDINO) - GroundingDINO 官方仓库，提供基础模型与算法实现。
  - [**GroundingDINO-TensorRT-and-ONNX-Inference**](https://github.com/wingdzero/GroundingDINO-TensorRT-and-ONNX-Inference) - 提供了 GroundingDINO 的 TensorRT 及 ONNX 推理部署参考实现。