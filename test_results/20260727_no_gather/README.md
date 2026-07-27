# 关闭 Gather 优化后的回归测试

测试日期：2026-07-27

## 修改内容

`migraphx_infer/modify_onnx.py` 不再调用 `optimize_gather_nodes(om)`。该优化会将部分
Gather 节点替换为 Slice，并导致 800x1200、bs=4 模型输出置信度异常下降。

## 测试配置

- 输入图片：`images/in/car_1.jpg`
- 文本输入：`car .`
- 检测阈值：0.35
- 输入尺度：544x960、800x1200
- Batch size：1、2、4、8
- 精度：FP16
- Python 性能测试：20 次预热、100 次正式测试
- DCU：544x960 使用设备 4，800x1200 使用设备 5

## 目录内容

- `544x960/driver_bs*.txt`：生成 MXR 时的 `migraphx-driver perf` 关键性能字段
- `544x960/python_bs*.txt`：Python 推理链路原始输出
- `544x960/batch_samples_bs*.txt`：bs=2、4、8 的逐样本精度检查
- `544x960/result_bs*.jpg`：检测结果图
- `800x1200/driver_bs*.txt`：生成 MXR 时的 `migraphx-driver perf` 关键性能字段
- `800x1200/python_bs*.txt`：Python 推理链路原始输出
- `800x1200/batch_samples_bs*.txt`：bs=2、4、8 的逐样本精度检查
- `800x1200/result_bs*.jpg`：检测结果图

汇总数据及统计边界见项目根目录 `README.md`。
