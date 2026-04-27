import cv2
import numpy as np
import time
import os
import migraphx
from typing import Tuple
import torch
import groundingdino.datasets.transforms as T
from PIL import Image

def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            # T.RandomResize([400], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _mgx_shape_to_numpy(shape):
    shape_str = str(shape)
    if "int64_type" in shape_str:
        dtype = np.int64
    elif "bool_type" in shape_str:
        dtype = np.bool_
    elif "half_type" in shape_str:
        dtype = np.float16
    else:
        dtype = np.float32
    try:
        dims = list(shape.dims())
    except Exception:
        dims = []
    try:
        lens = list(shape.lens())
    except Exception:
        lens = []
    return dtype, (dims if len(dims) > 0 else lens)
    

# =========================
# 🚀 MIGraphX 推理类（带缓存与生命周期管理）
# =========================
class MIGraphXModel:
    def __init__(self, onnx_path, cache_path="weights/ground_opt.mxr", force_recompile=False, device_id=0):
        self.cache_path = cache_path

        if os.path.exists(cache_path) and not force_recompile:
            print(f"⚡ 直接加载已编译模型: {cache_path}")
            self.model = migraphx.load(cache_path)
        else:
            print("🔍 从 ONNX 构建 MIGraphX")
            self.model = migraphx.parse_onnx(onnx_path) 
            
            print(f"⚙️ 编译 MIGraphX（GPU {device_id}）")
            self.model.compile(t=migraphx.get_target("gpu"), device_id=device_id)
            
            print(f"💾 保存编译模型到: {cache_path}")
            migraphx.save(self.model, cache_path)

        self.input_shapes = self.model.get_inputs()

    def infer(self, input_dict):
        mgx_inputs = {}
        
        # 【关键修复区】：用于保持 NumPy 数组存活，防止 Python 垃圾回收导致底层指针失效
        self._keep_alive_cache = {} 
        
        provided_names = set(input_dict.keys())
        required_names = {
            k for k in self.input_shapes.keys()
            if not str(k).startswith("main:#output")
        }

        for name in required_names:
            shape = self.input_shapes[name]
            target_dtype, lens = _mgx_shape_to_numpy(shape)
            
            if name in provided_names:
                # 1. 必须转为连续内存！防止 PyTorch 转过来的 array 内存步长不一致
                arr = np.ascontiguousarray(input_dict[name])
                # 2. 强制类型转换
                if arr.dtype != target_dtype:
                    arr = arr.astype(target_dtype)
            else:
                # 缺失的输入用 0 补齐
                arr = np.zeros(lens, dtype=target_dtype)
            
            # 3. 将数组塞进字典，强行续命！
            self._keep_alive_cache[name] = arr
            
            # 4. 安全地将指针移交给 migraphx
            mgx_inputs[name] = migraphx.argument(arr)

        start = time.time()
        result = self.model.run(mgx_inputs)
        infer_time = time.time() - start

        outputs = [np.array(r) for r in result]
        
        # 推理结束，释放内存
        self._keep_alive_cache.clear()
        
        return outputs, infer_time


# =========================
# 推理函数 (硬编码输入，无 Tokenizer)
# =========================
def predict(model, image, box_threshold, is_benchmark=False):

    input_dict = {
        "img": np.expand_dims(np.asarray(image), axis=0),
        "position_ids": np.array([[0, 0, 1, 0]]),
        "input_ids": np.array([[101, 2482, 1012, 102]]),
        "token_type_ids": np.array([[0, 0, 0, 0]]),
        "text_token_mask": np.array([[
            [True, False, False, False],
            [False, True, True, False],
            [False, True, True, False],
            [False, False, False, True]
        ]]),
        "attention_mask": np.array([[True, True, True, True]])
    }

    outputs, infer_time = model.infer(input_dict)

    if not is_benchmark:
        print(f"Inference time: {infer_time*1000:.2f} ms")

    logits = sigmoid(outputs[0][0])
    boxes = outputs[1][0]

    max_values = np.max(logits, axis=1)
    mask = max_values > box_threshold

    logits = logits[mask]
    boxes = boxes[mask]

    phrases = ["car"] * len(boxes)

    return boxes, np.max(logits, axis=1), phrases


# =========================
# Benchmark
# =========================
def benchmark(model, image, box_th, warmup=5, runs=10):
    print("\n🔥 预热")
    for _ in range(warmup):
        predict(model, image, box_th, True)

    print("\n🚀 测试")
    times = []
    for i in range(runs):
        start = time.time()
        predict(model, image, box_th, True)
        times.append(time.time() - start)

    print(f"\n平均耗时: {np.mean(times)*1000:.2f} ms")
    print(f"FPS: {1/np.mean(times):.2f}")


# =========================
# 主函数
# =========================
# if __name__ == "__main__":

#     model_path = "../weights/ground_opt.onnx"
#     cache_path = "../weights/ground_opt.mxr" 
#     img_path = "../images/in/car_1.jpg"

#     BOX_TRESHOLD = 0.35
#     DEVICE_ID = 5 # 匹配你之前报错堆栈里的 device: 5 / 0 的情况，按需修改

#     model = MIGraphXModel(
#         model_path,
#         cache_path=cache_path,
#         force_recompile=False,
#         device_id=DEVICE_ID
#     )

#     image_source, image = load_image(img_path)

#     benchmark(model, image, BOX_TRESHOLD)

#     boxes, confs, phrases = predict(model, image, BOX_TRESHOLD)

#     print("检测结果:", phrases)

def test_like_perf(model):
    print("\n" + "="*60)
    print("🛠️ 模拟 perf 工具：生成完美对齐的 Dummy 数据测试")
    print("="*60)
    
    mgx_inputs = {}
    keep_alive_cache = [] # 强行续命池
    
    # 1. 严格按照模型要求的形状造假数据
    for name, shape in model.get_inputs().items():
        if str(name).startswith("main:#output"):
            continue
            
        # 解析真实需要的类型和形状
        target_dtype, lens = _mgx_shape_to_numpy(shape)
        
        print(f"  📦 分配 {name}: shape={lens}, dtype={target_dtype.__name__}")
        
        # 生成分毫不差的全零矩阵（完美模拟 migraphx-driver）
        dummy_data = np.zeros(lens, dtype=target_dtype)
        keep_alive_cache.append(dummy_data)
        
        # 移交指针
        mgx_inputs[name] = migraphx.argument(dummy_data)
        
    print("\n🚀 开始 Dummy 推理测试...")
    try:
        start = time.time()
        model.run(mgx_inputs)
        print(f"✅ Python 端 Dummy 推理成功！没有任何 VMFault！耗时: {(time.time()-start)*1000:.2f}ms")
    except Exception as e:
        print(f"❌ 依然报错: {e}")
        
# ------------------
# 在主函数里这样调用：
# ------------------
if __name__ == "__main__":
    model_path = "../weights/ground_opt.onnx"
    cache_path = "../weights/ground_opt.mxr" 
    
    model = migraphx.load(cache_path) # 直接加载你确定没问题的 mxr
    
    # 运行模拟测试
    test_like_perf(model)