import json
import sys
from collections import defaultdict

def analyze_profile(json_path):
    print(f"🔍 正在解析性能文件: {json_path}\n")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # 兼容不同的 JSON 根节点格式
    events = data if isinstance(data, list) else data.get('traceEvents', [])

    # 按“算子类型”(如 MatMul, Conv) 统计总耗时
    op_type_times = defaultdict(float)
    # 按“具体节点名”(如 /transformer/encoder/MatMul_1) 统计总耗时
    node_name_times = defaultdict(float)

    total_inference_time = 0.0

    for event in events:
        # 只统计包含持续时间(dur)和参数(args)的事件
        if 'dur' in event and 'args' in event:
            args = event['args']
            # ORT 通常把算子类型记录在 args 里的 op_name
            if 'op_name' in args:
                op_type = args['op_name']
                # event['name'] 通常包含完整的节点路径
                node_name = event.get('name', 'Unknown_Node')
                
                # JSON 里的 dur 单位是微秒 (microseconds)，转成毫秒 (ms)
                dur_ms = event['dur'] / 1000.0 
                
                op_type_times[op_type] += dur_ms
                node_name_times[node_name] += dur_ms
                total_inference_time += dur_ms

    # 对字典进行降序排序
    sorted_op_types = sorted(op_type_times.items(), key=lambda x: x[1], reverse=True)
    sorted_nodes = sorted(node_name_times.items(), key=lambda x: x[1], reverse=True)

    print("="*50)
    print("🏆 按【算子类型 (OpType)】耗时总和排名 Top 10")
    print("="*50)
    for i, (op, time_ms) in enumerate(sorted_op_types[:10]):
        percentage = (time_ms / total_inference_time) * 100 if total_inference_time > 0 else 0
        print(f"{i+1:2d}. {op:<20} | 耗时: {time_ms:>8.3f} ms | 占比: {percentage:>5.2f}%")

    print("\n" + "="*50)
    print("🎯 按【单个具体节点 (Node)】耗时排名 Top 15")
    print("="*50)
    for i, (node, time_ms) in enumerate(sorted_nodes[:15]):
        percentage = (time_ms / total_inference_time) * 100 if total_inference_time > 0 else 0
        print(f"{i+1:2d}. 耗时: {time_ms:>8.3f} ms ({percentage:>5.2f}%) | 节点: {node}")

if __name__ == "__main__":
    # 把这里换成你刚刚生成的 json 文件名
    profile_file = "./onnxruntime_profile__2026-04-27_13-58-17.json" 
    
    if len(sys.argv) > 1:
        profile_file = sys.argv[1]
        
    analyze_profile(profile_file)