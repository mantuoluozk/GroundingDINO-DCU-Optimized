import torch
import onnx
from onnxsim import simplify

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

config_file = './groundingdino/config/GroundingDINO_SwinB_cfg.py'
checkpoint_path = './weights/groundingdino_swinb_cogcoor.pth'

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"

    # modified config
    args.use_checkpoint = False
    args.use_transformer_ckpt = False

    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

# 加载模型
model = load_model(config_file, checkpoint_path, cpu_only=True)

# ===================== 核心修改：batch_size=4 =====================
BATCH_SIZE = 8
# 正式推理时使用的提示词，以及相关的mask
caption = "car ."

# 1. 文本输入扩展到batch_size=4
# 重复caption BATCH_SIZE次，构建批量文本输入
input_ids = model.tokenizer([caption]*BATCH_SIZE, return_tensors="pt", padding="longest")["input_ids"]
seq_len = input_ids.shape[1]  # 获取序列长度（适配不同caption）

# 2. 扩展position_ids到batch_size=4
position_ids = torch.tensor([[0, 0, 1, 0]]).repeat(BATCH_SIZE, 1)
# 确保position_ids长度匹配seq_len（截断/补零）
if position_ids.shape[1] < seq_len:
    pad_len = seq_len - position_ids.shape[1]
    position_ids = torch.cat([position_ids, torch.zeros(BATCH_SIZE, pad_len, dtype=torch.long)], dim=1)
else:
    position_ids = position_ids[:, :seq_len]

# 3. 扩展token_type_ids到batch_size=4
token_type_ids = torch.tensor([[0, 0, 0, 0]]).repeat(BATCH_SIZE, 1)
if token_type_ids.shape[1] < seq_len:
    pad_len = seq_len - token_type_ids.shape[1]
    token_type_ids = torch.cat([token_type_ids, torch.zeros(BATCH_SIZE, pad_len, dtype=torch.long)], dim=1)
else:
    token_type_ids = token_type_ids[:, :seq_len]

# 4. 扩展attention_mask到batch_size=4
attention_mask = torch.tensor([[True, True, True, True]]).repeat(BATCH_SIZE, 1)
if attention_mask.shape[1] < seq_len:
    pad_len = seq_len - attention_mask.shape[1]
    attention_mask = torch.cat([attention_mask, torch.ones(BATCH_SIZE, pad_len, dtype=torch.bool)], dim=1)
else:
    attention_mask = attention_mask[:, :seq_len]

# 5. 扩展text_token_mask到batch_size=4
text_token_mask = torch.tensor([[[True, False, False, False],
                                 [False,  True,  True,  False],
                                 [False,  True,  True,  False],
                                 [False,  False, False, True]]]).repeat(BATCH_SIZE, 1, 1)
# 调整mask维度匹配seq_len
if text_token_mask.shape[1] < seq_len:
    pad_len = seq_len - text_token_mask.shape[1]
    # 补全mask的行和列
    pad_row = torch.zeros(BATCH_SIZE, pad_len, text_token_mask.shape[2], dtype=torch.bool)
    text_token_mask = torch.cat([text_token_mask, pad_row], dim=1)
    pad_col = torch.zeros(BATCH_SIZE, seq_len, pad_len, dtype=torch.bool)
    text_token_mask = torch.cat([text_token_mask, pad_col], dim=2)
else:
    text_token_mask = text_token_mask[:, :seq_len, :seq_len]

# 6. 扩展图像输入到batch_size=4 (1,3,800,1200) -> (4,3,800,1200)
img = torch.randn(BATCH_SIZE, 3, 800, 1200)

# 打印输入形状，验证batch_size=8
print("="*50)
print("输入形状验证（batch_size=8）:")
print(f"img: {img.shape}")
print(f"input_ids: {input_ids.shape}")
print(f"attention_mask: {attention_mask.shape}")
print(f"position_ids: {position_ids.shape}")
print(f"token_type_ids: {token_type_ids.shape}")
print(f"text_token_mask: {text_token_mask.shape}")
print("="*50)

# onnx模型可以支持动态输入，在转换engine时建议注销
dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "seq_len"},
    "attention_mask": {0: "batch_size", 1: "seq_len"},
    "position_ids": {0: "batch_size", 1: "seq_len"},
    "token_type_ids": {0: "batch_size", 1: "seq_len"},
    "text_token_mask": {0: "batch_size", 1: "seq_len", 2: "seq_len"},
    "img": {0: "batch_size", 2: "height", 3: "width"},
    "logits": {0: "batch_size"},
    "boxes": {0: "batch_size"}
}

# 导出原始ONNX模型
onnx_output_path = "weights/ground_bs8.onnx"
simplified_onnx_path = "weights/ground_simplified_bs8.onnx"

torch.onnx.export(
    model,
    f=onnx_output_path,
    args=(img, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask),
    input_names=["img", "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"],
    output_names=["logits", "boxes"],
    # dynamic_axes=dynamic_axes,  # 转换engine时建议注释
    opset_version=17,
    verbose=False,
    do_constant_folding=True  # 常量折叠优化，提升简化效果
)

# # 使用onnxsim简化模型
# print(f"\n开始简化ONNX模型: {onnx_output_path}")
# try:
#     # 加载原始ONNX模型
#     onnx_model = onnx.load(onnx_output_path)
    
#     # 简化模型（enable_fuse_bn=True 融合批归一化层，更彻底的简化）
#     simplified_model, check = simplify(
#         onnx_model,
#         dynamic_input_shape=False,  # 因为固定了batch_size和分辨率，设为False
#         input_shapes={  # 指定batch_size=4的输入形状
#             "img": (BATCH_SIZE, 3, 800, 1200),
#             "input_ids": tuple(input_ids.shape),
#             "attention_mask": tuple(attention_mask.shape),
#             "position_ids": tuple(position_ids.shape),
#             "token_type_ids": tuple(token_type_ids.shape),
#             "text_token_mask": tuple(text_token_mask.shape)
#         }
#     )
    
#     # 验证简化后的模型
#     assert check, "简化后的ONNX模型验证失败！"
    
#     # 保存简化后的模型
#     onnx.save(simplified_model, simplified_onnx_path)
#     print(f"ONNX模型简化完成，已保存至: {simplified_onnx_path}")
    
# except Exception as e:
#     print(f"ONNX简化过程出错: {e}")
#     print("将使用原始未简化的ONNX模型")