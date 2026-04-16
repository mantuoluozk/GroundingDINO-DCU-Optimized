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

# 正式推理时使用的提示词，以及相关的mask，可以提前使用get_caption_mask.py生成得到
caption = "car ."
input_ids = model.tokenizer([caption], return_tensors="pt")["input_ids"]
position_ids = torch.tensor([[0, 0, 1, 0]])
token_type_ids = torch.tensor([[0, 0, 0, 0]])
attention_mask = torch.tensor([[True, True, True, True]])
text_token_mask = torch.tensor([[[True, False, False, False],
                                 [False,  True,  True,  False],
                                 [False,  True,  True,  False],
                                 [False,  False, False, True]]])

# 固定输入分辨率
img = torch.randn(1, 3, 800, 1200)
# img = torch.randn(1, 3, 400, 600)

# 导出原始ONNX模型
onnx_output_path = "weights/ground.onnx"
simplified_onnx_path = "weights/ground_simplified1.onnx"


torch.onnx.export(
    model,
    f=onnx_output_path,
    args=(img, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask),
    input_names=["img", "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"],
    output_names=["logits", "boxes"],
    dynamic_axes=None, # 静态维度导出
    opset_version=17,
    verbose=False,  # 关闭详细日志，如需调试可改为True
    do_constant_folding=True  # 常量折叠优化，提升简化效果
)
print(f"ONNX模型已成功导出到: {onnx_output_path}")

# # 使用onnxsim简化模型
# print(f"开始简化ONNX模型: {onnx_output_path}")
# try:
#     # 加载原始ONNX模型
#     onnx_model = onnx.load(onnx_output_path)
    
#     # 简化模型（enable_fuse_bn=True 融合批归一化层，更彻底的简化）
#     simplified_model, check = simplify(
#         onnx_model,
#         skip_fuse_bn=True,
#         skip_constant_folding=True,
#         dynamic_input_shape=False, 
#         input_shapes={  # 指定输入形状，确保简化准确
#             "img": (1, 3, 800, 1200),
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