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

# 正式推理时使用的提示词，以及相关的mask
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
onnx_output_path = "weights/ground.onnx"

torch.onnx.export(
    model,
    f=onnx_output_path,
    args=(img, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask),
    input_names=["img", "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"],
    output_names=["logits", "boxes"],
    opset_version=17,
    verbose=False,  # 关闭详细日志，如需调试可改为True
    do_constant_folding=True  # 常量折叠优化，提升简化效果
)
