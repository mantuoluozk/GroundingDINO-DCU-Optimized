import torch
import onnx
from onnxsim import simplify

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

from torch.onnx import register_custom_op_symbolic
from groundingdino.models.GroundingDINO.ms_deform_attn import MultiScaleDeformableAttnFunction

def ms_deform_attn_symbolic(g, value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
    return g.op("custom::ms_deform_attn", 
                value, spatial_shapes, level_start_index, sampling_locations, attention_weights,
                im2col_step_i=im2col_step)

MultiScaleDeformableAttnFunction.symbolic = ms_deform_attn_symbolic

config_file = '../groundingdino/config/GroundingDINO_SwinB_cfg.py'
checkpoint_path = '../weights/groundingdino_swinb_cogcoor.pth'

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    args.use_checkpoint = False
    args.use_transformer_ckpt = False
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

# 加载模型
model = load_model(config_file, checkpoint_path, cpu_only=False)  # <-- 必须 cuda，否则算子不触发
device = "cuda"
model = model.to(device)

caption = "car ."
input_ids = model.tokenizer([caption], return_tensors="pt")["input_ids"].to(device)
position_ids = torch.tensor([[0, 0, 1, 0]]).to(device)
token_type_ids = torch.tensor([[0, 0, 0, 0]]).to(device)
attention_mask = torch.tensor([[True, True, True, True]]).to(device)
text_token_mask = torch.tensor([[[True, False, False, False],
                                 [False,  True,  True,  False],
                                 [False,  True,  True,  False],
                                 [False,  False, False, True]]]).to(device)

img = torch.randn(1, 3, 800, 1200).to(device)

# 导出 ONNX
onnx_output_path = "../weights/ground_deform.onnx"

torch.onnx.export(
    model,
    args=(img, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask),
    f=onnx_output_path,
    input_names=["img", "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"],
    output_names=["logits", "boxes"],
    opset_version=17,
    custom_opsets={"custom": 1},
    export_custom_ops=True,
    do_constant_folding=True,
)

print("✅ ONNX 导出成功！")