import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util import get_tokenlizer
from groundingdino.models.GroundingDINO.bertwarper import (
    generate_masks_with_special_tokens_and_transfer_map,
)


@dataclass
class TextInputs:
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    text_self_attention_masks: torch.Tensor


class GroundingDINOOnnxWrapper(torch.nn.Module):
    """
    ONNX 导出用 wrapper：
    - 让 ONNX 输入完全是 Tensor（image + tokenized）
    - 复用 GroundingDINO 原始 backbone/transformer/head
    - 跳过 Python tokenizer/文本mask生成（这些在导出前就做好，作为输入喂进来）
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        image: torch.Tensor,  # [B,3,H,W] float32
        input_ids: torch.Tensor,  # [B,S] int64
        token_type_ids: torch.Tensor,  # [B,S] int64
        attention_mask: torch.Tensor,  # [B,S] int64 (用于 text_token_mask)
        position_ids: torch.Tensor,  # [B,S] int64
        text_self_attention_masks: torch.Tensor,  # [B,S,S] bool/int64 (用于 sub_sentence_present)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ---- 文本编码（等价于 GroundingDINO.forward 内部实现）----
        if self.model.sub_sentence_present:
            tokenized_for_encoder: Dict[str, torch.Tensor] = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": text_self_attention_masks,
                "position_ids": position_ids,
            }
        else:
            tokenized_for_encoder = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
            }

        bert_output = self.model.bert(**tokenized_for_encoder)
        encoded_text = self.model.feat_map(bert_output["last_hidden_state"])
        text_token_mask = attention_mask.to(torch.bool)

        text_dict = {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks.to(torch.bool),
        }

        # ---- 视觉编码 + transformer（基本照抄 GroundingDINO.forward）----
        if isinstance(image, (list, torch.Tensor)):
            from groundingdino.util.misc import nested_tensor_from_tensor_list

            samples = nested_tensor_from_tensor_list(image)
        else:
            samples = image

        self.model.set_image_tensor(samples)

        import torch.nn.functional as F
        from groundingdino.util.misc import NestedTensor

        srcs = []
        masks = []
        for l, feat in enumerate(self.model.features):
            src, mask = feat.decompose()
            srcs.append(self.model.input_proj[l](src))
            masks.append(mask)
        if self.model.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.model.num_feature_levels):
                if l == _len_srcs:
                    src = self.model.input_proj[l](self.model.features[-1].tensors)
                else:
                    src = self.model.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                self.model.poss.append(pos_l)

        hs, reference, _, _, _ = self.model.transformer(
            srcs, masks, None, self.model.poss, None, None, text_dict
        )

        from groundingdino.util.misc import inverse_sigmoid

        outputs_coord_list = []
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(
            reference[:-1], self.model.bbox_embed, hs
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.model.class_embed, hs)
            ]
        )

        pred_logits = outputs_class[-1]
        pred_boxes = outputs_coord_list[-1]

        self.model.unset_image_tensor()
        return pred_logits, pred_boxes


def load_torch_model(model_config_path: str, model_checkpoint_path: str, device: str):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    model.to(device)

    # ONNX trace 不兼容 gradient checkpoint，导出时统一关闭
    for m in model.modules():
        if hasattr(m, "use_checkpoint"):
            try:
                setattr(m, "use_checkpoint", False)
            except Exception:
                pass
    return model, args


def build_text_inputs(
    tokenizer,
    caption: str,
    device: str,
    max_text_len: int,
    special_token_ids,
) -> TextInputs:
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    tokenized = tokenizer([caption], padding="longest", return_tensors="pt")
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    text_self_attention_masks, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, special_token_ids, tokenizer
    )

    # 按 max_text_len 裁剪（与模型 forward 行为一致）
    if text_self_attention_masks.shape[1] > max_text_len:
        s = max_text_len
        text_self_attention_masks = text_self_attention_masks[:, :s, :s]
        position_ids = position_ids[:, :s]
        tokenized["input_ids"] = tokenized["input_ids"][:, :s]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, :s]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :s]

    return TextInputs(
        input_ids=tokenized["input_ids"].to(torch.int64),
        token_type_ids=tokenized["token_type_ids"].to(torch.int64),
        attention_mask=tokenized["attention_mask"].to(torch.int64),
        position_ids=position_ids.to(torch.int64),
        text_self_attention_masks=text_self_attention_masks,
    )


def main():
    parser = argparse.ArgumentParser("Export GroundingDINO to ONNX", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True)
    parser.add_argument("--output_onnx", "-o", type=str, required=True, help="输出 onnx 路径")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="用于构建 dummy 文本输入（影响 seq_len）")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--dynamic", action="store_true", help="启用动态 H/W 与 seq_len 轴（更通用但可能更慢/更难优化）")
    parser.add_argument("--simplify", action="store_true", help="尝试用 onnxsim 简化（需要安装 onnxsim）")
    parser.add_argument("--image_hw", type=str, default="800,1333", help="dummy image H,W（默认与 transform 常见输出一致）")
    args = parser.parse_args()

    device = "cpu" if args.cpu_only else ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_torch_model(args.config_file, args.checkpoint_path, device=device)

    tokenizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    special_token_ids = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

    text_inputs = build_text_inputs(
        tokenizer=tokenizer,
        caption=args.text_prompt,
        device=device,
        max_text_len=getattr(cfg, "max_text_len", 256),
        special_token_ids=special_token_ids,
    )

    h_str, w_str = args.image_hw.split(",")
    H, W = int(h_str), int(w_str)
    dummy_image = torch.randn(1, 3, H, W, device=device, dtype=torch.float32)

    wrapper = GroundingDINOOnnxWrapper(model)
    wrapper.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_onnx)) or ".", exist_ok=True)

    input_names = [
        "image",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "position_ids",
        "text_self_attention_masks",
    ]
    output_names = ["pred_logits", "pred_boxes"]

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "image": {0: "batch", 2: "height", 3: "width"},
            "input_ids": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "position_ids": {0: "batch", 1: "seq"},
            "text_self_attention_masks": {0: "batch", 1: "seq", 2: "seq"},
            "pred_logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (
                dummy_image,
                text_inputs.input_ids,
                text_inputs.token_type_ids,
                text_inputs.attention_mask,
                text_inputs.position_ids,
                text_inputs.text_self_attention_masks,
            ),
            args.output_onnx,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify

            onnx_model = onnx.load(args.output_onnx)
            simplified_model, ok = simplify(onnx_model)
            if ok:
                onnx.save(simplified_model, args.output_onnx)
                print(f"✅ onnxsim 简化完成: {args.output_onnx}")
            else:
                print("⚠️ onnxsim 简化失败（模型未修改）")
        except Exception as e:
            print(f"⚠️ onnxsim 简化跳过: {e}")

    print(f"✅ 导出完成: {args.output_onnx}")


if __name__ == "__main__":
    main()

