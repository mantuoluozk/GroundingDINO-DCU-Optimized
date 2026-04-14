# python3 -m onnxruntime.transformers.optimizer \
#     --input ../weights/ground.onnx \
#     --output ./mha.onnx \
#     --use_multi_head_attention \
#     # --num_heads 12 \
#     # --hidden_size 256 \
#     --model_type bert \
#     --disable_skip_layer_norm \
#     --disable_gelu \
#     --use_gpu \
#     --disable_embed_layer_norm \
#     --use_mask_index \
#     --use_raw_attention_mask
python3 -m onnxruntime.transformers.optimizer --input ../weights/ground.onnx --output ./ground.onnx --model_type bert --use_gpu