CUDA_VISIBLE_DEVICES=0 \
python demo/test_ap_on_coco_onnx.py \
 -c groundingdino/config/GroundingDINO_SwinB_cfg.py \
 -p weights/ground.onnx \
 --anno_path /workspace/coco_2017/annotations/instances_val2017.json \
 --image_dir /workspace/coco_2017/val2017