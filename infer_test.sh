HIP_VISIBLE_DEVICES=4 python demo/infer_torch.py \
-c groundingdino/config/GroundingDINO_SwinB_cfg.py \
-p weights/groundingdino_swinb_cogcoor.pth \
-i images/in/car_1.jpg \
-o images/out \
-t "car"