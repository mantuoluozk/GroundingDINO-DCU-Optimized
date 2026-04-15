**Installation:**


1. Install the required dependencies in the current directory.

```bash
pip install -e .
```

2. Download pre-trained model weights.

```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

```bash
nvidia-smi
```
Replace `{GPU ID}`, `image_you_want_to_detect.jpg`, and `"dir you want to save the output"` with appropriate values in the following command
```bash
CUDA_VISIBLE_DEVICES={GPU ID} python demo/inference_on_a_image.py \
-c groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p weights/groundingdino_swint_ogc.pth \
-i image_you_want_to_detect.jpg \
-o "dir you want to save the output" \
-t "chair"
 [--cpu-only] # open it for cpu mode
```

If you would like to specify the phrases to detect, here is a demo:
```bash
CUDA_VISIBLE_DEVICES={GPU ID} python demo/inference_on_a_image.py \
-c groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p ./groundingdino_swint_ogc.pth \
-i .asset/cat_dog.jpeg \
-o logs/1111 \
-t "There is a cat and a dog in the image ." \
--token_spans "[[[9, 10], [11, 14]], [[19, 20], [21, 24]]]"
 [--cpu-only] # open it for cpu mode
```
The token_spans specify the start and end positions of a phrases. For example, the first phrase is `[[9, 10], [11, 14]]`. `"There is a cat and a dog in the image ."[9:10] = 'a'`, `"There is a cat and a dog in the image ."[11:14] = 'cat'`. Hence it refers to the phrase `a cat` . Similarly, the `[[19, 20], [21, 24]]` refers to the phrase `a dog`.

See the `demo/inference_on_a_image.py` for more details.

**Running with Python:**

```python
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = "weights/dog-3.jpeg"
TEXT_PROMPT = "chair . person . dog ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)
```
**Web UI**

We also provide a demo code to integrate Grounding DINO with Gradio Web UI. See the file `demo/gradio_app.py` for more details.

**Notebooks**

- We release [demos](demo/image_editing_with_groundingdino_gligen.ipynb) to combine [Grounding DINO](https://arxiv.org/abs/2303.05499) with [GLIGEN](https://github.com/gligen/GLIGEN)  for more controllable image editings.
- We release [demos](demo/image_editing_with_groundingdino_stablediffusion.ipynb) to combine [Grounding DINO](https://arxiv.org/abs/2303.05499) with [Stable Diffusion](https://github.com/Stability-AI/StableDiffusion) for image editings.

## COCO Zero-shot Evaluations

We provide an example to evaluate Grounding DINO zero-shot performance on COCO. The results should be **48.5**.

```bash
CUDA_VISIBLE_DEVICES=0 \
python demo/test_ap_on_coco.py \
 -c groundingdino/config/GroundingDINO_SwinT_OGC.py \
 -p weights/groundingdino_swint_ogc.pth \
 --anno_path /path/to/annoataions/ie/instances_val2017.json \
 --image_dir /path/to/imagedir/ie/val2017
```
