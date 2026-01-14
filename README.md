# Triarch Foundry: DINOv3 Multi-Task Vision Stack

Triarch Foundry is a unified pipeline for image classification, semantic segmentation, and object detection built on Meta AI's DINOv3 backbones. It is designed for a CLI-first workflow with clean entry points for training and inference.

## Highlights
- One stack for three tasks: classification, segmentation, detection.
- Swappable backbones with DINOv3 ViT and ConvNeXt families.
- Config-driven training and inference via YAML files.
- Clear project layout and repeatable runs.

## Project layout
```text
dinov3_stack/
  assets/results/
    object_detection/
    semantic_segmentation/
  classification_configs/
  detection_configs/
  segmentation_configs/
  src/
    img_cls/
    img_seg/
    detection/
  train_classifier.py
  train_segmentation.py
  train_detection.py
  infer_classifier.py
  infer_seg_image.py
  infer_det_image.py
```

## Setup
### 1. Clone the repositories
```bash
git clone https://github.com/HCWorld69/dinov3-triarch-foundry.git
cd dinov3-triarch-foundry
git clone https://github.com/facebookresearch/dinov3.git
```

### 2. Configure .env
Create a `.env` file in the repository root:
```bash
DINOv3_REPO="dinov3"
DINOv3_WEIGHTS="weights"
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

## Training and inference
### Image classification
```bash
python train_classifier.py \
  --train-dir path/to/train \
  --valid-dir path/to/valid \
  --epochs 50 \
  --weights path/to/dinov3_weights.pth \
  --repo-dir path/to/dinov3 \
  --model-name dinov3_vits16
```

### Semantic segmentation
```bash
python train_segmentation.py \
  --train-images path/to/train_images \
  --train-masks path/to/train_masks \
  --valid-images path/to/valid_images \
  --valid-masks path/to/valid_masks \
  --config segmentation_configs/voc.yaml \
  --weights path/to/dinov3_weights.pth \
  --model-name dinov3_convnext_tiny \
  --epochs 50 \
  --imgsz 640 640 \
  --batch 12
```

### Object detection
```bash
python train_detection.py \
  --weights path/to/dinov3_weights.pth \
  --model-name dinov3_convnext_tiny \
  --config detection_configs/voc.yaml \
  --imgsz 640 640 \
  --epochs 30 \
  --batch 8 \
  --fine-tune
```

## Results
### Semantic segmentation
| Example 1 | Example 2 |
| --- | --- |
| ![Semantic segmentation result 1](assets/results/semantic_segmentation/image_1.jpg) | ![Semantic segmentation result 2](assets/results/semantic_segmentation/image_2.jpg) |

### Object detection
| Example 1 | Example 2 |
| --- | --- |
| ![Object detection result 1](assets/results/object_detection/image_3.jpg) | ![Object detection result 2](assets/results/object_detection/image_4.jpg) |

## Credits
This stack is maintained by Sovit Ranjan Rath (DebuggerCafe). It builds on the official DINOv3 research and model releases from Meta AI.

## License
This repository is provided under a mix of MIT and DINOv3 licensing. See `License` for details.
