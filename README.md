# Triarch Foundry: DINOv3 Unified Computer Vision Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C.svg)](https://pytorch.org/)
[![License: Mixed](https://img.shields.io/badge/License-Mixed-yellow.svg)](License)
[![Backbone: DINOv3](https://img.shields.io/badge/Backbone-DINOv3-green.svg)](https://github.com/facebookresearch/dinov3)

Triarch Foundry is a production-ready codebase designed to bridge the gap between Meta AI's DINOv3 foundation models and practical downstream computer vision tasks. This repository provides a unified interface for image classification, semantic segmentation, and object detection.

---

## Key Features
- Multi-task integration: Support for image classification, semantic segmentation (FPN/Linear), and object detection (RetinaNet/Faster R-CNN).
- Flexible backbones: Seamlessly switch between ViT (Small, Base, Large, Giant) and ConvNeXt (Tiny to Large) architectures.
- Environment driven: Configuration via `.env` files for clean path management.
- Advanced preprocessing: Built-in support for Albumentations and ImageNet-standard normalization.
- Hardware optimized: Supports AMP (automatic mixed precision) and modern PyTorch optimizations.

---

## Project Architecture
```text
.
|-- dinov3/                 # Official DINOv3 repository (local clone)
|-- input/                  # Datasets (Pascal VOC, ImageNet-style, custom)
|-- outputs/                # Training logs, checkpoints, and inference results
|-- assets/
|   `-- results/
|       |-- object_detection/
|       `-- semantic_segmentation/
|-- classification_configs/ # YAML files for classification parameters
|-- detection_configs/      # YAML files for detection parameters
|-- segmentation_configs/   # YAML files for segmentation parameters
|-- src/                    # Task-specific logic
|   |-- img_cls/            # Classification heads and engine
|   |-- img_seg/            # Segmentation decoders and loss functions
|   `-- detection/          # Object detection utilities and models
|-- weights/                # Pretrained DINOv3 .pth files
|-- .env                    # Path configs (DINOv3_REPO and DINOv3_WEIGHTS)
|-- train_classifier.py     # Entry point for classification
|-- train_segmentation.py   # Entry point for segmentation
|-- train_detection.py      # Entry point for detection
|-- infer_classifier.py     # Image classification inference
|-- infer_seg_image.py      # Image segmentation inference
|-- infer_seg_video.py      # Video segmentation inference
|-- infer_det_image.py      # Image detection inference
`-- infer_det_video.py      # Video detection inference
```

---

## Setup and Installation
### 1. Repository setup
Clone this repository and the official DINOv3 backbone:

```bash
git clone https://github.com/HCWorld69/dinov3-triarch-foundry.git
cd dinov3-triarch-foundry
git clone https://github.com/facebookresearch/dinov3.git
```

### 2. Configure environment
The project uses a `.env` file to locate the backbone code and pretrained weights. Create it in the root directory:

```bash
# Example .env content
DINOv3_REPO="dinov3"
DINOv3_WEIGHTS="weights"
```

### 3. Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "albumentations>=2.0.8" opencv-python matplotlib python-dotenv pyyaml
```

If you want to install everything from the pinned list, run:

```bash
pip install -r requirements.txt
```

---

## Usage Guide
### 1. Image Classification
Train a classification head on top of a frozen or unfrozen DINOv3 backbone.

```bash
python train_classifier.py \
  --train-dir input/your_dataset/train \
  --valid-dir input/your_dataset/valid \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --weights path/to/dinov3_weights.pth \
  --repo-dir path/to/dinov3 \
  --model-name dinov3_vits16
```

### 2. Semantic Segmentation
Requires a YAML configuration file from `segmentation_configs/`.

```bash
python train_segmentation.py \
  --train-images input/your_dataset/train_images \
  --train-masks input/your_dataset/train_masks \
  --valid-images input/your_dataset/valid_images \
  --valid-masks input/your_dataset/valid_masks \
  --config segmentation_configs/voc.yaml \
  --weights path/to/dinov3_weights.pth \
  --model-name dinov3_convnext_tiny \
  --epochs 50 \
  --imgsz 640 640 \
  --batch 12
```

### 3. Object Detection
Supports DINOv3 ConvNeXt and ViT backbones with RetinaNet or Faster R-CNN heads.

```bash
python train_detection.py \
  --weights path/to/dinov3_weights.pth \
  --model-name dinov3_convnext_tiny \
  --config detection_configs/voc.yaml \
  --imgsz 640 640 \
  --epochs 30 \
  --lr 0.001 \
  --batch 8 \
  --fine-tune
```

---

## Pretrained Weights
Weights must be downloaded from the official DINOv3 release and placed in the `weights/` directory.

| Model Type | Identifier | Filename Example |
| --- | --- | --- |
| ViT-Small | `dinov3_vits16` | `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` |
| ConvNeXt-T | `dinov3_convnext_tiny` | `dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth` |

---

## Technical Details
- Normalization: Standard ImageNet stats `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
- Model Loading: Models are loaded via `torch.hub.load` using the `local` source pointing to the `dinov3/` directory.
- Logging: Training metrics are saved in CSV format and visualized using Matplotlib in the `outputs/` folder.

---

## Results
### Semantic segmentation
<table>
  <tr>
    <td align="center">
      <img src="assets/results/semantic_segmentation/image_1.jpg" alt="Semantic segmentation result 1" width="420">
    </td>
    <td align="center">
      <img src="assets/results/semantic_segmentation/image_2.jpg" alt="Semantic segmentation result 2" width="420">
    </td>
  </tr>
</table>

### Object detection
<table>
  <tr>
    <td align="center">
      <img src="assets/results/object_detection/image_3.jpg" alt="Object detection result 1" width="420">
    </td>
    <td align="center">
      <img src="assets/results/object_detection/image_4.jpg" alt="Object detection result 2" width="420">
    </td>
  </tr>
</table>

---

## Credits
This stack is maintained by Sovit Ranjan Rath (DebuggerCafe). It leverages the research and model architectures provided by the Meta AI DINOv3 team.

## License
This project is licensed under a mix of MIT and DINOv3 licensing. See `License` for details.
