# OCR Architecture Report

This report outlines the architecture of the Multi-Frame OCR system found in the `OCR` directory.

## 1. High-Level Overview
The system is designed for **Multi-Frame License Plate Recognition**. Instead of processing a single image, it takes a sequence of images (frames) from a track, extracts features from each, fuses them, and then recognizes the text using a Transformer-based sequence model with a CTC head.

**Key Characteristics:**
- **Input**: Sequence of 5 frames (images) per license plate track.
- **Model Type**: `ResTranOCR` (ResNet + Transformer + CTC).
- **Task**: Sequence-to-Sequence (Image Sequence -> Character Sequence).
- **Loss**: CTC Loss.

## 2. Directory Structure
The codebase follows a modular design:
- `configs/`: YAML configuration files and Python dataclass definitions.
- `src/models/`: Model implementation (`backbones`, `necks`, `heads`, and assembly in `model.py`).
- `src/dataloader/`: Data loading logic (`icpr.py` for dataset, transforms).
- `trainer.py`: Training loop, validation, and inference logic.
- `train.py`: Main entry point for CLI.

## 3. Model Architecture: `ResTranOCR`
The core model is defined in `src/models/model.py`. It consists of the following pipeline:

1.  **Input Processing**:
    -   Input tensor shape: `[Batch, Frames, 3, H, W]` (Default $F=5, H=32, W=128$).
    -   The batch and frame dimensions are flattened to apply 2D operations to each frame individually.

2.  **Spatial Transformer Network (STN) [Optional]**:
    -   Rectifies the image before feature extraction to handle rotation/perspective distortion.

3.  **Backbone (Feature Extractor)**:
    -   **ResNet34**: Extracts visual features from each frame.
    -   Output: Feature maps for each of the 5 frames.

4.  **Attention Fusion**:
    -   Module: `AttentionFusion` (likely in `src/models/necks/`).
    -   Function: Aggregates features from the 5 separate frames into a single fused feature sequence. This allows the model to leverage information from better quality frames or combine partial information.

5.  **Sequence Modeling (Transformer)**:
    -   **Positional Encoding**: Adds sequence order information.
    -   **Transformer Encoder**: stack of Transformer layers (Self-Attention + Feed Forward).
    -   Refines the fused features and models character dependencies.

6.  **Prediction Head**:
    -   **CTC Head**: Projects the transformer output to the character class space.
    -   Output: Logits for Connectionist Temporal Classification (CTC) loss.

## 4. Data Pipeline
Defined in `src/dataloader/icpr.py`:
-   **Dataset**: `ICPR_LPR_Datatset`.
-   **Input**: Reads "tracks" (folders containing multiple images of the same plate).
-   **Sampling**: Selects 5 frames per track. If fewer exist, the last frame is repeated. If more exist, it samples evenly.
-   **Augmentation**:
    -   Supports "light" and "full" augmentation levels.
    -   **Synthetic Data**: Can apply on-the-fly "degradation" to clean HR images to simulate real-world conditions (LR).

## 5. Training Process
Defined in `trainer.py`:
-   **Optimizer**: AdamW.
-   **Scheduler**: OneCycleLR (warmup + annealing).
-   **Precision**: Mixed Precision (AMP) is used for efficiency.
-   **Loss**: `CTCLoss` (handling variable length targets).
-   **Validation**: Calculates Accuracy and CER (Character Error Rate).
-   **Inference**: Supports generating submission files using `decode_with_confidence`.

## 6. Configuration
-   Managed via `configs/icpr.yaml`.
-   Key defaults:
    -   `img_height`: 32, `img_width`: 128
    -   `num_frames`: 5
    -   `model_type`: "restran"
    -   `transformer_layers`: 3, `heads`: 8
