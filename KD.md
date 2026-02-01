# Knowledge Distillation Implementation Report

This document details the architecture, methodology, and implementation of the Knowledge Distillation (KD) pipeline for the ICPR 2026 OCR task.

## 1. Overview
The goal is to improve the performance of a **Student** model on Low-Resolution (LR) images by leveraging valid feature representations from a **Teacher** model trained on High-Resolution (HR) images. This setup aims to make the Student robust to degradation (blur, noise, low resolution) by encouraging it to mimic the internal activations of the Teacher.

## 2. Architecture Setup

### Models
*   **Architecture**: `ResTranOCR` (ResNet Backbone + Transformer Encoder + CTC Head).
*   **Teacher**: A pre-trained `ResTranOCR` model trained **exclusively on clean HR images**. Its weights are **frozen** during the distillation phase.
*   **Student**: An identical `ResTranOCR` architecture initialized with random weights (or pre-trained weights) trained on **LR images**.

### Implementation Class
*   **`ResTranOCR_Distill`** (`src/models/model_distill.py`): A modified wrapper around the base architecture that allows accessing intermediate feature maps.
    *   **Returns**: Final Logits + Dictionary of features (`backbone_out`, `transformer_out`).

## 3. Distillation Methodology

We employ a **Feature-Based Knowledge Distillation** strategy using **L2 Loss (MSE)**. The Student is trained to minimize the distance between its features and the Teacher's features at multiple stages of the network.

### Loss Function
The total loss is a weighted sum of the standard CTC loss and the Distillation loss.

$$ L_{total} = L_{CTC} + \lambda \cdot (L_{backbone} + L_{transformer} + L_{logits}) $$

Where:
1.  **$L_{CTC}$**: Standard supervised connectionist temporal classification loss on the Student's predictions vs. Ground Truth text.
2.  **$L_{backbone}$**: MSE Loss between the Student's Backbone (ResNet+Refiner) output and the Teacher's Backbone output.
    *   *Purpose*: Aligns low-level visual feature extraction.
3.  **$L_{transformer}$**: MSE Loss between the Student's Transformer Encoder output and the Teacher's Transformer Encoder output.
    *   *Purpose*: Aligns high-level sequence context and semantic features.
4.  **$L_{logits}$** (Character-wise): MSE Loss between the Student's final logits and the Teacher's final logits.
    *   *Purpose*: Aligns the final probability distribution per character time-step.

## 4. Training Pipeline

### Unified Training Script: `train_kd.py`
We implemented a single script that handles both modes via the `--mode` argument.

#### Mode 1: Teacher Training (`--mode teacher`)
*   **Input**: HR Images only.
*   **Dataset**: `ICPR_LPR_HR_Dataset`.
*   **Augmentation**: Standard training usage (or Clean HR specific).
*   **Output**: A clean, high-performance Teacher checkpoint (e.g., `results/restran_hr_teacher_best.pth`).

#### Mode 2: Student Distillation (`--mode student`)
*   **Input**: Paired (LR, HR) images.
    *   **Student Input**: LR Image (Real LR or degraded/augmented HR).
    *   **Teacher Input**: Corresponding Clean HR Image.
*   **Dataset**: `ICPR_LPR_Distill_Dataset` (`src/dataloader/icpr_distill.py`).
    *   Ensures geometric consistency (or loose alignment) so that spatial features map correctly between Teacher and Student.
    *   Uses Fallback logic to handle both standard and distillation-specific batch structures.
*   **Trainer**: `TrainerDistill` (`trainer_distill.py`).
    *   Loads and freezes the Teacher model.
    *   Computes the multi-component L2 loss.
    *   Updates only the Student parameters.

## 5. Configuration (`configs/icpr_distill.yaml`)

Key hyperparameters controlling the process:
*   `teacher_checkpoint`: Path to the frozen Teacher weights.
*   `distill_weight`: Scalar (`lambda`) to balance CTC vs. L2 loss (Default: 1.0).
*   `model_type` / `student_model_type`: Defines the architecture variants.
*   `clean_hr_guided`: Boolean to enable loading clean HR images for the teacher during Student training.

## 6. Usage Commands

### Step 1: Train Teacher
```bash
python train_kd.py --mode teacher --config configs/icpr_hr.yaml
```

### Step 2: Train Student
```bash
python train_kd.py --mode student --config configs/icpr_distill.yaml --teacher-ckpt results/restran_hr_teacher_best.pth
```
