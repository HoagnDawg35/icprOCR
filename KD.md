# Knowledge Distillation Training Commands

This document provides the commands to run the Knowledge Distillation (KD) pipeline using the unified `train_kd.py` script.

## 1. Train HR Teacher (HR-Only)

Train a Teacher model on clean High-Resolution images. This model will be frozen and used to guide the student.

```bash
python train_kd.py --mode teacher --config configs/icpr_hr.yaml
```

**Output**: The best checkpoint will be saved to `results/`. Look for the file ending in `_best.pth` (e.g., `results/restran_hr_teacher_best.pth`).

## 2. Train Student (Distillation)

Train the Student model on Low-Resolution images using L2 feature guidance from the Teacher.

**Command**:
```bash
python train_kd.py --mode student --config configs/icpr_distill.yaml --teacher-ckpt results/restran_hr_teacher_best.pth
```

**Arguments**:
- `--teacher-ckpt`: Path to the teacher checkpoint trained in Step 1.
- `--distill-weight`: (Optional) Weight of the L2 loss (default: 1.0).

## 3. Inference (Optional)

To generate a submission file using the trained student model:

```bash
python train.py --mode infer --config configs/icpr_distill.yaml --checkpoint results/restran_distill_l2_best.pth
```
