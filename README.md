# ICPR 2026 OCR

For training:
- Validating
`python train.py --mode train --use-validation`

- Augmented:
`python train.py --mode train --aug-level full`

- HR images:
`python train.py --mode train --hr-only --data-root data/train --use-validation`

- HR Guided Training:
`python train.py --mode train --hr-guided --checkpoint results/hr_baseline_v1.pth --epochs 1 --batch-size 4 --num-workers 0 --experiment-name verification_run`

For inference:
`python train.py --mode infer --checkpoint results/v1_3frame.pth`

For evaluation:
`python train.py --mode eval --checkpoint results/v1_3frame.pth`
