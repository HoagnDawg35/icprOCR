# ICPR 2026 OCR

For training:
- Validating
`python train.py --mode train --use-validation`

- Augmented:
`python train.py --mode train --aug-level full`

- HR images:
`python train.py --mode train --hr-only --data-root data/train --use-validation`

- HR Guided Training (Student-Teacher):
`python train.py --mode train --hr-guided --teacher-checkpoint results/hr_baseline_v1.pth --experiment-name student_v1`
*(This trains a fresh Student model guided by the HR-pretrained Teacher)*

For inference:
`python train.py --mode infer --checkpoint results/v1_3frame.pth`

For evaluation:
`python train.py --mode eval --checkpoint results/v1_3frame.pth`
