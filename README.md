# ICPR 2026 OCR

For training:
- Validating
`python train.py --mode train --use-validation`

- Augmented:
`python train.py --mode train --aug-level full`

- HR images:
`python train.py --mode train --hr-only --data-root data/train --use-validation`

For inference:
`python train.py --mode infer --checkpoint results/v1_3frame.pth`

For evaluation:
`python train.py --mode eval --checkpoint results/v1_3frame.pth`

With TBSRN:
1. Training: `python train.py --mode train --use-tbsrn --experiment-name restran_tbsrn_v1 --use-validation --gpu 1 --batch-size 32`
2. Evaluation: `python train.py --mode eval --checkpoint results/restran_tbsrn_v1.pth`