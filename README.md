# ICPR 2026 OCR

For training:
`python train.py --mode train --use-validation`
`python train.py --mode train --aug-level full`

For inference:
`python train.py --mode infer --checkpoint results/v1_3frame.pth`

For evaluation:
`python train.py --mode eval --checkpoint results/v1_3frame.pth`
