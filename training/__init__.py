"""
training/
=========
AMPVNet + AdaFace Training Pipeline for Palm Vein Recognition.

This package is TRAINING-ONLY. It must NOT be imported from app/.

File overview:
  model.py                  — AMPVNet backbone (1.61M params, 512-dim L2 embedding)
  adaface_loss.py           — AdaFace adaptive margin head + AMPVNetWithNorm wrapper
  augmentations.py          — RPT + RGA training transforms; inference transform
  dataset.py                — PalmVeinDataset; CASIA + own-data DataLoader factories
  train.py                  — Main training script (--phase pretrain|finetune)
  export_onnx.py            — PyTorch -> ONNX export with numerical validation
  evaluate.py               — EER, Rank-1, threshold calibration report
  requirements-training.txt — Training-only pip dependencies

Usage:
  # 1. Install training deps (on desktop / cloud, not Pi)
  pip install -r training/requirements-training.txt

  # 2. Place CASIA data in training/data_raw/casia/<subject>/<image>.*
  #    Place own data in training/data_raw/own/<username>/<image>.png

  # 3. Phase 1: Pretrain on CASIA
  python training/train.py --phase pretrain --data_root training/data_raw/casia

  # 4. Phase 2: Fine-tune on own data
  python training/train.py --phase finetune --data_root training/data_raw/own

  # 5. Export to ONNX for Pi 5 deployment
  python training/export_onnx.py --checkpoint training/checkpoints/finetune_best.pt

  # 6. (Optional) Evaluate and calibrate threshold
  python training/evaluate.py --checkpoint training/checkpoints/finetune_best.pt

See system_architecture.md Section 6 for the complete pipeline specification.
"""
