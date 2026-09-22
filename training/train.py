#!/usr/bin/env python3
"""
training/train.py
-----------------
Training Script for AMPVNet with AdaFace Loss on Palm Vein Biometrics.

Implements all specifications from system_architecture.md Section 7 & Luo et al. (IEEE TIFS 2024):
  - Optimizer: Adam with momentum 0.9 (betas=(0.9, 0.999)), initial LR 0.001.
  - Scheduler: CosineAnnealingLR with T_max=epochs and eta_min=0.0001.
  - Augmentations: RPT (RandomPerspectiveTransform) and RGA (RandomGammaAdjustment).
  - Loss: AdaFace loss with adaptive quality margin (Eq. 6-10).
  - Validation: EER and TAR@FAR=0.01 evaluated every 5 epochs per Step 5 & 6.
  - Checkpointing: Saves lowest val EER checkpoint to checkpoints/best.pt and checkpoints/final.pt.
  - Hyperparameters: Fully configurable via CLI arguments (no hardcoding).
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

from model import AMPVNet
from adaface_loss import AdaFace
from dataset import PalmVeinDataset
from eval import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AMPVNet with AdaFace Loss for Palm Vein Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset & directories
    parser.add_argument("--data_dir", type=str, default="training/data",
                        help="Root directory containing subject subdirectories")
    parser.add_argument("--output_dir", type=str, default="training/checkpoints",
                        help="Directory to save checkpoints (best.pt, final.pt)")
    parser.add_argument("--split_ratio", type=float, default=0.5,
                        help="Subject-independent split ratio (5:5 protocol = 0.5)")

    # Training loop
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--learning_rate", "--lr", type=float, default=0.001,
                        help="Initial learning rate for Adam")
    parser.add_argument("--lr_min", type=float, default=0.0001,
                        help="Minimum learning rate floor for CosineAnnealingLR")
    parser.add_argument("--eval_interval", type=int, default=5,
                        help="Evaluate EER every N epochs")

    # Augmentations (RPT + RGA)
    parser.add_argument("--r_rpt", type=float, default=0.4,
                        help="Random Perspective Transform (RPT) distortion scale")
    parser.add_argument("--p_rpt", type=float, default=0.5,
                        help="Probability of applying RPT")
    parser.add_argument("--gamma_rga", type=float, default=0.6,
                        help="Random Gamma Adjustment (RGA) gamma half-range")
    parser.add_argument("--p_rga", type=float, default=0.3,
                        help="Probability of applying RGA")

    # AdaFace loss hyperparameters
    parser.add_argument("--adaface_m", type=float, default=0.55,
                        help="AdaFace base margin parameter m (paper default: 0.55)")
    parser.add_argument("--adaface_h", type=float, default=0.29,
                        help="AdaFace quality sensitivity scale h (paper default: 0.29)")
    parser.add_argument("--adaface_s", type=float, default=50.0,
                        help="AdaFace feature scale factor s (paper default: 50.0)")
    parser.add_argument("--adaface_t_alpha", type=float, default=0.99,
                        help="AdaFace EMA momentum for batch statistics (default: 0.99)")

    # System & execution
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device (cuda or cpu)")

    return parser.parse_args()


def save_checkpoint(
    path: Path,
    model: AMPVNet,
    head: AdaFace,
    optimizer: Adam,
    epoch: int,
    val_eer: float,
    config: Dict[str, Any],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "head_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_eer": val_eer,
        "config": config,
    }, path)


def train():
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[train.py] Starting AMPVNet training on device: {device}")
    print(f"[train.py] Config: epochs={args.epochs}, lr={args.learning_rate}, batch_size={args.batch_size}")
    print(f"[train.py] AdaFace: m={args.adaface_m}, h={args.adaface_h}, s={args.adaface_s}")
    print(f"[train.py] Augmentation: RPT(r={args.r_rpt}, p={args.p_rpt}), RGA(g={args.gamma_rga}, p={args.p_rga})")

    # Data availability check
    data_path = Path(args.data_dir)
    has_data = data_path.exists() and any(d.is_dir() and not d.name.startswith(".") for d in data_path.iterdir())

    if not has_data:
        raise FileNotFoundError(
            f"Dataset directory '{args.data_dir}' does not exist or contains no subject subdirectories.\n"
            f"Please place palm vein dataset images under: {args.data_dir}/<subject_id>/<image_file>"
        )

    # Datasets & Loaders
    train_dataset = PalmVeinDataset(
        data_dir=str(data_path),
        split="train",
        split_ratio=args.split_ratio,
        seed=args.seed,
        r_rpt=args.r_rpt,
        p_rpt=args.p_rpt,
        gamma_rga=args.gamma_rga,
        p_rga=args.p_rga,
    )
    val_dataset = PalmVeinDataset(
        data_dir=str(data_path),
        split="val",
        split_ratio=args.split_ratio,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=0 if device.type == "cpu" else args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(args.batch_size, len(val_dataset)),
        shuffle=False,
        num_workers=0 if device.type == "cpu" else args.num_workers,
    )

    num_classes = train_dataset.num_classes
    print(f"[train.py] Loaded {len(train_dataset)} train samples across {num_classes} subjects.")
    print(f"[train.py] Loaded {len(val_dataset)} val samples across {val_dataset.num_classes} subjects.")

    # Model & Loss Head
    model = AMPVNet(embedding_dim=512, dropout_p=0.2).to(device)
    head = AdaFace(
        num_classes=num_classes,
        embedding_dim=512,
        m=args.adaface_m,
        h=args.adaface_h,
        s=args.adaface_s,
        t_alpha=args.adaface_t_alpha,
    ).to(device)

    # Optimizer: Adam with momentum 0.9 (betas=(0.9, 0.999))
    optimizer = Adam(
        list(model.parameters()) + list(head.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    # Scheduler: CosineAnnealingLR with minimum LR matching 0.0001 floor
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr_min,
    )

    best_val_eer = float("inf")
    best_ckpt_path = output_dir / "best.pt"
    final_ckpt_path = output_dir / "final.pt"

    print("\n--- Starting Training Loop ---")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()

        total_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass: extract L2-normalized embeddings and pre-norm feature norms
            embeddings, norms = model(images, return_norm=True)

            # AdaFace loss: compute cross-entropy on margin-adjusted logits
            loss = head(embeddings, norms, labels, return_loss=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

        scheduler.step()
        epoch_loss = total_loss / max(1, total_samples)
        current_lr = scheduler.get_last_lr()[0]

        # Log every epoch
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] Loss: {epoch_loss:.4f} | LR: {current_lr:.6f}")

        # Evaluate EER every eval_interval epochs (and at epoch 1 and last epoch)
        if epoch % args.eval_interval == 0 or epoch == 1 or epoch == args.epochs:
            val_metrics = evaluate_model(model, val_loader, device)
            val_eer = val_metrics["eer_percent"]
            tar01 = val_metrics["tar_at_far_01_percent"]
            print(f"  --> Val Metrics [Epoch {epoch:03d}]: EER = {val_eer:.2f}% | TAR@FAR=0.01 = {tar01:.2f}% (thresh = {val_metrics['threshold']:.4f})")

            # Checkpoint best model (lowest val EER)
            if val_eer < best_val_eer:
                best_val_eer = val_eer
                save_checkpoint(
                    best_ckpt_path,
                    model,
                    head,
                    optimizer,
                    epoch,
                    val_eer,
                    vars(args),
                )
                print(f"  ★ Saved new best checkpoint to: {best_ckpt_path} (EER: {best_val_eer:.2f}%)")

    # Save final checkpoint
    save_checkpoint(
        final_ckpt_path,
        model,
        head,
        optimizer,
        args.epochs,
        best_val_eer,
        vars(args),
    )
    print(f"\n[train.py] Saved final checkpoint to: {final_ckpt_path}")

    # If best.pt was never saved (e.g. 1 epoch smoke test without eval), save final as best
    if not best_ckpt_path.exists():
        save_checkpoint(best_ckpt_path, model, head, optimizer, args.epochs, best_val_eer, vars(args))

    elapsed = time.time() - start_time
    print(f"[train.py] Training completed in {elapsed:.1f}s. Best Val EER: {best_val_eer:.2f}%")


if __name__ == "__main__":
    train()
