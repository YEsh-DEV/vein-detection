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
from typing import Dict, Any, Optional

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
                        help="Root directory containing subject subdirectories (when train_dir is not used)")
    parser.add_argument("--train_dir", type=str, default=None,
                        help="Explicit directory containing train class subdirectories")
    parser.add_argument("--val_dir", type=str, default=None,
                        help="Explicit directory containing val class subdirectories")
    parser.add_argument("--max_train_classes", type=int, default=None,
                        help="Optional cap on number of training classes (useful for pre-flight testing)")
    parser.add_argument("--max_val_classes", type=int, default=None,
                        help="Optional cap on number of validation classes")
    parser.add_argument("--output_dir", type=str, default="training/checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--ckpt_prefix", type=str, default="pretrain_stage1",
                        help="Prefix for saved checkpoint files")
    parser.add_argument("--log_file", type=str, default="training/logs/train_metrics.json",
                        help="Path to save per-epoch metrics JSON")
    parser.add_argument("--split_ratio", type=float, default=0.5,
                        help="Subject-independent split ratio when data_dir is used")

    # Checkpoint loading & fine-tuning
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="Path to pretrained model checkpoint (.pt) to load backbone weights from")
    parser.add_argument("--freeze_stages", type=str, default="none",
                        choices=["none", "stem_stage1", "stem_stage1_stage2", "backbone_except_fc"],
                        help="Stages to freeze during fine-tuning (e.g. stem_stage1_stage2 for Experiment A)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for Adam optimizer")

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
    val_metrics: Optional[Dict[str, Any]] = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "head_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_eer": val_eer,
        "config": config,
    }
    if val_metrics is not None:
        payload["val_metrics"] = val_metrics
    torch.save(payload, path)


def train():
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[train.py] Starting AMPVNet training on device: {device}")
    print(f"[train.py] Config: epochs={args.epochs}, lr={args.learning_rate}, batch_size={args.batch_size}")
    print(f"[train.py] AdaFace: m={args.adaface_m}, h={args.adaface_h}, s={args.adaface_s}")
    print(f"[train.py] Augmentation: RPT(r={args.r_rpt}, p={args.p_rpt}), RGA(g={args.gamma_rga}, p={args.p_rga})")

    # Datasets & Loaders
    if args.train_dir and args.val_dir:
        train_path = Path(args.train_dir)
        val_path = Path(args.val_dir)
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(f"Specified train_dir ({train_path}) or val_dir ({val_path}) does not exist.")

        train_dataset = PalmVeinDataset(
            data_dir=str(train_path),
            split="all",
            is_train=True,
            max_classes=args.max_train_classes,
            seed=args.seed,
            r_rpt=args.r_rpt,
            p_rpt=args.p_rpt,
            gamma_rga=args.gamma_rga,
            p_rga=args.p_rga,
        )
        val_dataset = PalmVeinDataset(
            data_dir=str(val_path),
            split="all",
            is_train=False,
            max_classes=args.max_val_classes,
            seed=args.seed,
        )
    else:
        data_path = Path(args.data_dir)
        has_data = data_path.exists() and any(d.is_dir() and not d.name.startswith(".") for d in data_path.iterdir())
        if not has_data:
            raise FileNotFoundError(
                f"Dataset directory '{args.data_dir}' does not exist or contains no subject subdirectories.\n"
                f"Please place palm vein dataset images under: {args.data_dir}/<subject_id>/<image_file>"
            )

        train_dataset = PalmVeinDataset(
            data_dir=str(data_path),
            split="train",
            split_ratio=args.split_ratio,
            max_classes=args.max_train_classes,
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
            max_classes=args.max_val_classes,
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

    # Phase 1: Load ONLY AMPVNet backbone weights from pretrained checkpoint
    if args.pretrained_weights:
        p_weights = Path(args.pretrained_weights)
        if not p_weights.exists():
            raise FileNotFoundError(f"Pretrained weights file not found: {p_weights}")
        print(f"\n[train.py] ================= CHECKPOINT LOADING (PHASE 1) =================")
        print(f"[train.py] Loading pretrained backbone from: {p_weights}")
        ckpt = torch.load(p_weights, map_location=device)
        model_state = ckpt.get("model_state_dict", ckpt)

        model_dict = model.state_dict()
        matched_dict = {}
        for k, v in model_state.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    matched_dict[k] = v
                else:
                    print(f"[train.py] Shape mismatch for {k}: ckpt {v.shape} vs model {model_dict[k].shape} -> SKIPPED")

        model.load_state_dict(matched_dict, strict=True)
        print(f"[train.py] Verified & loaded {len(matched_dict)}/{len(model_dict)} tensors into AMPVNet backbone.")
        print(f"[train.py] Verified stem tensor shape : {model.stem[0].weight.shape}")
        print(f"[train.py] Verified stage1 tensor shape: {model.stage1[0].block[0].weight.shape}")
        print(f"[train.py] Verified stage4 tensor shape: {model.stage4[0].block[0].weight.shape}")
        print(f"[train.py] Verified fc tensor shape    : {model.fc.weight.shape}")

        if "head_state_dict" in ckpt:
            ckpt_head_w = ckpt["head_state_dict"].get("weight", None)
            ckpt_head_classes = ckpt_head_w.shape[0] if ckpt_head_w is not None else "N/A"
            print(f"[train.py] Pretrained AdaFace head detected: {ckpt_head_classes} classes.")
            print(f"[train.py] Pretrained AdaFace head was DISCARDED. Zero weights transferred to local head.")
        print(f"[train.py] ====================================================================\n")

    # Initialize NEW AdaFace head with local training classes (Phase 1)
    head = AdaFace(
        num_classes=num_classes,
        embedding_dim=512,
        m=args.adaface_m,
        h=args.adaface_h,
        s=args.adaface_s,
        t_alpha=args.adaface_t_alpha,
    ).to(device)
    print(f"[train.py] Initialized NEW AdaFace head for {num_classes} local identities (W shape: [{num_classes}, 512]).")

    # Phase 2: Stage Freezing Strategy
    if args.freeze_stages == "stem_stage1_stage2":
        for module in [model.stem, model.stage1, model.stage2]:
            for p in module.parameters():
                p.requires_grad = False
        print(f"[train.py] Freezing Strategy 'stem_stage1_stage2' active: stem, stage1, stage2 are FROZEN.")
    elif args.freeze_stages == "stem_stage1":
        for module in [model.stem, model.stage1]:
            for p in module.parameters():
                p.requires_grad = False
        print(f"[train.py] Freezing Strategy 'stem_stage1' active: stem, stage1 are FROZEN.")
    elif args.freeze_stages == "backbone_except_fc":
        for module in [model.stem, model.stage1, model.stage2, model.stage3, model.stage4]:
            for p in module.parameters():
                p.requires_grad = False
        print(f"[train.py] Freezing Strategy 'backbone_except_fc' active: all conv stages are FROZEN; fc is TRAINABLE.")
    elif args.freeze_stages == "none":
        print(f"[train.py] Freezing Strategy 'none' active: All backbone parameters are trainable.")
    else:
        raise ValueError(f"Unknown freeze_stages option: {args.freeze_stages}")

    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in head.parameters() if p.requires_grad)
    frozen_params_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[train.py] Parameter Summary: Total = {trainable_params_count + frozen_params_count:,} | Trainable = {trainable_params_count:,} | Frozen = {frozen_params_count:,}")

    # Optimizer: optimize ONLY trainable parameters
    trainable_params = [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad]
    optimizer = Adam(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # Scheduler: CosineAnnealingLR with minimum LR floor
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr_min,
    )

    best_val_eer = float("inf")
    best_val_metrics = None
    best_ckpt_path = output_dir / f"{args.ckpt_prefix}_best.pt"
    final_ckpt_path = output_dir / f"{args.ckpt_prefix}_final.pt"

    history = {
        "config": vars(args),
        "num_classes": num_classes,
        "epochs": [],
    }

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

        epoch_record = {
            "epoch": epoch,
            "loss": float(epoch_loss),
            "lr": float(current_lr),
            "val_eer": None,
            "tar_at_far_01": None,
            "d_prime": None,
        }

        # Evaluate EER every eval_interval epochs (and at epoch 1 and last epoch)
        if epoch % args.eval_interval == 0 or epoch == 1 or epoch == args.epochs:
            val_metrics = evaluate_model(model, val_loader, device)
            val_eer = val_metrics["eer_percent"]
            tar01 = val_metrics["tar_at_far_01_percent"]
            d_prime = val_metrics.get("d_prime", 0.0)
            epoch_record["val_eer"] = float(val_eer)
            epoch_record["tar_at_far_01"] = float(tar01)
            epoch_record["d_prime"] = float(d_prime)
            print(f"  --> Val Metrics [Epoch {epoch:03d}]: EER = {val_eer:.2f}% | TAR@FAR=0.01 = {tar01:.2f}% | d' = {d_prime:.4f} (thresh = {val_metrics['threshold']:.4f})")

            # Checkpoint best model (lowest val EER)
            if val_eer < best_val_eer:
                best_val_eer = val_eer
                best_val_metrics = val_metrics
                save_checkpoint(
                    best_ckpt_path,
                    model,
                    head,
                    optimizer,
                    epoch,
                    val_eer,
                    vars(args),
                    val_metrics=val_metrics,
                )
                print(f"  ★ Saved new best checkpoint to: {best_ckpt_path} (EER: {best_val_eer:.2f}%)")

        history["epochs"].append(epoch_record)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    # Save final checkpoint
    save_checkpoint(
        final_ckpt_path,
        model,
        head,
        optimizer,
        args.epochs,
        best_val_eer,
        vars(args),
        val_metrics=best_val_metrics,
    )
    print(f"\n[train.py] Saved final checkpoint to: {final_ckpt_path}")

    # If best.pt was never saved (e.g. 1 epoch smoke test without eval), save final as best
    if not best_ckpt_path.exists():
        save_checkpoint(best_ckpt_path, model, head, optimizer, args.epochs, best_val_eer, vars(args), val_metrics=best_val_metrics)

    elapsed = time.time() - start_time
    print(f"[train.py] Training completed in {elapsed:.1f}s. Best Val EER: {best_val_eer:.2f}%")


if __name__ == "__main__":
    train()
