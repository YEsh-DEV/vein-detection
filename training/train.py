#!/usr/bin/env python3
"""
training/train.py
-----------------
AMPVNet + AdaFace Training Script.

Supports two training phases as described in system_architecture.md Section 6.2:

  --phase pretrain   Phase 1: Train AMPVNet from scratch on CASIA data.
                               Uses full AdaFace margin, RPT+RGA augmentation.
                               Produces: checkpoints/pretrain_best.pt

  --phase finetune   Phase 2: Fine-tune pretrained backbone on own captured data.
                               Loads pretrain_best.pt, freezes Stages 1-2,
                               trains Stages 3-4 + head with lower LR.
                               Produces: checkpoints/finetune_best.pt

Usage:
  # Phase 1 — Pretraining on CASIA (run on a machine with GPU or fast CPU)
  python training/train.py --phase pretrain --data_root training/data_raw/casia

  # Phase 2 — Fine-tuning on captured Pi data
  python training/train.py --phase finetune --data_root training/data_raw/own \\
      --pretrain_ckpt training/checkpoints/pretrain_best.pt

Checkpoint format (all phases):
  {
      "epoch":        int,
      "model_state":  OrderedDict,    (AMPVNet backbone state_dict)
      "head_state":   OrderedDict,    (AdaFace head state_dict)
      "optimizer":    dict,           (optimizer state_dict)
      "scheduler":    dict,           (scheduler state_dict)
      "best_val_acc": float,
      "phase":        str,
      "num_classes":  int,
      "config":       dict,           (all CLI args as dict)
  }
"""

import sys
import os
import time
import json
import argparse
import logging
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

# ---------------------------------------------------------------------------
# Add training directory to path so local imports work regardless of cwd
# ---------------------------------------------------------------------------
_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TRAINING_DIR)

from model import AMPVNet
from adaface_loss import AdaFace, AMPVNetWithNorm
from augmentations import build_train_transform, build_inference_transform
from dataset import build_casia_loaders, build_own_loaders


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str, phase: str) -> logging.Logger:
    """Creates a logger that writes to both stdout and a JSONL file."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{phase}_{ts}.log")

    logger = logging.getLogger("ampvnet_train")
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S"))

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Computes Top-1 accuracy from logits and ground-truth labels."""
    preds = logits.argmax(dim=1)
    correct = (preds == labels).float().sum().item()
    return correct / labels.size(0)


@torch.no_grad()
def evaluate(
    wrapped_backbone: AMPVNetWithNorm,
    head: AdaFace,
    val_loader,
    device: torch.device,
    logger: logging.Logger,
) -> float:
    """
    Runs one full validation epoch.

    Returns:
        Average top-1 accuracy across all validation batches.
    """
    wrapped_backbone.eval()
    head.eval()

    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(val_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        embeddings, norms = wrapped_backbone(images)
        logits = head(embeddings, norms, labels)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    acc = total_correct / max(total_samples, 1)
    return acc


# ---------------------------------------------------------------------------
# Freeze helpers for fine-tuning phase
# ---------------------------------------------------------------------------

def _freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False


def _unfreeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = True


def freeze_for_finetuning(backbone: AMPVNet, logger: logging.Logger):
    """
    Phase 2 fine-tuning freeze strategy:
      Freeze: stem, stage1, stage2  (coarse vascular features, large CASIA-trained)
      Train:  stage3, stage4, gap, dropout, fc  (identity-specific fine details)

    Rationale: stem + stage1/2 learn generalizable NIR vein edge filters from CASIA.
    stage3/4 learn identity-discriminative features — these need adaptation to
    deployment-domain lighting and palm geometry.
    """
    _freeze_module(backbone.stem)
    _freeze_module(backbone.stage1)
    _freeze_module(backbone.stage2)
    _unfreeze_module(backbone.stage3)
    _unfreeze_module(backbone.stage4)
    _unfreeze_module(backbone.fc)

    frozen_params   = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    logger.info(f"[Freeze] stem + stage1 + stage2 frozen ({frozen_params:,} params).")
    logger.info(f"[Freeze] stage3 + stage4 + fc trainable ({trainable_params:,} params).")


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    wrapped_backbone: AMPVNetWithNorm,
    head: AdaFace,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    train_loader,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    use_amp: bool = False,
) -> dict:
    """
    Runs one training epoch.

    Returns:
        dict with keys: loss_avg, acc_avg, duration_s
    """
    wrapped_backbone.train()
    head.train()

    running_loss   = 0.0
    running_correct = 0
    running_total  = 0
    t0 = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with autocast():
                embeddings, norms = wrapped_backbone(images)
                logits = head(embeddings, norms, labels)
                loss   = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(wrapped_backbone.parameters()) + list(head.parameters()),
                max_norm=5.0
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            embeddings, norms = wrapped_backbone(images)
            logits = head(embeddings, norms, labels)
            loss   = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(wrapped_backbone.parameters()) + list(head.parameters()),
                max_norm=5.0
            )
            optimizer.step()

        acc = compute_top1_accuracy(logits.detach(), labels)
        running_loss    += loss.item()
        running_correct += int(acc * labels.size(0))
        running_total   += labels.size(0)

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
            logger.info(
                f"  Epoch {epoch:03d} [{batch_idx+1:04d}/{len(train_loader):04d}] "
                f"loss={running_loss / (batch_idx+1):.4f}  "
                f"acc={running_correct / max(running_total, 1):.4f}"
            )

    duration_s = time.time() - t0
    return {
        "loss_avg":   running_loss  / max(len(train_loader), 1),
        "acc_avg":    running_correct / max(running_total, 1),
        "duration_s": duration_s,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    backbone: AMPVNet,
    head: AdaFace,
    optimizer,
    scheduler,
    epoch: int,
    best_val_acc: float,
    phase: str,
    num_classes: int,
    config: dict,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":        epoch,
        "model_state":  backbone.state_dict(),
        "head_state":   head.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "phase":        phase,
        "num_classes":  num_classes,
        "config":       config,
    }, path)


def load_pretrain_checkpoint(
    path: str,
    backbone: AMPVNet,
    logger: logging.Logger,
) -> dict:
    """
    Loads only the backbone weights from a pretrain checkpoint.
    Returns the full checkpoint dict for reference.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pretrain checkpoint not found: {path}\n"
            "Run Phase 1 (pretrain) first and copy the checkpoint."
        )
    ckpt = torch.load(path, map_location="cpu")
    backbone.load_state_dict(ckpt["model_state"])
    logger.info(
        f"[Checkpoint] Loaded pretrained backbone from {path} "
        f"(epoch={ckpt.get('epoch', '?')}, val_acc={ckpt.get('best_val_acc', '?'):.4f})"
    )
    return ckpt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AMPVNet + AdaFace Training — Palm Vein Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--phase", required=True, choices=["pretrain", "finetune"],
                   help="Training phase: 'pretrain' (CASIA) or 'finetune' (own data).")
    p.add_argument("--data_root", default=None,
                   help="Root directory of dataset. Auto-detected if not provided.")
    p.add_argument("--pretrain_ckpt",
                   default="training/checkpoints/pretrain_best.pt",
                   help="Path to pretrain checkpoint for Phase 2 fine-tuning.")
    p.add_argument("--ckpt_dir", default="training/checkpoints",
                   help="Directory to save checkpoints.")
    p.add_argument("--log_dir", default="training/logs",
                   help="Directory to save training logs.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Number of epochs. Default: 50 (pretrain), 30 (finetune).")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Batch size. Default: 16 (pretrain), 8 (finetune).")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate. Default: 1e-3 (pretrain), 1e-4 (finetune).")
    p.add_argument("--workers", type=int, default=2,
                   help="DataLoader worker processes.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility.")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable automatic mixed precision (AMP). Use on CPU or older GPU.")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume training from.")
    return p.parse_args()


def main():
    args = parse_args()

    # -------------------------------------------------------------------
    # Seed for reproducibility
    # -------------------------------------------------------------------
    torch.manual_seed(args.seed)

    # -------------------------------------------------------------------
    # Phase-specific defaults
    # -------------------------------------------------------------------
    if args.phase == "pretrain":
        epochs     = args.epochs     or 50
        batch_size = args.batch_size or 16
        lr         = args.lr         or 1e-3
        data_root  = args.data_root  or "training/data_raw/casia"
    else:  # finetune
        epochs     = args.epochs     or 30
        batch_size = args.batch_size or 8
        lr         = args.lr         or 1e-4
        data_root  = args.data_root  or "training/data_raw/own"

    config = {
        "phase": args.phase, "data_root": data_root, "epochs": epochs,
        "batch_size": batch_size, "lr": lr, "seed": args.seed,
    }

    logger = setup_logger(args.log_dir, args.phase)
    logger.info(f"=== AMPVNet Training — Phase: {args.phase.upper()} ===")
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # -------------------------------------------------------------------
    # Device selection
    # -------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and (device.type == "cuda")
    logger.info(f"Device: {device} | AMP: {use_amp}")

    # -------------------------------------------------------------------
    # Build transforms
    # -------------------------------------------------------------------
    train_tf = build_train_transform()
    val_tf   = build_inference_transform()

    # -------------------------------------------------------------------
    # Build data loaders
    # -------------------------------------------------------------------
    if args.phase == "pretrain":
        train_loader, val_loader, num_classes = build_casia_loaders(
            data_root, train_tf, val_tf, batch_size=batch_size,
            num_workers=args.workers,
        )
    else:
        train_loader, val_loader, num_classes = build_own_loaders(
            data_root, train_tf, val_tf, batch_size=batch_size,
            num_workers=args.workers,
        )

    logger.info(f"Subjects (classes): {num_classes}")

    # -------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------
    backbone  = AMPVNet(embedding_dim=512, dropout_p=0.2).to(device)
    wrapped   = AMPVNetWithNorm(backbone).to(device)
    head      = AdaFace(embedding_size=512, classnum=num_classes).to(device)

    # --- Phase 2: Load pretrained weights and freeze early stages ---
    if args.phase == "finetune":
        load_pretrain_checkpoint(args.pretrain_ckpt, backbone, logger)
        freeze_for_finetuning(backbone, logger)

    total_params = backbone.count_parameters()
    logger.info(f"AMPVNet total trainable params: {total_params:,}")

    # -------------------------------------------------------------------
    # Optimizer & Scheduler
    # -------------------------------------------------------------------
    optimizer = AdamW(
        list(filter(lambda p: p.requires_grad, wrapped.parameters()))
        + list(head.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler    = GradScaler() if use_amp else None

    # -------------------------------------------------------------------
    # Optional: resume from checkpoint
    # -------------------------------------------------------------------
    start_epoch   = 1
    best_val_acc  = 0.0

    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        backbone.load_state_dict(ckpt["model_state"])
        head.load_state_dict(ckpt["head_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        logger.info(f"Resumed from epoch {ckpt['epoch']}, best_val_acc={best_val_acc:.4f}")

    # -------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------
    best_ckpt_path = os.path.join(args.ckpt_dir, f"{args.phase}_best.pt")
    last_ckpt_path = os.path.join(args.ckpt_dir, f"{args.phase}_last.pt")

    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(start_epoch, epochs + 1):
        logger.info(f"\n--- Epoch {epoch}/{epochs} ---")

        # Training
        train_metrics = train_one_epoch(
            wrapped, head, optimizer, scaler, train_loader,
            device, epoch, logger, use_amp=use_amp,
        )

        # Validation
        val_acc = evaluate(wrapped, head, val_loader, device, logger)

        # Scheduler step
        scheduler.step()

        # Checkpoint: always save last
        save_checkpoint(
            last_ckpt_path, backbone, head, optimizer, scheduler,
            epoch, best_val_acc, args.phase, num_classes, config,
        )

        # Checkpoint: save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                best_ckpt_path, backbone, head, optimizer, scheduler,
                epoch, best_val_acc, args.phase, num_classes, config,
            )
            logger.info(f"  ★ New best val_acc={best_val_acc:.4f} — checkpoint saved.")

        logger.info(
            f"  [Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss_avg']:.4f}  "
            f"train_acc={train_metrics['acc_avg']:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.6f}  "
            f"({train_metrics['duration_s']:.1f}s)"
        )

    logger.info(f"\n=== Training complete. Best val_acc: {best_val_acc:.4f} ===")
    logger.info(f"Best checkpoint saved to: {best_ckpt_path}")
    logger.info("Next step: run 'python training/export_onnx.py' to export to ONNX.")


if __name__ == "__main__":
    main()
