#!/usr/bin/env python3
"""
training/eval.py
----------------
Biometric Verification Evaluation Harness for AMPVNet.

Evaluates:
  1. Equal Error Rate (EER): Operating point where False Accept Rate (FAR) == False Reject Rate (FRR).
  2. TAR@FAR=0.01: True Accept Rate at 1% False Accept Rate.
  3. Balanced genuine (intra-class) and impostor (inter-class) pair formulation per
     Luo et al. (IEEE TIFS 2024, Section V-A).

Can be called directly from train.py during validation or run standalone against
a saved checkpoint (.pt).
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

from model import AMPVNet
from dataset import PalmVeinDataset


@torch.no_grad()
def extract_embeddings(
    model: AMPVNet,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts 512-dim L2-normalized embeddings for all images in dataloader.

    Returns:
        embeddings: (N, 512) float32 numpy array.
        labels: (N,) int64 numpy array.
    """
    model.eval()
    embeddings_list = []
    labels_list = []

    for images, labels in dataloader:
        images = images.to(device)
        emb = model(images)  # (B, 512), L2-normalized
        embeddings_list.append(emb.cpu().numpy())
        labels_list.append(labels.numpy())

    if not embeddings_list:
        return np.empty((0, 512), dtype=np.float32), np.empty((0,), dtype=np.int64)

    all_embeddings = np.vstack(embeddings_list)
    all_labels = np.concatenate(labels_list)
    return all_embeddings, all_labels


def form_balanced_pairs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forms balanced genuine pairs (same subject) and impostor pairs (different subjects)
    per Luo et al. Section V-A.

    Returns:
        genuine_scores: 1D array of cosine similarities for intra-class pairs.
        impostor_scores: 1D array of cosine similarities for inter-class pairs.
    """
    n_samples = len(labels)
    if n_samples < 2:
        return np.array([1.0]), np.array([0.0])

    genuine_scores = []
    impostor_candidates = []

    # Collect all possible pairs
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # Cosine similarity is dot product since embeddings are L2-normalized
            cos_sim = float(np.dot(embeddings[i], embeddings[j]))
            if labels[i] == labels[j]:
                genuine_scores.append(cos_sim)
            else:
                impostor_candidates.append(cos_sim)

    if not genuine_scores:
        # Fallback if every sample has unique identity
        genuine_scores = [1.0]

    n_genuine = len(genuine_scores)
    rng = random.Random(seed)

    # Balance impostor pairs to equal count of genuine pairs (Section V-A)
    if len(impostor_candidates) > n_genuine:
        impostor_scores = rng.sample(impostor_candidates, n_genuine)
    else:
        impostor_scores = impostor_candidates if impostor_candidates else [0.0]

    return np.array(genuine_scores, dtype=np.float32), np.array(impostor_scores, dtype=np.float32)


def compute_eer_and_tar(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    num_thresholds: int = 2000,
) -> Dict[str, float]:
    """
    Sweeps decision threshold across [-1.0, 1.0] to compute:
      - EER (Equal Error Rate where FAR == FRR)
      - EER threshold
      - TAR @ FAR = 0.01 (1% FAR)
    """
    thresholds = np.linspace(-1.0, 1.0, num_thresholds)
    n_gen = len(genuine_scores)
    n_imp = len(impostor_scores)

    far_list = []
    frr_list = []
    tar_list = []

    for thresh in thresholds:
        # False Accept: impostor score >= threshold
        fa = np.sum(impostor_scores >= thresh)
        far = fa / max(1, n_imp)

        # False Reject: genuine score < threshold
        fr = np.sum(genuine_scores < thresh)
        frr = fr / max(1, n_gen)

        tar = 1.0 - frr

        far_list.append(far)
        frr_list.append(frr)
        tar_list.append(tar)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    tar_arr = np.array(tar_list)

    # Equal Error Rate: point where |FAR - FRR| is minimal
    eer_idx = np.argmin(np.abs(far_arr - frr_arr))
    eer = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0)
    eer_thresh = float(thresholds[eer_idx])

    # TAR @ FAR <= 0.01
    # Filter thresholds where FAR <= 0.01
    valid_far_indices = np.where(far_arr <= 0.01)[0]
    if len(valid_far_indices) > 0:
        # Pick the lowest threshold that still satisfies FAR <= 0.01 (maximizes TAR)
        idx_far01 = valid_far_indices[0]
        tar_at_far01 = float(tar_arr[idx_far01])
        thresh_at_far01 = float(thresholds[idx_far01])
    else:
        tar_at_far01 = 0.0
        thresh_at_far01 = 1.0

    return {
        "eer": eer,
        "eer_percent": eer * 100.0,
        "threshold": eer_thresh,
        "tar_at_far_01": tar_at_far01,
        "tar_at_far_01_percent": tar_at_far01 * 100.0,
        "thresh_at_far_01": thresh_at_far01,
        "num_genuine": n_gen,
        "num_impostor": n_imp,
    }


def evaluate_model(
    model: AMPVNet,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates AMPVNet model on a validation DataLoader and returns biometric metrics.
    """
    embeddings, labels = extract_embeddings(model, val_loader, device)
    genuine_scores, impostor_scores = form_balanced_pairs(embeddings, labels)
    metrics = compute_eer_and_tar(genuine_scores, impostor_scores)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="AMPVNet Biometric EER Evaluation")
    parser.add_argument("--checkpoint", type=str, default="training/checkpoints/best.pt",
                        help="Path to saved model checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, default="training/data",
                        help="Path to validation data directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--split_ratio", type=float, default=0.5, help="Subject split ratio")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cuda or cpu)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[eval.py] Evaluation on device: {device}")

    # Load Model
    model = AMPVNet().to(device)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        print(f"[eval.py] Loading weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[eval.py] Checkpoint not found at {ckpt_path}. Running with initialized weights.")

    # Load Dataset
    val_dataset = PalmVeinDataset(
        data_dir=args.data_dir,
        split="val",
        split_ratio=args.split_ratio,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"[eval.py] Loaded {len(val_dataset)} validation samples across {val_dataset.num_classes} subjects.")

    metrics = evaluate_model(model, val_loader, device)

    print("\n" + "=" * 50)
    print("BIOMETRIC VERIFICATION PERFORMANCE REPORT")
    print("=" * 50)
    print(f"Genuine Pairs Evaluated  : {metrics['num_genuine']}")
    print(f"Impostor Pairs Evaluated : {metrics['num_impostor']}")
    print(f"Equal Error Rate (EER)   : {metrics['eer_percent']:.2f}% (thresh = {metrics['threshold']:.4f})")
    print(f"TAR @ FAR = 0.01 (1%)    : {metrics['tar_at_far_01_percent']:.2f}% (thresh = {metrics['thresh_at_far_01']:.4f})")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
