#!/usr/bin/env python3
"""
training/evaluate.py
--------------------
Offline evaluation harness for the AMPVNet embedding model.

Computes:
  1. Cosine Verification Accuracy   — 1:1 pair-wise similarity
  2. Open-Set Rank-1 Identification — 1:N gallery search
  3. Equal Error Rate (EER)         — FAR/FRR operating curve crossing
  4. Threshold Calibration Report   — suggested cosine threshold range
     that achieves target FAR/FRR trade-off (e.g. FAR@0.1% target)

IMPORTANT — ALL threshold values printed by this script are EMPIRICAL ESTIMATES
based on the specific dataset provided. Do NOT hard-code them into constants.py
or server.py without:
  1. Testing on your actual enrolled users on the Pi hardware.
  2. Running with at least 5 genuine pairs and 5 impostor pairs per user.
Printed values are labelled "PLACEHOLDER — measure on real Pi data" explicitly.

Usage:
  python training/evaluate.py \\
      --checkpoint training/checkpoints/finetune_best.pt \\
      --data_root  training/data_raw/own \\
      --report     training/logs/evaluation_report.txt

Requirements:
  pip install scikit-learn
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TRAINING_DIR)

from model import AMPVNet
from augmentations import build_inference_transform
from dataset import PalmVeinDataset

logger = logging.getLogger("evaluate")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    backbone: AMPVNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Extracts L2-normalized embeddings for all samples in `loader`.

    Returns:
        embeddings: np.ndarray (N, 512)
        labels:     np.ndarray (N,)   — class/subject indices
    """
    backbone.eval()
    all_embs   = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        emb = backbone(images)   # (B, 512) — L2 normalized
        all_embs.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.vstack(all_embs), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_cosine_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    Computes pairwise cosine similarity matrix.
    Since embeddings are already L2-normalized, cosine = dot product.

    Args:
        emb_a: (M, D)
        emb_b: (N, D)

    Returns:
        similarity matrix: (M, N), values in [-1, 1]
    """
    return emb_a @ emb_b.T


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple:
    """
    Computes Equal Error Rate (EER) from binary labels and cosine scores.

    Args:
        labels: (N,) binary array — 1=genuine pair, 0=impostor pair
        scores: (N,) cosine similarity scores

    Returns:
        (eer, threshold_at_eer) — values in [0, 1]
    """
    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        logger.warning("scikit-learn not installed. Skipping EER computation.")
        return float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr

    # EER: point where FPR ≈ FNR
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    return float(eer), float(thresholds[eer_idx])


def compute_rank1(
    gallery_emb: np.ndarray,
    gallery_labels: np.ndarray,
    probe_emb: np.ndarray,
    probe_labels: np.ndarray,
) -> float:
    """
    Computes Rank-1 Identification Rate.
    For each probe, finds the closest gallery embedding and checks label match.

    Args:
        gallery_emb:    (G, D) enrolled embeddings
        gallery_labels: (G,)   enrolled subject labels
        probe_emb:      (P, D) probe embeddings
        probe_labels:   (P,)   probe subject labels

    Returns:
        Rank-1 accuracy in [0, 1]
    """
    sim = compute_cosine_matrix(probe_emb, gallery_emb)  # (P, G)
    best_gallery_idx = sim.argmax(axis=1)                 # (P,)
    predicted_labels = gallery_labels[best_gallery_idx]   # (P,)
    rank1 = (predicted_labels == probe_labels).mean()
    return float(rank1)


def build_genuine_impostor_pairs(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> tuple:
    """
    Builds ALL genuine and impostor pair scores from the embedding matrix.

    Genuine pairs:  same subject, different samples
    Impostor pairs: different subjects (uniformly subsampled if too many)

    Returns:
        pair_scores: (N,) cosine similarity for each pair
        pair_labels: (N,) 1=genuine, 0=impostor
    """
    n = len(labels)
    unique_labels = np.unique(labels)

    genuine_scores  = []
    impostor_scores = []

    # --- Genuine pairs ---
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        if len(idx) < 2:
            continue
        # All pairs within this subject
        emb_subj = embeddings[idx]
        sim_mat  = emb_subj @ emb_subj.T
        # Take upper triangle (exclude diagonal)
        triu_idx = np.triu_indices(len(idx), k=1)
        genuine_scores.extend(sim_mat[triu_idx].tolist())

    # --- Impostor pairs (cross-subject) ---
    # Compute full similarity matrix and collect cross-subject pairs
    sim_full = embeddings @ embeddings.T  # (N, N)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                impostor_scores.append(sim_full[i, j])

    # Subsample impostors to match genuine count (class balance for EER)
    n_genuine = len(genuine_scores)
    if len(impostor_scores) > n_genuine:
        rng = np.random.default_rng(42)
        impostor_scores = rng.choice(
            impostor_scores, size=n_genuine, replace=False
        ).tolist()

    all_scores = np.array(genuine_scores + impostor_scores)
    all_labels = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))

    return all_scores, all_labels


def far_at_target_frr(
    pair_labels: np.ndarray,
    pair_scores: np.ndarray,
    target_frr: float = 0.01,
) -> tuple:
    """
    Finds the cosine threshold where FRR = target_frr, and reports the FAR
    at that operating point.

    Returns:
        (threshold, far_at_threshold, actual_frr_at_threshold)
        All values in [0, 1].
    """
    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        return float("nan"), float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(pair_labels, pair_scores)
    frr = 1.0 - tpr

    # Find threshold where FRR first drops below or equals target_frr
    idx = np.where(frr <= target_frr)[0]
    if len(idx) == 0:
        return float("nan"), float("nan"), float("nan")

    idx = idx[0]   # first threshold achieving FRR <= target
    return float(thresholds[idx]), float(fpr[idx]), float(frr[idx])


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AMPVNet Embedding Evaluation — EER, Rank-1, Threshold Report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default="training/checkpoints/finetune_best.pt",
                   help="Path to trained checkpoint (.pt).")
    p.add_argument("--data_root", default="training/data_raw/own",
                   help="Dataset root for evaluation.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers",    type=int, default=2)
    p.add_argument("--report", default="training/logs/evaluation_report.txt",
                   help="Path to write the evaluation report.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    ckpt     = torch.load(args.checkpoint, map_location=device)
    backbone = AMPVNet(embedding_dim=512, dropout_p=0.2)
    backbone.load_state_dict(ckpt["model_state"])
    backbone.to(device).eval()

    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # --- Load dataset ---
    val_tf = build_inference_transform()
    ds     = PalmVeinDataset(args.data_root, transform=val_tf, split="all")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    logger.info(f"Dataset: {ds.describe()}")

    # --- Extract embeddings ---
    logger.info("Extracting embeddings...")
    embeddings, labels = extract_embeddings(backbone, loader, device)
    logger.info(f"  Embeddings shape: {embeddings.shape}")

    # --- Build pairs ---
    pair_scores, pair_labels = build_genuine_impostor_pairs(embeddings, labels)
    n_genuine  = int(pair_labels.sum())
    n_impostor = int((pair_labels == 0).sum())
    logger.info(f"  Genuine pairs : {n_genuine}")
    logger.info(f"  Impostor pairs: {n_impostor}")

    # --- EER ---
    eer, eer_threshold = compute_eer(pair_labels, pair_scores)
    logger.info(f"  EER           : {eer * 100:.2f}%")
    logger.info(f"  EER threshold : {eer_threshold:.4f}")

    # --- Threshold recommendations ---
    thr_1pct,  far_1pct,  frr_1pct  = far_at_target_frr(pair_labels, pair_scores, 0.01)
    thr_01pct, far_01pct, frr_01pct = far_at_target_frr(pair_labels, pair_scores, 0.001)

    # --- Rank-1 (use first-half as gallery, second-half as probe) ---
    n = len(embeddings)
    mid = n // 2
    gallery_emb, gallery_lbl = embeddings[:mid], labels[:mid]
    probe_emb,   probe_lbl   = embeddings[mid:], labels[mid:]
    rank1 = compute_rank1(gallery_emb, gallery_lbl, probe_emb, probe_lbl)
    logger.info(f"  Rank-1 Identification: {rank1 * 100:.2f}%")

    # --- Report ---
    report_lines = [
        "=" * 60,
        "AMPVNet Evaluation Report",
        "=" * 60,
        f"Checkpoint     : {os.path.abspath(args.checkpoint)}",
        f"Dataset        : {args.data_root}",
        f"Subjects       : {ds.num_classes}",
        f"Images         : {len(ds)}",
        f"Genuine Pairs  : {n_genuine}",
        f"Impostor Pairs : {n_impostor}",
        "",
        "--- Performance Metrics ---",
        f"EER                    : {eer * 100:.2f}%",
        f"EER Threshold          : {eer_threshold:.4f}",
        f"Rank-1 Identification  : {rank1 * 100:.2f}%",
        "",
        "--- Operating Point Recommendations (PLACEHOLDER) ---",
        "!!! These thresholds are estimated from this dataset ONLY. !!!",
        "!!! Calibrate on real Pi hardware with actual enrolled users. !!!",
        "",
        f"FRR=1.0% operating point:",
        f"  Threshold  : {thr_1pct:.4f}  [PLACEHOLDER — verify on Pi]",
        f"  FAR        : {far_1pct * 100:.4f}%",
        f"  actual FRR : {frr_1pct * 100:.4f}%",
        "",
        f"FRR=0.1% operating point (high security):",
        f"  Threshold  : {thr_01pct:.4f}  [PLACEHOLDER — verify on Pi]",
        f"  FAR        : {far_01pct * 100:.4f}%",
        f"  actual FRR : {frr_01pct * 100:.4f}%",
        "",
        "To apply a threshold in production:",
        "  1. Set MATCH_THRESHOLD in app/constants.py to the chosen value.",
        "  2. Mark it with '# CALIBRATED: <date> on <dataset>' comment.",
        "  3. Re-run this evaluation after every new enrollment.",
        "=" * 60,
    ]

    report_str = "\n".join(report_lines)
    logger.info("\n" + report_str)

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as f:
        f.write(report_str + "\n")

    logger.info(f"\nReport saved to: {args.report}")


if __name__ == "__main__":
    main()
