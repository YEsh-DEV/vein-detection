#!/usr/bin/env python3
"""
training/scripts/evaluate_new_user.py
-------------------------------------
Rigorous biometric evaluation harness for Stage 2:
  - Phase 6: Threshold determination on validation split (frozen threshold).
  - Phase 7: New-User Enrollment (zero retraining) on held-out test identities.
  - Phase 8: Multiple enrollment strategies (1-image, 2-image, multi-sample mean).
  - Phase 9: Impostor discrimination and ROC / EER analysis.
  - Phase 10: Cross-session verification (Session 1 enrollment -> Session 2 probe).
  - Phase 11: Quantitative answers to the product question.
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

import torch
from torch.utils.data import DataLoader

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT_DIR / "training"))

from model import AMPVNet
from dataset import PalmVeinDataset, preprocess_image_to_tensor
from eval import extract_embeddings, compute_eer_and_tar


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 New-User Enrollment & Biometric Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to AMPVNet checkpoint (.pt)")
    parser.add_argument("--val_dir", type=str, default="training/data_processed/own_splits/validation",
                        help="Validation directory for threshold calibration")
    parser.add_argument("--test_dir", type=str, default="training/data_processed/own_splits/test",
                        help="Held-out test directory for unseen user evaluation")
    parser.add_argument("--output_json", type=str, default="training/logs/new_user_enrollment_results.json",
                        help="Output path for JSON results")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> AMPVNet:
    model = AMPVNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def get_validation_threshold(
    model: AMPVNet,
    val_dir: str,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    """Phase 6: Calibrate and freeze threshold exclusively on validation set."""
    val_dataset = PalmVeinDataset(data_dir=val_dir, split="all", is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    embeddings, labels = extract_embeddings(model, val_loader, device)
    
    # Form all pairs on validation set
    n = len(labels)
    sim_matrix = np.dot(embeddings, embeddings.T)
    i_u, j_u = np.triu_indices(n, k=1)
    pair_sims = sim_matrix[i_u, j_u]
    is_same = (labels[i_u] == labels[j_u])
    
    gen_scores = pair_sims[is_same]
    imp_scores = pair_sims[~is_same]
    
    val_metrics = compute_eer_and_tar(gen_scores, imp_scores)
    frozen_threshold = float(val_metrics["threshold"])
    return frozen_threshold, val_metrics


def run_test_pair_evaluation(
    model: AMPVNet,
    test_dir: str,
    frozen_threshold: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Phases 6 & 9: Full pairwise verification on unseen test split."""
    test_dataset = PalmVeinDataset(data_dir=test_dir, split="all", is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    embeddings, labels = extract_embeddings(model, test_loader, device)
    
    n = len(labels)
    sim_matrix = np.dot(embeddings, embeddings.T)
    i_u, j_u = np.triu_indices(n, k=1)
    pair_sims = sim_matrix[i_u, j_u]
    is_same = (labels[i_u] == labels[j_u])
    
    gen_scores = pair_sims[is_same]
    imp_scores = pair_sims[~is_same]
    
    test_metrics = compute_eer_and_tar(gen_scores, imp_scores)
    
    # Performance at the FROZEN validation threshold
    fa_at_frozen = np.sum(imp_scores >= frozen_threshold)
    fr_at_frozen = np.sum(gen_scores < frozen_threshold)
    far_frozen = float(fa_at_frozen / max(1, len(imp_scores)))
    frr_frozen = float(fr_at_frozen / max(1, len(gen_scores)))
    tar_frozen = float(1.0 - frr_frozen)
    
    # Threshold sensitivity analysis
    sensitivity = []
    for th in np.arange(0.10, 0.75, 0.05):
        th_float = float(round(th, 2))
        fa = np.sum(imp_scores >= th_float)
        fr = np.sum(gen_scores < th_float)
        sensitivity.append({
            "threshold": th_float,
            "far_percent": float(fa / max(1, len(imp_scores)) * 100.0),
            "frr_percent": float(fr / max(1, len(gen_scores)) * 100.0),
            "tar_percent": float((1.0 - fr / max(1, len(gen_scores))) * 100.0),
        })
    
    return {
        "test_eer_percent": test_metrics["eer_percent"],
        "test_eer_threshold": test_metrics["threshold"],
        "tar_at_far_01_percent": test_metrics["tar_at_far_01_percent"],
        "d_prime": test_metrics["d_prime"],
        "num_genuine_pairs": len(gen_scores),
        "num_impostor_pairs": len(imp_scores),
        "genuine_stats": {
            "mean": float(np.mean(gen_scores)),
            "std": float(np.std(gen_scores)),
            "min": float(np.min(gen_scores)),
            "max": float(np.max(gen_scores)),
            "median": float(np.median(gen_scores)),
        },
        "impostor_stats": {
            "mean": float(np.mean(imp_scores)),
            "std": float(np.std(imp_scores)),
            "min": float(np.min(imp_scores)),
            "max": float(np.max(imp_scores)),
            "median": float(np.median(imp_scores)),
        },
        "frozen_threshold": frozen_threshold,
        "metrics_at_frozen_threshold": {
            "far_percent": far_frozen * 100.0,
            "frr_percent": frr_frozen * 100.0,
            "tar_percent": tar_frozen * 100.0,
        },
        "threshold_sensitivity": sensitivity,
    }


def run_enrollment_simulations(
    model: AMPVNet,
    test_dir: str,
    frozen_threshold: float,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Phases 7 & 8: Simulate real enrollment workflows.
    Evaluates:
      A. 1-image enrollment
      B. 2-image enrollment (normalized mean template)
      C. 3-image enrollment (normalized mean template)
    """
    test_dataset = PalmVeinDataset(data_dir=test_dir, split="all", is_train=False)
    
    # Group samples by subject
    subjects: Dict[str, List[np.ndarray]] = {}
    sample_paths: Dict[str, List[str]] = {}
    for path_str, _, subj_id in test_dataset.samples:
        tensor = preprocess_image_to_tensor(path_str).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(tensor).squeeze(0).cpu().numpy()  # (512,) L2-normalized
        if subj_id not in subjects:
            subjects[subj_id] = []
            sample_paths[subj_id] = []
        subjects[subj_id].append(emb)
        sample_paths[subj_id].append(path_str)
        
    all_subject_ids = sorted(list(subjects.keys()))
    
    results = {}
    for strategy_name, k_enroll in [("1_image", 1), ("2_images", 2), ("3_images", 3)]:
        genuine_matches = 0
        genuine_attempts = 0
        impostor_matches = 0
        impostor_attempts = 0
        
        gen_sims = []
        imp_sims = []
        
        for subj_id in all_subject_ids:
            embs = subjects[subj_id]
            n_samples = len(embs)
            if n_samples <= k_enroll:
                continue
                
            # Iterate through enrollment combinations (or sequential slices)
            for enroll_idx in range(n_samples - k_enroll + 1):
                enroll_embs = embs[enroll_idx : enroll_idx + k_enroll]
                # Template aggregation: normalized mean vector
                template = np.mean(enroll_embs, axis=0)
                template = template / (np.linalg.norm(template) + 1e-8)
                
                # Genuine probes: all other samples of this subject
                probe_indices = [idx for idx in range(n_samples) if idx < enroll_idx or idx >= enroll_idx + k_enroll]
                for p_idx in probe_indices:
                    probe_emb = embs[p_idx]
                    sim = float(np.dot(template, probe_emb))
                    gen_sims.append(sim)
                    genuine_attempts += 1
                    if sim >= frozen_threshold:
                        genuine_matches += 1
                        
                # Impostor probes: samples from all other subjects
                for other_id in all_subject_ids:
                    if other_id == subj_id:
                        continue
                    for other_emb in subjects[other_id]:
                        sim = float(np.dot(template, other_emb))
                        imp_sims.append(sim)
                        impostor_attempts += 1
                        if sim >= frozen_threshold:
                            impostor_matches += 1
                            
        tar = float(genuine_matches / max(1, genuine_attempts))
        frr = 1.0 - tar
        far = float(impostor_matches / max(1, impostor_attempts))
        
        results[strategy_name] = {
            "samples_enrolled": k_enroll,
            "genuine_attempts": genuine_attempts,
            "genuine_matches": genuine_matches,
            "impostor_attempts": impostor_attempts,
            "impostor_matches": impostor_matches,
            "tar_percent": tar * 100.0,
            "frr_percent": frr * 100.0,
            "far_percent": far * 100.0,
            "mean_genuine_sim": float(np.mean(gen_sims)) if gen_sims else 0.0,
            "mean_impostor_sim": float(np.mean(imp_sims)) if imp_sims else 0.0,
        }
        
    return results


def run_cross_session_evaluation(
    model: AMPVNet,
    test_dir: str,
    frozen_threshold: float,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Phase 10: Explicit cross-session evaluation for Subject 037.
    Enroll using Session 1 captures -> Probe using Session 2 captures.
    """
    test_path = Path(test_dir)
    subj_037_dir = test_path / "037"
    if not subj_037_dir.exists():
        return {"status": "Subject 037 directory not found"}
        
    s1_files = sorted([f for f in subj_037_dir.glob("S1_*.png")])
    s2_files = sorted([f for f in subj_037_dir.glob("S2_*.png")])
    
    if not s1_files or not s2_files:
        return {"status": "Missing S1 or S2 files for Subject 037"}
        
    def get_emb(p: Path) -> np.ndarray:
        tensor = preprocess_image_to_tensor(str(p)).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(tensor).squeeze(0).cpu().numpy()
            
    s1_embs = [get_emb(f) for f in s1_files]
    s2_embs = [get_emb(f) for f in s2_files]
    
    # Strategy 1: Single sample enrollment across sessions (all pairs S1 x S2)
    cross_sims = []
    for e1 in s1_embs:
        for e2 in s2_embs:
            sim = float(np.dot(e1, e2))
            cross_sims.append(sim)
            
    # Strategy 2: Multi-sample enrollment (mean template of S1) probed with S2
    s1_template = np.mean(s1_embs, axis=0)
    s1_template = s1_template / (np.linalg.norm(s1_template) + 1e-8)
    
    template_to_s2_sims = [float(np.dot(s1_template, e2)) for e2 in s2_embs]
    
    # Impostor cross-session probes: S1_037 template vs all other subjects
    other_impostor_sims = []
    for other_subj_dir in test_path.iterdir():
        if other_subj_dir.name == "037" or not other_subj_dir.is_dir():
            continue
        for other_file in other_subj_dir.glob("*.png"):
            sim = float(np.dot(s1_template, get_emb(other_file)))
            other_impostor_sims.append(sim)
            
    s1_mean_acc = float(np.mean([s >= frozen_threshold for s in template_to_s2_sims])) * 100.0
    imp_fa_rate = float(np.mean([s >= frozen_threshold for s in other_impostor_sims])) * 100.0 if other_impostor_sims else 0.0
    
    return {
        "num_s1_enrollment_samples": len(s1_files),
        "num_s2_probe_samples": len(s2_files),
        "pairwise_s1_s2_scores": {
            "mean": float(np.mean(cross_sims)),
            "std": float(np.std(cross_sims)),
            "min": float(np.min(cross_sims)),
            "max": float(np.max(cross_sims)),
            "acceptance_rate_percent": float(np.mean([s >= frozen_threshold for s in cross_sims])) * 100.0,
        },
        "s1_template_to_s2_probes": {
            "scores": template_to_s2_sims,
            "mean": float(np.mean(template_to_s2_sims)),
            "min": float(np.min(template_to_s2_sims)),
            "max": float(np.max(template_to_s2_sims)),
            "tar_percent": s1_mean_acc,
        },
        "impostor_discrimination": {
            "num_impostor_probes": len(other_impostor_sims),
            "mean": float(np.mean(other_impostor_sims)) if other_impostor_sims else 0.0,
            "max": float(np.max(other_impostor_sims)) if other_impostor_sims else 0.0,
            "far_percent": imp_fa_rate,
        },
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"\n==================================================================")
    print(f"      STAGE 2 EVALUATION & UNSEEN NEW-USER RECOGNITION HARNESS")
    print(f"==================================================================")
    print(f"Checkpoint under test: {args.checkpoint}")
    print(f"Device               : {device}")
    
    model = load_model(args.checkpoint, device)
    
    # Phase 6: Validation threshold calibration
    print("\n--- PHASE 6: CALIBRATING FROZEN THRESHOLD ON VALIDATION SET ---")
    frozen_threshold, val_metrics = get_validation_threshold(model, args.val_dir, device)
    print(f"Validation EER         : {val_metrics['eer_percent']:.2f}%")
    print(f"Validation Decidability: d' = {val_metrics['d_prime']:.4f}")
    print(f"Frozen Operating Thresh: {frozen_threshold:.4f}")
    print(f"Validation TAR @ FAR=1%: {val_metrics['tar_at_far_01_percent']:.2f}%")
    
    # Phase 6 & 9: Full Pairwise Verification on Held-Out Test Set
    print("\n--- PHASES 6 & 9: UNSEEN TEST PAIRWISE VERIFICATION ---")
    test_eval = run_test_pair_evaluation(model, args.test_dir, frozen_threshold, device)
    print(f"Unseen Test EER        : {test_eval['test_eer_percent']:.2f}% (thresh = {test_eval['test_eer_threshold']:.4f})")
    print(f"Unseen Test d'         : {test_eval['d_prime']:.4f}")
    print(f"Genuine Mean / Std     : {test_eval['genuine_stats']['mean']:.4f} ± {test_eval['genuine_stats']['std']:.4f}")
    print(f"Impostor Mean / Std    : {test_eval['impostor_stats']['mean']:.4f} ± {test_eval['impostor_stats']['std']:.4f}")
    print(f"Performance at Frozen Threshold ({frozen_threshold:.4f}):")
    print(f"  TAR: {test_eval['metrics_at_frozen_threshold']['tar_percent']:.2f}%")
    print(f"  FAR: {test_eval['metrics_at_frozen_threshold']['far_percent']:.2f}%")
    print(f"  FRR: {test_eval['metrics_at_frozen_threshold']['frr_percent']:.2f}%")
    
    # Phase 7 & 8: New-User Enrollment Simulations
    print("\n--- PHASES 7 & 8: NEW-USER ENROLLMENT SIMULATION ---")
    enroll_results = run_enrollment_simulations(model, args.test_dir, frozen_threshold, device)
    for strat, res in enroll_results.items():
        print(f"Strategy: {strat:9s} | Samples: {res['samples_enrolled']} | TAR: {res['tar_percent']:.2f}% | FAR: {res['far_percent']:.2f}% | Gen Mean: {res['mean_genuine_sim']:.4f} | Imp Mean: {res['mean_impostor_sim']:.4f}")
        
    # Phase 10: Cross-Session Verification
    print("\n--- PHASE 10: CROSS-SESSION VERIFICATION (Subject 037 S1 -> S2) ---")
    cross_session = run_cross_session_evaluation(model, args.test_dir, frozen_threshold, device)
    print(f"S1 Enrollment Samples  : {cross_session['num_s1_enrollment_samples']}")
    print(f"S2 Probe Samples       : {cross_session['num_s2_probe_samples']}")
    print(f"S1 Template -> S2 TAR  : {cross_session['s1_template_to_s2_probes']['tar_percent']:.2f}% (Mean Sim = {cross_session['s1_template_to_s2_probes']['mean']:.4f})")
    print(f"Impostor FAR on S1 Tmpl: {cross_session['impostor_discrimination']['far_percent']:.2f}% (Mean Sim = {cross_session['impostor_discrimination']['mean']:.4f})")
    
    # Aggregate and Save Report
    full_report = {
        "checkpoint": args.checkpoint,
        "phase6_validation_calibration": {
            "val_metrics": val_metrics,
            "frozen_threshold": frozen_threshold,
        },
        "phase9_test_verification": test_eval,
        "phase7_8_enrollment_simulations": enroll_results,
        "phase10_cross_session": cross_session,
    }
    
    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[evaluate_new_user.py] Comprehensive results saved to: {out_p}")
    print("==================================================================\n")


if __name__ == "__main__":
    main()
