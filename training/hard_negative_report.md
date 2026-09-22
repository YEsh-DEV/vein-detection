# Phase 5: Hard-Negative Biometric Dataset Report

**Date:** 2026-09-23  
**Model Evaluated:** `models/ampvnet_finetuned.onnx` (Current Stage-2 Checkpoint)  
**Threshold Analyzed:** `0.2226` (Experimental Operating Point)  
**Biometric Classification Level:** **CATEGORY B: WORKING PROTOTYPE (DATA-LIMITED)**  

---

## 1. Executive Summary: Hard-Negative Quantification

Across **5,225** total real hardware impostor comparisons, exactly **2,167 (41.47%)** pairs produced cosine similarities exceeding the experimental threshold ($0.2226$).

> [!IMPORTANT]
> **Key Structural Finding:**
> The high false acceptance rate (41.47%) at `0.2226` is **not caused by a small number of anomalous outliers**.
> It is caused by the global position of the impostor similarity distribution ($\mu = 0.1690$, $\sigma = 0.2231$).
> Because `0.2226` is only $0.24\sigma$ above the mean impostor similarity, 41.5% of random impostors exceed this threshold.
> When the threshold is set to the true empirical EER operating point (**`0.4100`**), hard negatives drop to **891 (17.05%)**.
> At the commercial FAR = 1% operating point (**`0.6300`**), hard negatives drop to **< 1.0%**.

---

## 2. Threshold Step-Down Breakdown

| Threshold ($	au$) | Impostor Acceptances | False Acceptance Rate (FAR) | Security Level |
| :--- | :--- | :--- | :--- |
| `0.2226` (Current Exp.) | 2,167 / 5,225 | **41.47%** | Extreme Risk (Pilot Calib) |
| `0.3000` | 1,530 / 5,225 | **29.28%** | High Risk |
| `0.4000` (~EER) | 891 / 5,225 | **17.05%** | Equal Error Balance (~16%) |
| `0.5000` | 415 / 5,225 | **7.94%** | Medium Security (~6%) |
| `0.6000` | 91 / 5,225 | **1.74%** | High Security (~1.5%) |
| `0.7000` | 7 / 5,225 | **0.13%** | Very High Security (< 0.2%) |

---

## 3. Top 25 Hardest Impostor Pairs Audited

Below are the highest-scoring false positive pairs identified across the physical hardware benchmark:

| Rank | Score | Pair Type | Subject A (Hand/Sess) | Subject B (Hand/Sess) | Contrast A / B | Padding A / B | Root Cause Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `0.8091` | Cross-Person | `046` (L/S1) | `035` (L/S2) | 33.6 / 16.8 | 0.0% / 0.0% | Spatial/Vascular Feature Overlap |
| 2 | `0.7589` | Cross-Person | `020` (L/S1) | `032` (L/S1) | 14.4 / 15.4 | 0.0% / 0.0% | Spatial/Vascular Feature Overlap |
| 3 | `0.7230` | Cross-Person | `037` (R/S1) | `035` (R/S2) | 27.7 / 23.3 | 3.2% / 0.0% | Spatial/Vascular Feature Overlap |
| 4 | `0.7202` | Cross-Person | `046` (L/S1) | `065` (L/S1) | 33.6 / 39.6 | 0.0% / 0.0% | Spatial/Vascular Feature Overlap |
| 5 | `0.7167` | Cross-Person | `037` (R/S1) | `065` (L/S1) | 27.7 / 39.6 | 3.2% / 0.0% | Spatial/Vascular Feature Overlap |
| 6 | `0.7139` | Cross-Person | `020` (R/S1) | `044` (R/S1) | 22.7 / 15.4 | 0.0% / 0.0% | Spatial/Vascular Feature Overlap |
| 7 | `0.7135` | Cross-Person | `037` (R/S1) | `050` (L/S1) | 27.7 / 26.6 | 3.2% / 0.0% | Spatial/Vascular Feature Overlap |
| 8 | `0.6841` | Cross-Person | `004` (R/S1) | `080` (L/S2) | 24.1 / 15.5 | 7.1% / 14.1% | Spatial/Vascular Feature Overlap |
| 9 | `0.6810` | Cross-Person | `020` (R/S1) | `048` (R/S1) | 22.7 / 21.2 | 0.0% / 0.0% | Spatial/Vascular Feature Overlap |
| 10 | `0.6735` | Cross-Person | `037` (R/S1) | `046` (L/S1) | 27.7 / 33.6 | 3.2% / 0.0% | Spatial/Vascular Feature Overlap |
| 11 | `0.6729` | Cross-Person | `020` (R/S1) | `080` (L/S2) | 22.7 / 15.5 | 0.0% / 14.1% | Spatial/Vascular Feature Overlap |
| 12 | `0.6696` | Cross-Person | `004` (L/S1) | `020` (R/S1) | 21.6 / 17.4 | 11.0% / 0.0% | Spatial/Vascular Feature Overlap |
| 13 | `0.6676` | Cross-Person | `037` (R/S1) | `018` (R/S2) | 27.7 / 27.3 | 3.2% / 0.0% | Spatial/Vascular Feature Overlap |
| 14 | `0.6670` | Cross-Person | `020` (R/S1) | `044` (R/S1) | 17.4 / 15.4 | 0.0% / 0.0% | Spatial/Vascular Feature Overlap |
| 15 | `0.6667` | Cross-Person | `046` (L/S1) | `035` (L/S2) | 33.6 / 14.3 | 0.0% / 0.0% | Spatial/Vascular Feature Overlap |
| 16 | `0.6662` | Cross-Person | `050` (L/S1) | `035` (R/S2) | 26.6 / 36.3 | 0.0% / 0.0% | Spatial/Vascular Feature Overlap |
| 17 | `0.6661` | Cross-Person | `004` (L/S1) | `020` (R/S1) | 21.6 / 22.7 | 11.0% / 0.0% | Spatial/Vascular Feature Overlap |
| 18 | `0.6628` | Cross-Person | `018` (L/S2) | `035` (L/S2) | 17.6 / 16.8 | 1.2% / 0.0% | Spatial/Vascular Feature Overlap |
| 19 | `0.6592` | Cross-Person | `014` (R/S1) | `104` (R/S2) | 19.0 / 16.7 | 22.0% / 24.0% | Boundary Replicate Artifact |
| 20 | `0.6586` | Cross-Person | `050` (L/S1) | `018` (R/S2) | 26.6 / 27.3 | 0.0% / 0.0% | Spatial/Vascular Feature Overlap |
| 21 | `0.6573` | Cross-Person | `050` (L/S1) | `035` (R/S2) | 26.6 / 23.3 | 0.0% / 0.0% | Spatial/Vascular Feature Overlap |
| 22 | `0.6552` | Cross-Person | `037` (R/S1) | `018` (R/S2) | 27.7 / 34.8 | 3.2% / 0.0% | Spatial/Vascular Feature Overlap |
| 23 | `0.6550` | Cross-Person | `020` (R/S1) | `048` (R/S1) | 20.0 / 21.2 | 0.0% / 0.0% | Spatial/Vascular Feature Overlap |
| 24 | `0.6534` | Cross-Person | `046` (L/S1) | `018` (L/S2) | 33.6 / 17.6 | 0.0% / 1.2% | Spatial/Vascular Feature Overlap |
| 25 | `0.6528` | Cross-Person | `037` (R/S1) | `065` (L/S1) | 19.5 / 39.6 | 0.0% / 0.0% | Spatial/Vascular Feature Overlap |

---

## 4. Root Cause Inspection & Visual Findings

Detailed inspection of the top false-accept pairs reveals three primary drivers:

### Driver A: Bilateral Vascular Symmetry (Cross-Hand Impostors)
- In several subjects (e.g. `004` Left vs `004` Right, `020` Left vs `020` Right), palmar superficial venous arches exhibit bilateral geometric symmetry.
- AMPVNet extracts deep multi-scale spatial receptive fields ($3\times 3, 5\times 5, 7\times 7$). When both left and right hands possess similar principal palmar arches, cosine similarity reaches $0.45 - 0.65$.

### Driver B: Local Training Identity Scarcity
- The Stage-2 checkpoint was fine-tuned on only 19 physical subjects.
- While the 600 Tongji classes taught the network general palm structure, fine-tuning with only 19 classes allowed the linear projection layer in AdaFace to collapse inter-class margins on hardware-specific illumination patterns.

### Driver C: Boundary Padding Artifacts
- Even after filtering with the Phase 3 quality gate ($\le 25\%$ padding), ROIs with $15-22\%$ replicated edge padding introduce uniform horizontal/vertical edge frequency signatures that artificially elevate cosine similarity.

---

## 5. Is Model Retraining Justified?

### **YES, BUT STRICTLY SEQUENCED**

Per project constraints, **DO NOT launch model retraining immediately**.
The evidence dictates the following exact sequence:
1. **Phase 1 Data Expansion:** Collect expanded real hardware identities (targeting 50-100+ physical palm classes).
2. **Strict Test Set Isolation:** Keep validation and test identities completely disjoint from training.
3. **Experiment 1 (Controlled):** Retrain AMPVNet + AdaFace using the expanded hardware dataset with the Phase 3 quality gate enforced on all training samples.
4. **Experiment 2 (Mining):** If inter-class margin overlap persists, apply online hard-negative mining (triplet loss or ArcFace with hard margin penalty on top impostor pairs identified in this report).

---

## 6. Biometric Assurance Status

The system remains firmly classified as:
### **CATEGORY B: WORKING PROTOTYPE (DATA-LIMITED)**
No production, commercial, or bank-grade claims are permitted.
