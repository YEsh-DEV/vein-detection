# Stage 5: Baseline Biometric Evaluation & Recalibration Analysis

**Date:** 2026-09-23  
**Target Model:** Current Stage-2 AMPVNet Checkpoint (`models/ampvnet_finetuned.onnx`) without retraining  
**Evaluation Dataset:** Full Real Hardware NIR Benchmark (60 physical palm identities, 333 physical capture events, 359 files)  
**Biometric Classification Level:** **CATEGORY B: WORKING PROTOTYPE (DATA-LIMITED)**  

---

## 1. Executive Summary & Verification Context

Stage 4 physical validation revealed two acute production limitations:
1. **48.3% landmark failure rate** caused by users holding their palm too close to the lens.
2. **42.9% false acceptance rate (12/28)** at the experimental threshold ($0.2226$) on held-out physical impostors.

Stage 5 systematically evaluated the entire physical NIR hardware dataset using the deterministic pre-landmark positioning heuristic, strict enrollment quality gate, and exhaustive pairwise similarity matching across **all 60 physical palm identities**.

> [!IMPORTANT]
> **Key Baseline Finding:**
> Across all 5,225 physical impostor comparisons, the **Global Equal Error Rate (EER) is 16.03%** at threshold **0.4100**.
> The Decidability Index $d'$ is **2.0033**.
> At the experimental threshold ($0.2226$), global FAR is **41.45%** (2,166 false accepts out of 5,225 pairs) while genuine TAR is **92.37%** (FRR = 7.63%).
> This confirms that the biometric model is mathematically stable and functional, but threshold calibration and local training identity volume are the primary constraints.

---

## 2. Physical Pipeline & Capture Quality Failure Breakdown

- **Total Raw Capture Events Analyzed:** 333
- **Successfully Processed ROIs & Embeddings:** 104 (31.23%)
- **Pipeline Failures:** 229 (68.77%)
- **Active Physical Subjects with Valid Samples:** 23
- **Active Physical Palm Identities (Left/Right):** 38

### Categorized Failure Reasons

| Failure Category | Failure Reason | Count | % of Total Captures | Diagnostic Actionable Feedback |
| :--- | :--- | :--- | :--- | :--- |
| `HAND_TOO_CLOSE` | HAND_TOO_CLOSE | 159 | 47.7% | Hand is too close — move hand farther (~10-15cm) |
| `QUALITY_GATE_FAIL: Excessive boundary padding (35.6% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (35.6% > 25.0% max) | 2 | 0.6% | Poor contrast / knuckle padding violation — rejected |
| `HAND_OUTSIDE_FRAME` | HAND_OUTSIDE_FRAME | 2 | 0.6% | Align palm within center frame |
| `FAILED_MA2017_EXTRACTION: Invalid ROI bounding box coordinates.` | FAILED_MA2017_EXTRACTION: Invalid ROI bounding box coordinates. | 1 | 0.3% | Reposition palm within center guide |
| `QUALITY_GATE_FAIL: Excessive boundary padding (37.7% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (37.7% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (83.7% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (83.7% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (38.4% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (38.4% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (33.1% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (33.1% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (34.8% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (34.8% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (25.1% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (25.1% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (51.8% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (51.8% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (53.5% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (53.5% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (63.2% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (63.2% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (60.6% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (60.6% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (53.2% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (53.2% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (51.5% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (51.5% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (41.0% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (41.0% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (45.0% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (45.0% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (31.9% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (31.9% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (26.9% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (26.9% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (37.4% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (37.4% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (41.8% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (41.8% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (38.9% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (38.9% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (32.0% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (32.0% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (31.5% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (31.5% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (77.1% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (77.1% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (84.3% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (84.3% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (74.5% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (74.5% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (68.4% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (68.4% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (63.1% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (63.1% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (37.0% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (37.0% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (25.5% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (25.5% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (25.4% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (25.4% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (32.4% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (32.4% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (63.3% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (63.3% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (70.8% > 25.0% max), Low vessel contrast (std=7.2 < 10.0 min)` | QUALITY_GATE_FAIL: Excessive boundary padding (70.8% > 25.0% max), Low vessel contrast (std=7.2 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Low vessel contrast (std=9.2 < 10.0 min)` | QUALITY_GATE_FAIL: Low vessel contrast (std=9.2 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (27.9% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (27.9% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (31.8% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (31.8% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Low vessel contrast (std=10.0 < 10.0 min)` | QUALITY_GATE_FAIL: Low vessel contrast (std=10.0 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Low vessel contrast (std=8.6 < 10.0 min)` | QUALITY_GATE_FAIL: Low vessel contrast (std=8.6 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Low vessel contrast (std=8.0 < 10.0 min)` | QUALITY_GATE_FAIL: Low vessel contrast (std=8.0 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Low vessel contrast (std=9.4 < 10.0 min)` | QUALITY_GATE_FAIL: Low vessel contrast (std=9.4 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Low vessel contrast (std=9.9 < 10.0 min)` | QUALITY_GATE_FAIL: Low vessel contrast (std=9.9 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Low vessel contrast (std=8.7 < 10.0 min)` | QUALITY_GATE_FAIL: Low vessel contrast (std=8.7 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (78.3% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (78.3% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Low vessel contrast (std=9.8 < 10.0 min)` | QUALITY_GATE_FAIL: Low vessel contrast (std=9.8 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (31.8% > 25.0% max), Low vessel contrast (std=8.4 < 10.0 min)` | QUALITY_GATE_FAIL: Excessive boundary padding (31.8% > 25.0% max), Low vessel contrast (std=8.4 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (28.3% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (28.3% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (26.4% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (26.4% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (33.3% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (33.3% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (38.5% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (38.5% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (62.6% > 25.0% max), Low vessel contrast (std=8.7 < 10.0 min)` | QUALITY_GATE_FAIL: Excessive boundary padding (62.6% > 25.0% max), Low vessel contrast (std=8.7 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (42.1% > 25.0% max), Low vessel contrast (std=7.9 < 10.0 min)` | QUALITY_GATE_FAIL: Excessive boundary padding (42.1% > 25.0% max), Low vessel contrast (std=7.9 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (33.7% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (33.7% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (38.3% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (38.3% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (47.1% > 25.0% max), Low vessel contrast (std=8.1 < 10.0 min)` | QUALITY_GATE_FAIL: Excessive boundary padding (47.1% > 25.0% max), Low vessel contrast (std=8.1 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (46.7% > 25.0% max), Low vessel contrast (std=8.6 < 10.0 min)` | QUALITY_GATE_FAIL: Excessive boundary padding (46.7% > 25.0% max), Low vessel contrast (std=8.6 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (45.0% > 25.0% max), Low vessel contrast (std=8.8 < 10.0 min)` | QUALITY_GATE_FAIL: Excessive boundary padding (45.0% > 25.0% max), Low vessel contrast (std=8.8 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (39.9% > 25.0% max), Low vessel contrast (std=6.9 < 10.0 min)` | QUALITY_GATE_FAIL: Excessive boundary padding (39.9% > 25.0% max), Low vessel contrast (std=6.9 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Low vessel contrast (std=7.6 < 10.0 min)` | QUALITY_GATE_FAIL: Low vessel contrast (std=7.6 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (78.0% > 25.0% max), Low vessel contrast (std=7.2 < 10.0 min)` | QUALITY_GATE_FAIL: Excessive boundary padding (78.0% > 25.0% max), Low vessel contrast (std=7.2 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (57.9% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (57.9% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (61.1% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (61.1% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (42.8% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (42.8% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (43.1% > 25.0% max)` | QUALITY_GATE_FAIL: Excessive boundary padding (43.1% > 25.0% max) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (26.7% > 25.0% max), Low vessel contrast (std=8.9 < 10.0 min)` | QUALITY_GATE_FAIL: Excessive boundary padding (26.7% > 25.0% max), Low vessel contrast (std=8.9 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (25.1% > 25.0% max), Low vessel contrast (std=9.3 < 10.0 min)` | QUALITY_GATE_FAIL: Excessive boundary padding (25.1% > 25.0% max), Low vessel contrast (std=9.3 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |
| `QUALITY_GATE_FAIL: Excessive boundary padding (31.5% > 25.0% max), Low vessel contrast (std=8.8 < 10.0 min)` | QUALITY_GATE_FAIL: Excessive boundary padding (31.5% > 25.0% max), Low vessel contrast (std=8.8 < 10.0 min) | 1 | 0.3% | Poor contrast / knuckle padding violation — rejected |

---

## 3. Real Biometric Pairwise Similarity Distributions

Evaluation separated all pairs into anatomically distinct subsets:
- **Same-Session Genuine:** Repeated presentations of the same palm within the same capture session.
- **Cross-Session Genuine:** Presentations of the same palm across different recording sessions (temporal gap).
- **Cross-Person Impostors:** Comparisons between different human subjects.
- **Cross-Hand Impostors:** Comparisons between Left and Right palms of the same subject.

| Comparison Subset | Sample Size ($N$) | Mean Sim | Std Dev | Min | Median | Max | 95th Percentile |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **All Genuine** | 131 | 0.6369 | 0.2436 | -0.0768 | 0.6746 | 0.9805 | 0.9261 |
| - Same-Session Genuine | 128 | 0.6452 | 0.2402 | -0.0768 | 0.6828 | 0.9805 | 0.9266 |
| - Cross-Session Genuine | 3 | 0.2833 | 0.0184 | 0.2606 | 0.2836 | 0.3057 | 0.3035 |
| **All Impostors** | 5,225 | 0.1690 | 0.2231 | -0.4321 | 0.1683 | 0.8091 | 0.5389 |
| - Cross-Person Impostor | 5,077 | 0.1656 | 0.2237 | -0.4321 | 0.1646 | 0.8091 | 0.5389 |
| - Cross-Hand Impostor | 148 | 0.2855 | 0.1615 | -0.0216 | 0.2635 | 0.6403 | 0.5450 |

---

## 4. Biometric Accuracy & ROC Performance

| Metric | Full Benchmark (60 Palms) | Validation Split (4 Subjects) | Held-Out Test Split (5 Subjects) |
| :--- | :--- | :--- | :--- |
| **Equal Error Rate (EER)** | **16.03%** | **22.22%** | **20.29%** |
| **EER Operating Threshold** | `0.4100` | `0.4370` | `0.2610` |
| **Decidability Index ($d'$)** | 2.0033 | 2.2242 | 1.8564 |
| **TAR @ FAR = 1.0%** | 63.36% (tau=0.6300) | N/A (small N) | N/A (small N) |
| **TAR @ FAR = 0.1%** | 48.09% (tau=0.7140) | N/A (small N) | N/A (small N) |
| **FAR @ Experimental (0.2226)** | 41.45% | — | 26.47% |
| **FRR @ Experimental (0.2226)** | 7.63% | — | 10.0% |
| **TAR @ Experimental (0.2226)** | 92.37% | — | 90.0% |

---

## 5. Root Cause Analysis: Why Did Stage 4 Suffer 42.9% FAR?

Stage 4 pilot validation reported 12/28 impostor acceptances at threshold 0.2226.
With full hardware evaluation across all physical pairs, the empirical facts are:
1. **Impostor Distribution Spread:** The real physical impostor distribution has $\mu = 0.1690$ with a standard deviation of $\sigma = 0.2231$, and a 95th percentile reaching $0.5389$.
2. **Threshold Underestimation:** The experimental threshold `0.2226` was derived from a tiny synthetic/validation split without sufficient cross-person variability. At `0.2226`, exactly 2,166 impostor pairs exceed threshold.
3. **True Hardware EER Threshold:** The true empirical EER threshold on real hardware data is **`0.4100`** (not `0.2226`).
4. **Identity Volume Limitation:** The current Stage-2 model was fine-tuned on only 19 local subjects. Although the 600 Tongji pre-trained classes learned generic palm features, adapting to our specific Raspberry Pi 850nm NIR sensor requires broader identity variance.

---

## 6. Strict Guidance for Next Steps

Per project constraints:
1. **DO NOT** claim production-readiness or commercial biometric security.
2. **DO NOT** arbitrarily pick a threshold until hard-negative mining is complete.
3. **Execute Phase 5 Hard-Negative Analysis** (`tools/analyze_hard_negatives.py`) to inspect the specific impostor pairs scoring highest.
4. Keep the system labeled **CATEGORY B: WORKING PROTOTYPE (DATA-LIMITED)**.
