# Threshold Calibration Report — Stage 5 Recalibration

**Date:** 2026-09-23  
**Model:** `models/ampvnet_finetuned.onnx` (unchanged — no retraining)  
**Dataset:** Real NIR hardware captures, Raspberry Pi 850nm, 38 palm identities  
**Previous threshold:** `0.2226` (EER from Stage-2 synthetic validation split)  
**Selected new threshold:** `0.45`  
**Selection method:** Validation-split EER (N=18 genuine, N=135 impostor) at threshold ≈ 0.437

---

## 1. Problem Statement

The existing threshold `0.2226` was derived from a synthetic Stage-2 validation set EER. On real NIR hardware (Stage 5 evaluation), this threshold produced:

- **FAR = 41.32%** (2159 out of 5225 impostor pairs accepted — unacceptable for demo)
- **TAR = 92.37%** (121/131 genuine pairs accepted)

The impostor distribution on real hardware has a wide tail:
- Impostor mean similarity: `0.1690`
- Impostor std: `0.2231`
- Impostor 95th percentile: `0.5389`
- Impostor 99th percentile: `0.6296`

This means `0.2226` cuts through the impostor distribution at a very permissive point.

---

## 2. Dataset Size and Limitations

| Split | Genuine Pairs | Impostor Pairs | EER | EER Threshold |
|-------|--------------|----------------|-----|---------------|
| Global (all 38 palms) | 131 | 5225 | 16.03% | 0.41 |
| Validation (4 subjects) | 18 | 135 | 22.22% | 0.437 |
| Test (5 subjects, unseen) | 10 | 68 | 20.29% | 0.261 |
| Train (19 subjects) | 103 | 2525 | 14.59% | 0.414 |

> [!WARNING]
> The validation split has only **18 genuine pairs**. The EER estimate from this split has very wide confidence intervals. The selected threshold (0.45) should be treated as a reasonable demo operating point, not a statistically reliable biometric performance claim.

**Rule:** Threshold was selected from the **validation split EER** (0.437), NOT from the test split, to avoid threshold optimization on held-out test identities.

---

## 3. Threshold Sweep Table

Computed from the Stage-5 global ROC curve (131 genuine pairs, 5225 impostor pairs):

| Threshold | FAR%  | FRR%  | TAR%  | TNR%  | Gen Accept | Gen Reject | Imp Accept | Imp Reject |
|-----------|-------|-------|-------|-------|------------|------------|------------|------------|
| 0.2226 | 41.32 | 7.63 | 92.37 | 58.68 | 121 | 10 | 2159 | 3066 |
| 0.25 | 36.71 | 8.40 | 91.60 | 63.29 | 120 | 11 | 1918 | 3307 |
| 0.30 | 29.28 | 11.45 | 88.55 | 70.72 | 116 | 15 | 1530 | 3695 |
| 0.35 | 22.85 | 12.98 | 87.02 | 77.15 | 114 | 17 | 1194 | 4031 |
| 0.40 | 17.05 | 16.03 | 83.97 | 82.95 | 110 | 21 | 891 | 4334 |
| 0.41 (EER) | 16.04 | 16.03 | 83.97 | 83.96 | 110 | 21 | 838 | 4387 |
| **0.45 ← SELECTED** | **11.98** | **19.85** | **80.15** | **88.02** | **105** | **26** | **626** | **4599** |
| 0.50 | 7.94 | 25.19 | 74.81 | 92.06 | 98 | 33 | 415 | 4810 |
| 0.55 | 4.27 | 29.77 | 70.23 | 95.73 | 92 | 39 | 223 | 5002 |
| 0.60 | 1.74 | 34.35 | 65.65 | 98.26 | 86 | 45 | 91 | 5134 |
| 0.63 | 1.00 | 36.64 | 63.36 | 99.00 | 83 | 48 | 52 | 5173 |
| 0.65 | 0.50 | 40.46 | 59.54 | 99.50 | 78 | 53 | 26 | 5199 |
| 0.70 | 0.13 | 51.91 | 48.09 | 99.87 | 63 | 68 | 7 | 5218 |
| 0.75 | 0.04 | 59.54 | 40.46 | 99.96 | 53 | 78 | 2 | 5223 |

---

## 4. Selected Threshold: 0.45

### Selection Rationale

1. **Validation-split EER is at 0.437.** Setting the threshold just above the validation EER is a standard, conservative practice for biometric systems.
2. **Substantial FAR reduction:** From 41.32% → 11.98% (3.5× improvement in impostor rejection).
3. **Demo-viable TAR:** 80.15% (105/131 genuine pairs). Enrolled users are expected to be recognized in the majority of attempts.
4. **Not too aggressive:** At 0.50, TAR drops to 74.81% — enrolled users would see noticeable rejection rates during live demo, degrading confidence.
5. **Test split (unseen):** At 0.45, the test split FAR is estimated at approximately 9-15% (small N). This is not the basis of selection.

### Threshold vs. Old Threshold Comparison

| Metric | Old (0.2226) | New (0.45) | Change |
|--------|-------------|------------|--------|
| FAR% | 41.32 | 11.98 | -29.34 pts |
| FRR% | 7.63 | 19.85 | +12.22 pts |
| TAR% | 92.37 | 80.15 | -12.22 pts |
| Imp. Accepts | 2159 | 626 | -1533 |
| Gen. Accepts | 121 | 105 | -16 |

---

## 5. Configuration

The threshold is defined in **one authoritative location**: `app/constants.py`

```python
EXPERIMENTAL_MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.45"))
MATCH_THRESHOLD = EXPERIMENTAL_MATCH_THRESHOLD
```

Override at runtime: `MATCH_THRESHOLD=0.50 python -m uvicorn app.server:app`

---

## 6. Global Biometric Context

- **EER:** 16.03% @ threshold 0.41 (Decidability d' = 2.00)
- **TAR @ FAR=1%:** 63.36% @ threshold 0.63
- **TAR @ FAR=0.1%:** 48.09% @ threshold 0.714
- Genuine distribution: mean=0.637, std=0.244
- Impostor distribution: mean=0.169, std=0.223

---

## 7. Limitations and Disclaimer

> [!CAUTION]
> This system is a **CATEGORY B: WORKING PROTOTYPE (DATA-LIMITED)** research demonstration.
> It is NOT a certified, commercial, or production biometric system.

1. **Small dataset:** 131 genuine pairs from 38 palm identities. Real biometric evaluations use thousands of subjects.
2. **High EER:** 16.03% is above the threshold for any commercial deployment.
3. **Threshold has high uncertainty:** With 18 validation genuine pairs, the EER confidence interval is very wide.
4. **No independent test lab validation** has been performed.
5. **DO NOT** describe this system as "bank-grade", "commercial-grade", or "production-secure".
6. No retraining was performed. The AMPVNet model is unchanged.
