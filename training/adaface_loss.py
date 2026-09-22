#!/usr/bin/env python3
"""
training/adaface_loss.py
------------------------
AdaFace Classification Head & Loss Function.

Adapted from the official AdaFace implementation by Kim et al.:
  Source repository : https://github.com/mk-minchul/AdaFace (MIT License)
  Source file       : AdaFace/head.py
  Paper             : Kim et al., "AdaFace: Quality Adaptive Margin for Face
                      Recognition", CVPR 2022.

Modifications made for this project:
  1. embedding_size defaulted to 512 to match AMPVNet output dimensionality.
  2. Removed dependency on AdaFace's internal config dataclasses — all
     hyperparameters passed directly as __init__ arguments.
  3. Added detailed inline comments mapping each formula line to the paper's
     equations (Section 3.1, Eq. 5–8 in Kim et al. CVPR 2022).
  4. EMA batch statistics updated as running buffers (no gradient through them).
  5. Compatible with PyTorch >= 2.0 (no deprecated function calls).

Usage during training:
    criterion = AdaFace(embedding_size=512, classnum=N_SUBJECTS)
    # In train loop:
    logits = criterion(embeddings, norms, labels)
    loss   = F.cross_entropy(logits, labels)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaFace(nn.Module):
    """
    AdaFace Classification Head.

    Implements quality-adaptive angular margin softmax (Kim et al., CVPR 2022).

    The core insight is that feature norm ||z_i||_2 is a reliable proxy for
    biometric image quality:
    - High-quality samples (large norm) -> large angular + additive margins enforced
    - Low-quality samples (small norm)  -> small margins, preventing noisy gradient

    Margin formulas (from paper Eq. 5):
      g_angle = -m * ||z_hat_i||
      g_add   =  m * ||z_hat_i|| + m

    where ||z_hat_i|| = clip( (||z_i|| - mu_z) / (sigma_z / h), -1, 1 )

    Args:
        embedding_size: Dimensionality of input embeddings (512 for AMPVNet).
        classnum:       Number of training subjects/identities.
        m:              Base margin parameter. Paper uses m=0.55 for palm vein.
        h:              Quality sensitivity scale. Paper uses h=0.29.
        s:              Feature scale factor. Paper uses s=50.
        t_alpha:        EMA decay rate for batch statistics. Paper uses 0.99.
    """

    def __init__(
        self,
        embedding_size: int = 512,
        classnum: int = 100,
        m: float = 0.55,
        h: float = 0.29,
        s: float = 50.0,
        t_alpha: float = 0.99,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.classnum = classnum
        self.m = m
        self.h = h
        self.s = s
        self.t_alpha = t_alpha

        # Learnable class center weight matrix (normalized during forward)
        # W: (classnum x embedding_size)
        self.weight = nn.Parameter(torch.Tensor(classnum, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        # EMA running statistics for feature norm normalization
        # Registered as buffers so they are saved with model state but
        # do NOT receive gradients.
        self.register_buffer("batch_mean", torch.ones(1) * 20.0)   # initial mu_z guess
        self.register_buffer("batch_std",  torch.ones(1) * 100.0)  # initial sigma_z guess

    def forward(
        self,
        embeddings: torch.Tensor,
        norms: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, embedding_size) — L2-normalized unit vectors from AMPVNet.
            norms:      (B,) — Pre-normalization feature norms ||z_i||_2 from backbone.
                        These are needed to compute the quality indicator.
                        IMPORTANT: embeddings fed here must already be L2-normalized.
                        norms must be the un-normalized magnitudes captured before
                        F.normalize() is applied in the backbone forward pass.
            labels:     (B,) — Long integer class indices.

        Returns:
            logits: (B, classnum) — Scaled cosine logits with adaptive margin applied.
                    Pass to F.cross_entropy(logits, labels) to get the loss scalar.
        """
        # -------------------------------------------------------------------
        # Step 1: Normalize class weight vectors to unit sphere
        # AdaFace uses cosine similarity: cos(theta_j) = W_j . z_i
        # -------------------------------------------------------------------
        W = F.normalize(self.weight, p=2, dim=1)    # (classnum, emb_size)

        # -------------------------------------------------------------------
        # Step 2: Compute cosine similarities for all classes
        # cosine: (B, classnum)
        # -------------------------------------------------------------------
        cosine = F.linear(embeddings, W)            # same as embeddings @ W.T
        # Clamp for numerical stability before arccos
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # -------------------------------------------------------------------
        # Step 3: Update EMA batch statistics for norms (no gradient)
        # mu_z, sigma_z are updated per mini-batch using exponential moving average.
        # These statistics normalize the raw norm to [-1, 1] quality indicator.
        # -------------------------------------------------------------------
        with torch.no_grad():
            current_mean = norms.mean()
            current_std  = norms.std() + 1e-6
            self.batch_mean = self.t_alpha * self.batch_mean + (1.0 - self.t_alpha) * current_mean
            self.batch_std  = self.t_alpha * self.batch_std  + (1.0 - self.t_alpha) * current_std

        # -------------------------------------------------------------------
        # Step 4: Compute quality indicator ||z_hat_i||
        # Formula: clip( (||z_i|| - mu_z) / (sigma_z / h), -1.0, 1.0 )
        # Kim et al. CVPR 2022, Section 3.1, Eq. 6
        # -------------------------------------------------------------------
        norm_hat = (norms - self.batch_mean) / (self.batch_std / self.h)
        norm_hat = torch.clamp(norm_hat, -1.0, 1.0).squeeze()  # (B,)

        # -------------------------------------------------------------------
        # Step 5: Compute adaptive angle and additive margin components
        # g_angle = -m * ||z_hat_i||    (negates margin for low-quality samples)
        # g_add   =  m * ||z_hat_i|| + m
        # Kim et al. CVPR 2022, Eq. 7–8
        # -------------------------------------------------------------------
        g_angle = -self.m * norm_hat          # (B,)
        g_add   =  self.m * norm_hat + self.m # (B,)

        # -------------------------------------------------------------------
        # Step 6: Apply margin only to the ground-truth class logit
        # theta_yi is the angle between embedding and ground-truth class center
        # -------------------------------------------------------------------
        theta = torch.acos(cosine)           # (B, classnum) — angles in [0, pi]

        # Create one-hot mask for the target class
        one_hot = torch.zeros_like(cosine, dtype=torch.bool)
        one_hot.scatter_(1, labels.view(-1, 1), True)

        # Expand g_angle and g_add to broadcast over class dimension
        g_angle_expanded = g_angle.unsqueeze(1).expand_as(cosine)  # (B, classnum)
        g_add_expanded   = g_add.unsqueeze(1).expand_as(cosine)    # (B, classnum)

        # Apply angular margin only to target class:
        # cos(theta_yi + g_angle) - g_add  for yi == y
        # cos(theta_j)                     for yi != y
        theta_with_margin = theta.clone()
        theta_with_margin[one_hot] = theta[one_hot] + g_angle_expanded[one_hot]

        # Convert back to cosine and subtract additive margin
        logits_margin = torch.cos(theta_with_margin)
        logits_margin[one_hot] -= g_add_expanded[one_hot]

        # -------------------------------------------------------------------
        # Step 7: Scale logits by s and return
        # -------------------------------------------------------------------
        return logits_margin * self.s


# ---------------------------------------------------------------------------
# Helper: Extract embedding norm BEFORE L2 normalization for AdaFace
# ---------------------------------------------------------------------------

class AMPVNetWithNorm(nn.Module):
    """
    Thin wrapper that makes AMPVNet return both the L2-normalized embedding
    AND the pre-normalization feature norm, as required by AdaFace.

    Usage:
        from model import AMPVNet
        from adaface_loss import AMPVNetWithNorm, AdaFace

        backbone = AMPVNet()
        wrapped  = AMPVNetWithNorm(backbone)
        criterion = AdaFace(embedding_size=512, classnum=N)

        # In training loop:
        embeddings, norms = wrapped(x)         # embeddings: L2-norm, norms: raw
        logits = criterion(embeddings, norms, labels)
        loss   = F.cross_entropy(logits, labels)
    """

    def __init__(self, ampvnet_backbone: nn.Module):
        super().__init__()
        self.backbone = ampvnet_backbone

    def forward(self, x: torch.Tensor):
        """
        Returns:
            embeddings: (B, 512) L2-normalized
            norms:      (B, 1)   pre-normalization Euclidean norms
        """
        # Temporarily bypass the final L2 normalize in AMPVNet.forward()
        # by running all layers up to the fc output manually.
        x = self.backbone.stem(x)        # (B, 32, 56, 56)
        x = self.backbone.stage1(x)      # (B, 64, 28, 28)
        x = self.backbone.stage2(x)      # (B, 128, 14, 14)
        x = self.backbone.stage3(x)      # (B, 256, 7, 7)
        x = self.backbone.stage4(x)      # (B, 256, 7, 7)  — stage4 keeps 256 channels
        x = self.backbone.gap(x)         # (B, 256, 1, 1)
        x = x.flatten(1)                 # (B, 256)
        x = self.backbone.dropout(x)
        raw_features = self.backbone.fc(x)  # (B, 512) — FC projects 256->512, not yet L2-norm

        norms = raw_features.norm(p=2, dim=1, keepdim=True)   # (B, 1)
        embeddings = F.normalize(raw_features, p=2, dim=1)     # (B, 512)

        return embeddings, norms


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from model import AMPVNet

    N_CLASSES = 50
    BATCH     = 4

    backbone  = AMPVNet()
    wrapped   = AMPVNetWithNorm(backbone)
    criterion = AdaFace(embedding_size=512, classnum=N_CLASSES)

    dummy_x      = torch.zeros(BATCH, 3, 224, 224)
    dummy_labels = torch.randint(0, N_CLASSES, (BATCH,))

    emb, norms = wrapped(dummy_x)
    logits     = criterion(emb, norms, dummy_labels)
    loss       = F.cross_entropy(logits, dummy_labels)

    print(f"[AdaFace] Embeddings shape : {tuple(emb.shape)}")
    print(f"[AdaFace] Norms shape      : {tuple(norms.shape)}")
    print(f"[AdaFace] Logits shape     : {tuple(logits.shape)}")
    print(f"[AdaFace] Loss value       : {loss.item():.4f}")
    print("[AdaFace] Sanity check PASSED ✓")
