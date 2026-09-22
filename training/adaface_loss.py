#!/usr/bin/env python3
"""
training/adaface_loss.py
------------------------
AdaFace Classification Head & Loss Function for AMPVNet.

Ported directly from the official AdaFace reference implementation:
  Source Repository : https://github.com/mk-minchul/AdaFace (MIT License)
  Canonical File    : head.py
  Canonical Function: AdaFace.forward
  Paper References  :
    - Kim et al., "AdaFace: Quality Adaptive Margin for Face Recognition", CVPR 2022.
    - Luo et al., "Palm Vein Recognition Under Unconstrained and Weak-Cooperative
      Conditions", IEEE TIFS 2024 (Section IV-B2: AdaFace with m=0.55, h=0.29, s=50).
  System Spec       : system_architecture.md (Section 7, Eq. 6-10)

Key Implementation Details:
  1. W matrix of shape (num_classes, 512) is L2-normalized row-wise on each forward pass.
  2. cos(theta_j) is computed via linear projection of L2-normalized embeddings onto W.
  3. Feature norm ||z_i|| is computed BEFORE final L2 normalization in AMPVNet.
  4. Batch statistics mu_z and sigma_z are tracked via EMA (momentum=0.99) using
     registered buffers (persist in state_dict, move with .to(device), not trainable).
  5. Dynamic margins g_angle and g_add are computed per Eq. 8-9 and clipped to [-1, 1].
  6. Final loss: Softmax Cross-Entropy over scaled margin-adjusted logits.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_norm(tensor: torch.Tensor, axis: int = 1) -> torch.Tensor:
    """Ported from AdaFace/head.py:l2_norm"""
    norm = torch.norm(tensor, 2, dim=axis, keepdim=True)
    return tensor / (norm + 1e-8)


class AdaFace(nn.Module):
    """
    AdaFace Head ported from official AdaFace/head.py (Kim et al., CVPR 2022).

    Eq. 6-10 in system_architecture.md / Luo et al. IEEE TIFS 2024:
      ||z_hat_i|| = clip( (||z_i|| - mu_z) / (sigma_z / h), -1.0, 1.0 )
      g_angle = -m * ||z_hat_i||
      g_add   =  m * ||z_hat_i|| + m
      f(theta_j, m) = s * (cos(theta_j + g_angle) - g_add)   for j == y_i
                    = s * cos(theta_j)                       for j != y_i
    """

    def __init__(
        self,
        num_classes: int = 100,
        embedding_dim: int = 512,
        classnum: int = None,
        embedding_size: int = None,
        m: float = 0.55,
        h: float = 0.29,
        s: float = 50.0,
        t_alpha: float = 0.99,
    ):
        super().__init__()
        # Allow overridable hyperparameters and both naming conventions
        self.num_classes = classnum if classnum is not None else num_classes
        self.embedding_dim = embedding_size if embedding_size is not None else embedding_dim
        self.classnum = self.num_classes
        self.embedding_size = self.embedding_dim

        self.m = float(m)
        self.h = float(h)
        self.s = float(s)
        self.t_alpha = float(t_alpha)
        self.eps = 1e-3

        # Classifier weight matrix W: (num_classes, embedding_dim)
        # Ported from AdaFace/head.py: self.kernel (transposed to (num_classes, emb_dim) for F.linear)
        self.weight = nn.Parameter(torch.Tensor(self.num_classes, self.embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Batch statistics tracked via EMA as registered buffers (Eq. 10)
        # Not trainable parameters; persist with state_dict and move with .to(device)
        self.register_buffer("batch_mean", torch.ones(1) * 20.0)
        self.register_buffer("batch_std", torch.ones(1) * 100.0)

    def forward(
        self,
        embeddings: torch.Tensor,
        norms: torch.Tensor,
        labels: torch.Tensor,
        return_loss: bool = False,
    ) -> torch.Tensor:
        """
        Ported from AdaFace/head.py: AdaFace.forward.

        Args:
            embeddings: (B, 512) L2-normalized embedding vectors.
            norms: (B, 1) or (B,) pre-normalization feature norms ||z_i||.
            labels: (B,) class indices (torch.long).
            return_loss: If True, computes and returns scalar cross-entropy loss.
                         If False, returns scaled margin-adjusted logits (B, num_classes).
        """
        # Ensure norms are 1D (B,)
        norms = norms.view(-1)
        safe_norms = torch.clip(norms, min=0.001, max=100.0)

        # Step 1 & 2: L2-normalize W row-wise and compute cos(theta) = embeddings @ W.T
        W_norm = F.normalize(self.weight, p=2, dim=1)  # (num_classes, 512)
        cosine = F.linear(embeddings, W_norm)          # (B, num_classes)
        cosine = torch.clamp(cosine, -1.0 + self.eps, 1.0 - self.eps)

        # Step 4: EMA update of batch statistics (Eq. 10)
        if self.training:
            with torch.no_grad():
                mean = safe_norms.mean().detach()
                std = safe_norms.std().detach() if safe_norms.numel() > 1 else torch.zeros_like(mean)
                self.batch_mean = mean * (1.0 - self.t_alpha) + self.batch_mean * self.t_alpha
                self.batch_std = std * (1.0 - self.t_alpha) + self.batch_std * self.t_alpha

        # Step 5: Adaptive margin scaler and components (Eq. 8-9)
        # margin_scaler = clip((||z_i|| - mu_z) / (sigma_z / h), -1.0, 1.0)
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std / self.h + 1e-6)
        margin_scaler = torch.clip(margin_scaler, -1.0, 1.0)  # (B,)

        # g_angular = -m * ||z_hat_i||
        g_angular = -self.m * margin_scaler  # (B,)

        # Angular margin mask: scatter only to label[i]
        m_arc = torch.zeros_like(cosine)
        m_arc.scatter_(1, labels.view(-1, 1), 1.0)
        m_arc = m_arc * g_angular.view(-1, 1)

        theta = torch.acos(cosine)
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi - self.eps)
        cosine = torch.cos(theta_m)

        # g_additive = m * ||z_hat_i|| + m
        g_add = self.m + (self.m * margin_scaler)  # (B,)
        m_cos = torch.zeros_like(cosine)
        m_cos.scatter_(1, labels.view(-1, 1), 1.0)
        m_cos = m_cos * g_add.view(-1, 1)

        # Apply additive margin to target class
        cosine = cosine - m_cos

        # Step 7: Scale logits by s (Eq. 6)
        scaled_logits = cosine * self.s

        if return_loss:
            return F.cross_entropy(scaled_logits, labels)
        return scaled_logits


class AdaFaceLoss(nn.Module):
    """
    AdaFace Loss Module returning a scalar loss directly.
    Wraps AdaFace head and F.cross_entropy.
    """

    def __init__(
        self,
        num_classes: int = 100,
        embedding_dim: int = 512,
        classnum: int = None,
        embedding_size: int = None,
        m: float = 0.55,
        h: float = 0.29,
        s: float = 50.0,
        t_alpha: float = 0.99,
    ):
        super().__init__()
        self.head = AdaFace(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            classnum=classnum,
            embedding_size=embedding_size,
            m=m,
            h=h,
            s=s,
            t_alpha=t_alpha,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        norms: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Computes softmax cross-entropy over margin-adjusted logits."""
        return self.head(embeddings, norms, labels, return_loss=True)


class AMPVNetWithNorm(nn.Module):
    """
    Helper wrapper around AMPVNet that delegates to forward(x, return_norm=True).
    Maintained for clean backward compatibility.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor):
        return self.backbone(x, return_norm=True)
