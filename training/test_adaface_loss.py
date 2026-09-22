#!/usr/bin/env python3
"""
training/test_adaface_loss.py
-----------------------------
Unit test for AdaFace loss function.

Verifies:
  1. Forward pass with fake batch (batch_size=8, num_classes=10, embedding_dim=512).
  2. Embeddings are L2-normalized random vectors.
  3. Pre-normalization norms are positive tensors.
  4. Forward pass produces valid scalar loss without NaN or Inf.
  5. Backpropagation computes gradients for both backbone embeddings and AdaFace weight.
"""

import sys
import os
import torch
import torch.nn.functional as F

# Ensure training directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaface_loss import AdaFace, AdaFaceLoss


def test_adaface_loss():
    print("=== Testing AdaFace Loss (Step 2 Verification) ===")

    batch_size = 8
    num_classes = 10
    embedding_dim = 512

    # Instantiate loss with paper defaults
    criterion = AdaFaceLoss(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        m=0.55,
        h=0.29,
        s=50.0,
        t_alpha=0.99,
    )
    criterion.train()

    # Generate random normalized vectors for embeddings
    raw_emb = torch.randn(batch_size, embedding_dim, requires_grad=True)
    embeddings = F.normalize(raw_emb, p=2, dim=1)

    # Realistic pre-normalization norms (e.g. 15.0 - 25.0, typical for trained deep nets)
    norms = torch.empty(batch_size, 1).uniform_(15.0, 25.0)

    # Random target class labels
    labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

    # Run forward pass
    loss = criterion(embeddings, norms, labels)

    print(f"Batch Size     : {batch_size}")
    print(f"Num Classes    : {num_classes}")
    print(f"Embedding Dim  : {embedding_dim}")
    print(f"Loss Scalar    : {loss.item():.4f}")
    print(f"Loss is NaN    : {torch.isnan(loss).item()}")
    print(f"Loss is Inf    : {torch.isinf(loss).item()}")

    # Validations
    assert not torch.isnan(loss), "ERROR: AdaFace loss produced NaN!"
    assert not torch.isinf(loss), "ERROR: AdaFace loss produced Inf!"
    assert loss.dim() == 0, f"ERROR: Loss is not a scalar (dim={loss.dim()})"
    assert loss.item() > 0, f"ERROR: Loss is non-positive ({loss.item()})"

    # Backward pass check to ensure gradient flows
    loss.backward()
    assert criterion.head.weight.grad is not None, "AdaFace weight gradient is None!"
    assert raw_emb.grad is not None, "Embedding gradient is None!"

    print("Gradient Check : Weight grad norm = {:.4f}, Input grad norm = {:.4f}".format(
        criterion.head.weight.grad.norm().item(),
        raw_emb.grad.norm().item(),
    ))
    print("Batch Statistics: batch_mean = {:.2f}, batch_std = {:.2f}".format(
        criterion.head.batch_mean.item(),
        criterion.head.batch_std.item(),
    ))
    print("AdaFace Loss Unit Test PASSED ✓\n")


if __name__ == "__main__":
    test_adaface_loss()
