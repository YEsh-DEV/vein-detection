#!/usr/bin/env python3
"""
training/export_onnx.py
-----------------------
Export a trained AMPVNet PyTorch checkpoint to ONNX format for deployment
on the Raspberry Pi 5 via ONNX Runtime (CPU, ARM NEON acceleration).

Usage:
  # Export the Phase 2 fine-tuned checkpoint (recommended for deployment):
  python training/export_onnx.py \\
      --checkpoint training/checkpoints/finetune_best.pt \\
      --output models/ampvnet.onnx

  # Or export Phase 1 pretrained for testing:
  python training/export_onnx.py \\
      --checkpoint training/checkpoints/pretrain_best.pt \\
      --output models/ampvnet_pretrain.onnx

Output validation:
  The script runs a numerical equivalence check between the PyTorch and ONNX
  model outputs on a batch of synthetic inputs. Maximum absolute difference
  must be < 1e-5 for the export to be accepted.

Pi 5 deployment:
  Copy models/ampvnet.onnx to the Raspberry Pi and set:
      ONNX_MODEL_PATH = "models/ampvnet.onnx"  (in app/constants.py)
  The ONNX Runtime session in app/server.py loads this file directly.

  Expected inference latency on Pi 5 (Cortex-A76, 4 threads): 50–80ms
  (Luo et al. 2024 Table VI; actual values must be measured on your hardware)

Requirements:
  pip install onnx onnxruntime  (or onnxruntime-arm64 on the Pi)
"""

import sys
import os
import argparse
import logging
import numpy as np

import torch
import torch.nn as nn

_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TRAINING_DIR)

from model import AMPVNet

logger = logging.getLogger("onnx_export")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def check_onnxruntime():
    """Warns if onnxruntime is not installed but does not abort (ONNX validation optional)."""
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        logger.warning(
            "onnxruntime not installed. Skipping numerical validation.\n"
            "  Install with: pip install onnxruntime"
        )
        return False


def check_onnx():
    """Aborts if the `onnx` package is not installed."""
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        logger.error(
            "onnx package not installed. Cannot export.\n"
            "  Install with: pip install onnx"
        )
        return False


def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Loads the AMPVNet backbone from a training checkpoint.

    Returns:
        (backbone_model, checkpoint_dict)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first with: python training/train.py --phase pretrain"
        )

    ckpt = torch.load(checkpoint_path, map_location=device)

    backbone = AMPVNet(embedding_dim=512, dropout_p=0.2)
    backbone.load_state_dict(ckpt["model_state"])
    backbone.eval()
    backbone.to(device)

    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    logger.info(f"  Phase      : {ckpt.get('phase', 'unknown')}")
    logger.info(f"  Epoch      : {ckpt.get('epoch', 'unknown')}")
    logger.info(f"  best_val_acc: {ckpt.get('best_val_acc', float('nan')):.4f}")

    return backbone, ckpt


def export_to_onnx(
    backbone: nn.Module,
    output_path: str,
    input_size: tuple = (1, 3, 224, 224),
    opset_version: int = 14,
):
    """
    Exports AMPVNet to ONNX format with named dynamic batch axis.

    Args:
        backbone:      Trained AMPVNet model in eval mode.
        output_path:   Destination .onnx file path.
        input_size:    ONNX trace input shape (batch, channels, H, W).
        opset_version: ONNX opset. 14+ recommended for AvgPool and BatchNorm.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    dummy_input = torch.zeros(*input_size)

    logger.info(f"Exporting to ONNX (opset {opset_version})...")
    logger.info(f"  Input shape: {input_size}")
    logger.info(f"  Output: {output_path}")

    torch.onnx.export(
        backbone,
        dummy_input,
        output_path,
        input_names  = ["palm_roi"],
        output_names = ["embedding"],
        dynamic_axes = {
            "palm_roi":  {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        opset_version      = opset_version,
        do_constant_folding = True,
        export_params      = True,
    )
    logger.info(f"ONNX export complete: {output_path}")


def validate_onnx(
    onnx_path: str,
    backbone: nn.Module,
    n_samples: int = 4,
    tolerance: float = 1e-5,
):
    """
    Validates ONNX output against PyTorch output numerically.

    Runs `n_samples` random inputs through both PyTorch and ONNX Runtime,
    compares outputs. Max absolute diff must be < tolerance.

    Args:
        onnx_path:  Path to exported .onnx file.
        backbone:   PyTorch model (eval mode, same weights).
        n_samples:  Number of random test inputs.
        tolerance:  Maximum acceptable absolute difference.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnx or onnxruntime not installed. Skipping ONNX validation.")
        return

    # --- Check ONNX graph validity ---
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    logger.info(f"ONNX graph check passed ✓  (opset {model.opset_import[0].version})")

    # --- Create ONNX Runtime session ---
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1   # deterministic for comparison
    ort_session = ort.InferenceSession(onnx_path, sess_options=sess_options)
    input_name  = ort_session.get_inputs()[0].name

    max_diff = 0.0
    for i in range(n_samples):
        dummy = torch.randn(1, 3, 224, 224)

        # PyTorch forward
        with torch.no_grad():
            pt_out = backbone(dummy).numpy()

        # ONNX Runtime forward
        ort_out = ort_session.run(None, {input_name: dummy.numpy()})[0]

        diff = np.abs(pt_out - ort_out).max()
        max_diff = max(max_diff, diff)
        logger.info(f"  Sample {i+1}/{n_samples}: max_abs_diff = {diff:.2e}")

    logger.info(f"Max absolute difference across {n_samples} samples: {max_diff:.2e}")

    if max_diff < tolerance:
        logger.info(f"Numerical validation PASSED ✓ (tolerance={tolerance:.0e})")
    else:
        logger.error(
            f"Numerical validation FAILED — max diff {max_diff:.2e} exceeds "
            f"tolerance {tolerance:.0e}. Do NOT deploy this ONNX file."
        )
        sys.exit(1)


def log_model_metadata(onnx_path: str, ckpt: dict):
    """Appends human-readable metadata to a sidecar .json file next to the ONNX."""
    import json
    meta = {
        "model":         "AMPVNet",
        "embedding_dim": 512,
        "input_shape":   [1, 3, 224, 224],
        "input_dtype":   "float32",
        "input_range":   "normalized [-1, 1] approx via mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]",
        "output":        "L2-normalized 512-dim embedding on unit hypersphere",
        "output_note":   "Cosine similarity = dot product since both embeddings are unit vectors",
        "phase":         ckpt.get("phase", "unknown"),
        "epoch":         ckpt.get("epoch", "unknown"),
        "best_val_acc":  ckpt.get("best_val_acc", None),
        "num_classes":   ckpt.get("num_classes", None),
        # PLACEHOLDER threshold — must be calibrated with real enrolled data on Pi
        "match_threshold": "PLACEHOLDER — calibrate via tools/real_data_analysis.py",
        "paper":          "Luo et al., IEEE TIFS 2024, AMPVNet Table V",
        "target_device":  "Raspberry Pi 5, ONNX Runtime 1.18+ CPU",
        "expected_latency_ms": "50–80ms (from paper; measure on actual Pi hardware)",
    }
    meta_path = onnx_path.replace(".onnx", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Model metadata written to: {meta_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Export AMPVNet PyTorch checkpoint to ONNX for Raspberry Pi deployment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", default="training/checkpoints/finetune_best.pt",
        help="Path to the trained checkpoint (.pt). Use finetune_best.pt for production."
    )
    p.add_argument(
        "--output", default="models/ampvnet.onnx",
        help="Output ONNX file path. Deployment expects models/ampvnet.onnx."
    )
    p.add_argument(
        "--opset", type=int, default=14,
        help="ONNX opset version. 14+ required for BatchNorm + AvgPool."
    )
    p.add_argument(
        "--validate", action="store_true", default=True,
        help="Run numerical PyTorch vs ONNX Runtime validation after export."
    )
    p.add_argument(
        "--no_validate", dest="validate", action="store_false",
        help="Skip numerical validation (not recommended for production exports)."
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not check_onnx():
        sys.exit(1)

    device = torch.device("cpu")   # Export always on CPU for Pi compatibility

    # Load checkpoint
    backbone, ckpt = load_checkpoint(args.checkpoint, device)

    # Export to ONNX
    export_to_onnx(backbone, args.output, opset_version=args.opset)

    # Write metadata sidecar
    log_model_metadata(args.output, ckpt)

    # Validate
    if args.validate and check_onnxruntime():
        validate_onnx(args.output, backbone)

    logger.info("=" * 60)
    logger.info("EXPORT COMPLETE")
    logger.info(f"  ONNX file : {os.path.abspath(args.output)}")
    logger.info(f"  Copy this file to your Raspberry Pi under: models/ampvnet.onnx")
    logger.info(
        "  IMPORTANT: After deploying, run a short enrollment + recognition test\n"
        "  and calibrate MATCH_THRESHOLD via tools/real_data_analysis.py.\n"
        "  The threshold is NOT determined here — it must be measured on real hardware."
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
