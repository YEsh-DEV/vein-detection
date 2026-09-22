#!/usr/bin/env python3
"""
training/export_onnx.py
-----------------------
Exports trained AMPVNet PyTorch model to ONNX format (opset 17).

Specifications per Step 7:
  - Checkpoint: training/checkpoints/best.pt (from Step 5)
  - Output path: models/ampvnet.onnx (exact path expected by app/cnn_extractor.py)
  - Input shape: (1, 3, 224, 224)
  - Output shape: (1, 512) L2-normalized embedding
  - Opset: 17
  - Fixed batch size 1 (dynamic_axes=None for optimal Pi 5 inference)
  - Validation: onnxruntime.InferenceSession dummy forward, shape (1, 512), L2 norm ~1.0 (tol < 1e-3)
  - File size reporting: Target ~6.4MB (6-7MB range for 1.61M float32 params)
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn

_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

from model import AMPVNet


def export_ampvnet_onnx(
    checkpoint_path: str = "training/checkpoints/best.pt",
    output_path: str = "models/ampvnet.onnx",
    opset_version: int = 17,
) -> Path:
    ckpt_file = Path(checkpoint_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[export_onnx.py] Loading AMPVNet model...")
    model = AMPVNet(embedding_dim=512, dropout_p=0.2)

    if ckpt_file.exists():
        print(f"[export_onnx.py] Loading weights from checkpoint: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[export_onnx.py] WARNING: Checkpoint {ckpt_file} not found. Exporting initialized architecture.")

    model.eval()

    # Step 7 item 2: Input shape (1, 3, 224, 224), opset 17, fixed batch size 1
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    print(f"[export_onnx.py] Exporting to ONNX (opset {opset_version})...")
    print(f"  Input name : palm_roi (shape: {tuple(dummy_input.shape)})")
    print(f"  Output name: embedding (shape: (1, 512))")
    print(f"  Target file: {out_file}")

    torch.onnx.export(
        model,
        dummy_input,
        str(out_file),
        input_names=["palm_roi"],
        output_names=["embedding"],
        dynamic_axes=None,  # Fixed batch size 1 for Pi 5 optimization
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,  # Use classic TorchScript-based exporter for opset 17 and self-contained protobuf
    )

    print(f"[export_onnx.py] Export successful -> {out_file}")
    return out_file


def verify_onnx(
    onnx_path: Path,
    tolerance: float = 1e-3,
):
    print("\n[export_onnx.py] Verifying exported ONNX model with onnxruntime...")
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as e:
        print(f"ERROR: Missing onnx or onnxruntime: {e}")
        sys.exit(1)

    # 1. Structural graph check
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"  [1/4] ONNX graph check PASSED ✓ (opset {onnx_model.opset_import[0].version})")

    # 2. Run InferenceSession
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print(f"  [2/4] Input spec : name='{input_info.name}', shape={input_info.shape}, type={input_info.type}")
    print(f"        Output spec: name='{output_info.name}', shape={output_info.shape}, type={output_info.type}")

    assert input_info.shape == [1, 3, 224, 224], f"Unexpected input shape: {input_info.shape}"
    assert output_info.shape == [1, 512], f"Unexpected output shape: {output_info.shape}"

    # 3. Dummy forward pass with random input
    rng = np.random.RandomState(42)
    dummy_input = rng.randn(1, 3, 224, 224).astype(np.float32)

    ort_outputs = session.run([output_info.name], {input_info.name: dummy_input})
    emb = ort_outputs[0]

    assert emb.shape == (1, 512), f"Expected output shape (1, 512), got {emb.shape}"
    l2_norm = float(np.linalg.norm(emb, ord=2, axis=1)[0])
    norm_err = abs(l2_norm - 1.0)

    print(f"  [3/4] Dummy inference successful: output shape = {emb.shape}")
    print(f"        L2 norm of embedding = {l2_norm:.6f} (error = {norm_err:.2e}, tolerance = {tolerance})")
    assert norm_err < tolerance, f"Output not L2-normalized: norm = {l2_norm:.6f}, diff = {norm_err} >= {tolerance}"
    print(f"        L2 normalization verification PASSED ✓")

    # 4. File size check (Step 7 item 5: expected 6-7MB for 1.61M params)
    size_bytes = onnx_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"  [4/4] Exported file size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
    if 6.0 <= size_mb <= 7.0:
        print(f"        File size {size_mb:.2f} MB is within expected 6-7 MB range ✓")
    else:
        print(f"        WARNING: File size {size_mb:.2f} MB deviates from expected 6-7 MB range!")

    print("[export_onnx.py] STEP 7 ONNX VERIFICATION COMPLETE AND PASSED ✓\n")


def main():
    parser = argparse.ArgumentParser(description="Export AMPVNet to ONNX")
    parser.add_argument("--checkpoint", type=str, default="training/checkpoints/best.pt",
                        help="Path to checkpoint (.pt)")
    parser.add_argument("--output", type=str, default="models/ampvnet.onnx",
                        help="Path to save ONNX model")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    args = parser.parse_args()

    out_path = export_ampvnet_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
    )
    verify_onnx(out_path)


if __name__ == "__main__":
    main()
