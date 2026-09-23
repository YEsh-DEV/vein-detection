#!/usr/bin/env python3
"""
tests/test_preprocessing_equivalence.py
---------------------------------------
Automated regression tests verifying exact mathematical equivalence between:
  1. PyTorch training preprocessing (training/dataset.py:preprocess_image_to_tensor)
  2. Production ONNX preprocessing (app/ampvnet_inference.py:preprocess_roi)
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "training"))

try:
    import torch
    from dataset import preprocess_image_to_tensor
    from model import AMPVNet
    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TORCH_AVAILABLE = False

from app.ampvnet_inference import AMPVNetInference, extract_embedding, cosine_similarity


class TestPreprocessingEquivalence(unittest.TestCase):
    """Verifies that production ONNX preprocessing exactly matches training PyTorch preprocessing."""

    @classmethod
    def setUpClass(cls):
        if not TORCH_AVAILABLE:
            raise unittest.SkipTest(
                "PyTorch is not installed (training-only dependency; not required on Pi runtime)."
            )
        cls.engine = AMPVNetInference()
        cls.test_images = list((_PROJECT_ROOT / "training/data_processed/own_splits/test").glob("*/*.png"))
        if not cls.test_images:
            # Fallback to any sample image
            cls.test_images = list((_PROJECT_ROOT / "training/data_processed/own").glob("*/*.png"))

    def test_tensor_numerical_equivalence_on_real_samples(self):
        """Feeds real palm ROI images into both PyTorch and ONNX preprocessors and asserts identical tensors."""
        self.assertTrue(len(self.test_images) > 0, "No test images found for regression testing.")

        for img_path in self.test_images[:10]:
            img_bgr = cv2.imread(str(img_path))
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # PyTorch pipeline
            torch_tensor = preprocess_image_to_tensor(str(img_path))  # (3, 224, 224)
            torch_np = torch_tensor.unsqueeze(0).numpy()             # (1, 3, 224, 224)

            # Production ONNX pipeline
            onnx_tensor = self.engine.preprocess_roi(img_gray)       # (1, 3, 224, 224)

            # Check shapes
            self.assertEqual(torch_np.shape, onnx_tensor.shape)

            # Numerical delta
            max_diff = np.max(np.abs(torch_np - onnx_tensor))
            self.assertLess(
                max_diff, 1e-4,
                f"Preprocessing mismatch on {img_path.name}: max_diff = {max_diff:.6e}"
            )

    def test_tensor_range_and_normalization_bounds(self):
        """Checks that preprocessed tensors map correctly into [-1.0, 1.0]."""
        dummy_roi = np.full((224, 224), 128, dtype=np.uint8)
        tensor = self.engine.preprocess_roi(dummy_roi)

        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)

        # Min and max checks: 0 -> (0 - 0.5)/0.5 = -1.0; 255 -> (255/255 - 0.5)/0.5 = 1.0
        black_tensor = self.engine.preprocess_roi(np.zeros((224, 224), dtype=np.uint8))
        self.assertAlmostEqual(float(black_tensor.min()), -1.0, places=5)

        white_tensor = self.engine.preprocess_roi(np.full((224, 224), 255, dtype=np.uint8))
        self.assertAlmostEqual(float(white_tensor.max()), 1.0, places=5)

    def test_arbitrary_shape_resizing(self):
        """Verifies that non-224x224 inputs (e.g. 256x256) are bicubic-resized cleanly."""
        larger_roi = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        tensor = self.engine.preprocess_roi(larger_roi)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_channel_handling_grayscale_and_bgr(self):
        """Verifies handling of 2D, 3D (H,W,1), and 3D (H,W,3) inputs."""
        gray_2d = np.ones((224, 224), dtype=np.uint8) * 100
        gray_3d_single = gray_2d[:, :, np.newaxis]
        bgr_3d = cv2.cvtColor(gray_2d, cv2.COLOR_GRAY2BGR)

        t_2d = self.engine.preprocess_roi(gray_2d)
        t_3d_single = self.engine.preprocess_roi(gray_3d_single)
        t_bgr = self.engine.preprocess_roi(bgr_3d)

        np.testing.assert_allclose(t_2d, t_3d_single, atol=1e-5)
        np.testing.assert_allclose(t_2d, t_bgr, atol=1e-5)

    def test_pytorch_vs_onnx_embedding_numerical_identity(self):
        """Tests that PyTorch model and ONNX Runtime session yield virtually identical embeddings."""
        ckpt_path = _PROJECT_ROOT / "training/checkpoints/stage2_expA/stage2_expA_best.pt"
        if not ckpt_path.exists():
            self.skipTest(f"Checkpoint not found at {ckpt_path}")

        pt_model = AMPVNet(embedding_dim=512, dropout_p=0.2)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        pt_model.load_state_dict(ckpt["model_state_dict"])
        pt_model.eval()

        for img_path in self.test_images[:5]:
            img_bgr = cv2.imread(str(img_path))
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # PyTorch inference
            torch_in = preprocess_image_to_tensor(str(img_path)).unsqueeze(0)
            with torch.no_grad():
                pt_emb = pt_model(torch_in).numpy().flatten()

            # ONNX inference
            onnx_emb = self.engine.extract_embedding(img_gray)

            # Assert shape & L2 norm
            self.assertEqual(onnx_emb.shape, (512,))
            self.assertAlmostEqual(float(np.linalg.norm(onnx_emb)), 1.0, places=5)

            # Compare embeddings
            sim = cosine_similarity(pt_emb, onnx_emb)
            max_abs_diff = np.max(np.abs(pt_emb - onnx_emb))

            self.assertGreater(sim, 0.9999, f"Low cosine similarity ({sim}) on {img_path.name}")
            self.assertLess(max_abs_diff, 1e-4, f"High absolute diff ({max_abs_diff}) on {img_path.name}")


if __name__ == "__main__":
    unittest.main()
