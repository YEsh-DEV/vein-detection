#!/usr/bin/env python3
"""
cnn_extractor.py
----------------
Compatibility adapter delegating directly to app/ampvnet_inference.py.
Preserves existing import contracts across server and test suites
while ensuring a single authoritative ONNX inference pipeline.
"""

from app.ampvnet_inference import (
    AMPVNetInference,
    get_inference_engine,
    extract_embedding,
    cosine_similarity,
    MODEL_LOADED,
    MODEL_ERROR_DETAIL,
    EMBEDDING_DIM,
)

__all__ = [
    "AMPVNetInference",
    "get_inference_engine",
    "extract_embedding",
    "cosine_similarity",
    "MODEL_LOADED",
    "MODEL_ERROR_DETAIL",
    "EMBEDDING_DIM",
]
