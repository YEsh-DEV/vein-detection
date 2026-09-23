#!/usr/bin/env python3
"""
capture_errors.py
-----------------
Canonical structured capture failure exceptions and machine-readable error definitions.
Preserves backward compatibility with legacy string-based handlers by subclassing ValueError.
"""

from typing import Optional, Dict, Any

try:
    from app.constants import (
        CODE_HAND_TOO_CLOSE,
        CODE_HAND_TOO_FAR,
        CODE_HAND_OUTSIDE_FRAME,
        CODE_MEDIAPIPE_NO_LANDMARKS,
        CODE_INVALID_LANDMARKS,
        CODE_VALLEY_EXTRACTION_FAILED,
        CODE_ROI_EXTRACTION_FAILED,
        CODE_QUALITY_LOW_CONTRAST,
        CODE_QUALITY_EXCESSIVE_PADDING,
        CODE_MODEL_NOT_LOADED,
        CODE_CAMERA_ERROR,
        CODE_UNKNOWN_PIPELINE_ERROR,
    )
except ImportError:
    from constants import (
        CODE_HAND_TOO_CLOSE,
        CODE_HAND_TOO_FAR,
        CODE_HAND_OUTSIDE_FRAME,
        CODE_MEDIAPIPE_NO_LANDMARKS,
        CODE_INVALID_LANDMARKS,
        CODE_VALLEY_EXTRACTION_FAILED,
        CODE_ROI_EXTRACTION_FAILED,
        CODE_QUALITY_LOW_CONTRAST,
        CODE_QUALITY_EXCESSIVE_PADDING,
        CODE_MODEL_NOT_LOADED,
        CODE_CAMERA_ERROR,
        CODE_UNKNOWN_PIPELINE_ERROR,
    )


# Priority ranking for selecting the most informative error in a multi-frame burst:
# Higher value indicates more actionable diagnostic specificity.
ERROR_PRIORITY: Dict[str, int] = {
    CODE_QUALITY_LOW_CONTRAST: 50,
    CODE_QUALITY_EXCESSIVE_PADDING: 50,
    CODE_VALLEY_EXTRACTION_FAILED: 40,
    CODE_ROI_EXTRACTION_FAILED: 40,
    CODE_HAND_TOO_CLOSE: 30,
    CODE_HAND_TOO_FAR: 30,
    CODE_HAND_OUTSIDE_FRAME: 30,
    CODE_INVALID_LANDMARKS: 20,
    CODE_MEDIAPIPE_NO_LANDMARKS: 15,
    CODE_CAMERA_ERROR: 10,
    CODE_MODEL_NOT_LOADED: 10,
    CODE_UNKNOWN_PIPELINE_ERROR: 0,
}


class CaptureError(ValueError):
    """
    Structured domain exception for biometric capture pipeline failures.
    Subclasses ValueError so existing 'except ValueError:' blocks catch it without breakage.
    """

    def __init__(
        self,
        error_code: str,
        instruction: str,
        stage: str = "pipeline",
        diagnostics: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(instruction)
        self.error_code = error_code
        self.instruction = instruction
        self.stage = stage
        self.diagnostics = diagnostics or {}

    @property
    def priority(self) -> int:
        return ERROR_PRIORITY.get(self.error_code, 10)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes error into a standardized API payload.
        'detail' is kept as the user-facing instruction string to preserve
        backward compatibility with existing frontend string-matching logic.
        """
        return {
            "error_code": self.error_code,
            "reason": self.error_code,
            "stage": self.stage,
            "instruction": self.instruction,
            "detail": self.instruction,
            "diagnostics": self.diagnostics,
        }

    def __repr__(self) -> str:
        return f"CaptureError(code={self.error_code}, stage={self.stage}, instruction={self.instruction!r})"
