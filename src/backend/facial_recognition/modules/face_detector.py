"""
face_detector.py
-----------------
Module 2: Face Detection (+ embedding extraction)

Wraps InsightFace's FaceAnalysis application, which bundles:
  - SCRFD          -> fast, accurate face detection + 5-point keypoints
  - ArcFace (w600k) -> 512-d face recognition embedding
  - 2D-106 landmarks -> used by the liveness module for blink detection

This is the ONLY module that talks to InsightFace directly. Every other
module (enrollment, verification, liveness) depends on this class, not
on InsightFace itself — so the underlying recognition framework could
be swapped later with changes isolated to this one file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from insightface.app import FaceAnalysis

from src.backend.facial_recognition.config import (
    CTX_ID,
    DETECTION_SIZE,
    INSIGHTFACE_MODEL_PACK,
    MIN_DETECTION_CONFIDENCE,
)
from src.backend.facial_recognition.modules.exceptions import (
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)


@dataclass
class DetectedFace:
    """Lightweight container for the data we care about from an InsightFace result."""

    bbox: np.ndarray  # [x1, y1, x2, y2]
    det_score: float  # SCRFD detection confidence
    kps: np.ndarray  # 5-point keypoints (eyes, nose, mouth corners)
    embedding: (
        np.ndarray
    )  # 512-d ArcFace embedding (unnormalized, as given by InsightFace)
    landmark_106: Optional[np.ndarray]  # 106-point landmarks, if available


class FaceDetector:
    """
    Detects faces in a frame using SCRFD and extracts ArcFace embeddings.

    Usage:
        detector = FaceDetector()
        faces = detector.detect(frame)                  # 0..N faces
        face = detector.detect_single_face(frame)        # enforces exactly 1
    """

    def __init__(
        self,
        model_pack: str = INSIGHTFACE_MODEL_PACK,
        ctx_id: int = CTX_ID,
        detection_size: tuple[int, int] = DETECTION_SIZE,
    ) -> None:
        print(
            f"[FaceDetector] Loading InsightFace model pack '{model_pack}' "
            f"(this downloads weights on first run)..."
        )
        self._app = FaceAnalysis(name=model_pack)
        self._app.prepare(ctx_id=ctx_id, det_size=detection_size)
        print(
            "[FaceDetector] Model ready (SCRFD detector + ArcFace recognizer loaded)."
        )

    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        """
        Run SCRFD detection + ArcFace embedding extraction on a frame.

        Args:
            frame: BGR image as returned by CameraModule.read_frame().

        Returns:
            List of DetectedFace, one per face found (may be empty).
        """
        raw_faces = self._app.get(frame)

        results: List[DetectedFace] = []
        for f in raw_faces:
            if f.det_score < MIN_DETECTION_CONFIDENCE:
                continue  # skip low-confidence detections (likely false positives)
            results.append(
                DetectedFace(
                    bbox=f.bbox,
                    det_score=float(f.det_score),
                    kps=f.kps,
                    embedding=f.embedding,
                    landmark_106=getattr(f, "landmark_2d_106", None),
                )
            )
        return results

    def detect_single_face(self, frame: np.ndarray) -> DetectedFace:
        """
        Convenience wrapper that enforces exactly one face in the frame.

        This is the guard used by enrollment/verification, since both
        processes are meaningless if zero or multiple people are in frame.

        Raises:
            NoFaceDetectedError: if no face passes the confidence threshold.
            MultipleFacesDetectedError: if more than one face is detected.
        """
        faces = self.detect(frame)

        if len(faces) == 0:
            raise NoFaceDetectedError(
                "No face detected in frame. Ensure the subject is facing the "
                "camera with adequate lighting."
            )
        if len(faces) > 1:
            raise MultipleFacesDetectedError(
                f"Detected {len(faces)} faces in frame. Only one person should "
                "be in view during enrollment/verification."
            )
        return faces[0]
