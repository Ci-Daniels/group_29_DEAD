"""
verification.py
----------------
Module 4: Face Verification

Captures a live face, generates its embedding, and compares it against
all previously enrolled embeddings for a given Beneficiary ID using
cosine similarity. Returns a VERIFIED / NOT VERIFIED decision plus the
similarity score.

LIVE PREVIEW: a real-time OpenCV window shows every stage (searching for
a face, liveness check, capturing the embedding, and the final result).
SCRFD/ArcFace inference is decoupled from the display refresh rate:
inference only runs once every `inference_interval` displayed frames, and
the previous detection is reused in between so the preview stays smooth.
The cosine-similarity matching logic itself is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from backend.facial_recognition.config import (
    EMBEDDINGS_DIR,
    RESULT_SCREEN_DURATION_SECONDS,
    VERIFICATION_INFERENCE_INTERVAL,
    VERIFICATION_SEARCH_TIMEOUT_FRAMES,
    VERIFICATION_THRESHOLD,
)
from backend.facial_recognition.modules.camera import CameraModule
from backend.facial_recognition.modules.camera_stream import BufferedCameraStream
from backend.facial_recognition.modules.exceptions import (
    BeneficiaryNotEnrolledError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from backend.facial_recognition.modules.face_detector import DetectedFace, FaceDetector
from backend.facial_recognition.modules.liveness import LivenessCheck
from backend.facial_recognition.modules.preview import PreviewRenderer

_MODE_LABEL = "Facial Biometric Verification"
_COLOR_SUCCESS = (0, 200, 0)
_COLOR_ERROR = (0, 0, 220)


@dataclass
class VerificationResult:
    """Structured result of a verification attempt."""

    verified: bool
    similarity_score: float
    beneficiary_id: str
    threshold_used: float

    def __str__(self) -> str:
        status = "VERIFIED" if self.verified else "NOT VERIFIED"
        return (
            f"[{status}] beneficiary_id={self.beneficiary_id} "
            f"similarity={self.similarity_score:.4f} threshold={self.threshold_used:.4f}"
        )


class FaceVerification:
    """
    Orchestrates camera + face detector + liveness check + preview to
    verify a live face against a beneficiary's enrolled embeddings.
    """

    def __init__(
        self,
        detector: FaceDetector,
        liveness_check: LivenessCheck,
        embeddings_dir: Path = EMBEDDINGS_DIR,
        threshold: float = VERIFICATION_THRESHOLD,
        inference_interval: int = VERIFICATION_INFERENCE_INTERVAL,
        search_timeout_frames: int = VERIFICATION_SEARCH_TIMEOUT_FRAMES,
    ) -> None:
        self.detector = detector
        self.liveness_check = liveness_check
        self.embeddings_dir = embeddings_dir
        self.threshold = threshold
        self.inference_interval = inference_interval
        self.search_timeout_frames = search_timeout_frames

    '''def verify(self, beneficiary_id: str) -> VerificationResult:
        """
        Run the full verification flow for a beneficiary, with a live preview.

        Args:
            beneficiary_id: identifier whose enrolled embeddings we compare against.

        Returns:
            VerificationResult with the VERIFIED/NOT VERIFIED decision and score.

        Raises:
            BeneficiaryNotEnrolledError: if no embeddings exist for this ID.
            NoFaceDetectedError: if no face is found before the search times out.
        """
        enrolled_embeddings = self._load_enrolled_embeddings(beneficiary_id)

        print(f"\n=== Starting verification for Beneficiary ID: {beneficiary_id} ===")

        preview = PreviewRenderer(window_name=f"{_MODE_LABEL} - {beneficiary_id}")'''

    def verify(
        self, beneficiary_id: str, preview: Optional[PreviewRenderer] = None
    ) -> VerificationResult:
        """
        Run the full verification flow for a beneficiary, with a live preview.

        Args:
            beneficiary_id: identifier whose enrolled embeddings we compare against.
            preview: optional PreviewRenderer (or compatible subclass, e.g.
                WebPreviewRenderer) to reuse. When None (CLI usage), a native
                OpenCV window is created exactly as before.

        Returns:
            VerificationResult with the VERIFIED/NOT VERIFIED decision and score.

        Raises:
            BeneficiaryNotEnrolledError: if no embeddings exist for this ID.
            NoFaceDetectedError: if no face is found before the search times out.
        """
        enrolled_embeddings = self._load_enrolled_embeddings(beneficiary_id)

        print(f"\n=== Starting verification for Beneficiary ID: {beneficiary_id} ===")

        preview = preview or PreviewRenderer(
            window_name=f"{_MODE_LABEL} - {beneficiary_id}"
        )
        live_embedding: Optional[np.ndarray] = None
        similarity_score: float = 0.0
        verified: bool = False

        try:
            with BufferedCameraStream(CameraModule()) as camera:
                if hasattr(preview, "attach_camera_stream"):
                    preview.attach_camera_stream(camera)
                # Liveness gate before accepting a live sample for matching.
                self.liveness_check.run(
                    camera,
                    self.detector,
                    preview=preview,
                    mode_label=_MODE_LABEL,
                    beneficiary_id=beneficiary_id,
                )

                last_face: Optional[DetectedFace] = None
                status = "Looking for face..."
                frame_counter = 0

                # --- Face-acquisition loop: search until we get one clean,
                # single-face frame to embed (or time out). This only adds
                # a retry-with-preview loop around the same single-face
                # detection call used before; the matching logic below is
                # untouched. ---
                for _ in range(self.search_timeout_frames):
                    frame = camera.read_frame()
                    frame_counter += 1
                    run_inference = frame_counter % self.inference_interval == 0

                    if not run_inference:
                        annotated = preview.render(
                            frame,
                            _MODE_LABEL,
                            beneficiary_id,
                            status,
                            bbox=last_face.bbox if last_face else None,
                            kps=last_face.kps if last_face else None,
                            landmark_106=last_face.landmark_106 if last_face else None,
                            det_score=last_face.det_score if last_face else None,
                        )
                        preview.show(annotated)
                        continue

                    try:
                        face = self.detector.detect_single_face(frame)
                    except NoFaceDetectedError:
                        last_face = None
                        status = "Looking for face..."
                        preview.show(
                            preview.render(frame, _MODE_LABEL, beneficiary_id, status)
                        )
                        continue
                    except MultipleFacesDetectedError:
                        last_face = None
                        status = "Multiple faces detected"
                        preview.show(
                            preview.render(frame, _MODE_LABEL, beneficiary_id, status)
                        )
                        continue

                    last_face = face
                    status = "Capturing embedding..."
                    preview.show(
                        preview.render(
                            frame,
                            _MODE_LABEL,
                            beneficiary_id,
                            status,
                            bbox=face.bbox,
                            kps=face.kps,
                            landmark_106=face.landmark_106,
                            det_score=face.det_score,
                        )
                    )

                    live_embedding = face.embedding
                    break

                if live_embedding is None:
                    raise NoFaceDetectedError(
                        "No face detected during verification within the search window."
                    )

                # --- Matching logic: UNCHANGED cosine-similarity comparison ---
                similarity_score = self._best_cosine_similarity(
                    live_embedding, enrolled_embeddings
                )
                verified = similarity_score >= self.threshold

                status = (
                    "Verification successful" if verified else "Verification failed"
                )
                status_frame = camera.read_frame()
                preview.show(
                    preview.render(
                        status_frame,
                        _MODE_LABEL,
                        beneficiary_id,
                        status,
                        progress=f"Similarity: {similarity_score:.4f}",
                        bbox=last_face.bbox if last_face else None,
                        kps=last_face.kps if last_face else None,
                    )
                )

                # --- Final result screen, shown before the camera/preview close ---
                headline = "VERIFIED" if verified else "NOT VERIFIED"
                color = _COLOR_SUCCESS if verified else _COLOR_ERROR
                result_frame = preview.render_result_screen(
                    status_frame,
                    headline=headline,
                    color=color,
                    subtext=f"Similarity score: {similarity_score:.4f} (threshold {self.threshold:.4f})",
                )
                preview.hold(result_frame, seconds=RESULT_SCREEN_DURATION_SECONDS)
        finally:
            preview.close()

        result = VerificationResult(
            verified=verified,
            similarity_score=similarity_score,
            beneficiary_id=beneficiary_id,
            threshold_used=self.threshold,
        )

        print(f"=== Verification result: {result} ===\n")
        return result

    # -- internal helpers (unchanged) --------------------------------------
    def _load_enrolled_embeddings(self, beneficiary_id: str) -> List[np.ndarray]:
        """Load all stored .npy embeddings for a beneficiary."""
        beneficiary_dir = self.embeddings_dir / beneficiary_id

        if not beneficiary_dir.exists():
            raise BeneficiaryNotEnrolledError(
                f"No enrollment data found for beneficiary_id='{beneficiary_id}'. "
                "Run enrollment first."
            )

        embedding_files = sorted(beneficiary_dir.glob("*.npy"))
        if not embedding_files:
            raise BeneficiaryNotEnrolledError(
                f"Enrollment folder for '{beneficiary_id}' exists but contains no "
                "embeddings. Re-run enrollment."
            )

        return [np.load(f) for f in embedding_files]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Standard cosine similarity between two embedding vectors."""
        numerator = float(np.dot(a, b))
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator == 0.0:
            return 0.0
        return numerator / denominator

    def _best_cosine_similarity(
        self, live_embedding: np.ndarray, enrolled_embeddings: List[np.ndarray]
    ) -> float:
        """
        Compare the live embedding against every enrolled sample and
        return the maximum similarity. Using max (rather than mean)
        makes verification robust to pose variation across the ~20
        enrolled samples — the live face only needs to closely match
        ONE of the enrolled poses/expressions.
        """
        scores = [
            self._cosine_similarity(live_embedding, ref) for ref in enrolled_embeddings
        ]
        return max(scores)
