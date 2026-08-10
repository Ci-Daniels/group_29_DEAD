"""
enrollment.py
-------------
Module 3: Face Enrollment

Captures ~20 face samples across different poses/expressions for a given
Beneficiary ID, extracts ArcFace embeddings for each, and persists them
as individual .npy files organized by Beneficiary ID.

Storing every sample separately (rather than averaging into one vector)
preserves pose/expression diversity, which verification can later exploit
via max-similarity matching for better robustness.

LIVE PREVIEW: a real-time OpenCV window guides the user through the
process (status, progress, bounding box, landmarks). To keep the preview
smooth, SCRFD/ArcFace inference only runs once every `frame_interval`
displayed frames — exactly the same cadence already used to space out
samples for pose variety. On the frames in between, the most recent
detection result is simply reused for drawing, so the preview never
waits on the AI models to refresh.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    EMBEDDINGS_DIR,
    ENROLLMENT_FRAME_INTERVAL,
    ENROLLMENT_SAMPLE_COUNT,
    RESULT_SCREEN_DURATION_SECONDS,
)
from modules.camera import CameraModule
from modules.exceptions import MultipleFacesDetectedError, NoFaceDetectedError
from modules.face_detector import DetectedFace, FaceDetector
from modules.liveness import LivenessCheck
from modules.preview import PreviewRenderer

_MODE_LABEL = "Facial Biometric Enrollment"
_COLOR_SUCCESS = (0, 200, 0)


class FaceEnrollment:
    """
    Orchestrates camera + face detector + liveness check + preview to
    enroll a new beneficiary's face embeddings.
    """

    def __init__(
        self,
        detector: FaceDetector,
        liveness_check: LivenessCheck,
        embeddings_dir: Path = EMBEDDINGS_DIR,
        sample_count: int = ENROLLMENT_SAMPLE_COUNT,
        frame_interval: int = ENROLLMENT_FRAME_INTERVAL,
    ) -> None:
        self.detector = detector
        self.liveness_check = liveness_check
        self.embeddings_dir = embeddings_dir
        self.sample_count = sample_count
        self.frame_interval = frame_interval

    '''def enroll(self, beneficiary_id: str) -> Path:
        """
        Run the full enrollment flow for a beneficiary, with a live preview.

        Args:
            beneficiary_id: unique identifier for the person being enrolled.

        Returns:
            Path to the directory containing the saved .npy embeddings.
        """
        beneficiary_dir = self.embeddings_dir / beneficiary_id
        beneficiary_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Starting enrollment for Beneficiary ID: {beneficiary_id} ===")
        print(f"Target samples: {self.sample_count}. Please move your head slightly "
              f"and vary your expression between captures.\n")

        preview = PreviewRenderer(window_name=f"{_MODE_LABEL} - {beneficiary_id}")
        samples_collected = 0'''
    
    def enroll(self, beneficiary_id: str, preview: Optional[PreviewRenderer] = None) -> Path:
        """
        Run the full enrollment flow for a beneficiary, with a live preview.

        Args:
            beneficiary_id: unique identifier for the person being enrolled.
            preview: optional PreviewRenderer (or compatible subclass, e.g.
                WebPreviewRenderer) to reuse. When None (CLI usage), a native
                OpenCV window is created exactly as before.

        Returns:
            Path to the directory containing the saved .npy embeddings.
        """
        beneficiary_dir = self.embeddings_dir / beneficiary_id
        beneficiary_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Starting enrollment for Beneficiary ID: {beneficiary_id} ===")
        print(f"Target samples: {self.sample_count}. Please move your head slightly "
              f"and vary your expression between captures.\n")

        preview = preview or PreviewRenderer(window_name=f"{_MODE_LABEL} - {beneficiary_id}")
        samples_collected = 0

        try:
            with CameraModule() as camera:
                # Liveness gate before we start collecting biometric data.
                # The same preview window stays open and gives live feedback.
                self.liveness_check.run(
                    camera, self.detector, preview=preview,
                    mode_label=_MODE_LABEL, beneficiary_id=beneficiary_id,
                )

                frame_counter = 0
                status = "Looking for face..."

                # Cache of the most recent detection result, reused on the
                # displayed frames where we deliberately skip inference.
                last_face: Optional[DetectedFace] = None

                while samples_collected < self.sample_count:
                    frame = camera.read_frame()
                    frame_counter += 1
                    progress_text = f"Sample {samples_collected} / {self.sample_count}"

                    run_inference = (frame_counter % self.frame_interval == 0)

                    if not run_inference:
                        # Reuse the previous detection for a smooth, low-
                        # latency preview without re-running SCRFD/ArcFace.
                        annotated = preview.render(
                            frame, _MODE_LABEL, beneficiary_id, status, progress_text,
                            bbox=last_face.bbox if last_face else None,
                            kps=last_face.kps if last_face else None,
                            landmark_106=last_face.landmark_106 if last_face else None,
                            det_score=last_face.det_score if last_face else None,
                        )
                        preview.show(annotated)
                        continue

                    # --- Inference cycle: run SCRFD + ArcFace this frame ---
                    try:
                        face = self.detector.detect_single_face(frame)
                    except NoFaceDetectedError:
                        last_face = None
                        status = "Looking for face..."
                        preview.show(preview.render(frame, _MODE_LABEL, beneficiary_id, status, progress_text))
                        print("  [!] No face detected — please face the camera.")
                        continue
                    except MultipleFacesDetectedError:
                        last_face = None
                        status = "Multiple faces detected"
                        preview.show(preview.render(frame, _MODE_LABEL, beneficiary_id, status, progress_text))
                        print("  [!] Multiple faces detected — only one person should be in frame.")
                        continue

                    last_face = face
                    status = "Capturing embedding..."
                    preview.show(preview.render(
                        frame, _MODE_LABEL, beneficiary_id, status, progress_text,
                        bbox=face.bbox, kps=face.kps, landmark_106=face.landmark_106,
                        det_score=face.det_score,
                    ))

                    sample_path = beneficiary_dir / f"sample_{samples_collected:02d}.npy"
                    np.save(sample_path, face.embedding)

                    samples_collected += 1
                    status = "Embedding stored"
                    progress_text = f"Sample {samples_collected} / {self.sample_count}"
                    preview.show(preview.render(
                        frame, _MODE_LABEL, beneficiary_id, status, progress_text,
                        bbox=face.bbox, kps=face.kps, landmark_106=face.landmark_106,
                        det_score=face.det_score,
                    ))

                    print(f"  [Enrollment] Captured sample {samples_collected}/{self.sample_count} "
                          f"(det_score={face.det_score:.2f}) -> {sample_path.name}")

                    # Small delay so consecutive frames aren't visually identical.
                    time.sleep(0.15)

                # --- Success screen, shown before the camera/preview close ---
                final_frame = camera.read_frame()
                result_frame = preview.render_result_screen(
                    final_frame,
                    headline="ENROLLMENT COMPLETE",
                    color=_COLOR_SUCCESS,
                    subtext=f"{samples_collected} samples stored for {beneficiary_id}",
                )
                preview.hold(result_frame, seconds=RESULT_SCREEN_DURATION_SECONDS)
        finally:
            preview.close()

        print(f"\n=== Enrollment complete for '{beneficiary_id}'. "
              f"{samples_collected} embeddings saved to: {beneficiary_dir} ===\n")
        return beneficiary_dir
