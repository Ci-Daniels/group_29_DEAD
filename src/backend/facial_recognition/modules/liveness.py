"""
liveness.py
-----------
Module 5: Simple Liveness Check

Requires the user to blink or turn their head before enrollment/verification
proceeds. This is a basic, camera-only heuristic — NOT a robust anti-spoofing
solution (it will not reliably stop a printed photo or a video replay).

DESIGN NOTE: This module is intentionally isolated behind a single public
method, `run()`, which only depends on CameraModule and FaceDetector. It
does not know anything about enrollment or verification logic. This makes
it a drop-in replacement point: swap this class for an AI-based
anti-spoofing model later without touching any other module.

The optional `preview` parameter only adds live visual feedback (bounding
box, landmarks, and a "Liveness check..." status) on top of the SAME
per-frame detection this module already performs — it does not add any
extra inference calls or change the blink/head-turn detection logic below.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.backend.facial_recognition.config import (
    BLINK_EAR_THRESHOLD,
    HEAD_TURN_DISPLACEMENT_RATIO,
    LIVENESS_TIMEOUT_FRAMES,
)
from src.backend.facial_recognition.modules.camera import CameraModule
from src.backend.facial_recognition.modules.exceptions import (
    LivenessCheckFailedError,
    NoFaceDetectedError,
)
from src.backend.facial_recognition.modules.face_detector import (
    DetectedFace,
    FaceDetector,
)
from src.backend.facial_recognition.modules.preview import PreviewRenderer

# 106-point landmark indices around each eye (InsightFace 2d_106 layout).
# Used only for blink detection when landmarks are available.
_LEFT_EYE_IDX = [35, 41, 40, 42, 39, 37]
_RIGHT_EYE_IDX = [89, 95, 94, 96, 93, 91]


class LivenessCheck:
    """
    Prompts the user to perform a liveness gesture (head turn or blink)
    and confirms it was performed using simple geometric heuristics on
    face keypoints/landmarks.
    """

    def __init__(
        self,
        head_turn_ratio: float = HEAD_TURN_DISPLACEMENT_RATIO,
        blink_ear_threshold: float = BLINK_EAR_THRESHOLD,
        timeout_frames: int = LIVENESS_TIMEOUT_FRAMES,
    ) -> None:
        self.head_turn_ratio = head_turn_ratio
        self.blink_ear_threshold = blink_ear_threshold
        self.timeout_frames = timeout_frames

    def run(
        self,
        camera: CameraModule,
        detector: FaceDetector,
        preview: Optional[PreviewRenderer] = None,
        mode_label: str = "",
        beneficiary_id: str = "",
    ) -> bool:
        """
        Run the liveness challenge: watch the live feed until either a
        head turn or a blink is detected, or the timeout is reached.

        Args:
            camera: an already-opened CameraModule.
            detector: a FaceDetector used to locate keypoints each frame.
            preview: optional PreviewRenderer. When provided, each frame is
                drawn with the current bounding box/landmarks and a
                "Liveness check..." status so the user gets live feedback.
                Purely cosmetic — no detection behavior changes.
            mode_label: label shown in the preview panel (e.g. "Facial
                Biometric Enrollment"). Ignored if preview is None.
            beneficiary_id: beneficiary ID shown in the preview panel.
                Ignored if preview is None.

        Returns:
            True if a liveness gesture was detected.

        Raises:
            LivenessCheckFailedError: if no gesture is detected within
                the frame timeout.
        """
        print("[Liveness] Please BLINK or TURN YOUR HEAD to confirm you are live...")
        status = "Liveness check... please blink or turn your head"

        baseline_nose_x: Optional[float] = None
        baseline_face_width: Optional[float] = None
        ear_history: list[float] = []

        for frame_count in range(self.timeout_frames):
            frame = camera.read_frame()

            try:
                face = detector.detect_single_face(frame)
            except NoFaceDetectedError:
                # Face temporarily out of frame — just keep waiting.
                if preview is not None:
                    preview.show(
                        preview.render(
                            frame, mode_label, beneficiary_id, "Looking for face..."
                        )
                    )
                continue
            except Exception:
                if preview is not None:
                    preview.show(
                        preview.render(
                            frame, mode_label, beneficiary_id, "Multiple faces detected"
                        )
                    )
                continue

            # --- Head-turn detection (unchanged logic) ---------------------
            nose_x = float(face.kps[2][0])  # index 2 = nose tip in 5-pt kps
            face_width = float(face.bbox[2] - face.bbox[0])

            if baseline_nose_x is None:
                baseline_nose_x = nose_x
                baseline_face_width = face_width

            displacement = abs(nose_x - baseline_nose_x)
            normalized_displacement = displacement / max(baseline_face_width, 1e-6)

            if normalized_displacement >= self.head_turn_ratio:
                print(
                    f"[Liveness] Head turn detected (frame {frame_count}). Liveness CONFIRMED."
                )
                if preview is not None:
                    preview.show(
                        preview.render(
                            frame,
                            mode_label,
                            beneficiary_id,
                            "Liveness confirmed",
                            bbox=face.bbox,
                            kps=face.kps,
                            landmark_106=face.landmark_106,
                            det_score=face.det_score,
                        )
                    )
                return True

            # --- Blink detection (only if 106-pt landmarks available, unchanged logic) ---
            if face.landmark_106 is not None:
                ear = self._compute_average_ear(face)
                ear_history.append(ear)
                if len(ear_history) > 5 and self._blink_detected(ear_history):
                    print(
                        f"[Liveness] Blink detected (frame {frame_count}). Liveness CONFIRMED."
                    )
                    if preview is not None:
                        preview.show(
                            preview.render(
                                frame,
                                mode_label,
                                beneficiary_id,
                                "Liveness confirmed",
                                bbox=face.bbox,
                                kps=face.kps,
                                landmark_106=face.landmark_106,
                                det_score=face.det_score,
                            )
                        )
                    return True

            # Neither gesture detected yet this frame — show live feedback.
            if preview is not None:
                preview.show(
                    preview.render(
                        frame,
                        mode_label,
                        beneficiary_id,
                        status,
                        bbox=face.bbox,
                        kps=face.kps,
                        landmark_106=face.landmark_106,
                        det_score=face.det_score,
                    )
                )

        raise LivenessCheckFailedError(
            "Liveness check timed out. No blink or head turn detected. "
            "Please try again with clearer face visibility."
        )

    # -- internal helpers (unchanged) --------------------------------------
    @staticmethod
    def _eye_aspect_ratio(eye_points: np.ndarray) -> float:
        """
        Compute the Eye Aspect Ratio (EAR) for a set of 6 eye landmark points.
        EAR drops sharply when the eye closes.
        """
        p1, p2, p3, p4, p5, p6 = eye_points
        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        return (vertical_1 + vertical_2) / (2.0 * max(horizontal, 1e-6))

    def _compute_average_ear(self, face: DetectedFace) -> float:
        """Average EAR across both eyes using the 106-point landmark set."""
        landmarks = face.landmark_106
        left_eye = np.array([landmarks[i] for i in _LEFT_EYE_IDX])
        right_eye = np.array([landmarks[i] for i in _RIGHT_EYE_IDX])
        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        return (left_ear + right_ear) / 2.0

    def _blink_detected(self, ear_history: list[float]) -> bool:
        """
        A blink is a brief dip in EAR below the threshold, followed by a
        recovery. We check the last few frames for that dip-and-recover
        pattern.
        """
        recent = ear_history[-5:]
        return (
            min(recent) < self.blink_ear_threshold
            and recent[-1] > self.blink_ear_threshold
        )
