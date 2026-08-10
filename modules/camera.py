"""
camera.py
---------
Module 1: Camera

Responsible ONLY for opening the webcam and yielding raw BGR frames.
It knows nothing about faces, embeddings, or liveness — keeping it a
thin, reusable building block for every other module.
"""

from __future__ import annotations

from types import TracebackType
from typing import Optional, Type

import cv2
import numpy as np

from config import CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH
from modules.exceptions import CameraUnavailableError
from modules.face_detector import DetectedFace


class CameraModule:
    """
    Thin wrapper around cv2.VideoCapture.

    Implemented as a context manager so the camera device is always
    released, even if an exception occurs mid-capture:

        with CameraModule() as cam:
            frame = cam.read_frame()
    """

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._capture: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the webcam device and configure resolution."""
        self._capture = cv2.VideoCapture(self.camera_index)

        if not self._capture.isOpened():
            raise CameraUnavailableError(
                f"Could not open camera at index {self.camera_index}. "
                "Check that a webcam is connected and not in use by another app."
            )

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        print(f"[Camera] Opened device {self.camera_index} at {self.width}x{self.height}.")

    def read_frame(self) -> np.ndarray:
        """
        Grab a single frame from the camera.

        Returns:
            np.ndarray: BGR image frame.

        Raises:
            CameraUnavailableError: if the camera is not open or the
                frame could not be read (e.g. device disconnected).
        """
        if self._capture is None or not self._capture.isOpened():
            raise CameraUnavailableError("Camera is not open. Call open() first.")

        success, frame = self._capture.read()
        if not success or frame is None:
            raise CameraUnavailableError("Failed to read frame from camera.")

        return frame

    def release(self) -> None:
        """Release the camera device and close any preview windows."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        cv2.destroyAllWindows()
        print("[Camera] Released.")

    # -- context manager protocol -----------------------------------------
    def __enter__(self) -> "CameraModule":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.release()


class PreviewDisplay:
    """Lightweight OpenCV preview window for live camera-based flows."""

    def __init__(self, window_name: str = "Live Preview") -> None:
        self.window_name = window_name
        self._is_open = False

    def __enter__(self) -> "PreviewDisplay":
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._is_open = True
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def show(
        self,
        frame: np.ndarray,
        face: Optional[DetectedFace] = None,
        overlay_text: str = "",
    ) -> int | None:
        if not self._is_open:
            return

        display_frame = frame.copy()

        if face is not None:
            self._draw_face_overlay(display_frame, face)

        if overlay_text:
            self._draw_overlay_text(display_frame, overlay_text)

        cv2.imshow(self.window_name, display_frame)
        # Return key pressed so callers can optionally react (ESC to close)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            self.close()
            return key
        return key

    def close(self) -> None:
        if self._is_open:
            cv2.destroyWindow(self.window_name)
            self._is_open = False

    @staticmethod
    def _draw_face_overlay(frame: np.ndarray, face: DetectedFace) -> None:
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        score_text = f"conf: {face.det_score:.2f}"
        cv2.putText(
            frame,
            score_text,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_overlay_text(frame: np.ndarray, text: str) -> None:
        lines = text.split("\n")
        margin = 10
        line_height = 24
        width = frame.shape[1]
        height = frame.shape[0]
        box_height = line_height * len(lines) + margin

        cv2.rectangle(
            frame,
            (0, 0),
            (width, box_height),
            (0, 0, 0),
            thickness=cv2.FILLED,
        )

        for idx, line in enumerate(lines):
            position = (margin, margin + (idx + 1) * line_height - 8)
            cv2.putText(
                frame,
                line,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
