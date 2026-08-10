"""
web_preview.py
--------------
Flask integration adapter: a PreviewRenderer subclass that streams frames
to the browser instead of opening a native OpenCV window.

WHY THIS EXISTS
---------------
`modules/preview.py` (PreviewRenderer) draws bounding boxes, landmarks, and
the status panel onto a frame — but its window-management methods
(__init__, show, hold, close) assume a local desktop GUI (cv2.namedWindow /
cv2.imshow / cv2.waitKey). A headless Flask server has no display to show a
native window on, and even if it did, the frame would appear on the
server's screen, not in the user's browser.

WebPreviewRenderer overrides those GUI-bound methods. It still inherits
render() and render_result_screen() completely unchanged, so every pixel
drawn (bounding boxes, landmarks, info panel, result banner) is produced by
the exact same drawing code already used by the desktop CLI. Nothing about
detection, liveness, or verification logic is touched by this file.

REAL-TIME PREVIEW (attach_camera_stream)
-----------------------------------------
By default (attach_camera_stream never called), this class behaves exactly
like the original adapter: each preview.show(preview.render(...)) call from
the AI loop (enrollment.py) encodes and stores one frame. That's fine for
enrollment, where a slower cadence is intentional.

For verification, calling attach_camera_stream(camera_stream) starts a
SEPARATE background thread that continuously pulls the latest raw frame
from a BufferedCameraStream and redraws it with whatever AI overlay
(status/bbox/landmarks) was most recently reported — at a steady ~30fps,
completely decoupled from how fast SCRFD/ArcFace happens to be running.
This is what makes the browser feed look like a real-time camera preview
instead of stepping only as fast as inference allows: the video itself
never waits on the AI pipeline, only the overlay does.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from src.backend.facial_recognition.config import PREVIEW_REFRESH_INTERVAL_SECONDS
from src.backend.facial_recognition.modules.preview import PreviewRenderer


class WebPreviewRenderer(PreviewRenderer):
    """
    Drop-in replacement for PreviewRenderer used by the Flask app.

    Instead of drawing to a native OpenCV window, each rendered frame is
    JPEG-encoded and stored thread-safely so app.py's MJPEG route can
    stream it to the browser. The latest status/progress text is also
    cached so app.py's status-polling endpoint can report it as JSON.
    """

    def __init__(self, session_id: str) -> None:
        # Deliberately does NOT call super().__init__() -- that method
        # calls cv2.namedWindow(), which must never run on a headless server.
        self.session_id = session_id
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None

        # Latest status/progress text, read by the Flask status endpoint.
        self.status: str = "Initializing..."
        self.progress: Optional[str] = None

        # -- real-time refresh plumbing (opt-in, see attach_camera_stream) --
        self._camera_stream = None
        self._overlay_lock = threading.Lock()
        self._overlay: dict = {
            "mode_label": "",
            "beneficiary_id": "",
            "status": self.status,
            "progress": None,
            "bbox": None,
            "kps": None,
            "landmark_106": None,
            "det_score": None,
        }
        self._refresh_thread: Optional[threading.Thread] = None
        self._refresh_running = False

    # -- real-time refresh (used by verification.py) ------------------------
    def attach_camera_stream(self, camera_stream) -> None:
        """
        Start a background thread that continuously redraws the latest raw
        camera frame with the most recently cached AI overlay, at a fixed
        cadence independent of the AI loop calling render()/show(). Pass any
        object exposing read_frame() (BufferedCameraStream is the intended one).
        """
        self._camera_stream = camera_stream
        self._refresh_running = True
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()

    def detach_camera_stream(self) -> None:
        """Stop the background refresh thread (e.g. before freezing on a result screen)."""
        self._refresh_running = False
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=1.0)
            self._refresh_thread = None
        self._camera_stream = None

    def _refresh_loop(self) -> None:
        """Runs on its own thread: redraw + re-encode at a fixed cadence, forever."""
        while self._refresh_running:
            try:
                frame = self._camera_stream.read_frame()
            except Exception:
                time.sleep(PREVIEW_REFRESH_INTERVAL_SECONDS)
                continue

            with self._overlay_lock:
                overlay = dict(self._overlay)

            annotated = super().render(
                frame,
                overlay["mode_label"],
                overlay["beneficiary_id"],
                overlay["status"],
                overlay["progress"],
                bbox=overlay["bbox"],
                kps=overlay["kps"],
                landmark_106=overlay["landmark_106"],
                det_score=overlay["det_score"],
            )
            self._encode_and_store(annotated)
            time.sleep(PREVIEW_REFRESH_INTERVAL_SECONDS)

    # -- overridden GUI-bound methods --------------------------------------
    def render(
        self,
        frame: np.ndarray,
        mode_label: str,
        beneficiary_id: str,
        status: str,
        progress: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Cache status/progress + overlay params for the background refresh
        thread (if attached), then draw this exact frame using the parent's
        unchanged drawing logic -- same behavior as before when no camera
        stream is attached (i.e. during enrollment).
        """
        self.status = status
        self.progress = progress
        with self._overlay_lock:
            self._overlay = {
                "mode_label": mode_label,
                "beneficiary_id": beneficiary_id,
                "status": status,
                "progress": progress,
                "bbox": kwargs.get("bbox"),
                "kps": kwargs.get("kps"),
                "landmark_106": kwargs.get("landmark_106"),
                "det_score": kwargs.get("det_score"),
            }
        return super().render(
            frame, mode_label, beneficiary_id, status, progress, **kwargs
        )

    def show(self, frame: np.ndarray, wait_ms: int = 1) -> int:
        """
        Encode this exact frame and store it. When a camera stream is
        attached, the background thread already handles continuous
        encoding -- this call still fires for important one-off moments
        (e.g. "Liveness confirmed") so they're never missed between
        refresh ticks.
        """
        self._encode_and_store(frame)
        return -1  # no keyboard input exists in a web context

    def hold(self, frame: np.ndarray, seconds: float) -> None:
        """
        Freeze on the final result frame for `seconds`. Stops the live
        refresh thread first so it can't overwrite the result banner with
        a fresh, unannotated frame.
        """
        self.detach_camera_stream()
        end_time = time.time() + seconds
        while time.time() < end_time:
            self.show(frame)
            time.sleep(0.03)

    def close(self) -> None:
        """Stop any running refresh thread. The Flask app owns everything else's lifecycle."""
        self.detach_camera_stream()

    # -- internal + Flask-facing accessors -----------------------------------
    def _encode_and_store(self, frame: np.ndarray) -> None:
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if success:
            with self._lock:
                self._latest_jpeg = buffer.tobytes()

    def get_jpeg(self) -> Optional[bytes]:
        """Thread-safe read of the most recently rendered frame."""
        with self._lock:
            return self._latest_jpeg
