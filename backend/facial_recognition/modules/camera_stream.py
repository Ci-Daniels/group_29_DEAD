"""
camera_stream.py
-----------------
Decouples physical camera reads from AI inference speed.

WHY THIS EXISTS
---------------
Both liveness.py and verification.py call camera.read_frame() once per loop
iteration, immediately followed by SCRFD+ArcFace inference on that frame.
Because CameraModule.read_frame() blocks until it grabs a new device frame,
the whole loop -- and by extension, whatever calls preview.show() with that
frame -- runs no faster than inference allows. On a typical CPU that means
the browser preview visibly "steps" instead of playing like smooth video.

BufferedCameraStream fixes this WITHOUT touching liveness.py's or
verification.py's detection/matching logic: it wraps a CameraModule with a
single dedicated background thread that continuously reads the physical
device into a thread-safe single-slot buffer. read_frame() here returns
whatever the most recent frame is -- essentially instantly -- instead of
blocking on the device. This is what lets modules/web_preview.py refresh
the browser at a steady ~30fps that's completely independent of how fast
the AI pipeline happens to be running.

Exposes the exact same public interface as CameraModule (open / read_frame /
release / context manager), so it's a drop-in replacement wherever
CameraModule is used today -- callers don't need to know or care that reads
are now buffered.
"""

from __future__ import annotations

import threading
import time
from types import TracebackType
from typing import Optional, Type

import numpy as np

from backend.facial_recognition.modules.camera import CameraModule
from backend.facial_recognition.modules.exceptions import CameraUnavailableError

# Max time to wait for the first frame to arrive after opening the device.
_FIRST_FRAME_TIMEOUT_SECONDS = 5.0


class BufferedCameraStream:
    """
    Wraps a CameraModule with a background capture thread, so read_frame()
    never blocks on the physical device -- it just returns the latest
    frame already sitting in the buffer.
    """

    def __init__(self, camera: CameraModule) -> None:
        self._camera = camera
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._capture_error: Optional[Exception] = None

    def open(self) -> None:
        """Open the underlying device and start the background capture thread."""
        self._camera.open()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # Block briefly until the first real frame lands, so callers can
        # start reading immediately after open() returns.
        deadline = time.time() + _FIRST_FRAME_TIMEOUT_SECONDS
        while time.time() < deadline:
            with self._lock:
                if self._latest_frame is not None:
                    return
                if self._capture_error is not None:
                    raise self._capture_error
            time.sleep(0.02)

        raise CameraUnavailableError("Timed out waiting for the first camera frame.")

    def _capture_loop(self) -> None:
        """Runs on a dedicated thread: reads the device as fast as it delivers frames."""
        while self._running:
            try:
                frame = self._camera.read_frame()
            except CameraUnavailableError as exc:
                with self._lock:
                    self._capture_error = exc
                break
            with self._lock:
                self._latest_frame = frame

    def read_frame(self) -> np.ndarray:
        """
        Return the most recently captured frame (near-instant, does not
        block on the physical device). Same contract as CameraModule.
        """
        with self._lock:
            if self._latest_frame is None:
                raise CameraUnavailableError("No camera frame available yet.")
            return self._latest_frame.copy()

    def release(self) -> None:
        """Stop the capture thread and release the underlying device."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._camera.release()

    # -- context manager protocol (mirrors CameraModule) --------------------
    def __enter__(self) -> "BufferedCameraStream":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.release()
