"""
preview.py
----------
Presentation-only module: renders the real-time OpenCV preview window used
during enrollment and verification.

This module owns NO AI logic whatsoever — it only knows how to draw frames,
bounding boxes, landmarks, and status/info panels that are handed to it by
the modules that orchestrate the actual pipeline (enrollment.py,
verification.py, liveness.py). Keeping it fully decoupled from SCRFD/ArcFace
means the preview can be redesigned or replaced without ever touching
detection, embedding, or matching logic.

Performance note: this class does not run any inference itself. Callers
are responsible for deciding when to run detection (e.g. once every N
frames) and can simply re-call render() with a cached/stale detection
result on the frames in between — this is what keeps the preview smooth
and decoupled from the cost of SCRFD/ArcFace.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Visual constants
# --------------------------------------------------------------------------
_FONT = cv2.FONT_HERSHEY_SIMPLEX

_COLOR_TEXT = (255, 255, 255)      # white
_COLOR_PANEL_BG = (20, 20, 20)     # near-black
_COLOR_BOX = (0, 220, 0)           # green bounding box
_COLOR_LANDMARK = (0, 200, 255)    # amber landmark dots
_COLOR_SUCCESS = (0, 200, 0)       # green
_COLOR_WARNING = (0, 200, 255)     # amber
_COLOR_ERROR = (0, 0, 220)         # red

_PANEL_ALPHA = 0.55                # semi-transparent info-panel background


def _status_color(status: str) -> Tuple[int, int, int]:
    """Classify a status string into a semantic display color (BGR)."""
    lowered = status.lower()

    error_keywords = ("failed", "not verified", "multiple faces", "no face", "timed out")
    if any(keyword in lowered for keyword in error_keywords):
        return _COLOR_ERROR

    success_keywords = ("stored", "successful", "confirmed", "complete")
    if "verified" in lowered and "not verified" not in lowered:
        return _COLOR_SUCCESS
    if any(keyword in lowered for keyword in success_keywords):
        return _COLOR_SUCCESS

    return _COLOR_WARNING


class PreviewRenderer:
    """
    Owns a single OpenCV window and draws the biometric preview UI:
    an information panel (title, beneficiary ID, progress, status),
    the SCRFD bounding box, and landmarks when available.
    """

    def __init__(self, window_name: str) -> None:
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 520)

    # -- main per-frame render ---------------------------------------------
    def render(
        self,
        frame: np.ndarray,
        mode_label: str,
        beneficiary_id: str,
        status: str,
        progress: Optional[str] = None,
        bbox: Optional[np.ndarray] = None,
        kps: Optional[np.ndarray] = None,
        landmark_106: Optional[np.ndarray] = None,
        det_score: Optional[float] = None,
    ) -> np.ndarray:
        """
        Draw the full overlay (info panel + bounding box + landmarks) on a
        COPY of the given frame and return it. The original frame is left
        untouched so callers can still use it for saving embeddings, etc.

        `bbox`/`kps`/`landmark_106`/`det_score` may be None (e.g. while no
        face is currently detected) or may be a CACHED result from a
        previous inference cycle — this method has no opinion on freshness,
        it just draws whatever it's given.
        """
        annotated = frame.copy()

        if bbox is not None:
            self._draw_bbox(annotated, bbox, det_score)
        if landmark_106 is not None:
            self._draw_points(annotated, landmark_106, radius=1)
        elif kps is not None:
            self._draw_points(annotated, kps, radius=3)

        self._draw_panel(annotated, mode_label, beneficiary_id, status, progress)
        return annotated

    # -- final result screen -------------------------------------------------
    def render_result_screen(
        self,
        frame: np.ndarray,
        headline: str,
        color: Tuple[int, int, int],
        subtext: Optional[str] = None,
    ) -> np.ndarray:
        """
        Build a final "result" frame with a large centered banner
        (e.g. VERIFIED / NOT VERIFIED / ENROLLMENT COMPLETE) plus an
        optional subtext line (e.g. the similarity score).
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Dim the whole frame so the banner stands out clearly.
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)

        font_scale = 1.6
        thickness = 3
        (text_w, text_h), _ = cv2.getTextSize(headline, _FONT, font_scale, thickness)
        x = max((w - text_w) // 2, 10)
        y = h // 2
        pad = 20

        cv2.rectangle(
            annotated, (x - pad, y - text_h - pad), (x + text_w + pad, y + pad),
            _COLOR_PANEL_BG, -1,
        )
        cv2.rectangle(
            annotated, (x - pad, y - text_h - pad), (x + text_w + pad, y + pad),
            color, 3,
        )
        cv2.putText(annotated, headline, (x, y), _FONT, font_scale, color, thickness, cv2.LINE_AA)

        if subtext:
            (sub_w, sub_h), _ = cv2.getTextSize(subtext, _FONT, 0.8, 2)
            sub_x = max((w - sub_w) // 2, 10)
            sub_y = y + pad + sub_h + 20
            cv2.putText(annotated, subtext, (sub_x, sub_y), _FONT, 0.8, _COLOR_TEXT, 2, cv2.LINE_AA)

        return annotated

    # -- window I/O -----------------------------------------------------------
    def show(self, frame: np.ndarray, wait_ms: int = 1) -> int:
        """
        Display a frame and pump the GUI event loop.

        Returns:
            The pressed key code (masked to 8 bits), or -1/255 if none.
        """
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(wait_ms) & 0xFF

    def hold(self, frame: np.ndarray, seconds: float) -> None:
        """
        Keep a static frame on screen for a given duration while still
        pumping the GUI event loop (so the window doesn't appear frozen
        or gray out on some platforms).
        """
        end_time = time.time() + seconds
        while time.time() < end_time:
            self.show(frame, wait_ms=30)

    def close(self) -> None:
        """Close only this preview window (the camera device is managed elsewhere)."""
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass  # window may already be closed (e.g. by CameraModule.release())

    # -- drawing helpers --------------------------------------------------------
    @staticmethod
    def _draw_bbox(frame: np.ndarray, bbox: np.ndarray, det_score: Optional[float]) -> None:
        """Draw the SCRFD bounding box, with the detection confidence as a label."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), _COLOR_BOX, 2)
        if det_score is not None:
            label = f"{det_score:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 15)), _FONT, 0.6, _COLOR_BOX, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_points(frame: np.ndarray, points: np.ndarray, radius: int) -> None:
        """Draw keypoints/landmarks as small filled circles."""
        for (px, py) in points:
            cv2.circle(frame, (int(px), int(py)), radius, _COLOR_LANDMARK, -1)

    def _draw_panel(
        self,
        frame: np.ndarray,
        mode_label: str,
        beneficiary_id: str,
        status: str,
        progress: Optional[str],
    ) -> None:
        """Draw the semi-transparent info panel in the top-left corner."""
        lines = [mode_label, f"Beneficiary ID: {beneficiary_id}"]
        if progress:
            lines.append(progress)
        lines.append(f"Status: {status}")

        font_scale = 0.7
        thickness = 2
        line_height = 34
        padding = 18

        text_sizes = [cv2.getTextSize(line, _FONT, font_scale, thickness)[0] for line in lines]
        panel_width = max(w for w, _ in text_sizes) + padding * 2
        panel_height = line_height * len(lines) + padding * 2

        # Semi-transparent background so the panel never fully obscures the feed.
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), _COLOR_PANEL_BG, -1)
        frame[:] = cv2.addWeighted(overlay, _PANEL_ALPHA, frame, 1 - _PANEL_ALPHA, 0)

        y = 10 + padding + 22
        for line in lines:
            color = _status_color(status) if line is lines[-1] else _COLOR_TEXT
            cv2.putText(frame, line, (10 + padding, y), _FONT, font_scale, color, thickness, cv2.LINE_AA)
            y += line_height
