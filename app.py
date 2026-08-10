"""
app.py
------
Flask web layer for the facial biometric prototype.

This file is intentionally thin: it contains NO detection, liveness, or
matching logic. It only:
  1. Serves the dashboard / register / verify pages.
  2. Starts enrollment/verification as background threads (they run for
     several seconds and must not block the Flask server or its own
     video-streaming route).
  3. Streams the live annotated preview to the browser as MJPEG.
  4. Exposes a small JSON status endpoint so the frontend can show
     real-time stage text ("Looking for face...", "Liveness passed", etc.).

All AI work is delegated to modules/enrollment.py, modules/verification.py,
modules/face_detector.py, modules/liveness.py, and modules/camera.py --
none of which had their recognition logic modified for this integration.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Dict, Optional

from flask import Flask, Response, jsonify, render_template, request

from config import EMBEDDINGS_DIR
from modules.enrollment import FaceEnrollment
from modules.exceptions import (
    BeneficiaryNotEnrolledError,
    CameraUnavailableError,
    LivenessCheckFailedError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from modules.face_detector import FaceDetector
from modules.liveness import LivenessCheck
from modules.verification import FaceVerification
from modules.web_preview import WebPreviewRenderer

app = Flask(__name__)

# --------------------------------------------------------------------------
# Shared AI components
# --------------------------------------------------------------------------
# FaceDetector loads the InsightFace model pack once at startup (this is the
# expensive step). It is safely reused across requests/threads -- the
# underlying ONNX Runtime session is stateless per detect() call.
_detector = FaceDetector()
_liveness_check = LivenessCheck()

# --------------------------------------------------------------------------
# In-memory session registry
# --------------------------------------------------------------------------
# Keyed by a random session_id per registration/verification attempt. This
# is a local, single-user PoC, so an in-memory dict guarded by a lock is
# sufficient -- a production version would use a real job queue/store.
#
# NOTE: because CameraModule opens the physical webcam device, only one
# session should be run at a time. Starting a second session while one is
# active will fail to open the camera -- out of scope for this PoC.
_sessions: Dict[str, dict] = {}
_sessions_lock = threading.Lock()


def _new_session(kind: str, beneficiary_id: str) -> str:
    """Create and register a new session, returning its ID."""
    session_id = uuid.uuid4().hex
    with _sessions_lock:
        _sessions[session_id] = {
            "kind": kind,  # "register" | "verify"
            "beneficiary_id": beneficiary_id,
            "preview": WebPreviewRenderer(session_id),
            "done": False,
            "error": None,
            "result": None,  # populated on success
        }
    return session_id


def _get_session(session_id: str) -> Optional[dict]:
    with _sessions_lock:
        return _sessions.get(session_id)


# --------------------------------------------------------------------------
# Background workers (run the existing, unmodified AI workflows)
# --------------------------------------------------------------------------
_EXPECTED_ERRORS = (
    CameraUnavailableError,
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    LivenessCheckFailedError,
    BeneficiaryNotEnrolledError,
)


def _run_enrollment(session_id: str, metadata: dict) -> None:
    """Runs FaceEnrollment.enroll() on a background thread."""
    session = _get_session(session_id)
    beneficiary_id = session["beneficiary_id"]

    # Persist beneficiary metadata (name/relationship/phone/email) alongside
    # the embeddings. modules/enrollment.py itself only ever writes .npy
    # embedding files -- this write is Flask-side only and does not touch
    # the enrollment module's storage logic.
    beneficiary_dir = EMBEDDINGS_DIR / beneficiary_id
    beneficiary_dir.mkdir(parents=True, exist_ok=True)
    (beneficiary_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    enrollment = FaceEnrollment(detector=_detector, liveness_check=_liveness_check)
    try:
        enrollment.enroll(beneficiary_id, preview=session["preview"])
        session["result"] = {"success": True}
    except _EXPECTED_ERRORS as exc:
        session["error"] = str(exc)
    finally:
        session["done"] = True


def _run_verification(session_id: str) -> None:
    """Runs FaceVerification.verify() on a background thread."""
    session = _get_session(session_id)
    beneficiary_id = session["beneficiary_id"]

    verification = FaceVerification(detector=_detector, liveness_check=_liveness_check)
    try:
        result = verification.verify(beneficiary_id, preview=session["preview"])
        session["result"] = {
            "verified": result.verified,
            "similarity_score": result.similarity_score,
            "threshold_used": result.threshold_used,
        }
    except _EXPECTED_ERRORS as exc:
        session["error"] = str(exc)
    finally:
        session["done"] = True


# --------------------------------------------------------------------------
# Page routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    """Dashboard homepage."""
    return render_template("index.html")


@app.route("/register")
def register_page():
    """Beneficiary registration page."""
    return render_template("register.html")


@app.route("/verify")
def verify_page():
    """Beneficiary verification page."""
    return render_template("verify.html")


# --------------------------------------------------------------------------
# API: registration
# --------------------------------------------------------------------------
@app.route("/api/register/start", methods=["POST"])
def api_register_start():
    """Start a background enrollment session for the submitted beneficiary."""
    data = request.get_json(force=True) or {}
    beneficiary_id = (data.get("beneficiary_id") or "").strip()
    if not beneficiary_id:
        return jsonify({"error": "Beneficiary ID is required."}), 400

    if not data.get("consent_given"):
        return jsonify({"error": "Biometric data consent is required before enrollment can proceed."}), 400

    metadata = {
        "beneficiary_id": beneficiary_id,
        "name": (data.get("name") or "").strip(),
        "relationship": (data.get("relationship") or "").strip(),
        "phone": (data.get("phone") or "").strip(),
        "email": (data.get("email") or "").strip(),
        "consent_given": True,
        "consent_timestamp": data.get("consent_timestamp", ""),
    }

    session_id = _new_session("register", beneficiary_id)
    thread = threading.Thread(target=_run_enrollment, args=(session_id, metadata), daemon=True)
    thread.start()

    return jsonify({"session_id": session_id})


# --------------------------------------------------------------------------
# API: verification
# --------------------------------------------------------------------------
@app.route("/api/verify/start", methods=["POST"])
def api_verify_start():
    """Start a background verification session for the submitted beneficiary."""
    data = request.get_json(force=True) or {}
    beneficiary_id = (data.get("beneficiary_id") or "").strip()
    if not beneficiary_id:
        return jsonify({"error": "Beneficiary ID is required."}), 400

    session_id = _new_session("verify", beneficiary_id)
    thread = threading.Thread(target=_run_verification, args=(session_id,), daemon=True)
    thread.start()

    return jsonify({"session_id": session_id})


# --------------------------------------------------------------------------
# API: shared status polling (used by both register.js and verify.js)
# --------------------------------------------------------------------------
@app.route("/api/session/<session_id>/status")
def api_session_status(session_id: str):
    """Poll the live status/progress/result of any session (register or verify)."""
    session = _get_session(session_id)
    if session is None:
        return jsonify({"error": "Unknown session."}), 404

    preview: WebPreviewRenderer = session["preview"]
    return jsonify(
        {
            "status": preview.status,
            "progress": preview.progress,
            "done": session["done"],
            "error": session["error"],
            "result": session["result"],
        }
    )


# --------------------------------------------------------------------------
# Live MJPEG video feed
# --------------------------------------------------------------------------
def _mjpeg_stream(session_id: str):
    """Generator yielding multipart JPEG frames for a given session."""
    boundary = b"--frame"
    while True:
        session = _get_session(session_id)
        if session is None:
            break

        frame = session["preview"].get_jpeg()
        if frame is not None:
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        if session["done"]:
            # Keep the last (result) frame visible briefly, then end the stream.
            time.sleep(1.0)
            break

        time.sleep(0.03)  # ~30 fps cap, independent of AI inference cadence


@app.route("/video_feed/<session_id>")
def video_feed(session_id: str):
    """MJPEG live preview endpoint, consumed by an <img> tag in the browser."""
    return Response(
        _mjpeg_stream(session_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# --------------------------------------------------------------------------
# API: registered beneficiaries
# --------------------------------------------------------------------------
@app.route("/api/beneficiaries")
def api_beneficiaries():
    """List enrolled beneficiaries by reading data/embeddings/ subfolders."""
    if not EMBEDDINGS_DIR.exists():
        return jsonify([])

    beneficiaries = []
    for folder in sorted(EMBEDDINGS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        entry = {"beneficiary_id": folder.name}
        metadata_file = folder / "metadata.json"
        if metadata_file.exists():
            entry.update(json.loads(metadata_file.read_text()))
        beneficiaries.append(entry)

    return jsonify(beneficiaries)


if __name__ == "__main__":
    # threaded=True is required: the MJPEG route holds a long-lived
    # connection open and must not block status polling or page requests.
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
