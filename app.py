"""app.py.
------
Flask web layer for the facial biometric prototype.

This file:
1. Serves the dashboard / register / verify / Gmail pages.
2. Starts enrollment/verification as background threads.
3. Streams the live annotated preview to the browser as MJPEG.
4. Exposes JSON status endpoints.
5. Provides Gmail OAuth + Gmail API integration.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import uuid

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from src.backend.email_scanning.main import EmailEvaluation, EmailEvaluationError
from src.backend.facial_recognition.config import EMBEDDINGS_DIR
from src.backend.facial_recognition.modules.enrollment import FaceEnrollment
from src.backend.facial_recognition.modules.exceptions import (
    BeneficiaryNotEnrolledError,
    CameraUnavailableError,
    LivenessCheckFailedError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from src.backend.facial_recognition.modules.face_detector import FaceDetector
from src.backend.facial_recognition.modules.liveness import LivenessCheck
from src.backend.facial_recognition.modules.verification import FaceVerification
from src.backend.facial_recognition.modules.web_preview import WebPreviewRenderer

# --------------------------------------------------------------------------
# Flask application
# --------------------------------------------------------------------------

app = Flask(
    __name__,
    template_folder="src/frontend/templates",
    static_folder="src/frontend/static/",
)

# Required for Flask session storage.
# In production, set FLASK_SECRET_KEY in the environment.
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "dev-only-secret-change-this",
)


# --------------------------------------------------------------------------
# Shared AI components
# --------------------------------------------------------------------------

# FaceDetector loads the InsightFace model pack once at startup.
# It is safely reused across requests/threads.
_detector = FaceDetector()
_liveness_check = LivenessCheck()


# --------------------------------------------------------------------------
# In-memory session registry
# --------------------------------------------------------------------------

_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()


def _new_session(kind: str, beneficiary_id: str) -> str:
    """Create and register a new biometric session."""
    session_id = uuid.uuid4().hex

    with _sessions_lock:
        _sessions[session_id] = {
            "kind": kind,
            "beneficiary_id": beneficiary_id,
            "preview": WebPreviewRenderer(session_id),
            "done": False,
            "error": None,
            "result": None,
        }

    return session_id


def _get_session(session_id: str) -> dict | None:
    """Retrieve a biometric session."""
    with _sessions_lock:
        return _sessions.get(session_id)


# --------------------------------------------------------------------------
# Background workers
# --------------------------------------------------------------------------

_EXPECTED_ERRORS = (
    CameraUnavailableError,
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    LivenessCheckFailedError,
    BeneficiaryNotEnrolledError,
)


def _run_enrollment(session_id: str, metadata: dict) -> None:
    """Run FaceEnrollment.enroll() on a background thread."""
    session_data = _get_session(session_id)

    if session_data is None:
        return

    beneficiary_id = session_data["beneficiary_id"]

    beneficiary_dir = EMBEDDINGS_DIR / beneficiary_id
    beneficiary_dir.mkdir(parents=True, exist_ok=True)

    (beneficiary_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    enrollment = FaceEnrollment(
        detector=_detector,
        liveness_check=_liveness_check,
    )

    try:
        enrollment.enroll(
            beneficiary_id,
            preview=session_data["preview"],
        )

        session_data["result"] = {"success": True}

    except _EXPECTED_ERRORS as exc:
        session_data["error"] = str(exc)

    finally:
        session_data["done"] = True


def _run_verification(session_id: str) -> None:
    """Run FaceVerification.verify() on a background thread."""
    session_data = _get_session(session_id)

    if session_data is None:
        return

    beneficiary_id = session_data["beneficiary_id"]

    verification = FaceVerification(
        detector=_detector,
        liveness_check=_liveness_check,
    )

    try:
        result = verification.verify(
            beneficiary_id,
            preview=session_data["preview"],
        )

        session_data["result"] = {
            "verified": result.verified,
            "similarity_score": result.similarity_score,
            "threshold_used": result.threshold_used,
        }

    except _EXPECTED_ERRORS as exc:
        session_data["error"] = str(exc)

    finally:
        session_data["done"] = True


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


@app.route("/scan_email")
def scan_email():
    """Gmail scanning page."""
    return render_template(
        "scan_emails.html",
        gmail_connected=email_evaluation.credentials_available(),
    )


@app.route("/emails")
def emails_page():
    """Display the first 10 inbox emails after Gmail is connected."""
    if not email_evaluation.credentials_available():
        return redirect(url_for("scan_email"))

    messages = []
    error_message = None

    try:
        messages = email_evaluation.fetch_recent_messages(max_results=10)
    except EmailEvaluationError as exc:
        error_message = str(exc)
    except Exception as exc:
        error_message = f"Unexpected Gmail error: {exc}"

    return render_template(
        "emails.html",
        messages=messages,
        error_message=error_message,
    )


@app.route("/verify")
def verify_page():
    """Beneficiary verification page."""
    return render_template("verify.html")


# --------------------------------------------------------------------------
# API: registration
# --------------------------------------------------------------------------


@app.route("/api/register/start", methods=["POST"])
def api_register_start():
    """Start a background enrollment session."""
    data = request.get_json(force=True) or {}

    beneficiary_id = (data.get("beneficiary_id") or "").strip()

    if not beneficiary_id:
        return jsonify({"error": "Beneficiary ID is required."}), 400

    if not data.get("consent_given"):
        return jsonify(
            {
                "error": (
                    "Biometric data consent is required before enrollment can proceed."
                )
            }
        ), 400

    metadata = {
        "beneficiary_id": beneficiary_id,
        "name": (data.get("name") or "").strip(),
        "relationship": (data.get("relationship") or "").strip(),
        "phone": (data.get("phone") or "").strip(),
        "email": (data.get("email") or "").strip(),
        "consent_given": True,
        "consent_timestamp": data.get(
            "consent_timestamp",
            "",
        ),
    }

    session_id = _new_session(
        "register",
        beneficiary_id,
    )

    thread = threading.Thread(
        target=_run_enrollment,
        args=(session_id, metadata),
        daemon=True,
    )

    thread.start()

    return jsonify({"session_id": session_id})


# --------------------------------------------------------------------------
# API: verification
# --------------------------------------------------------------------------


@app.route("/api/verify/start", methods=["POST"])
def api_verify_start():
    """Start a background verification session."""
    data = request.get_json(force=True) or {}

    beneficiary_id = (data.get("beneficiary_id") or "").strip()

    if not beneficiary_id:
        return jsonify({"error": "Beneficiary ID is required."}), 400

    session_id = _new_session(
        "verify",
        beneficiary_id,
    )

    thread = threading.Thread(
        target=_run_verification,
        args=(session_id,),
        daemon=True,
    )

    thread.start()

    return jsonify({"session_id": session_id})


# --------------------------------------------------------------------------
# API: biometric session status
# --------------------------------------------------------------------------


@app.route("/api/session/<session_id>/status")
def api_session_status(session_id: str):
    """Return current registration/verification status."""
    session_data = _get_session(session_id)

    if session_data is None:
        return jsonify({"error": "Unknown session."}), 404

    preview: WebPreviewRenderer = session_data["preview"]

    return jsonify(
        {
            "status": preview.status,
            "progress": preview.progress,
            "done": session_data["done"],
            "error": session_data["error"],
            "result": session_data["result"],
        }
    )


# --------------------------------------------------------------------------
# Live MJPEG video feed
# --------------------------------------------------------------------------


def _mjpeg_stream(session_id: str):
    """Generate multipart JPEG frames."""
    boundary = b"--frame"

    while True:
        session_data = _get_session(session_id)

        if session_data is None:
            break

        frame = session_data["preview"].get_jpeg()

        if frame is not None:
            yield (
                boundary
                + b"\r\n"
                + b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

        if session_data["done"]:
            time.sleep(1.0)
            break

        time.sleep(0.03)


@app.route("/video_feed/<session_id>")
def video_feed(session_id: str):
    """MJPEG live preview endpoint."""
    return Response(
        _mjpeg_stream(session_id),
        mimetype=("multipart/x-mixed-replace; boundary=frame"),
    )


# --------------------------------------------------------------------------
# API: registered beneficiaries
# --------------------------------------------------------------------------


@app.route("/api/beneficiaries")
def api_beneficiaries():
    """List enrolled beneficiaries."""
    if not EMBEDDINGS_DIR.exists():
        return jsonify([])

    beneficiaries = []

    for folder in sorted(EMBEDDINGS_DIR.iterdir()):
        if not folder.is_dir():
            continue

        entry = {"beneficiary_id": folder.name}

        metadata_file = folder / "metadata.json"

        if metadata_file.exists():
            with contextlib.suppress(json.JSONDecodeError):
                entry.update(json.loads(metadata_file.read_text()))

        beneficiaries.append(entry)

    return jsonify(beneficiaries)


# ==========================================================================
# Gmail OAuth + API
# ==========================================================================

email_evaluation = EmailEvaluation()


@app.route("/authorize")
def authorize():
    """Start the Google OAuth authorization flow."""
    try:
        authorization_url = email_evaluation.start_authorization(session)
        return redirect(authorization_url)
    except EmailEvaluationError as exc:
        return str(exc), exc.status_code


@app.route("/oauth2callback")
def oauth2callback():
    """Receive Google's OAuth callback and exchange
    the authorization code for credentials.
    """
    try:
        email_evaluation.complete_authorization(
            flask_session=session,
            authorization_response_url=request.url,
        )
        return redirect(url_for("emails_page"))
    except EmailEvaluationError as exc:
        return str(exc), exc.status_code


@app.route("/api/emails")
def api_emails():
    """List Gmail inbox messages.

    Optional query parameters:

        max_results
        page_token

    Example:
        /api/emails?max_results=20

    """
    if not email_evaluation.credentials_available():
        return jsonify(
            {
                "error": "Gmail is not connected.",
                "authorize_url": url_for("authorize"),
            }
        ), 401

    try:
        max_results = request.args.get(
            "max_results",
            default=2000,
            type=int,
        )
        page_token = request.args.get("page_token")

        result = email_evaluation.fetch_inbox_messages(
            max_results=max_results,
            page_token=page_token,
        )

        return jsonify(result)

    except EmailEvaluationError as exc:
        return jsonify(
            {
                "error": str(exc),
            }
        ), exc.status_code

    except Exception as exc:
        return jsonify(
            {
                "error": ("Unexpected Gmail error."),
                "details": str(exc),
            }
        ), 500


@app.route("/logout/google")
def logout_google():
    """Remove locally stored Gmail credentials.

    This disconnects the application locally.
    It does not revoke the application's authorization
    from the user's Google account.
    """
    email_evaluation.disconnect()

    return redirect(url_for("scan_email"))


# --------------------------------------------------------------------------
# Application entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
