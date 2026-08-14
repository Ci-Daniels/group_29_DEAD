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
from functools import wraps
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()

import google_auth_oauthlib.flow
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
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from werkzeug.security import check_password_hash, generate_password_hash

from src.backend.database_model.models import User, AuditLog, db
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
# Gmail configuration
# --------------------------------------------------------------------------

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

CLIENT_SECRETS_FILE = os.environ.get(
    "GOOGLE_CLIENT_SECRETS_FILE",
    "credentials.json",
)

TOKEN_FILE = os.environ.get(
    "GOOGLE_TOKEN_FILE",
    "token.json",
)

# Optional explicit callback URI override (must match Google Auth UI exactly).
GOOGLE_OAUTH_REDIRECT_URI = os.environ.get(
    "GOOGLE_OAUTH_REDIRECT_URI",
    "http://127.0.0.1:5000/oauth2callback",
)

# OAuthlib enforces HTTPS token exchange by default.
# For local loopback development callbacks only, allow HTTP transport.
if GOOGLE_OAUTH_REDIRECT_URI.startswith(("http://127.0.0.1", "http://localhost")):
    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")


# --------------------------------------------------------------------------
# Role-Based Access Control (RBAC)
# --------------------------------------------------------------------------

def require_role(*allowed_roles):
    """Decorator to restrict endpoint access by user role.
    
    Usage:
        @app.route("/admin/users")
        @require_role("admin")
        def admin_users():
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_role = session.get("user_role")
            user_id = session.get("user_id")
            
            if not user_id or not user_role:
                return jsonify({"error": "Unauthorized. Please log in."}), 401
            
            if user_role not in allowed_roles:
                return jsonify({"error": "Forbidden. Insufficient permissions."}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def log_audit(actor_id: str, action: str):
    """Log an action to the audit log for admin tracking."""
    try:
        audit = AuditLog(user_id=actor_id, actor=actor_id, action=action)
        db.session.add(audit)
        db.session.commit()
    except Exception as e:
        print(f"Warning: Failed to log audit: {e}")


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

_sessions: Dict[str, dict] = {}
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


def _get_session(session_id: str) -> Optional[dict]:
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
        gmail_connected=_gmail_credentials_available(),
    )


@app.route("/emails")
def emails_page():
    """Display the first 10 inbox emails after Gmail is connected."""
    credentials = _get_gmail_credentials()

    if credentials is None:
        return redirect(url_for("scan_email"))

    messages: list[dict] = []
    error_message = None

    try:
        service = discovery.build(
            "gmail",
            "v1",
            credentials=credentials,
            cache_discovery=False,
        )

        response = (
            service.users()
            .messages()
            .list(
                userId="me",
                labelIds=["INBOX"],
                maxResults=10,
                q="-category:promotions",
            )
            .execute()
        )

        for message_ref in response.get("messages", []):
            try:
                messages.append(_get_message_metadata(service, message_ref["id"]))
            except HttpError:
                continue

    except HttpError as exc:
        error_message = f"Gmail API request failed: {exc}"
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


def _gmail_credentials_available() -> bool:
    """
    Check whether we have usable Gmail credentials.

    A token.json may contain an expired access token while still
    having a refresh token. That is still considered connected.
    """

    if not os.path.exists(TOKEN_FILE):
        return False

    try:
        credentials = Credentials.from_authorized_user_file(
            TOKEN_FILE,
            GMAIL_SCOPES,
        )

        return bool(credentials and (credentials.valid or credentials.refresh_token))

    except Exception:
        return False


def _get_gmail_credentials() -> Optional[Credentials]:
    """
    Load Gmail credentials from token.json.

    Refresh the access token automatically when necessary.
    """

    if not os.path.exists(TOKEN_FILE):
        return None

    try:
        credentials = Credentials.from_authorized_user_file(
            TOKEN_FILE,
            GMAIL_SCOPES,
        )
    except Exception:
        return None

    if credentials.valid:
        return credentials

    if credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())

            with open(
                TOKEN_FILE,
                "w",
                encoding="utf-8",
            ) as token:
                token.write(credentials.to_json())

            return credentials

        except Exception:
            return None

    return None


def _oauth_redirect_uri() -> str:
    """Return a stable OAuth callback URI for Google."""
    return GOOGLE_OAUTH_REDIRECT_URI


@app.route("/authorize")
def authorize():
    """Start the Google OAuth authorization flow."""
    if not os.path.exists(CLIENT_SECRETS_FILE):
        return (
            f"Google OAuth credentials file not found. Expected: {CLIENT_SECRETS_FILE}",
            500,
        )

    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=GMAIL_SCOPES,
        autogenerate_code_verifier=True,
    )

    # IMPORTANT:
    # This is the URI that must be registered
    # in Google Cloud Console.
    flow.redirect_uri = _oauth_redirect_uri()

    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    # OAuth state must survive the redirect to Google
    # and the later callback request.
    session["oauth_state"] = state
    session["oauth_code_verifier"] = flow.code_verifier

    return redirect(authorization_url)


@app.route("/oauth2callback")
def oauth2callback():
    """Receive Google's OAuth callback and exchange
    the authorization code for credentials.
    """
    state = session.get("oauth_state")
    code_verifier = session.get("oauth_code_verifier")

    if not state:
        return (
            "Missing OAuth state. Please start the authorization process again.",
            400,
        )

    if not code_verifier:
        return (
            "Missing OAuth code verifier. Please start the authorization process again.",
            400,
        )

    if not os.path.exists(CLIENT_SECRETS_FILE):
        return (
            "Google OAuth credentials file not found.",
            500,
        )

    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=GMAIL_SCOPES,
        state=state,
        code_verifier=code_verifier,
    )

    flow.redirect_uri = _oauth_redirect_uri()

    try:
        flow.fetch_token(authorization_response=request.url)
    except Exception as exc:
        return (
            "Google authorization failed: " + str(exc),
            400,
        )

    credentials = flow.credentials

    # Persist credentials locally for this prototype.
    #
    # IMPORTANT:
    # Do not commit token.json to source control.
    with open(
        TOKEN_FILE,
        "w",
        encoding="utf-8",
    ) as token:
        token.write(credentials.to_json())

    session.pop("oauth_state", None)
    session.pop("oauth_code_verifier", None)

    return redirect(url_for("emails_page"))


# --------------------------------------------------------------------------
# Gmail API helpers
# --------------------------------------------------------------------------


def _get_header(
    message: dict,
    header_name: str,
) -> str:
    """Extract a Gmail message header."""

    payload = message.get(
        "payload",
        {},
    )

    headers = payload.get(
        "headers",
        [],
    )

    for header in headers:
        if header.get("name", "").lower() == header_name.lower():
            return header.get(
                "value",
                "",
            )

    return ""


def _get_message_metadata(
    service,
    message_id: str,
) -> dict:
    """
    Fetch useful metadata for a single Gmail message.
    """

    message = (
        service.users()
        .messages()
        .get(
            userId="me",
            id=message_id,
            format="metadata",
            metadataHeaders=[
                "From",
                "To",
                "Subject",
                "Date",
            ],
        )
        .execute()
    )

    return {
        "id": message.get("id"),
        "thread_id": message.get("threadId"),
        "snippet": message.get(
            "snippet",
            "",
        ),
        "from": _get_header(
            message,
            "From",
        ),
        "to": _get_header(
            message,
            "To",
        ),
        "subject": _get_header(
            message,
            "Subject",
        ),
        "date": _get_header(
            message,
            "Date",
        ),
        "label_ids": message.get(
            "labelIds",
            [],
        ),
    }


# --------------------------------------------------------------------------
# Gmail API: list messages
# --------------------------------------------------------------------------


@app.route("/api/emails")
def api_emails():
    """
    List Gmail inbox messages.

    Optional query parameters:

        max_results
        page_token

    Example:

        /api/emails?max_results=20
    """

    credentials = _get_gmail_credentials()

    if credentials is None:
        return jsonify(
            {
                "error": "Gmail is not connected.",
                "authorize_url": url_for("authorize"),
            }
        ), 401

    try:
        service = discovery.build(
            "gmail",
            "v1",
            credentials=credentials,
            cache_discovery=False,
        )

        max_results = request.args.get(
            "max_results",
            default=200,
            type=int,
        )

        # Keep the API request sensible.
        max_results = max(
            1,
            min(max_results, 100),
        )

        page_token = request.args.get("page_token")

        request_params = {
            "userId": "me",
            "labelIds": ["INBOX"],
            "maxResults": max_results,
            "q": "-category:promotions",
        }

        if page_token:
            request_params["pageToken"] = page_token

        response = service.users().messages().list(**request_params).execute()

        message_refs = response.get(
            "messages",
            [],
        )

        messages = []

        for message_ref in message_refs:
            try:
                messages.append(
                    _get_message_metadata(
                        service,
                        message_ref["id"],
                    )
                )
            except HttpError:
                # If an individual message cannot
                # be retrieved, skip it rather than
                # failing the whole request.
                continue

        return jsonify(
            {
                "messages": messages,
                "next_page_token": response.get("nextPageToken"),
                "result_size_estimate": response.get(
                    "resultSizeEstimate",
                    len(messages),
                ),
            }
        )

    except HttpError as exc:
        return jsonify(
            {
                "error": ("Gmail API request failed."),
                "details": str(exc),
            }
        ), 502

    except Exception as exc:
        return jsonify(
            {
                "error": ("Unexpected Gmail error."),
                "details": str(exc),
            }
        ), 500


# --------------------------------------------------------------------------
# Backwards-compatible Gmail endpoint
# --------------------------------------------------------------------------


@app.route("/request_email")
def email_api_request():
    """Backwards-compatible route.

    Existing code that calls /request_email will now
    receive the same response as /api/emails.
    """
    return api_emails()


# --------------------------------------------------------------------------
# Gmail disconnect
# --------------------------------------------------------------------------


@app.route("/logout/google")
def logout_google():
    """Remove locally stored Gmail credentials.

    This disconnects the application locally.
    It does not revoke the application's authorization
    from the user's Google account.
    """
    if os.path.exists(TOKEN_FILE):
        with contextlib.suppress(OSError):
            os.remove(TOKEN_FILE)

    return redirect(url_for("scan_email"))


# --------------------------------------------------------------------------
# Application entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)


#----------------------------------------------------------------------------
# Login and signup
#----------------------------------------------------------------------------

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/group_29_db",
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    db.create_all()


@app.route("/api/signup", methods=["POST"])
def api_signup():
    print(">>> API SIGNUP ROUTE WAS CALLED SAFELY")
    data = request.get_json(silent=True) or {}

    full_name = (data.get("full_name") or "").strip()
    country = (data.get("country") or "").strip()
    national_id = (data.get("national_id") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not all([full_name, country, national_id, email, password]):
        return jsonify({"error": "full_name, country, national_id, email, and password are required."}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "User already exists."}), 400

    user = User(
        full_name=full_name,
        country=country,
        national_id=national_id,
        email=email,
        password_hash=generate_password_hash(password),
        consent_status=True,
        role="user",
    )

    db.session.add(user)
    db.session.commit()

    return jsonify({
        "message": "User created successfully.",
        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "country": user.country,
            "national_id": user.national_id,
            "email": user.email,
            "role": user.role,
        },
    }), 201


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}

    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "email and password are required."}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid email or password."}), 401

    session.clear()
    session["logged_in"] = True
    session["user_id"] = user.id
    session["user_email"] = user.email
    session["user_role"] = user.role

    return jsonify({
        "message": "Login successful.",
        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "country": user.country,
            "national_id": user.national_id,
            "email": user.email,
            "role": user.role,
        },
    }), 200


@app.route("/api/logout", methods=["POST"])
def api_logout():
    """Logout endpoint: clears the user session."""
    user_id = session.get("user_id")
    if user_id:
        log_audit(user_id, f"User logged out")
    
    session.clear()
    return jsonify({"message": "Logout successful."}), 200


# --------------------------------------------------------------------------
# Admin-Only Endpoints
# --------------------------------------------------------------------------

@app.route("/api/admin/users", methods=["GET"])
@require_role("admin")
def admin_list_users():
    """Admin endpoint: list all users in the system."""
    admin_id = session.get("user_id")
    log_audit(admin_id, "Admin viewed all users")
    
    users = User.query.all()
    return jsonify({
        "count": len(users),
        "users": [
            {
                "id": u.id,
                "full_name": u.full_name,
                "email": u.email,
                "country": u.country,
                "role": u.role,
                "created_at": u.created_at.isoformat() if u.created_at else None,
                "is_deceased": u.is_deceased,
            }
            for u in users
        ],
    }), 200


@app.route("/api/admin/logs", methods=["GET"])
@require_role("admin")
def admin_view_logs():
    """Admin endpoint: view audit logs."""
    admin_id = session.get("user_id")
    log_audit(admin_id, "Admin viewed audit logs")
    
    # Get optional query parameter for filtering by user_id
    user_id_filter = request.args.get("user_id")
    
    if user_id_filter:
        logs = AuditLog.query.filter_by(user_id=user_id_filter).all()
    else:
        logs = AuditLog.query.all()
    
    return jsonify({
        "count": len(logs),
        "logs": [
            {
                "id": log.id,
                "user_id": log.user_id,
                "actor": log.actor,
                "action": log.action,
                "created_at": log.created_at.isoformat() if log.created_at else None,
            }
            for log in logs
        ],
    }), 200


@app.route("/api/admin/users/<user_id>/role", methods=["PATCH"])
@require_role("admin")
def admin_update_user_role(user_id):
    """Admin endpoint: update a user's role."""
    admin_id = session.get("user_id")
    data = request.get_json(silent=True) or {}
    
    new_role = (data.get("role") or "").strip().lower()
    
    if new_role not in ["user", "admin"]:
        return jsonify({"error": "Role must be 'user' or 'admin'."}), 400
    
    user = User.query.filter_by(id=user_id).first()
    if not user:
        return jsonify({"error": "User not found."}), 404
    
    old_role = user.role
    user.role = new_role
    db.session.commit()
    
    log_audit(admin_id, f"Admin changed user {user_id} role from {old_role} to {new_role}")
    
    return jsonify({
        "message": f"User role updated to {new_role}.",
        "user": {
            "id": user.id,
            "email": user.email,
            "role": user.role,
        },
    }), 200
