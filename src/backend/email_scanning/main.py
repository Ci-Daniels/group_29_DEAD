"""Email scanning service.

Encapsulates Gmail OAuth and Gmail API operations so the Flask app layer can
remain thin and only delegate requests.
"""

from __future__ import annotations

import contextlib
import os

import google_auth_oauthlib.flow
from asset_identifier import classify_emails
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient import discovery
from googleapiclient.errors import HttpError


class EmailEvaluationError(Exception):
    """Domain error for Gmail integration with an HTTP-friendly status code."""

    def __init__(self, message: str, status_code: int = 500):
        """Initialize an email-domain error with a corresponding HTTP status."""
        super().__init__(message)
        self.status_code = status_code


class EmailEvaluation:
    """Service that manages OAuth, credentials, and inbox queries."""

    def __init__(self):
        """Initialize Gmail OAuth and token storage configuration."""
        self.scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
        self.client_secrets_file = os.environ.get(
            "GOOGLE_CLIENT_SECRETS_FILE",
            "credentials.json",
        )
        self.token_file = os.environ.get(
            "GOOGLE_TOKEN_FILE",
            "token.json",
        )
        self.redirect_uri = os.environ.get(
            "GOOGLE_OAUTH_REDIRECT_URI",
            "http://127.0.0.1:5000/oauth2callback",
        )

        if self.redirect_uri.startswith(("http://127.0.0.1", "http://localhost")):
            os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

    def _ensure_client_secrets_file(self) -> None:
        if not os.path.exists(self.client_secrets_file):
            raise EmailEvaluationError(
                (
                    "Google OAuth credentials file not found. "
                    f"Expected: {self.client_secrets_file}"
                ),
                status_code=500,
            )
        self.user_id = 00

    def start_authorization(self, flask_session: dict) -> str:
        """Start OAuth and store PKCE + state in session."""
        self._ensure_client_secrets_file()

        flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
            self.client_secrets_file,
            scopes=self.scopes,
            autogenerate_code_verifier=True,
        )
        flow.redirect_uri = self.redirect_uri

        authorization_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )

        flask_session["oauth_state"] = state
        flask_session["oauth_code_verifier"] = flow.code_verifier

        return authorization_url

    def complete_authorization(
        self,
        flask_session: dict,
        authorization_response_url: str,
    ) -> None:
        """Exchange auth code for credentials and persist token."""
        state = flask_session.get("oauth_state")
        code_verifier = flask_session.get("oauth_code_verifier")

        if not state:
            raise EmailEvaluationError(
                "Missing OAuth state. Please start authorization again.",
                status_code=400,
            )

        if not code_verifier:
            raise EmailEvaluationError(
                "Missing OAuth code verifier. Please start authorization again.",
                status_code=400,
            )

        self._ensure_client_secrets_file()

        flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
            self.client_secrets_file,
            scopes=self.scopes,
            state=state,
            code_verifier=code_verifier,
        )
        flow.redirect_uri = self.redirect_uri

        try:
            flow.fetch_token(authorization_response=authorization_response_url)
        except Exception as exc:
            raise EmailEvaluationError(
                f"Google authorization failed: {exc}",
                status_code=400,
            ) from exc

        with open(self.token_file, "w", encoding="utf-8") as token:
            token.write(flow.credentials.to_json())

        flask_session.pop("oauth_state", None)
        flask_session.pop("oauth_code_verifier", None)

    def get_credentials(self) -> Credentials | None:
        """Load Gmail credentials and refresh if required."""
        if not os.path.exists(self.token_file):
            return None

        try:
            credentials = Credentials.from_authorized_user_file(
                self.token_file,
                self.scopes,
            )
        except Exception:
            return None

        if credentials.valid:
            return credentials

        if credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())

                with open(self.token_file, "w", encoding="utf-8") as token:
                    token.write(credentials.to_json())

                return credentials
            except Exception:
                return None

        return None

    def credentials_available(self) -> bool:
        """Check whether OAuth is connected enough to call Gmail APIs."""
        if not os.path.exists(self.token_file):
            return False

        try:
            credentials = Credentials.from_authorized_user_file(
                self.token_file,
                self.scopes,
            )
            return bool(
                credentials and (credentials.valid or credentials.refresh_token)
            )
        except Exception:
            return False

    def disconnect(self) -> None:
        """Delete locally persisted OAuth token."""
        if os.path.exists(self.token_file):
            with contextlib.suppress(OSError):
                os.remove(self.token_file)

    def _build_service(self):
        credentials = self.get_credentials()
        if credentials is None:
            raise EmailEvaluationError("Gmail is not connected.", status_code=401)

        return discovery.build(
            "gmail",
            "v1",
            credentials=credentials,
            cache_discovery=False,
        )

    def _get_header(self, message: dict, header_name: str) -> str:
        payload = message.get("payload", {})
        headers = payload.get("headers", [])

        for header in headers:
            if header.get("name", "").lower() == header_name.lower():
                return header.get("value", "")

        return ""

    def _get_message_metadata(self, service, message_id: str) -> dict:
        message = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=message_id,
                format="metadata",
                metadataHeaders=["From", "To", "Subject", "Date"],
            )
            .execute()
        )

        return {
            "id": message.get("id"),
            "thread_id": message.get("threadId"),
            "snippet": message.get("snippet", ""),
            "from": self._get_header(message, "From"),
            "to": self._get_header(message, "To"),
            "subject": self._get_header(message, "Subject"),
            "date": self._get_header(message, "Date"),
            "label_ids": message.get("labelIds", []),
        }

    def fetch_inbox_messages(
        self,
        max_results: int = 100,
        page_token: str | None = None,
        query: str = "-category:promotions",
    ) -> dict:
        """Fetch inbox messages with metadata and optional pagination."""
        service = self._build_service()

        max_results = max(1, min(max_results, 100))

        request_params = {
            "userId": "me",
            "labelIds": ["INBOX"],
            "maxResults": max_results,
            "q": query,
        }

        if page_token:
            request_params["pageToken"] = page_token

        try:
            response = service.users().messages().list(**request_params).execute()
        except HttpError as exc:
            raise EmailEvaluationError(
                f"Gmail API request failed: {exc}",
                status_code=502,
            ) from exc

        messages = []

        for message_ref in response.get("messages", []):
            try:
                messages.append(self._get_message_metadata(service, message_ref["id"]))
            except HttpError:
                continue

        return {
            "messages": messages,
            "next_page_token": response.get("nextPageToken"),
            "result_size_estimate": response.get("resultSizeEstimate", len(messages)),
        }

    def fetch_recent_messages(self, max_results: int = 10) -> list[dict]:
        """Fetch recent messages for view rendering without pagination details."""
        return self.fetch_inbox_messages(max_results=max_results)["messages"]

    def filter_emails(self, max_results: int = 100) -> list[dict]:
        """Scan the inbox and return emails classified as digital assets.

        Fetches messages, runs them through the local NLP asset classifier,
        and returns a list of dicts each containing asset_provider,
        surety_percentage, category, and reasoning. Emails that don't clear
        the confidence threshold are dropped.
        """

        messages = self.fetch_inbox_messages(max_results=max_results)["messages"]
        findings = classify_emails(messages)

        return [finding.to_dict() for finding in findings]
