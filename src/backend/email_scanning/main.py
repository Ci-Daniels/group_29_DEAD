"""Email scanning service.

Encapsulates Gmail OAuth and Gmail API operations so the Flask app layer can
remain thin and only delegate requests.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import socket
import ssl
import threading
import time
from collections.abc import Callable

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib import flow as oauth_flow
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from src.backend.email_scanning.asset_identifier import (
    FINANCIAL_EVENTS,
    classify_email,
    dedupe_providers,
)

_DOMAIN_RE = re.compile(r"@([\w.-]+)")


def _normalize_provider_key(value: str) -> str:
    """Normalize provider names for dedupe/skipping comparisons."""
    normalized = re.sub(r"\s+", " ", (value or "").strip().lower())
    if not normalized:
        return ""
    for suffix in (" bank", " inc", " llc", " ltd", " group", " holdings"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()
    return normalized


def _provider_skip_key_from_from_header(from_header: str) -> str:
    """Build a deterministic skip key from the sender domain when available."""
    if not from_header:
        return ""

    match = _DOMAIN_RE.search(from_header)
    if not match:
        return ""

    domain = match.group(1).lower().strip(".")
    return f"domain:{domain}" if domain else ""


def _provider_name_key_from_from_header(from_header: str) -> str:
    """Approximate provider key from sender domain root for early skipping."""
    if not from_header:
        return ""

    match = _DOMAIN_RE.search(from_header)
    if not match:
        return ""

    domain = match.group(1).lower().strip(".")
    if not domain:
        return ""

    provider_guess = _normalize_provider_key(domain.split(".")[0].replace("-", " "))
    return f"provider:{provider_guess}" if provider_guess else ""


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
        self.default_query = os.environ.get(
            "GMAIL_DEFAULT_QUERY",
            "-category:promotions -label:^smartlabel_promo "
            "-category:forums -label:^smartlabel_group",
        )

        if self.redirect_uri.startswith(("http://127.0.0.1", "http://localhost")):
            os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

        self.gmail_api_max_retries = int(os.environ.get("GMAIL_API_MAX_RETRIES", "4"))
        self.gmail_api_backoff_seconds = float(
            os.environ.get("GMAIL_API_BACKOFF_SECONDS", "1.0")
        )
        self._provider_skip_keys: set[str] = set()
        self._provider_skip_keys_lock = threading.Lock()

    def _provider_skip_keys_snapshot(self) -> set[str]:
        """Read a thread-safe copy of persisted provider skip keys."""
        with self._provider_skip_keys_lock:
            return set(self._provider_skip_keys)

    def _merge_provider_skip_keys(self, keys: set[str]) -> None:
        """Persist newly learned provider skip keys for future scan batches."""
        if not keys:
            return
        with self._provider_skip_keys_lock:
            self._provider_skip_keys.update(keys)

    def _ensure_client_secrets_file(self) -> None:
        if not os.path.exists(self.client_secrets_file):
            raise EmailEvaluationError(
                (
                    "Google OAuth credentials file not found. "
                    f"Expected: {self.client_secrets_file}"
                ),
                status_code=500,
            )

        try:
            with open(self.client_secrets_file, encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            raise EmailEvaluationError(
                "Google OAuth credentials file is not valid JSON.",
                status_code=500,
            ) from exc

        oauth_kind = (
            "web"
            if "web" in payload
            else "installed"
            if "installed" in payload
            else None
        )
        if oauth_kind is None:
            raise EmailEvaluationError(
                "Google OAuth credentials must include a 'web' or 'installed' client.",
                status_code=500,
            )

        oauth_config = payload.get(oauth_kind, {})
        redirect_uris = oauth_config.get("redirect_uris") or []
        if not redirect_uris:
            raise EmailEvaluationError(
                (
                    "Google OAuth credentials are missing redirect URIs. "
                    "Add your callback URL in Google Cloud Console and "
                    "download the OAuth client JSON again."
                ),
                status_code=500,
            )

        if self.redirect_uri not in redirect_uris:
            raise EmailEvaluationError(
                (
                    "Configured redirect URI is not in credentials.json. "
                    f"Configured: {self.redirect_uri}. "
                    "Update Google Cloud Console > OAuth client > Authorized "
                    "redirect URIs, then download a fresh credentials file."
                ),
                status_code=500,
            )

    def start_authorization(self, flask_session: dict) -> str:
        """Start OAuth and store PKCE + state in session."""
        self._ensure_client_secrets_file()

        flow = oauth_flow.Flow.from_client_secrets_file(
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

        flow = oauth_flow.Flow.from_client_secrets_file(
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

    def _is_retryable_http_error(self, exc: HttpError) -> bool:
        status = getattr(getattr(exc, "resp", None), "status", None)
        return status in {408, 429, 500, 502, 503, 504}

    def _is_retryable_network_error(self, exc: Exception) -> bool:
        if isinstance(
            exc, (TimeoutError, socket.timeout, ssl.SSLError, ConnectionError)
        ):
            return True
        message = str(exc).lower()
        return "timed out" in message or "handshake" in message

    def _execute_with_retries(self, request, operation_name: str) -> dict:
        max_attempts = max(1, self.gmail_api_max_retries)
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return request.execute()
            except HttpError as exc:
                last_exc = exc
                if not self._is_retryable_http_error(exc) or attempt == max_attempts:
                    raise EmailEvaluationError(
                        f"Gmail API request failed during {operation_name}: {exc}",
                        status_code=502,
                    ) from exc
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable_network_error(exc) or attempt == max_attempts:
                    raise EmailEvaluationError(
                        (
                            "Gmail network request timed out during "
                            f"{operation_name}. Please retry in a few seconds. "
                            f"Details: {exc}"
                        ),
                        status_code=504,
                    ) from exc

            time.sleep(self.gmail_api_backoff_seconds * (2 ** (attempt - 1)))

        raise EmailEvaluationError(
            (
                "Gmail request failed after retries during "
                f"{operation_name}. Last error: {last_exc}"
            ),
            status_code=504,
        )

    def _get_header(self, message: dict, header_name: str) -> str:
        payload = message.get("payload", {})
        headers = payload.get("headers", [])

        for header in headers:
            if header.get("name", "").lower() == header_name.lower():
                return header.get("value", "")

        return ""

    def _get_message_metadata(self, service, message_id: str) -> dict:
        request = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=message_id,
                format="metadata",
                metadataHeaders=["From", "To", "Subject", "Date"],
            )
        )
        message = self._execute_with_retries(
            request,
            operation_name=f"message metadata fetch ({message_id})",
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
        query: str | None = None,
    ) -> dict:
        """Fetch inbox messages with metadata."""
        service = self._build_service()

        max_results = max(1, min(max_results, 500))

        request_params = {
            "userId": "me",
            "labelIds": ["INBOX"],
            "maxResults": max_results,
            "q": query if query is not None else self.default_query,
        }

        if page_token:
            request_params["pageToken"] = page_token

        try:
            request = service.users().messages().list(**request_params)
            response = self._execute_with_retries(
                request,
                operation_name="inbox list fetch",
            )
        except EmailEvaluationError:
            raise

        messages = []

        for message_ref in response.get("messages", []):
            try:
                message = self._get_message_metadata(service, message_ref["id"])
                label_ids = message.get("label_ids", [])
                if "CATEGORY_PROMOTIONS" in label_ids or "CATEGORY_FORUMS" in label_ids:
                    continue
                messages.append(message)
            except HttpError:
                continue

        return {
            "messages": messages,
            "next_page_token": response.get("nextPageToken"),
            "result_size_estimate": response.get("resultSizeEstimate", len(messages)),
        }

    def fetch_all_inbox_messages(
        self,
        max_results: int | None = None,
        query: str | None = None,
        page_size: int = 500,
    ) -> list[dict]:
        """Fetch multiple inbox pages until max_results or mailbox end is reached."""
        page_size = max(1, min(page_size, 500))

        all_messages: list[dict] = []
        next_page_token: str | None = None

        while True:
            remaining = None if max_results is None else max_results - len(all_messages)
            if remaining is not None and remaining <= 0:
                break

            page_limit = page_size if remaining is None else min(page_size, remaining)
            page = self.fetch_inbox_messages(
                max_results=page_limit,
                page_token=next_page_token,
                query=query,
            )
            all_messages.extend(page.get("messages", []))

            next_page_token = page.get("next_page_token")
            if not next_page_token:
                break

        return all_messages

    def fetch_recent_messages(self, max_results: int = 10) -> list[dict]:
        """Fetch recent messages for view rendering without pagination details."""
        return self.fetch_inbox_messages(max_results=max_results)["messages"]

    def filter_emails(
        self,
        max_results: int = 2000,
        progress_callback: Callable[[int, str], None] | None = None,
        findings_callback: Callable[[list[dict], int], None] | None = None,
    ) -> list[dict]:
        """Scan the inbox and return emails classified as digital assets.

        Fetches messages, runs them through the local NLP asset classifier,
        and returns a list of dicts each containing asset_provider,
        surety_percentage, category, and reasoning. Emails that don't clear
        the confidence threshold are dropped.
        """
        effective_max_results = None if max_results <= 0 else max_results
        page_size = int(os.environ.get("EMAIL_SCAN_PAGE_SIZE", "100"))
        page_size = max(10, min(page_size, 500))
        classification_chunk_size = int(
            os.environ.get("EMAIL_CLASSIFICATION_CHUNK_SIZE", "25")
        )
        classification_chunk_size = max(5, min(classification_chunk_size, page_size))
        findings = []
        next_page_token: str | None = None
        scanned_count = 0
        skipped_provider_count = 0
        seen_provider_keys = self._provider_skip_keys_snapshot()

        if progress_callback is not None:
            progress_callback(2, "Starting inbox scan...")
            progress_callback(5, "Connecting to Gmail and preparing classifier...")

        while True:
            remaining = (
                None
                if effective_max_results is None
                else effective_max_results - scanned_count
            )
            if remaining is not None and remaining <= 0:
                break

            page_limit = page_size if remaining is None else min(page_size, remaining)
            page = self.fetch_inbox_messages(
                max_results=page_limit,
                page_token=next_page_token,
                query=self.default_query,
            )
            page_messages = page.get("messages", [])
            current_page_count = len(page_messages)

            if progress_callback is not None:
                progress_callback(
                    max(8, min(90, 8 + (scanned_count // 50))),
                    (
                        "Fetched "
                        f"{current_page_count} emails. "
                        "Classifying current batch..."
                    ),
                )

            scanned_count += len(page_messages)

            if page_messages:
                for chunk_start in range(
                    0, current_page_count, classification_chunk_size
                ):
                    chunk_end = min(
                        chunk_start + classification_chunk_size, current_page_count
                    )
                    chunk = page_messages[chunk_start:chunk_end]
                    chunk_findings = []
                    for email in chunk:
                        domain_skip_key = _provider_skip_key_from_from_header(
                            email.get("from", "")
                        )
                        provider_name_skip_key = _provider_name_key_from_from_header(
                            email.get("from", "")
                        )
                        if (
                            domain_skip_key and domain_skip_key in seen_provider_keys
                        ) or (
                            provider_name_skip_key
                            and provider_name_skip_key in seen_provider_keys
                        ):
                            skipped_provider_count += 1
                            continue

                        finding = classify_email(email, categories=FINANCIAL_EVENTS)
                        if finding is None:
                            continue

                        chunk_findings.append(finding)

                        if domain_skip_key:
                            seen_provider_keys.add(domain_skip_key)

                        if provider_name_skip_key:
                            seen_provider_keys.add(provider_name_skip_key)

                        provider_key = _normalize_provider_key(finding.asset_provider)
                        if provider_key:
                            seen_provider_keys.add(f"provider:{provider_key}")

                    findings.extend(chunk_findings)

                    if findings_callback is not None and chunk_findings:
                        findings_callback(
                            [finding.to_dict() for finding in chunk_findings],
                            scanned_count,
                        )

                    if progress_callback is not None:
                        progress_callback(
                            max(10, min(92, 10 + (scanned_count // 50))),
                            (
                                f"Classifying current batch: {chunk_end}/"
                                f"{current_page_count} emails"
                            ),
                        )

            estimate = page.get("result_size_estimate", scanned_count)
            if effective_max_results is None:
                total_expected = max(estimate, scanned_count)
            else:
                total_expected = min(
                    effective_max_results, max(estimate, scanned_count)
                )

            if progress_callback is not None:
                progress = 95
                if total_expected > 0:
                    progress = int(min(95, (scanned_count / total_expected) * 95))
                progress_callback(
                    progress,
                    (
                        f"Scanned {scanned_count} inbox emails "
                        f"({skipped_provider_count} skipped due to provider dedupe)..."
                    ),
                )

            next_page_token = page.get("next_page_token")
            if not next_page_token:
                break

        canonical_map = dedupe_providers([f.asset_provider for f in findings])
        for finding in findings:
            finding.asset_provider = canonical_map.get(
                finding.asset_provider,
                finding.asset_provider,
            )

        self._merge_provider_skip_keys(seen_provider_keys)

        if progress_callback is not None:
            progress_callback(
                100,
                (
                    f"Scan complete. {scanned_count} emails scanned, "
                    f"{skipped_provider_count} skipped after provider identification."
                ),
            )

        return [finding.to_dict() for finding in findings]
