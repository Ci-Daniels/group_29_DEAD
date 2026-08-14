"""Local NLP classification of emails into digital-asset categories.

Runs entirely on local Hugging Face models (no external API calls / cost).
Given an email (from EmailEvaluation.fetch_inbox_messages) and the
financial_events taxonomy defined , this module
produces, per email:
    - asset_provider: best-guess institution/company that holds the asset
    - surety_percentage: 0-100 confidence the email represents a real asset
    - reasoning: short natural-language justification
    - category: which financial_events key it matched

Models used (all local, CPU-friendly):
    - MoritzLaurer/deberta-v3-base-zeroshot-v2  (zero-shot category match)
    - dslim/bert-base-NER                        (organization extraction)
    - sentence-transformers/all-MiniLM-L6-v2      (provider name de-duplication)
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

FINANCIAL_EVENTS = {
    "account_statement": {
        "description": "A periodic statement showing an account's "
        "activity, balance, holdings or transactions."
    },
    "transaction_notification": {
        "description": "A notification that money, securities or "
        "digital assets were transferred or traded."
    },
    "balance_notification": {
        "description": "A message reporting a current or historical "
        "financial account balance."
    },
    "portfolio_statement": {
        "description": "A statement showing investments, securities belonging to the registered email user. "
        "fund holdings or portfolio value."
    },
    "investment_confirmation": {
        "description": "Confirmation that an investment was purchased by the given email user. "
        "sold or allocated."
    },
    "dividend_payment": {
        "description": "Notification of dividends or distributions paid to the email user."
    },
    "interest_payment": {
        "description": "Notification of interest earned or paid to the email user."
    },
    "pension_statement": {
        "description": "Statement showing pension or retirement savings paid or owed to the email user."
    },
    "insurance_statement": {
        "description": "Statement or notification relating to an "
        "insurance policy or its cash value."
    },
    "crypto_transaction": {
        "description": "Cryptocurrency deposit, withdrawal, purchase, sale or transfer."
    },
    "crypto_statement": {
        "description": "Statement or account summary for cryptocurrency holdings."
    },
    "account_opening": {
        "description": "Confirmation that a financial account has been opened."
    },
    "account_application": {
        "description": "An application or request to open a financial "
        "account that may not yet exist."
    },
    "payment": {"description": "Confirmation of a payment for goods or services."},
    "loan_statement": {
        "description": "Statement concerning borrowed money or a loan by a given user."
    },
}


ZERO_SHOT_MODELS = [
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    "facebook/bart-large-mnli",
]
NER_MODEL = "dslim/bert-base-NER"  # recognize named entities i.e. providers
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # useful for deduplication
SURETY_THRESHOLD = 55.0
OWNERSHIP_THRESHOLD = float(os.environ.get("ASSET_OWNERSHIP_THRESHOLD", "60.0"))

_MODEL_LOAD_ERRORS: dict[str, str] = {}

_CATEGORY_KEYWORDS = {
    "account_statement": ["statement", "monthly statement", "account summary"],
    "transaction_notification": ["transaction", "transfer", "debit", "credit"],
    "balance_notification": ["balance", "available balance", "current balance"],
    "portfolio_statement": ["portfolio", "holdings", "aum", "fund"],
    "investment_confirmation": ["trade confirmation", "filled", "order executed"],
    "dividend_payment": ["dividend", "distribution"],
    "interest_payment": ["interest payment", "interest earned", "apy"],
    "pension_statement": ["retirement", "401k", "pension", "ira"],
    "insurance_statement": ["policy", "premium", "insurance"],
    "crypto_transaction": ["wallet", "blockchain", "withdrawal", "deposit", "crypto"],
    "crypto_statement": ["crypto statement", "exchange", "coin", "token"],
    "account_opening": ["account opened", "welcome", "new account"],
    "account_application": ["application", "pending approval", "verify identity"],
    "marketing": ["offer", "promotion", "bonus", "apply now"],
    "payment": ["receipt", "invoice", "payment received", "payment sent"],
    "loan_statement": ["loan", "mortgage", "emi", "installment"],
}

_OWNERSHIP_POSITIVE_CUES = {
    "your account",
    "your balance",
    "statement period",
    "transaction id",
    "account ending",
    "available balance",
    "portfolio value",
    "policy number",
    "dividend paid",
    "interest credited",
}

_OWNERSHIP_NEGATIVE_CUES = {
    "learn more",
    "subscribe",
    "unsubscribe",
    "sponsored",
    "offer",
    "promotion",
    "apply now",
    "limited time",
    "ad",
}


def _load_env_token_from_dotenv() -> str | None:
    """Best-effort .env token loader for local development workflows."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip().strip('"').strip("'")

    candidate_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[3] / ".env",
    ]
    for env_path in candidate_paths:
        if not env_path.exists():
            continue

        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if not line.startswith("HF_TOKEN="):
                    continue

                _, value = line.split("=", 1)
                token = value.strip().strip('"').strip("'")
                if token:
                    os.environ.setdefault("HF_TOKEN", token)
                    return token
        except OSError:
            continue

    return None


def _hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or _load_env_token_from_dotenv()
    )


@contextmanager
def _hf_offline_mode(enabled: bool):
    """Temporarily force offline loading for Hugging Face libraries."""
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}

    if enabled:
        for key in keys:
            os.environ[key] = "1"

    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass
class AssetFinding:
    """Structured result of classifying one email."""

    message_id: str
    asset_provider: str
    provider_confidence: float
    category: str
    surety_percentage: float
    reasoning: str
    subject: str = ""
    date: str = ""
    raw_scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-ify a finding."""
        return {
            "message_id": self.message_id,
            "asset_provider": self.asset_provider,
            "provider_confidence": self.provider_confidence,
            "category": self.category,
            "surety_percentage": self.surety_percentage,
            "reasoning": self.reasoning,
            "subject": self.subject,
            "date": self.date,
        }


# ---- model loading (cached so we only load once per process) -------------


@lru_cache(maxsize=1)
def _get_zero_shot_pipeline():
    local_only = os.environ.get("EMAIL_CLASSIFIER_LOCAL_ONLY", "1") == "1"
    token = _hf_token()
    last_error: Exception | None = None
    for model_name in ZERO_SHOT_MODELS:
        try:
            with _hf_offline_mode(local_only):
                result = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=-1,
                    token=token,
                )
            _MODEL_LOAD_ERRORS.pop("zero_shot", None)
            return result
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        _MODEL_LOAD_ERRORS["zero_shot"] = str(last_error)
    return None


@lru_cache(maxsize=1)
def _get_ner_pipeline():
    local_only = os.environ.get("EMAIL_CLASSIFIER_LOCAL_ONLY", "1") == "1"
    token = _hf_token()
    try:
        with _hf_offline_mode(local_only):
            result = pipeline(
                "ner",
                model=NER_MODEL,
                aggregation_strategy="simple",
                device=-1,
                token=token,
            )
        _MODEL_LOAD_ERRORS.pop("ner", None)
        return result
    except Exception as exc:
        _MODEL_LOAD_ERRORS["ner"] = str(exc)
        return None


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    local_only = os.environ.get("EMAIL_CLASSIFIER_LOCAL_ONLY", "1") == "1"
    token = _hf_token()
    try:
        with _hf_offline_mode(local_only):
            result = SentenceTransformer(
                EMBED_MODEL,
                token=token,
            )
        _MODEL_LOAD_ERRORS.pop("embedder", None)
        return result
    except Exception as exc:
        _MODEL_LOAD_ERRORS["embedder"] = str(exc)
        return None


_DOMAIN_RE = re.compile(r"@([\w.-]+)")
# Common consumer domains that are never the "asset provider" themselves.
_GENERIC_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com",
}


def extract_asset_provider(email: dict) -> tuple[str, float]:
    """Best-effort extraction of the institution/company holding the asset.

    Tries NER on the From/Subject text first (highest precision), falls back
    to the sending domain (cheap, deterministic, decent recall).
    Returns (provider_name, confidence_0_to_1).
    """
    from_header = email.get("from", "")
    subject = email.get("subject", "")

    ner_pipe = _get_ner_pipeline()
    if ner_pipe is not None:
        try:
            entities = ner_pipe(f"{from_header} {subject}")
            orgs = [e["word"] for e in entities if e.get("entity_group") == "ORG"]
            if orgs:
                # Prefer the longest org span found; usually the most specific name.
                best = max(orgs, key=len)
                return best.strip(), 0.9
        except Exception:
            pass

    domain_match = _DOMAIN_RE.search(from_header)
    if domain_match:
        domain = domain_match.group(1).lower()
        if domain not in _GENERIC_DOMAINS:
            provider = domain.split(".")[0].replace("-", " ").title()
            return provider, 0.6

    return "Unknown", 0.0


def dedupe_providers(
    providers: list[str], similarity_threshold: float = 0.75
) -> dict[str, str]:
    """Cluster near-duplicate provider names (e.g. 'Chase' / 'Chase Bank').

    Returns a mapping from each original name to a canonical representative,
    using local sentence-embedding cosine similarity (no API calls).
    """
    if not providers:
        return {}

    unique = list(dict.fromkeys(providers))  # preserve order, dedupe exact matches
    embedder = _get_embedder()
    if embedder is None:
        # Fast fallback when embedding model is unavailable locally.
        suffixes = (" bank", " inc", " llc", " ltd", " group", " holdings")
        canonical_by_key: dict[str, str] = {}
        mapping: dict[str, str] = {}
        for name in unique:
            key = re.sub(r"\s+", " ", name.strip().lower())
            for suffix in suffixes:
                if key.endswith(suffix):
                    key = key[: -len(suffix)]
            key = key.strip()
            if not key:
                key = name.strip().lower()
            canonical = canonical_by_key.setdefault(key, name)
            mapping[name] = canonical
        return mapping

    embeddings = embedder.encode(unique, convert_to_tensor=True)

    canonical: dict[str, str] = {}
    assigned = [False] * len(unique)

    for i, name in enumerate(unique):
        if assigned[i]:
            continue
        canonical[name] = name
        assigned[i] = True
        for j in range(i + 1, len(unique)):
            if assigned[j]:
                continue
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim >= similarity_threshold:
                canonical[unique[j]] = name
                assigned[j] = True

    return canonical


# ---- category classification ----------------------------------------------


def _classify_category_heuristic(
    email: dict, categories: dict
) -> tuple[str, float, dict]:
    """Keyword-based category fallback used when HF models are unavailable."""
    text = f"{email.get('subject', '')} {email.get('snippet', '')}".lower()
    if not text.strip():
        return "unknown", 0.0, {}

    has_amount = bool(re.search(r"\$\s?\d|\d+\.\d{2}|usd|eur|gbp", text))

    scores: dict[str, float] = {}
    for key in categories:
        keywords = _CATEGORY_KEYWORDS.get(key, [])
        hits = sum(1 for token in keywords if token in text)
        base = 15.0 + (15.0 * hits)
        if has_amount and hits > 0:
            base += 10.0
        scores[key] = min(base, 95.0)

    top_label = max(scores, key=scores.get)
    return top_label, round(scores[top_label], 1), scores


def _classify_ownership_heuristic(email: dict) -> tuple[bool, float, str]:
    """Heuristic ownership gate fallback when zero-shot is unavailable."""
    text = f"{email.get('subject', '')} {email.get('snippet', '')}".lower()
    if not text.strip():
        return False, 0.0, "Email has no usable content."

    positive_hits = sum(1 for cue in _OWNERSHIP_POSITIVE_CUES if cue in text)
    negative_hits = sum(1 for cue in _OWNERSHIP_NEGATIVE_CUES if cue in text)

    ownership_score = 45.0 + (12.0 * positive_hits) - (14.0 * negative_hits)
    ownership_score = max(0.0, min(95.0, ownership_score))

    if ownership_score >= OWNERSHIP_THRESHOLD:
        return True, round(ownership_score, 1), "Ownership cues found in email content."

    return (
        False,
        round(ownership_score, 1),
        "Insufficient ownership evidence or promotional language detected.",
    )


def classify_ownership_gate(email: dict) -> tuple[bool, float, str]:
    """Stage 1 gate: determine if email is about recipient's real owned assets."""
    text = f"{email.get('subject', '')}. {email.get('snippet', '')}".strip()
    if not text:
        return False, 0.0, "Email has no usable content."

    classifier = _get_zero_shot_pipeline()
    if classifier is None:
        return _classify_ownership_heuristic(email)

    candidate_labels = [
        (
            "A direct record or update about the recipient's own current financial "
            "account, balance, holding, transaction, payout, or policy"
        ),
        (
            "General marketing, informational content, or non-personal financial "
            "content not proving the recipient currently owns the asset"
        ),
    ]

    try:
        result = classifier(
            text,
            candidate_labels=candidate_labels,
            hypothesis_template="This email is {}.",
        )
        ranked = list(zip(result["labels"], result["scores"]))
        top_label, top_score = ranked[0]
        ownership_score = round(top_score * 100, 1)

        is_owned_asset = (
            top_label == candidate_labels[0] and ownership_score >= OWNERSHIP_THRESHOLD
        )

        if is_owned_asset:
            return (
                True,
                ownership_score,
                "Ownership gate passed for recipient asset signal.",
            )

        return (
            False,
            ownership_score,
            "Ownership gate failed; email appears promotional or non-personal.",
        )
    except Exception:
        return _classify_ownership_heuristic(email)


def classify_category(email: dict, categories: dict) -> tuple[str, float, dict]:
    """Zero-shot match an email against the financial_events taxonomy.

    `categories` is the EmailEvaluation.financial_events dict (or a subset).
    Returns (top_category, surety_0_to_100, all_scores).
    """
    text = f"{email.get('subject', '')}. {email.get('snippet', '')}".strip()
    if not text:
        return "unknown", 0.0, {}

    labels = list(categories.keys())
    hypothesis_template = "This email is about {}."
    # Use each category's description as the candidate label text so the
    # model matches against meaning, not just the short key name.
    label_to_desc = {k: v["description"] for k, v in categories.items()}
    candidate_labels = [label_to_desc[k] for k in labels]

    classifier = _get_zero_shot_pipeline()
    if classifier is None:
        return _classify_category_heuristic(email, categories)

    try:
        result = classifier(
            text,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
        )

        desc_to_label = {v: k for k, v in label_to_desc.items()}
        ranked = list(zip(result["labels"], result["scores"]))
        top_desc, top_score = ranked[0]
        top_label = desc_to_label[top_desc]

        all_scores = {
            desc_to_label[desc]: round(score * 100, 1) for desc, score in ranked
        }
        return top_label, round(top_score * 100, 1), all_scores
    except Exception:
        return _classify_category_heuristic(email, categories)


def build_reasoning(
    email: dict, category: str, description: str, surety: float, provider: str
) -> str:
    """Template-based reasoning statement (no generative model needed)."""
    subject = email.get("subject", "(no subject)")
    return (
        f"Subject \"{subject}\" was matched to category '{category}' with "
        f"{surety:.1f}% confidence. This category covers: {description} "
        f"The sender/domain suggests '{provider}' as the holding institution."
    )


# ---- public entry point ----------------------------------------------------


def preload_models(force_download: bool = False) -> dict:
    """Warm local model caches for faster, offline-ready email classification.

    If force_download=True, temporarily allows remote fetches so missing
    Hugging Face models are downloaded and cached on disk.
    """
    previous_local_only = os.environ.get("EMAIL_CLASSIFIER_LOCAL_ONLY")

    if force_download:
        os.environ["EMAIL_CLASSIFIER_LOCAL_ONLY"] = "0"

    # Clear caches so we load with the current environment setting.
    _get_zero_shot_pipeline.cache_clear()
    _get_ner_pipeline.cache_clear()
    _get_embedder.cache_clear()
    _MODEL_LOAD_ERRORS.clear()

    token = _hf_token()
    if token:
        os.environ.setdefault("HF_TOKEN", token)

    zero_shot = _get_zero_shot_pipeline()
    ner = _get_ner_pipeline()
    embedder = _get_embedder()

    if force_download:
        if previous_local_only is None:
            os.environ.pop("EMAIL_CLASSIFIER_LOCAL_ONLY", None)
        else:
            os.environ["EMAIL_CLASSIFIER_LOCAL_ONLY"] = previous_local_only

    zero_shot_model = None
    if zero_shot is not None:
        zero_shot_model = getattr(
            getattr(zero_shot, "model", None),
            "name_or_path",
            None,
        ) or getattr(
            getattr(getattr(zero_shot, "model", None), "config", None),
            "_name_or_path",
            None,
        )

    return {
        "hf_token_detected": bool(token),
        "zero_shot_loaded": zero_shot is not None,
        "zero_shot_model": zero_shot_model,
        "ner_loaded": ner is not None,
        "ner_model": NER_MODEL if ner is not None else None,
        "embedder_loaded": embedder is not None,
        "embedder_model": EMBED_MODEL if embedder is not None else None,
        "model_errors": dict(_MODEL_LOAD_ERRORS),
    }


def classify_email(email: dict, categories: dict) -> AssetFinding | None:
    """Classify a single email dict (as returned by fetch_inbox_messages).

    Returns None if the email doesn't clear the surety threshold, i.e. it's
    judged unlikely to represent a real digital/financial asset.
    """
    is_owned_asset, ownership_surety, ownership_reason = classify_ownership_gate(email)
    if not is_owned_asset:
        return None

    category, surety, all_scores = classify_category(email, categories)
    if surety < SURETY_THRESHOLD:
        return None

    provider, provider_conf = extract_asset_provider(email)
    reasoning = (
        f"{ownership_reason} (ownership confidence {ownership_surety:.1f}%). "
        + build_reasoning(
            email, category, categories[category]["description"], surety, provider
        )
    )

    return AssetFinding(
        message_id=email.get("id", ""),
        asset_provider=provider,
        provider_confidence=round(provider_conf * 100, 1),
        category=category,
        surety_percentage=surety,
        reasoning=reasoning,
        subject=email.get("subject", ""),
        date=email.get("date", ""),
        raw_scores=all_scores,
    )


def classify_emails(
    emails: list[dict], categories=FINANCIAL_EVENTS
) -> list[AssetFinding]:
    """Classify a batch of emails, dropping ones below the surety threshold.

    Remove duplicate providers across the batch afterward so 'Chase' and
    'Chase Bank' collapse to a single canonical provider name.
    """
    findings = [
        f for f in (classify_email(e, categories) for e in emails) if f is not None
    ]

    canonical_map = dedupe_providers([f.asset_provider for f in findings])
    for f in findings:
        f.asset_provider = canonical_map.get(f.asset_provider, f.asset_provider)

    return findings
