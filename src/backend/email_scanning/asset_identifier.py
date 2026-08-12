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

import re
from dataclasses import dataclass, field
from functools import lru_cache

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
        "description": "A statement showing investments, securities, "
        "fund holdings or portfolio value."
    },
    "investment_confirmation": {
        "description": "Confirmation that an investment was purchased, "
        "sold or allocated."
    },
    "dividend_payment": {"description": "Notification of dividends or distributions."},
    "interest_payment": {"description": "Notification of interest earned or paid."},
    "pension_statement": {
        "description": "Statement showing pension or retirement savings."
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
    "marketing": {
        "description": "Promotional content encouraging the recipient "
        "to use or purchase a financial service."
    },
    "payment": {"description": "Confirmation of a payment for goods or services."},
    "loan_statement": {"description": "Statement concerning borrowed money or a loan."},
}


ZERO_SHOT_MODEL = (
    "MoritzLaurer/deberta-v3-base-zeroshot-v2"  # identify possibility of assets
)
NER_MODEL = "dslim/bert-base-NER"  # recognize named entities i.e. providers
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # useful for deduplication
SURETY_THRESHOLD = 55.0


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
    return pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL, device=-1)


@lru_cache(maxsize=1)
def _get_ner_pipeline():
    return pipeline("ner", model=NER_MODEL, aggregation_strategy="simple", device=-1)


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


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

    entities = _get_ner_pipeline()(f"{from_header} {subject}")
    orgs = [e["word"] for e in entities if e.get("entity_group") == "ORG"]
    if orgs:
        # Prefer the longest org span found; usually the most specific name.
        best = max(orgs, key=len)
        return best.strip(), 0.9

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
    result = classifier(
        text, candidate_labels=candidate_labels, hypothesis_template=hypothesis_template
    )

    desc_to_label = {v: k for k, v in label_to_desc.items()}
    ranked = list(zip(result["labels"], result["scores"]))
    top_desc, top_score = ranked[0]
    top_label = desc_to_label[top_desc]

    all_scores = {desc_to_label[desc]: round(score * 100, 1) for desc, score in ranked}
    return top_label, round(top_score * 100, 1), all_scores


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


def classify_email(email: dict, categories: dict) -> AssetFinding | None:
    """Classify a single email dict (as returned by fetch_inbox_messages).

    Returns None if the email doesn't clear the surety threshold, i.e. it's
    judged unlikely to represent a real digital/financial asset.
    """
    category, surety, all_scores = classify_category(email, categories)
    if surety < SURETY_THRESHOLD:
        return None

    provider, provider_conf = extract_asset_provider(email)
    reasoning = build_reasoning(
        email, category, categories[category]["description"], surety, provider
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
