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
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

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
