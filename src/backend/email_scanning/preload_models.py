"""Preload local email-classifier models.

Usage:
    /path/to/python -m src.backend.email_scanning.preload_models --download

- Without --download: verifies locally cached models can be loaded.
- With --download: allows network fetch and caches missing models.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _configure_repo_local_model_cache() -> Path:
    """Point Hugging Face/SentenceTransformer caches into this repository."""
    repo_root = Path(__file__).resolve().parents[3]
    models_root = repo_root / "src" / "backend" / "email_scanning" / "assets" / "models"
    hf_home = models_root / "hf_home"
    st_home = models_root / "sentence_transformers"
    torch_home = models_root / "torch"

    for directory in (models_root, hf_home, st_home, torch_home):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(st_home))
    os.environ.setdefault("TORCH_HOME", str(torch_home))

    return models_root


def main() -> int:
    """Run model preload and print readiness diagnostics."""
    models_root = _configure_repo_local_model_cache()

    from src.backend.email_scanning.asset_identifier import preload_models

    parser = argparse.ArgumentParser(description="Preload email scanning models")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing Hugging Face models before caching.",
    )
    args = parser.parse_args()

    print(f"Model cache directory: {models_root}")
    status = preload_models(force_download=args.download)
    print(json.dumps(status, indent=2))

    required_loaded = (
        status.get("zero_shot_loaded")
        and status.get("ner_loaded")
        and status.get("embedder_loaded")
    )

    if required_loaded:
        print("Email models are ready for local scanning.")
        return 0

    errors = status.get("model_errors") or {}
    if errors:
        print("\nModel load failures:")
        for model_key, message in errors.items():
            print(f"- {model_key}: {message}")

    print("Some models are missing. Re-run with --download to fetch and cache.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
