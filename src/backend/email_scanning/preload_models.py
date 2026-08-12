"""Preload local email-classifier models.

Usage:
    /path/to/python -m src.backend.email_scanning.preload_models --download

- Without --download: verifies locally cached models can be loaded.
- With --download: allows network fetch and caches missing models.
"""

from __future__ import annotations

import argparse
import json

from src.backend.email_scanning.asset_identifier import preload_models


def main() -> int:
    parser = argparse.ArgumentParser(description="Preload email scanning models")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing Hugging Face models before caching.",
    )
    args = parser.parse_args()

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

    print("Some models are missing. Re-run with --download to fetch and cache.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
