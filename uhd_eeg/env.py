"""Load environment variables from .env file."""

import json
import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_raw_root() -> Path:
    """Get RAW_ROOT path from .env."""
    val = os.environ.get("RAW_ROOT")
    if val is None:
        raise ValueError("RAW_ROOT not found in .env")
    return Path(val)


def get_bids_root() -> Path:
    """Get BIDS_ROOT path from .env (NAS BIDS output directory)."""
    val = os.environ.get("BIDS_ROOT")
    if val is None:
        raise ValueError("BIDS_ROOT not found in .env")
    return Path(val)


def load_participant_mapping() -> Dict[str, str]:
    """Load name→BIDS-ID mapping from path specified in .env."""
    mapping_path = os.environ.get("PARTICIPAT_MAPPING_PATH")
    if mapping_path is None:
        raise ValueError("PARTICIPAT_MAPPING_PATH not found in .env")
    with open(mapping_path) as f:
        return json.load(f)
