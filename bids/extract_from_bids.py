#!/usr/bin/env python3
"""CLI: extract per-trial arrays from OpenNeuro ds007591 BIDS data.

Examples
--------
Using ``configs/paths.yaml``:

```bash
uv run python bids/extract_from_bids.py
```

Override paths explicitly:

```bash
uv run python bids/extract_from_bids.py \
    --bids-root /path/to/ds007591 \
    --output-root /path/to/derived
```

Restrict subjects:

```bash
uv run python bids/extract_from_bids.py --subjects sub-1 sub-3
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uhd_eeg.bids.constants import SUBJECT_IDS
from uhd_eeg.bids.extract import extract_dataset
from uhd_eeg.paths import get_bids_root, get_output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=None,
        help="OpenNeuro ds007591 root (default: configs/paths.yaml bids_root).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for per-trial arrays (default: configs/paths.yaml output_root).",
    )
    parser.add_argument(
        "--paths-file",
        type=Path,
        default=None,
        help="Optional YAML path config (default: configs/paths.yaml).",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help=f"Subject IDs to process (default: {' '.join(SUBJECT_IDS)}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bids_root = args.bids_root or get_bids_root(args.paths_file)
    output_root = args.output_root or get_output_root(args.paths_file)
    subjects = args.subjects or list(SUBJECT_IDS)

    print(f"BIDS root:   {bids_root}")
    print(f"Output root: {output_root}")
    print(f"Subjects:    {', '.join(subjects)}")
    print()

    manifest = extract_dataset(bids_root, output_root, subject_ids=subjects)
    print(f"Extracted {manifest['n_runs']} runs.")
    print(f"Wrote manifest to {Path(output_root) / 'manifest.json'}")


if __name__ == "__main__":
    main()
