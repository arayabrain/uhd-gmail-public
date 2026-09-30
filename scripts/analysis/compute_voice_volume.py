#!/usr/bin/env python3
"""Compute trial-level voice volume from microphone signals or audio files.

Inputs may be:

- the bipolar microphone (MIC) channels from ds007591 derived trial arrays, or
- a folder of audio files via ``--wav-root``.

Absolute dB scales can differ by recording source; only subject-centered values
are comparable across subjects. The manuscript panel uses the derived tables in
``data/voice_volume/``.

Example:

```bash
uv run python scripts/analysis/compute_voice_volume.py \
    --output-dir outputs/voice_volume_recomputed

uv run python scripts/analysis/compute_voice_volume.py \
    --wav-root /path/to/audio \
    --output-dir outputs/voice_volume_recomputed
```
"""


from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from mne.filter import filter_data

from scripts.figures._bids_runs import OFFLINE_RUNS, ONLINE_RUNS
from scripts.figures.plot_fig1_fig2_rms import compute_voice_volume

MIC_SUBJECTS = ("sub-1", "sub-2", "sub-3")
WAV_SUBJECTS = ("sub-4", "sub-5", "sub-6", "sub-7", "sub-8", "sub-9")
CONDITIONS = ("overt", "minimally_overt", "covert")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root with per-subject derived arrays (eeg_before_preproc).",
    )
    parser.add_argument(
        "--wav-root",
        type=Path,
        default=None,
        help="Optional folder of audio files for volume computation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/voice_volume_recomputed"),
    )
    parser.add_argument("--sr", type=int, default=256)
    parser.add_argument("--low", type=float, default=100.0)
    parser.add_argument("--high", type=float, default=127.0)
    return parser.parse_args()


def condition_from_task(task: str) -> str:
    if task in {"minimallyovert", "minimally overt", "minimally_overt"}:
        return "minimally_overt"
    return task


def iter_mic_trial_paths(data_root: Path) -> list[tuple[str, str, Path]]:
    """Return (subject, condition, npy_path) for offline+online MIC subjects."""
    rows: list[tuple[str, str, Path]] = []
    for run in (*OFFLINE_RUNS, *ONLINE_RUNS):
        if run.subject not in MIC_SUBJECTS:
            continue
        date = run.session.removeprefix("ses-")
        session_glob = data_root / run.subject / date / "eeg_before_preproc"
        if not session_glob.is_dir():
            continue
        for session_dir in sorted(session_glob.glob(f"{date}_*")):
            # Infer condition from sibling metadata when present; else from run.task.
            condition = condition_from_task(run.task)
            for npy_path in sorted(session_dir.glob("*.npy")):
                rows.append((run.subject, condition, npy_path))
    return rows


def volume_from_wav(path: Path, sr: int, low: float, high: float) -> float:
    audio, file_sr = librosa.load(path, sr=None, mono=True)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    filtered = filter_data(audio.astype(float), sr, low, high)
    rms = librosa.feature.rms(y=filtered)
    return float(np.mean(20.0 * np.log10(np.maximum(rms, 1e-12))))


def iter_wav_trials(wav_root: Path) -> list[tuple[str, str, Path]]:
    """Best-effort discovery: ``wav_root/sub-N/<condition>/*.wav``."""
    rows: list[tuple[str, str, Path]] = []
    for subject in WAV_SUBJECTS:
        subject_dir = wav_root / subject
        if not subject_dir.is_dir():
            continue
        for condition in CONDITIONS:
            for folder_name in (condition, condition.replace("_", "-"), condition.replace("_", " ")):
                folder = subject_dir / folder_name
                if not folder.is_dir():
                    continue
                for wav_path in sorted(folder.glob("*.wav")):
                    rows.append((subject, condition, wav_path))
    return rows


def subject_center(trial_df: pd.DataFrame) -> pd.DataFrame:
    """Average trials within subject×condition, then subject-center the three means."""
    subject_level = (
        trial_df.groupby(["subject", "condition"], as_index=False)
        .agg(n_trials=("volume_db", "size"), volume_db=("volume_db", "mean"), source=("source", "first"))
    )
    rows = []
    for subject, group in subject_level.groupby("subject"):
        means = {row.condition: float(row.volume_db) for row in group.itertuples()}
        if set(CONDITIONS) - set(means):
            continue
        center = float(np.mean([means[c] for c in CONDITIONS]))
        for row in group.itertuples():
            rows.append(
                {
                    "subject": subject,
                    "condition": row.condition,
                    "n_trials": int(row.n_trials),
                    "volume_db_subject_centered": float(row.volume_db) - center,
                    "volume_db": float(row.volume_db),
                    "source": row.source,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_rows: list[dict] = []

    for subject, condition, path in iter_mic_trial_paths(args.data_root):
        eeg = np.load(path)
        trial_rows.append(
            {
                "subject": subject,
                "condition": condition,
                "trial": None,  # filled below per group
                "volume_db": compute_voice_volume(eeg),
                "source": "mic_channel",
                "_path": str(path),
            }
        )

    if args.wav_root is not None:
        wav_root = Path(args.wav_root)
        if not wav_root.is_dir():
            raise FileNotFoundError(f"--wav-root not found: {wav_root}")
        for subject, condition, path in iter_wav_trials(wav_root):
            trial_rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "trial": None,
                    "volume_db": volume_from_wav(path, args.sr, args.low, args.high),
                    "source": "audio_recording",
                    "_path": str(path),
                }
            )
    else:
        print(
            "Note: --wav-root not set; only microphone-channel trials under "
            "--data-root are included. For the manuscript panel, use "
            "data/voice_volume/."
        )

    if not trial_rows:
        raise SystemExit(
            "No trials found. Provide derived MIC arrays under --data-root "
            "and/or private WAVs via --wav-root."
        )

    trial_df = pd.DataFrame(trial_rows)
    # Assign zero-based trial index within subject×condition in discovery order.
    trial_df["trial"] = (
        trial_df.groupby(["subject", "condition"]).cumcount().astype(int)
    )
    trial_out = trial_df.loc[:, ["subject", "condition", "trial", "volume_db", "source"]]
    subject_out = subject_center(trial_out)

    trial_path = args.output_dir / "voice_volume_trial.csv"
    subject_path = args.output_dir / "voice_volume_subject.csv"
    trial_out.to_csv(trial_path, index=False)
    subject_out.to_csv(subject_path, index=False)
    print(f"Wrote {trial_path} ({len(trial_out)} trials)")
    print(f"Wrote {subject_path} ({len(subject_out)} subject×condition rows)")
    print(
        "For the manuscript panel, prefer data/voice_volume/voice_volume_subject.csv."
    )


if __name__ == "__main__":
    main()
