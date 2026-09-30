"""Smoke tests for scrubbed figure helpers (CPU, no dataset)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.figures._adaptive_filter_waveforms import (
    TrialBundle,
    load_or_compute_bundle,
    slice_window,
)
from scripts.figures.plot_adaptive_filter_waveforms_trace import (
    plot_trace_variant,
    trace_rows,
)
from scripts.figures.plot_intersubject_variability_supplement import (
    load,
    significance_marker,
    to_public_subject_id,
)


def test_normalize_subject_id():
    """Original name; equivalent to ``test_public_subject_ids``."""
    assert to_public_subject_id("subject7") == "sub-7"
    assert to_public_subject_id("sub-7") == "sub-7"
    assert to_public_subject_id("3") == "sub-3"


def test_public_subject_ids():
    test_normalize_subject_id()


def test_fig_s1_plot_from_synthetic_arrays(tmp_path: Path):
    """Original name; same coverage as ``test_fig_s1_plot_from_precomputed_arrays``."""
    test_fig_s1_plot_from_precomputed_arrays(tmp_path)


def test_fig_s1_plot_from_precomputed_arrays(tmp_path: Path):
    rng = np.random.default_rng(0)
    n_samp = 320
    before = tmp_path / "before.npy"
    after = tmp_path / "after.npy"
    emg = tmp_path / "emg.npy"
    np.save(before, rng.normal(size=(128, n_samp)))
    np.save(after, rng.normal(size=(128, n_samp)))
    np.save(emg, rng.normal(size=(3, n_samp)))
    bundle = load_or_compute_bundle(
        eeg_before_npy=before,
        eeg_after_npy=after,
        emg_npy=emg,
        trial_index=49,
    )
    assert isinstance(bundle, TrialBundle)
    time, win = slice_window(bundle, fs=256, start_sec=0.0, duration_sec=1.0)
    rows = trace_rows(bundle, affected_channel=64, minimal_channel=47, win=win)
    out = tmp_path / "figS1"
    plot_trace_variant(
        subject="sub-7",
        condition="overt",
        recording_type="offline",
        trial_index=49,
        time=time,
        rows=rows,
        scale_bar_sec=1.0,
        scale_bar_value=2.0,
        scale_bar_label="z-score",
        output_suffix="trace_zscore",
        output_dir=out,
        formats=("png",),
    )
    assert any(out.glob("*.png"))


def test_intersubject_load_shipped_csvs():
    measures = Path("data/intersubject/subject_measures.csv")
    summary = Path("data/intersubject/subject_condition_summary.csv")
    ages = Path("data/subject_demographics.example.csv")
    if not (measures.is_file() and summary.is_file() and ages.is_file()):
        pytest.skip("shipped intersubject tables missing")
    df = load("offline", ages_csv=ages)
    assert set(df["subject"]) == {f"sub-{i}" for i in range(1, 10)}
    assert "snr" in df.columns
    assert "emg_rms" in df.columns
    assert "eeg_emg_mutual_information" in df.columns
    assert "dominant_hand" not in df.columns
    assert significance_marker(0.0005, 0.0005) == "*"
    assert significance_marker(0.04, 0.04) == "*"
    assert significance_marker(0.04, 0.12) == ""
