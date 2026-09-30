"""CPU smoke tests: figure/table plotters write output from synthetic arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.image import imread

from scripts.figures._temporal_ig_helpers import CandidateRecord
from scripts.figures.make_temporal_ig_paper_trace import save_trace_figure
from scripts.figures.plot_eegnet_spatial_contribution import plot_montage as plot_fig4_montage
from scripts.figures.plot_eegnet_wo_adapt_diff_spatial_contribution import (
    plot_montage as plot_fig5_montage,
)
from scripts.figures.plot_fig1_fig2_rms import plot_emg_stats, plot_volume_stats
from scripts.figures.plot_fig1_preprocessing import (
    plot_eeg_emg_waveform_from_epoch,
    plot_speech_waveform_from_epoch,
    show_likelihood_example,
)
from scripts.figures.plot_fig2_mutual_information import show_montage as show_mi_montage
from scripts.figures.plot_intersubject_variability_publication import draw_figure
from scripts.figures.plot_spatial_channel_correlation import plot_matrix as plot_channel_matrix
from scripts.figures.plot_spatial_contribution_mi_correlation import (
    CorrResult,
    plot_corr_grid,
)
from scripts.figures.plot_spatial_correlation_between_tasks import (
    MatrixResult as BetweenTaskMatrix,
)
from scripts.figures.plot_spatial_correlation_between_tasks import plot_single as plot_between_tasks
from scripts.figures.plot_spatial_shift_correlation_between_tasks import (
    MatrixResult as ShiftMatrix,
)
from scripts.figures.plot_spatial_shift_correlation_between_tasks import (
    plot_single as plot_shift_corr,
)
from scripts.figures.plot_table_decoding_accs import (
    plot_accs_decimation_save,
    plot_accs_save,
    save_accs_table,
)
from scripts.figures.plot_temporal_contribution import write_table as write_temporal_table
from scripts.figures.plot_temporal_jaccard_trial_based import plot_single as plot_jaccard
from scripts.figures.weighted_corr import WeightedCorr
from uhd_eeg.analysis.electrode_subsets_kmedoids import (
    DENSITIES,
    EXPECTED_SUBSETS,
    write_subsets_json,
)

ASSETS = Path("scripts/figures/assets")
ERROR_CONFIG = {"lw": 0.5, "capsize": 1.4, "capthick": 0.5}


def _matrix_result(n: int = 3) -> BetweenTaskMatrix:
    rng = np.random.default_rng(0)
    matrix = np.eye(n) * 0.0
    matrix += rng.normal(0, 0.2, size=(n, n))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1.0)
    sem = np.full((n, n), 0.05)
    p = np.full((n, n), 0.01)
    np.fill_diagonal(p, 1.0)
    return BetweenTaskMatrix(
        matrix=matrix,
        sem=sem,
        p_values=p,
        p_values_corrected=p,
        significant=p < 0.05,
        n_units=np.full((n, n), 9),
    )


def _shift_matrix(n: int = 3) -> ShiftMatrix:
    base = _matrix_result(n)
    return ShiftMatrix(
        matrix=base.matrix,
        sem=base.sem,
        p_values=base.p_values,
        p_values_corrected=base.p_values_corrected,
        significant=base.significant,
        n_units=base.n_units,
    )


def _synthetic_accs() -> pd.DataFrame:
    rows = []
    models = ["EEGNet", "LSTM", "CovTanSVM", "EEGNet_with_mask_4ch", "EEGNet_with_mask_8ch"]
    for subject in ("sub-1", "sub-2", "sub-3"):
        for task in ("overt", "minimally overt", "covert"):
            for model in models:
                row = {
                    "model": model,
                    "subject": subject,
                    "date": "20230511",
                    "sub_idx": 1,
                    "task": task,
                    "online_acc": 0.6,
                    "online_balanced_acc": 0.55,
                }
                for cv in range(10):
                    row[f"offline_acc_{cv}"] = 0.5 + 0.01 * cv
                    row[f"offline_balanced_acc_{cv}"] = 0.48 + 0.01 * cv
                rows.append(row)
    return pd.DataFrame(rows)


def test_fig1_likelihood_schematic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "figures" / "fig1").mkdir(parents=True)
    show_likelihood_example()
    assert (tmp_path / "figures" / "fig1" / "likelihood_example.png").is_file()


def test_fig1_preprocessing_from_synthetic_epoch(tmp_path: Path):
    """CPU smoke: Fig. 1 speech / EEG+EMG panels from a synthetic BIDS-like epoch."""
    rng = np.random.default_rng(0)
    n_samp = 2880
    epoch = rng.normal(scale=50.0, size=(139, n_samp)).astype(np.float64)
    # Speech mic + EMG bipolar pairs need non-zero differentials.
    epoch[130] += 200.0 * np.sin(np.linspace(0, 40 * np.pi, n_samp))
    epoch[132] += 80.0 * np.sin(np.linspace(0, 60 * np.pi, n_samp))
    epoch[134] += 60.0 * np.sin(np.linspace(0, 50 * np.pi, n_samp))
    epoch[136] += 40.0 * np.sin(np.linspace(0, 45 * np.pi, n_samp))

    out = tmp_path / "fig1"
    plot_speech_waveform_from_epoch(epoch, out, "overt")
    plot_eeg_emg_waveform_from_epoch(epoch, out)
    assert (out / "speech_waveform_overt.png").is_file()
    assert (out / "raw_eeg_emg.png").is_file()
    assert (out / "filtered_eeg_emg.png").is_file()
    assert (out / "filtered_avg_eeg.png").is_file()


def test_fig1_voice_volume_and_fig2_emg_rms(tmp_path: Path):
    from uhd_eeg.analysis.voice_volume import load_subject_voice_volume

    csv_path = Path("data/voice_volume/voice_volume_subject.csv")
    if not csv_path.is_file():
        pytest.skip("shipped voice_volume_subject.csv missing")
    voice = load_subject_voice_volume(csv_path)
    emg = pd.DataFrame(
        {
            "subject": ["sub-1", "sub-1", "sub-1", "sub-2", "sub-2", "sub-2"],
            "task": ["overt", "minimally_overt", "covert"] * 2,
            "EOG": np.linspace(0.1, 0.9, 6),
            "EMG upper": np.linspace(0.2, 1.0, 6),
            "EMG lower": np.linspace(0.15, 0.8, 6),
        }
    )
    out1 = tmp_path / "fig1"
    out2 = tmp_path / "fig2"
    out1.mkdir()
    out2.mkdir()
    plot_volume_stats(voice, out1, ERROR_CONFIG)
    plot_emg_stats(emg, out2, ERROR_CONFIG)
    assert (out1 / "voice_volume.png").is_file()
    assert (out2 / "rms_EOG.png").is_file()


def test_fig2_mi_montage(tmp_path: Path):
    color = np.linspace(0, 1, 128)
    out = tmp_path / "mi_montage.png"
    show_mi_montage(color, out, show_colorbar=True, vmax=1.0)
    assert out.is_file()


def test_fig3b_trace_from_synthetic_arrays(tmp_path: Path):
    n_times = 64
    traces = {
        name: np.sin(np.linspace(0, 2 * np.pi, n_times) + i)
        for i, name in enumerate(
            ["Denoised EEG", "min preprocessed EEG", "EOG", "EMG upper", "EMG lower"]
        )
    }
    record = CandidateRecord(
        rank=1,
        selection_type="dissimilar",
        condition="sub-6-overt",
        subject="sub-6",
        task="overt",
        label=2,
        color="orange",
        source_trial="33",
        trial_file=tmp_path / "dummy.npy",
        cv=0,
        pred_probability=0.9,
        run_date="20200101",
        run_time="000000",
    )
    args = argparse.Namespace(
        output_dir=tmp_path / "fig3b",
        dpi=100,
        time_bar_sec=0.1,
        amp_bar_z=2.0,
        mode="averaged",
        top_name="top1",
    )
    paths = save_trace_figure(
        args,
        record,
        traces,
        fs=256,
        eegnet_channels=np.array([35]),
        wo_channels=np.array([78]),
        spacing_scale=2.0,
        height_scale=2.0,
    )
    assert any(p.suffix == ".png" and p.is_file() for p in paths)


def test_fig3_temporal_contribution_table(tmp_path: Path):
    out = tmp_path / "temporal_contribution.csv"
    write_temporal_table(out, np.linspace(0, 1, 16), n_trials=3)
    assert out.is_file()
    assert "temporal_contribution" in out.read_text(encoding="utf-8")


def test_fig3_temporal_jaccard_grid(tmp_path: Path):
    observed = np.eye(5)
    significant = np.zeros((5, 5), dtype=bool)
    significant[0, 1] = significant[1, 0] = True
    p = np.full((5, 5), 0.2)
    p[0, 1] = p[1, 0] = 0.01
    result = {
        "observed": observed,
        "significant": significant,
        "p_values_corrected": p,
        "metric": "jaccard",
    }
    out = tmp_path / "jaccard.png"
    plot_jaccard(out, result, title="smoke")
    assert out.is_file()


def test_fig4_spatial_ig_montage(tmp_path: Path):
    coordinates = np.load(ASSETS / "coordinates_colorless.npy")
    image = imread(ASSETS / "montage_colorless.png")
    values = np.linspace(-1, 1, 128)
    out = tmp_path / "fig4_montage.png"
    plot_fig4_montage(out, values, coordinates, image, -1, 1, "smoke")
    assert out.is_file()


def test_fig4_between_task_correlation(tmp_path: Path):
    out = tmp_path / "between_tasks.png"
    plot_between_tasks(
        out,
        _matrix_result(),
        ["overt", "minimally_overt", "covert"],
        "smoke",
        -1.0,
        1.0,
    )
    assert out.is_file()


def test_fig4_channel_level_matrix(tmp_path: Path):
    matrix = np.eye(3)
    pvals = np.full((3, 3), 0.2)
    pvals[0, 1] = pvals[1, 0] = 0.01
    out = plot_channel_matrix(
        matrix,
        pvals,
        ["overt", "minimally_overt", "covert"],
        vmin=-1.0,
        vmax=1.0,
        alpha=0.05,
        output_dir=tmp_path,
    )
    assert out.with_suffix(".png").is_file()


def test_fig4_ig_vs_mi_correlation(tmp_path: Path):
    n_t, n_e = 3, 3
    values = np.linspace(-0.5, 0.5, n_t * n_e).reshape(n_t, n_e)
    p = np.full((n_t, n_e), 0.01)
    result = CorrResult(
        values=values,
        p_values=p,
        p_values_corrected=p,
        significant=p < 0.05,
        n_channels=np.full((n_t, n_e), 128),
    )
    out = tmp_path / "ig_mi.png"
    plot_corr_grid(
        out,
        result,
        ["overt", "minimally_overt", "covert"],
        ["EOG", "EMG_upper", "EMG_lower"],
        "avg",
        -1.0,
        1.0,
    )
    assert out.is_file()


def test_fig5_adapt_filter_diff_montage(tmp_path: Path):
    coordinates = np.load(ASSETS / "coordinates_colorless.npy")
    image = imread(ASSETS / "montage_colorless.png")
    out = tmp_path / "fig5_diff.png"
    plot_fig5_montage(out, np.linspace(-1, 1, 128), coordinates, image, -1.0, 1.0, "diff")
    assert out.is_file()


def test_fig5_s6_shift_correlation(tmp_path: Path):
    out = tmp_path / "shift_corr.png"
    plot_shift_corr(
        out,
        _shift_matrix(),
        ["overt", "minimally_overt", "covert"],
        "smoke",
        -1.0,
        1.0,
    )
    assert out.is_file()


def test_fig_s2_electrode_subsets_json(tmp_path: Path):
    out = tmp_path / "electrode_subsets.json"
    write_subsets_json(EXPECTED_SUBSETS, out)
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    for density in DENSITIES:
        assert str(density) in text


def test_fig_s2a_electrode_subset_montage(tmp_path: Path):
    from scripts.figures.plot_fig_s2a_electrode_montage import load_subsets, show_montage

    channels = load_subsets()[4]
    out = tmp_path / "montage_4"
    show_montage(channels, out)
    assert out.with_suffix(".png").is_file()


def test_fig_s7_jitter_plot_from_synthetic_table(tmp_path: Path):
    from scripts.figures._jitter_ablation_helpers import (
        compute_statistical_tests,
        plot_condition,
        summarize_condition,
    )

    rows = []
    for sbj in ("sub-1", "sub-2"):
        for behavior in ("overt", "minimally_overt", "covert"):
            for jitter in ("random_jitter_±100ms", "no_jitter"):
                rows.append(
                    {
                        "sbj": sbj,
                        "behavior": behavior,
                        "jitter_condition": jitter,
                        "model_name": "EEGNet",
                        "ensemble_method": "single",
                        "balanced_acc_test": 0.35 if jitter == "no_jitter" else 0.45,
                    }
                )
    data_subj = pd.DataFrame(rows)
    stats = compute_statistical_tests(data_subj)
    summary = summarize_condition(data_subj)
    out = tmp_path / "s7_jitter.png"
    plot_condition(summary, data_subj, out, stats)
    assert out.is_file()


def test_fig_s2b_channel_decimation_from_summary_csv(tmp_path: Path):
    from scripts.figures._channel_decimation_helpers import plot_decimation_from_summary_csvs

    group = pd.DataFrame(
        {
            "behavior": ["overt"] * 5,
            "n_channels": [4, 8, 16, 32, 128],
            "balanced_acc_mean": [0.30, 0.40, 0.50, 0.55, 0.60],
            "n_subjects": [2] * 5,
        }
    )
    subject = pd.DataFrame(
        {
            "subject": ["sub-1", "sub-1", "sub-2", "sub-2"],
            "behavior": ["overt", "overt", "overt", "overt"],
            "n_channels": [4, 128, 4, 128],
            "balanced_acc_test_mean": [0.28, 0.58, 0.32, 0.62],
        }
    )
    group_csv = tmp_path / "group.csv"
    subject_csv = tmp_path / "subject.csv"
    group.to_csv(group_csv, index=False)
    subject.to_csv(subject_csv, index=False)
    figures_dir = tmp_path / "figures"
    plot_decimation_from_summary_csvs(
        group_csv,
        subject_csv,
        figures_dir,
        figure_stem="smoke_decimation",
    )
    assert (figures_dir / "smoke_decimation.png").is_file()


def test_fig_s3_s4_publication_plot(tmp_path: Path):
    measures = Path("data/intersubject/subject_measures.csv")
    summary = Path("data/intersubject/subject_condition_summary.csv")
    ages = Path("data/subject_demographics.example.csv")
    if not (measures.is_file() and summary.is_file() and ages.is_file()):
        pytest.skip("shipped intersubject tables missing")
    for recording_type, tag in (("offline", "S3"), ("online", "S4")):
        fig = draw_figure(
            recording_type,
            show_subject_labels=False,
            measures_csv=measures,
            summary_csv=summary,
            ages_csv=ages,
        )
        out = tmp_path / f"fig_{tag}.png"
        fig.savefig(out, dpi=100)
        assert out.is_file()


def test_fig_s5_wo_adapt_between_task_plot(tmp_path: Path):
    """Fig. S5 uses the same plotter as Fig. 4 with wo-adapt model outputs."""
    out = tmp_path / "s5_between_tasks.png"
    plot_between_tasks(
        out,
        _matrix_result(),
        ["overt", "minimally_overt", "covert"],
        "EEGNet_wo_adapt_filt",
        -1.0,
        1.0,
    )
    assert out.is_file()


def test_fig_s6_weighted_corr_helper():
    rng = np.random.default_rng(0)
    x = rng.normal(size=128)
    y = x + rng.normal(scale=0.1, size=128)
    w = np.abs(rng.normal(size=128)) + 0.1
    r, p = WeightedCorr(num_shuffle=99, seed=0)(x, y, w)
    assert np.isfinite(r)
    assert 0.0 <= p <= 1.0


def test_tables_decoding_accs_plot_and_csv(tmp_path: Path):
    accs = _synthetic_accs()
    subjects = ["sub-1", "sub-2", "sub-3"]
    plot_dir = tmp_path / "tables"
    plot_accs_save(
        accs,
        plot_dir,
        acc_type="balanced_acc",
        online=True,
        offline=True,
        subjects=subjects,
    )
    assert any(plot_dir.glob("*.png"))
    save_accs_table(
        accs.assign(subject=accs["subject"].str.replace("sub-", "subject")),
        plot_dir,
        acc_type="balanced_acc",
        offline=True,
        models=["EEGNet", "LSTM", "CovTanSVM"],
    )
    assert any(plot_dir.glob("*.csv"))
    plot_accs_decimation_save(
        accs,
        plot_dir / "decimation",
        acc_type="balanced_acc",
        online=True,
        offline=True,
        show_ratio=False,
        subjects=subjects,
    )
    assert any((plot_dir / "decimation").glob("*.png"))
