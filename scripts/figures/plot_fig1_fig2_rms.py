"""plot voice volume (Fig. 1b) and EMG RMS (Fig. 2a).

Fig. 1b loads ``data/voice_volume/voice_volume_subject.csv`` (subject-centered dB),
plots individual subjects and group means, and annotates pairwise Wilcoxon
comparisons with ``*`` when Bonferroni-adjusted P < 0.05.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
from mne.filter import filter_data
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from termcolor import cprint

from scripts.figures._bids_runs import OFFLINE_RUNS
from scripts.figures._lib.default_plt import (
    cm_to_inch,
    dark_blue,
    light_blue,
    medium_blue,
    plt,
)
from uhd_eeg.analysis.stats import manuscript_star
from uhd_eeg.analysis.voice_volume import (
    CONDITIONS,
    CONDITION_LABELS,
    DEFAULT_SUBJECT_CSV,
    load_subject_voice_volume,
    subject_centered_pivot,
    voice_volume_stats,
)

CONDITION_COLORS = {
    "overt": dark_blue,
    "minimally_overt": medium_blue,
    "covert": light_blue,
}


def compute_voice_volume(eeg: np.ndarray) -> float:
    """Volume (dB) from MIC+/MIC− bipolar channels (indices 130/131)."""
    speech = eeg[130, :] - eeg[131, :]
    speech = filter_data(speech, 256, 100, 127)
    rms = librosa.feature.rms(y=speech)
    volume_db = 20 * np.log10(rms)
    return float(np.mean(volume_db))


def volume_stats_save(
    data_root: Path, subject: str, date: str, session_name: str, save_path: Path
) -> np.ndarray:
    before_preproc_paths = list(
        (
            data_root / subject / date / "eeg_before_preproc" / f"{date}_{session_name}"
        ).glob("*.npy")
    )
    volume_session = []
    for path in before_preproc_paths:
        eeg = np.load(path)
        volume_session.append(compute_voice_volume(eeg))
    volume_session = np.asarray(volume_session, dtype=float)
    np.save(str(data_root / subject / date / f"voice_volumes_{session_name}.npy"), volume_session)
    return volume_session


def load_emg_rms(
    sub_dates: List[Tuple[str, str]],
    data_root: Path,
    n_ch_eeg: int = 128,
    n_ch_emg: int = 3,
    highpass: int = 60,
) -> pd.DataFrame:
    emg_rmss = pd.DataFrame(columns=("subject", "task", "EOG", "EMG upper", "EMG lower"))
    for subject, date in sub_dates:
        with open(str(data_root / subject / date / "metadata.json")) as f:
            metadata = json.load(f)
        metadata = {key: metadata[key] for key in metadata.keys() if key != "subject"}
        for session_name, meta in metadata.items():
            task = meta["task"]
            if task == "minimally overt":
                task_key = "minimally_overt"
            elif task == "minimallyovert":
                task_key = "minimally_overt"
            else:
                task_key = task
            paths = list(
                (
                    data_root / subject / date / "eeg_before_preproc" / f"{date}_{session_name}"
                ).glob("*.npy")
            )
            rows = []
            for path in paths:
                eeg = np.load(path)
                emg = eeg[n_ch_eeg : n_ch_eeg + n_ch_emg]
                emg = filter_data(emg.astype(float), 256, highpass, None)
                rms = np.sqrt(np.mean(emg**2, axis=1))
                rows.append(rms)
            if not rows:
                continue
            emg_rms = np.asarray(rows)
            emg_rmss = pd.concat(
                [
                    emg_rmss,
                    pd.DataFrame(
                        {
                            "subject": [subject] * emg_rms.shape[0],
                            "task": [task_key] * emg_rms.shape[0],
                            "EOG": emg_rms[:, 0],
                            "EMG upper": emg_rms[:, 1],
                            "EMG lower": emg_rms[:, 2],
                        }
                    ),
                ],
                ignore_index=True,
            )
    return emg_rmss


def add_significance_bracket(ax, x1: float, x2: float, y: float, height: float, text: str) -> None:
    if not text:
        return
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="black", lw=0.8)
    ax.text((x1 + x2) / 2, y + height * 1.25, text, ha="center", va="bottom", fontsize=8)


def plot_volume_stats(
    voice_subjects: pd.DataFrame,
    save_path: Path,
    error_config: Dict[str, float] | None = None,
    *,
    stats: dict | None = None,
) -> Path:
    """Plot Fig. 1b from subject-centered dB (one point per subject × condition)."""
    del error_config  # retained for smoke-test call compatibility
    pivot = subject_centered_pivot(voice_subjects)
    if stats is None:
        stats = voice_volume_stats(voice_subjects)

    fig, ax = plt.subplots(figsize=(4.0 * cm_to_inch, 5.0 * cm_to_inch))
    x = np.arange(len(CONDITIONS))
    subject_ids = list(pivot.index)
    offsets = np.linspace(-0.16, 0.16, max(len(subject_ids), 1))
    offset_by_subject = dict(zip(subject_ids, offsets))

    means = []
    for i, condition in enumerate(CONDITIONS):
        values = pivot[condition].to_numpy(dtype=float)
        means.append(float(np.mean(values)))
        xs = [i + offset_by_subject[subject] for subject in subject_ids]
        ax.scatter(
            xs,
            values,
            s=18,
            color=CONDITION_COLORS[condition],
            edgecolor="white",
            linewidth=0.4,
            alpha=0.9,
            zorder=3,
        )
        ax.hlines(
            means[-1],
            i - 0.22,
            i + 0.22,
            color="black",
            linewidth=2.4,
            zorder=4,
        )

    finite = np.concatenate([pivot[c].to_numpy(dtype=float) for c in CONDITIONS])
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    y_span = max(y_max - y_min, 1.0)
    bracket_height = y_span * 0.04
    y_base = y_max + y_span * 0.08
    pair_index = {c: i for i, c in enumerate(CONDITIONS)}
    significant = [row for row in stats["pairwise"] if row["reject"]]
    significant = sorted(significant, key=lambda row: row["p_value_corrected"])
    for bracket_i, row in enumerate(significant):
        x1 = pair_index[row["condition1"]]
        x2 = pair_index[row["condition2"]]
        if x1 > x2:
            x1, x2 = x2, x1
        add_significance_bracket(
            ax,
            x1,
            x2,
            y_base + bracket_i * y_span * 0.12,
            bracket_height,
            manuscript_star(row["p_value_corrected"]),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], rotation=40, ha="right")
    ax.set_ylabel("Volume (dB, subject-centered)")
    y_top = max(y_max, y_base + max(len(significant), 1) * y_span * 0.12)
    ax.set_ylim(y_min - y_span * 0.08, y_top + y_span * 0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / "voice_volume.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def write_voice_volume_stats(stats: dict, save_path: Path) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "statistic": stats["friedman"]["statistic"],
                "p_value": stats["friedman"]["p_value"],
                "n_subjects": stats["n_subjects"],
            }
        ]
    ).to_csv(save_path / "friedman_volume.csv", index=False)
    pd.DataFrame(stats["pairwise"]).to_csv(save_path / "stats_volume.csv", index=False)
    pd.DataFrame(
        [{"condition": k, "mean_subject_centered_db": v} for k, v in stats["means"].items()]
    ).to_csv(save_path / "means_volume.csv", index=False)


def emg_subject_means(emg_rmss: pd.DataFrame) -> pd.DataFrame:
    """One row per subject × condition with mean RMS per EMG channel."""
    task_map = {
        "overt": "overt",
        "minimally overt": "minimally_overt",
        "minimallyovert": "minimally_overt",
        "minimally_overt": "minimally_overt",
        "covert": "covert",
    }
    df = emg_rmss.copy()
    df["condition"] = df["task"].map(lambda t: task_map.get(str(t), str(t)))
    return (
        df.groupby(["subject", "condition"], as_index=False)[["EOG", "EMG upper", "EMG lower"]]
        .mean()
    )


def statistical_test_emg_subject(
    emg_subjects: pd.DataFrame, save_path: Path, alpha: float = 0.05
) -> dict[str, dict]:
    """Friedman + Wilcoxon (Bonferroni ×3) on subject-mean EMG RMS."""
    save_path.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict] = {}
    for emg_type in ("EOG", "EMG upper", "EMG lower"):
        pivot = emg_subjects.pivot(index="subject", columns="condition", values=emg_type)
        for condition in CONDITIONS:
            if condition not in pivot.columns:
                raise ValueError(f"Missing {condition} for {emg_type}")
        pivot = pivot.loc[:, list(CONDITIONS)]
        friedman = friedmanchisquare(*(pivot[c].to_numpy() for c in CONDITIONS))
        pair_rows = []
        raw_p = []
        for left, right in combinations(CONDITIONS, 2):
            result = wilcoxon(pivot[left], pivot[right], alternative="two-sided")
            raw_p.append(float(result.pvalue))
            pair_rows.append(
                {
                    "condition1": left,
                    "condition2": right,
                    "statistic": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "n_subjects": int(len(pivot)),
                }
            )
        adjusted = multipletests(raw_p, method="bonferroni", alpha=alpha)[1]
        for row, p_adj in zip(pair_rows, adjusted):
            row["p_value_corrected"] = float(p_adj)
            row["reject"] = bool(p_adj < alpha)
            row["star"] = manuscript_star(float(p_adj), alpha=alpha)
        stats = {
            "friedman": {"statistic": float(friedman.statistic), "p_value": float(friedman.pvalue)},
            "pairwise": pair_rows,
            "means": {c: float(pivot[c].mean()) for c in CONDITIONS},
            "n_subjects": int(len(pivot)),
        }
        out[emg_type] = stats
        pd.DataFrame(
            [
                {
                    "statistic": stats["friedman"]["statistic"],
                    "p_value": stats["friedman"]["p_value"],
                    "n_subjects": stats["n_subjects"],
                }
            ]
        ).to_csv(save_path / f"friedman_{emg_type}.csv", index=False)
        pd.DataFrame(pair_rows).to_csv(save_path / f"stats_{emg_type}.csv", index=False)
    return out


def plot_emg_stats(
    emg_rmss: pd.DataFrame,
    save_path: Path,
    error_config: Dict[str, float],
    *,
    stats_by_channel: dict[str, dict] | None = None,
) -> None:
    """Plot Fig. 2a: subject means as points + group mean; ``*`` for P_adj < 0.05."""
    del error_config
    save_path.mkdir(parents=True, exist_ok=True)
    subjects = emg_subject_means(emg_rmss)
    if stats_by_channel is None:
        stats_by_channel = statistical_test_emg_subject(subjects, save_path)

    for emg_type in ("EOG", "EMG upper", "EMG lower"):
        pivot = subjects.pivot(index="subject", columns="condition", values=emg_type)
        pivot = pivot.loc[:, list(CONDITIONS)]
        stats = stats_by_channel[emg_type]
        fig, ax = plt.subplots(figsize=(4.0 * cm_to_inch, 5.0 * cm_to_inch))
        subject_ids = list(pivot.index)
        offsets = np.linspace(-0.16, 0.16, max(len(subject_ids), 1))
        offset_by_subject = dict(zip(subject_ids, offsets))
        means = []
        for i, condition in enumerate(CONDITIONS):
            values = pivot[condition].to_numpy(dtype=float)
            means.append(float(np.mean(values)))
            xs = [i + offset_by_subject[s] for s in subject_ids]
            ax.scatter(
                xs,
                values,
                s=18,
                color=CONDITION_COLORS[condition],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.9,
                zorder=3,
            )
            ax.hlines(means[-1], i - 0.22, i + 0.22, color="black", linewidth=2.4, zorder=4)

        finite = np.concatenate([pivot[c].to_numpy(dtype=float) for c in CONDITIONS])
        y_min = min(0.0, float(np.min(finite)))
        y_max = float(np.max(finite))
        y_span = max(y_max - y_min, 1e-6)
        bracket_height = y_span * 0.04
        y_base = y_max + y_span * 0.08
        pair_index = {c: i for i, c in enumerate(CONDITIONS)}
        significant = sorted(
            [row for row in stats["pairwise"] if row["reject"]],
            key=lambda row: row["p_value_corrected"],
        )
        for bracket_i, row in enumerate(significant):
            x1 = pair_index[row["condition1"]]
            x2 = pair_index[row["condition2"]]
            if x1 > x2:
                x1, x2 = x2, x1
            add_significance_bracket(
                ax,
                x1,
                x2,
                y_base + bracket_i * y_span * 0.12,
                bracket_height,
                manuscript_star(row["p_value_corrected"]),
            )
        ax.set_xticks(np.arange(len(CONDITIONS)))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], rotation=40, ha="right")
        ax.set_ylabel("RMS")
        y_top = max(y_max, y_base + max(len(significant), 1) * y_span * 0.12)
        ax.set_ylim(y_min, y_top + y_span * 0.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        out = save_path / f"rms_{emg_type}.png"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voice-subject-csv",
        type=Path,
        default=DEFAULT_SUBJECT_CSV,
        help="Published subject-level voice-volume CSV (Fig. 1b).",
    )
    parser.add_argument(
        "--fig1-dir",
        type=Path,
        default=Path("figures/fig1"),
    )
    parser.add_argument(
        "--fig2-dir",
        type=Path,
        default=Path("figures/fig2/RMS"),
    )
    parser.add_argument(
        "--skip-emg",
        action="store_true",
        help="Only plot Fig. 1b from the shipped voice-volume CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    voice = load_subject_voice_volume(args.voice_subject_csv)
    stats = voice_volume_stats(voice)
    cprint(
        f"Fig. 1b means (subject-centered dB): "
        + ", ".join(f"{k}={v:.2f}" for k, v in stats["means"].items()),
        "cyan",
    )
    cprint(
        f"Friedman chi2={stats['friedman']['statistic']:.2f} "
        f"p={stats['friedman']['p_value']:.6g}",
        "cyan",
    )
    for row in stats["pairwise"]:
        cprint(
            f"{row['condition1']} vs {row['condition2']}: "
            f"P_adj={row['p_value_corrected']:.4g} {row['star'] or '(n.s.)'}",
            "cyan" if row["reject"] else "green",
        )

    args.fig1_dir.mkdir(parents=True, exist_ok=True)
    plot_volume_stats(voice, args.fig1_dir, {}, stats=stats)
    write_voice_volume_stats(stats, args.fig1_dir)

    if args.skip_emg:
        return

    sub_dates = sorted(
        {(run.subject, run.session.removeprefix("ses-")) for run in OFFLINE_RUNS if run.subject in {"sub-1", "sub-2", "sub-3"}}
    )
    data_root = Path("data/")
    try:
        emg_rmss = load_emg_rms(sub_dates, data_root)
    except FileNotFoundError as exc:
        cprint(f"Skipping Fig. 2a EMG RMS (missing preproc inputs): {exc}", "yellow")
        return
    if emg_rmss.empty:
        cprint("Skipping Fig. 2a EMG RMS (no trial arrays found under data/).", "yellow")
        return
    args.fig2_dir.mkdir(parents=True, exist_ok=True)
    subjects = emg_subject_means(emg_rmss)
    emg_stats = statistical_test_emg_subject(subjects, args.fig2_dir)
    plot_emg_stats(emg_rmss, args.fig2_dir, {}, stats_by_channel=emg_stats)


if __name__ == "__main__":
    main()
