"""Fig. 1 schematic: speech / EEG+EMG waveforms from BIDS-derived trials.

Loads per-trial arrays written by ``bids/extract_from_bids.py`` under
``configs/paths.yaml`` ``output_root``, addressed by BIDS entities only
(``sub``, ``ses``, ``task``, ``acq``, ``run``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from mne.filter import filter_data
from scipy.stats import zscore
from termcolor import cprint

from scripts.figures._bids_runs import OFFLINE_RUNS, BidsRun
from scripts.figures.make_preproc_files import preproc
from scripts.figures._lib.default_plt import cm_to_inch, plt
from uhd_eeg.paths import get_output_root

# Manuscript Fig. 1 example panels (sub-1 calibration runs).
FIG1_SPEECH_RUNS: tuple[tuple[BidsRun, str], ...] = (
    (BidsRun("sub-1", "ses-20230529", "overt", "calibration"), "overt"),
    (BidsRun("sub-1", "ses-20230511", "minimallyovert", "calibration"), "min-overt"),
    (BidsRun("sub-1", "ses-20230529", "covert", "calibration"), "covert"),
)
FIG1_EEG_EMG_RUN = BidsRun(
    "sub-1", "ses-20230511", "minimallyovert", "calibration"
)
DEFAULT_TRIAL_INDEX = 2
N_CH_EEG = 128
N_CH_EMG = 3
DURA_SAMP = 320  # 1.25 s at 256 Hz


def run_dir(run: BidsRun, output_root: Path) -> Path:
    """Directory for one extracted BIDS run under ``output_root``."""
    return (
        Path(output_root)
        / run.subject
        / run.session
        / f"task-{run.task}_acq-{run.acq}_run-{run.run}"
    )


def trial_array_dir(run: BidsRun, output_root: Path) -> Path:
    """Return the directory that holds per-trial ``*.npy`` files."""
    base = run_dir(run, output_root)
    nested = base / "trials"
    if nested.is_dir():
        return nested
    return base


def load_trial_epoch(
    run: BidsRun,
    trial_index: int,
    output_root: Path | None = None,
) -> np.ndarray:
    """Load one trial epoch ``(n_ch, n_samp)`` for a BIDS run."""
    root = Path(output_root) if output_root is not None else get_output_root()
    d = trial_array_dir(run, root)
    candidates = [d / f"{trial_index:03d}.npy", d / f"{trial_index}.npy"]
    for path in candidates:
        if path.is_file():
            return np.load(path)
    raise FileNotFoundError(
        f"No trial {trial_index} under {d} (tried {[p.name for p in candidates]})"
    )


def check_run_existence(run: BidsRun, output_root: Path | None = None) -> None:
    """Print whether derived trial arrays exist for one BIDS run."""
    root = Path(output_root) if output_root is not None else get_output_root()
    base = run_dir(run, root)
    trials = trial_array_dir(run, root)
    npy_paths = sorted(trials.glob("*.npy"))
    meta = base / "run.json"
    cprint(f"{run.key}: {len(npy_paths)} trials under {trials}", "cyan")
    if meta.is_file():
        with open(meta, encoding="utf-8") as f:
            info = json.load(f)
        cprint(f"  run.json n_trials={info.get('n_trials')}", "cyan")
    if npy_paths:
        eeg = np.load(npy_paths[0])
        cprint(f"  first trial shape={eeg.shape}", "cyan")
        cprint("OK", "green")
    else:
        cprint("  missing derived trials", "yellow")


def plot_speech_waveform_from_epoch(
    epoch: np.ndarray, save_path: Path, task: str
) -> None:
    """Plot bipolar speech mic channels from a raw epoch."""
    speech = epoch[130, :] - epoch[131, :]
    speech = filter_data(speech, 256, 100, 127)

    fig, _ax = plt.subplots(figsize=(1.25 * cm_to_inch, 0.22 * cm_to_inch))
    fig.subplots_adjust()
    plt.plot(speech)
    plt.axis("off")
    plt.ylim(-2500, 2500)
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / f"speech_waveform_{task}.png")
    plt.clf()
    plt.close()


def plot_speech_waveform(
    run: BidsRun,
    save_path: Path,
    task: str,
    trial_index: int = DEFAULT_TRIAL_INDEX,
    output_root: Path | None = None,
) -> None:
    """Load a BIDS-derived trial and plot the speech waveform."""
    epoch = load_trial_epoch(run, trial_index, output_root=output_root)
    plot_speech_waveform_from_epoch(epoch, save_path, task)


def plot_eeg_emg_waveform_from_epoch(
    epoch: np.ndarray,
    save_path: Path,
    avg_eeg: np.ndarray | None = None,
) -> None:
    """Plot raw / filtered EEG+EMG stacks and optional averaged EEG panel."""
    blue = tuple(np.array([47, 85, 151]) / 255)

    eeg_z = zscore(epoch[:N_CH_EEG, :], axis=1)
    emg = np.vstack(
        (
            epoch[132, :] - epoch[133, :],
            epoch[134, :] - epoch[135, :],
            epoch[136, :] - epoch[137, :],
        )
    )
    emg = filter_data(emg, 256, 30, 127)
    emg_z = zscore(emg, axis=1)

    eeg_filtered, _emg_filtered = preproc(epoch, wo_adapt_filt=False)
    eeg_filtered_z = zscore(eeg_filtered[:N_CH_EEG, :], axis=1)
    emg_filtered_z = zscore(emg, axis=1)

    save_path.mkdir(parents=True, exist_ok=True)
    for eeg_to_show, emg_to_show, name in zip(
        [eeg_z, eeg_filtered_z], [emg_z, emg_filtered_z], ["raw", "filtered"]
    ):
        fig, _ax = plt.subplots(figsize=(3.0 * cm_to_inch, 2.5 * cm_to_inch))
        fig.subplots_adjust()
        ch = 0
        for ch in range(N_CH_EEG):
            plt.plot(eeg_to_show[ch, :] - ch * 2, c="k")
        for i in range(N_CH_EMG):
            plt.plot(emg_to_show[i, :] - (ch + 3) * 2 - i * 4, c=blue)
        plt.axis("off")
        plt.savefig(save_path / f"{name}_eeg_emg.png")
        plt.clf()
        plt.close()

    if avg_eeg is None:
        # Schematic average: last analysis window of the filtered EEG.
        avg_eeg = eeg_filtered[:N_CH_EEG, -DURA_SAMP:]
    fig, _ax = plt.subplots(figsize=(0.6 * cm_to_inch, 2.5 * cm_to_inch))
    fig.subplots_adjust()
    for ch in range(N_CH_EEG):
        plt.plot(avg_eeg[ch, :] - ch * 2, c="k")
    plt.axis("off")
    plt.savefig(save_path / "filtered_avg_eeg.png")
    plt.clf()
    plt.close()


def plot_eeg_emg_waveform(
    run: BidsRun,
    save_path: Path,
    trial_index: int = DEFAULT_TRIAL_INDEX,
    output_root: Path | None = None,
) -> None:
    """Load a BIDS-derived trial and plot EEG/EMG preprocessing panels."""
    epoch = load_trial_epoch(run, trial_index, output_root=output_root)
    plot_eeg_emg_waveform_from_epoch(epoch, save_path)


def show_likelihood_example(save_path: Path | None = None) -> None:
    """Schematic likelihood bars (not derived from real data)."""
    green = tuple(np.array([0, 176, 80]) / 255)
    magenta = tuple(np.array([208, 0, 149]) / 255)
    orange = tuple(np.array([237, 125, 49]) / 255)
    violet = tuple(np.array([112, 48, 160]) / 255)
    yellow = tuple(np.array([255, 192, 0]) / 255)

    likelihood = [0.45, 0.23, 0.07, 0.17, 0.08]
    colors = [green, magenta, orange, violet, yellow]
    fig, ax = plt.subplots(figsize=(3.0 * cm_to_inch, 2.5 * cm_to_inch))
    fig.subplots_adjust()

    rect = ax.bar(np.arange(5), likelihood)
    for i, color in enumerate(colors):
        rect[i].set_color(color)
    plt.ylim([0, 1])
    plt.yticks([])
    plt.xticks([])
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    out = Path("figures/fig1") if save_path is None else Path(save_path)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "likelihood_example.png")
    plt.clf()
    plt.close()


def main(output_root: Path | None = None) -> None:
    save_path = Path("figures/fig1")
    save_path.mkdir(parents=True, exist_ok=True)
    root = Path(output_root) if output_root is not None else get_output_root()

    seen: set[tuple[str, str]] = set()
    for run in OFFLINE_RUNS:
        key = (run.subject, run.session)
        if key in seen:
            continue
        seen.add(key)
        cprint(f"{run.subject}, {run.session}", "cyan")
        check_run_existence(run, output_root=root)

    for run, task_label in FIG1_SPEECH_RUNS:
        plot_speech_waveform(run, save_path, task_label, output_root=root)

    plot_eeg_emg_waveform(FIG1_EEG_EMG_RUN, save_path, output_root=root)
    show_likelihood_example(save_path)


if __name__ == "__main__":
    main()
