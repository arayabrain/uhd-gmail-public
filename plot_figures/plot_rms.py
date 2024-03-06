"""plot voice volume and emg rms (Fig. 1, 2A)"""

import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
from mne.filter import filter_data
from scipy.stats import kruskal, ranksums
from statsmodels.stats.multitest import multipletests
from termcolor import cprint

from plot_figures.src.default_plt import (
    cm_to_inch,
    dark_blue,
    light_blue,
    medium_blue,
    plt,
)


def compute_voice_volume(eeg: np.ndarray) -> float:
    """Calculate the volume of audio

    Args:
        eeg (np.ndarray): eeg

    Returns:
        float: Volume of audio
    """
    speech = eeg[130, :] - eeg[131, :]
    speech = filter_data(speech, 256, 100, 127)
    rms = librosa.feature.rms(y=speech)
    volume_db = 20 * np.log10(rms)
    volume_db = np.mean(volume_db)
    return volume_db


def volume_stats_save(
    data_root: Path, subject: str, date: str, session_name: str, save_path: Path
) -> np.ndarray:
    """voice volume calculation and storage

    Args:
        data_root (Path): root directory of data
        subject (str): subject name
        date (str): date
        session_name (str): session name
        save_path (Path): path to save

    Returns:
        np.ndarray: voice volume
    """
    before_preproc_paths = list(
        (
            data_root / subject / date / "eeg_before_preproc" / f"{date}_{session_name}"
        ).glob("*.npy")
    )
    volume_session = []
    for path in before_preproc_paths:
        eeg = np.load(path)
        volume_db = compute_voice_volume(eeg)
        volume_session.append(volume_db)
    volume_session = np.array(volume_session)
    np.save(
        str(data_root / subject / date / f"voice_volumes_{session_name}.npy"),
        volume_session,
    )
    return volume_session


def load_voice_volumes(
    sub_dates: List[Tuple[str, str]], data_root: Path
) -> pd.DataFrame:
    """load voice volume

    Args:
        sub_dates (List[Tuple[str, str]]): list of subject and date
        data_root (Path): root directory of data

    Returns:
        pd.DataFrame: voice volume
    """
    voice_volumes = pd.DataFrame(columns=("subject", "task", "volume"))
    for subject, date in sub_dates:
        with open(str(data_root / subject / date / "metadata.json")) as f:
            metadata = json.load(f)
        metadata = {key: metadata[key] for key in metadata.keys() if key != "subject"}
        exp_names = list(metadata.keys())
        tasks = [metadata[key]["task"] for key in exp_names]
        for task, session_name in zip(tasks, exp_names):
            volume_session = np.load(
                str(data_root / subject / date / f"voice_volumes_{session_name}.npy")
            )
            voice_volumes = pd.concat(
                [
                    voice_volumes,
                    pd.DataFrame(
                        {
                            "subject": [subject] * len(volume_session),
                            "task": [task] * len(volume_session),
                            "volume": volume_session,
                        }
                    ),
                ],
                ignore_index=True,
            )
    return voice_volumes


def load_emg_rms(
    sub_dates: List[Tuple[str, str]],
    data_root: Path,
    n_ch_eeg: int = 128,
    n_ch_emg: int = 3,
    highpass: int = 60,
) -> pd.DataFrame:
    """load emg rms

    Args:
        sub_dates (list[tuple[str, str]]): list of subject and date
        data_root (Path): root directory of data
        n_ch_eeg (int, optional): number of channels of EEG. Defaults to 128.
        n_ch_emg (int, optional): number of channels of EMG. Defaults to 3.
        highpass (int, optional): highpass filter cut off frequency (Hz). Defaults to 60.

    Returns:
        pd.DataFrame: rms of EMG
    """
    emg_rmss = pd.DataFrame(
        columns=("subject", "task", "EOG", "EMG upper", "EMG lower")
    )
    for subject, date in sub_dates:
        with open(str(data_root / subject / date / "metadata.json")) as f:
            metadata = json.load(f)
        metadata = {key: metadata[key] for key in metadata.keys() if key != "subject"}
        exp_names = list(metadata.keys())
        tasks = [metadata[key]["task"] for key in exp_names]
        for task, session_name in zip(tasks, exp_names):
            emg_rms = np.load(
                data_root
                / subject
                / date
                / f"emg_rms_highpass{highpass}_{session_name}.npy"
            )
            emg_rmss = pd.concat(
                [
                    emg_rmss,
                    pd.DataFrame(
                        {
                            "subject": [subject] * emg_rms.shape[0],
                            "task": [task] * emg_rms.shape[0],
                            "EOG": emg_rms[:, 0],
                            "EMG upper": emg_rms[:, 1],
                            "EMG lower": emg_rms[:, 2],
                        }
                    ),
                ],
                ignore_index=True,
            )
    return emg_rmss


def plot_volume_stats(
    voice_volumes: pd.DataFrame, save_path: Path, error_config: Dict[str, float]
):
    """plot volume stats

    Args:
        voice_volumes (pd.DataFrame): voice volume
        save_path (Path): save path
        error_config (Dict[str, float]): error config
    """
    fig, ax = plt.subplots(figsize=(4.0 * cm_to_inch, 5.0 * cm_to_inch))
    fig.subplots_adjust()
    tasks = ["overt", "minimally overt", "covert"]
    tasks_labels = ["overt", "min overt", "covert"]
    colors = [dark_blue, medium_blue, light_blue]
    for i, task in enumerate(tasks):
        me = voice_volumes[voice_volumes["task"] == task]["volume"].mean()
        sd = voice_volumes[voice_volumes["task"] == task]["volume"].std()
        plt.bar(
            i,
            height=me,
            yerr=sd,
            label=tasks_labels[i],
            color=colors[i],
            error_kw=error_config,
        )

    plt.xlim(-1, 3)
    plt.xticks(
        [0, 1, 2],
        tasks_labels,
        rotation=40,
        ha="right",
        va="top",
    )
    # yticks = np.arange(0.1, 0.751, 0.05)
    # plt.ylim(0.1, 0.75)
    # plt.yticks(yticks)
    # plt.legend(prop={"size": 5})
    plt.ylabel("Volume (dB)")
    plt.tight_layout()
    plt.savefig(
        str(save_path / "voice_volume.png"),
        dpi=600,
        bbox_inches="tight",
    )


def plot_emg_stats(
    emg_rmss: pd.DataFrame, save_path: Path, error_config: Dict[str, float]
):
    """plot emg stats

    Args:
        emg_rmss (pd.DataFrame): emg rms
        save_path (Path): save path
        error_config (Dict[str, float]): error config
    """
    EMG_types = ["EOG", "EMG upper", "EMG lower"]
    for emg_type in EMG_types:
        fig, ax = plt.subplots(figsize=(4.0 * cm_to_inch, 5.0 * cm_to_inch))
        fig.subplots_adjust()
        tasks = ["overt", "minimally overt", "covert"]
        tasks_labels = ["overt", "min overt", "covert"]
        colors = [dark_blue, medium_blue, light_blue]
        for i, task in enumerate(tasks):
            me = emg_rmss[emg_rmss["task"] == task][emg_type].mean()
            sd = emg_rmss[emg_rmss["task"] == task][emg_type].std()
            plt.bar(
                i,
                height=me,
                yerr=sd,
                label=tasks_labels[i],
                color=colors[i],
                error_kw=error_config,
            )

        plt.xlim(-1, 3)
        plt.xticks(
            [0, 1, 2],
            tasks_labels,
            rotation=40,
            ha="right",
            va="top",
        )
        plt.ylim(0, plt.ylim()[1])
        # yticks = np.arange(0.1, 0.751, 0.05)
        # plt.ylim(0.1, 0.75)
        # plt.yticks(yticks)
        # plt.legend(prop={"size": 5})
        plt.ylabel("RMS")
        # plt.title(emg_type)
        plt.tight_layout()
        plt.savefig(
            str(save_path / f"rms_{emg_type}.png"),
            dpi=600,
            bbox_inches="tight",
        )


def statistical_test_volume(
    voice_volumes: pd.DataFrame, save_path: Path, alpha: int = 0.05
):
    """statistical test for volume

    Args:
        voice_volumes (pd.DataFrame): voice volume
        save_path (Path): save path
        alpha (int, optional): alpha. Defaults to 0.05.
    """
    tasks = ["overt", "minimally overt", "covert"]
    tasks_labels = ["overt", "min overt", "covert"]
    # kruskal
    results_kruskal = kruskal(
        voice_volumes[voice_volumes["task"] == tasks[0]]["volume"],
        voice_volumes[voice_volumes["task"] == tasks[1]]["volume"],
        voice_volumes[voice_volumes["task"] == tasks[2]]["volume"],
    )
    statistic_kruskal = results_kruskal.statistic
    p_kruskal = results_kruskal.pvalue
    cprint(f"statustuc: {statistic_kruskal}, kruskal: {p_kruskal}", "cyan")
    p_values_before_corrected = []
    for task1, task2 in combinations(tasks, 2):
        p_value = ranksums(
            voice_volumes[voice_volumes["task"] == task1]["volume"],
            voice_volumes[voice_volumes["task"] == task2]["volume"],
        ).pvalue
        p_values_before_corrected.append(p_value)
    rejects, p_values_corrected, _, _ = multipletests(
        p_values_before_corrected, method="bonferroni", alpha=alpha
    )
    for i, (task1, task2) in enumerate(combinations(tasks, 2)):
        reject = rejects[i]
        p_value_corrected = p_values_corrected[i]
        cprint(
            f"{task1} vs {task2}: {reject} ({p_value_corrected})",
            "cyan" if reject else "green",
        )
    # convert results_kruskal into pd.DataFrame
    n_samples_dict = {
        f"n_{task}": len(voice_volumes[voice_volumes["task"] == task]) for task in tasks
    }
    results_kruskal = pd.DataFrame(
        {
            "statistic": [statistic_kruskal],
            "p_value": [p_kruskal],
            **n_samples_dict,
        }
    )
    results_kruskal.to_csv(save_path / "kruskal_volume.csv", index=False)
    stats_results = pd.DataFrame(
        {
            "task1": [task1 for task1, _ in combinations(tasks, 2)],
            "task2": [task2 for _, task2 in combinations(tasks, 2)],
            "p_value_before_corrected": p_values_before_corrected,
            "p_value_corrected": p_values_corrected,
            "reject": rejects,
        }
    )
    stats_results.to_csv(save_path / "stats_volume.csv", index=False)


def statistical_test_emg(emg_rmss: pd.DataFrame, save_path: Path, alpha: int = 0.05):
    """statistical test for emg

    Args:
        emg_rmss (pd.DataFrame): emg rms
        save_path (Path): save path
        alpha (int, optional): alpha. Defaults to 0.05.
    """
    EMG_types = ["EOG", "EMG upper", "EMG lower"]
    tasks = ["overt", "minimally overt", "covert"]
    tasks_labels = ["overt", "min overt", "covert"]
    # kruskal
    for emg_type in EMG_types:
        results_kruskal = kruskal(
            emg_rmss[emg_rmss["task"] == tasks[0]][emg_type],
            emg_rmss[emg_rmss["task"] == tasks[1]][emg_type],
            emg_rmss[emg_rmss["task"] == tasks[2]][emg_type],
        )
        statistic_kruskal = results_kruskal.statistic
        p_kruskal = results_kruskal.pvalue
        cprint(f"statustuc: {statistic_kruskal}, kruskal: {p_kruskal}", "cyan")
        p_values_before_corrected = []
        for task1, task2 in combinations(tasks, 2):
            p_value = ranksums(
                emg_rmss[emg_rmss["task"] == task1][emg_type],
                emg_rmss[emg_rmss["task"] == task2][emg_type],
            ).pvalue
            p_values_before_corrected.append(p_value)
        rejects, p_values_corrected, _, _ = multipletests(
            p_values_before_corrected, method="bonferroni", alpha=alpha
        )
        for i, (task1, task2) in enumerate(combinations(tasks, 2)):
            reject = rejects[i]
            p_value_corrected = p_values_corrected[i]
            cprint(
                f"{task1} vs {task2}: {reject} ({p_value_corrected})",
                "cyan" if reject else "green",
            )
        # convert results_kruskal into pd.DataFrame
        n_samples_dict = {
            f"n_{task}": len(emg_rmss[emg_rmss["task"] == task]) for task in tasks
        }
        results_kruskal = pd.DataFrame(
            {
                "statistic": [statistic_kruskal],
                "p_value": [p_kruskal],
                **n_samples_dict,
            }
        )
        results_kruskal.to_csv(save_path / f"kruskal_{emg_type}.csv", index=False)
        stats_results = pd.DataFrame(
            {
                "task1": [task1 for task1, _ in combinations(tasks, 2)],
                "task2": [task2 for _, task2 in combinations(tasks, 2)],
                "p_value_before_corrected": p_values_before_corrected,
                "p_value_corrected": p_values_corrected,
                "reject": rejects,
            }
        )
        stats_results.to_csv(save_path / f"stats_{emg_type}.csv", index=False)


def main():
    sub_dates = [
        ("subject1", "20230511"),
        ("subject1", "20230529"),
        ("subject2", "20230512"),
        ("subject2", "20230516"),
        ("subject3", "20230523"),
        ("subject3", "20230524"),
    ]
    data_root = Path("data/")
    cm_to_point = 28.3465
    error_config = {
        "lw": 0.02 * cm_to_point,
        "capsize": 0.05 * cm_to_point,
        "capthick": 0.02 * cm_to_point,
    }

    voice_volumes = load_voice_volumes(sub_dates, data_root)
    emg_rmss = load_emg_rms(sub_dates, data_root)
    cprint(len(voice_volumes), "cyan")
    cprint(voice_volumes.head(), "cyan")
    cprint(len(emg_rmss), "cyan")
    cprint(emg_rmss.head(), "cyan")

    save_path = Path("figures/fig1")
    save_path.mkdir(parents=True, exist_ok=True)
    plot_volume_stats(voice_volumes, save_path, error_config)
    statistical_test_volume(voice_volumes, save_path, alpha=0.05)

    save_path = Path("figures/fig2/RMS/")
    save_path.mkdir(parents=True, exist_ok=True)
    plot_emg_stats(emg_rmss, save_path, error_config)
    statistical_test_emg(emg_rmss, save_path, alpha=0.05)


if __name__ == "__main__":
    main()
