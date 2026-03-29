"""functions for loading ephys data"""

import json
from pathlib import Path
from typing import Tuple

# import librosa
import mne
import numpy as np
import pandas as pd
import torch
from mne.filter import filter_data
from natsort import natsorted
from scipy import signal

# from termcolor import cprint

mne.set_log_level("WARNING")
data_root = Path("data/")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_ephys(
    DATE: str, sub_idx: int, subject: str, ephys_type: str
) -> Tuple[np.ndarray, pd.DataFrame]:
    """load online ephys data

    Args:
        DATE (str): date of experiment
        sub_idx (int): index of experiment
        subject (str): subject name
        ephys_type (str): eeg or emg

    Raises:
        ValueError: _description_

    Returns:
        Tupele[np.ndarray, pd.DataFrame]: eegs_all, eegs_table
    """
    eegs_table = pd.DataFrame(columns=("date", "subject", "task", "label"))
    eegs_all = []
    if "eeg" in ephys_type:
        preproc_dir = "eeg_after_preproc"
        if "wo_adapt_filt" in ephys_type:
            preproc_dir = "eeg_after_preproc_wo_adapt_filt"
        else:
            preproc_dir = "eeg_after_preproc"
    elif "emg" in ephys_type:
        preproc_dir = "emg_after_preproc"
        if "highpass30" in ephys_type:
            preproc_dir += "_highpass30"
        elif "highpass60" in ephys_type:
            preproc_dir += "_highpass60"
        else:
            pass
    elif "speech" in ephys_type:
        preproc_dir = "eeg_before_preproc"
    else:
        raise ValueError("ephys_type must be eeg or emg")
    with open(str(data_root / subject / DATE / "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    metadata = {key: metadata[key] for key in metadata.keys() if key != "subject"}
    exp_names = list(metadata.keys())
    tasks = [metadata[key]["task"] for key in exp_names]
    calib_with = [
        (
            metadata[key]["calibrated with"]
            if "calibrated with" in metadata[key].keys()
            else ""
        )
        for key in exp_names
    ]
    online_idx = [
        i
        for i, value in enumerate(calib_with)
        if value == f"backup_calibrated_{sub_idx}"
    ]
    online_names = [exp_names[idx] for idx in online_idx]
    tasks = [tasks[idx] for idx in online_idx]

    for task, online_name in zip(tasks, online_names):
        csv_path = str(data_root / subject / DATE / f"word_list_{online_name}.csv")
        words = np.loadtxt(csv_path, delimiter=",", dtype=int)
        # cprint(
        #     f"searching:\n{str(data_root / subject / DATE / preproc_dir / f'{DATE}_{online_name}')}",
        #     "green",
        # )
        if "speech" in ephys_type:
            eeg_paths = natsorted(
                (
                    data_root / subject / DATE / preproc_dir / f"{DATE}_{online_name}"
                ).glob("*.npy")
            )
            eegs = np.stack(
                [compute_voice_envelope_avg(np.load(str(path))) for path in eeg_paths]
            )
        else:
            eeg_paths = natsorted(
                (
                    data_root / subject / DATE / preproc_dir / f"{DATE}_{online_name}"
                ).glob("*.pt")
            )
            eegs = np.stack(
                [
                    torch.load(str(path)).to("cpu").detach().numpy().copy()
                    for path in eeg_paths
                ]
            )

        n_eegs = len(eegs)
        eegs_table = pd.concat(
            [
                eegs_table,
                pd.DataFrame(
                    {
                        "date": [DATE] * n_eegs,
                        "subject": [subject] * n_eegs,
                        "task": [task] * n_eegs,
                        "label": words,
                    }
                ),
            ]
        )
        eegs_all.append(np.squeeze(eegs))
    eegs_all = np.concatenate(eegs_all)
    # cprint(f"eegs_all.shape: {eegs_all.shape}", "green")
    return eegs_all, eegs_table


def load_ephys_before_avg(
    DATE: str, sub_idx: int, subject: str, ephys_type: str
) -> Tuple[np.ndarray, pd.DataFrame]:
    """load online and offline ephys data before trial averaging

    Args:
        DATE (str): date of experiment
        sub_idx (int): index of experiment
        subject (str): subject name
        ephys_type (str): eeg_raw

    Returns:
        Tupele[np.ndarray, pd.DataFrame]: eegs_all, eegs_table
    """
    eegs_table = pd.DataFrame(columns=("date", "subject", "task", "label", "exp_name"))
    eegs_all = []
    if "eeg_raw" in ephys_type:
        preproc_dir = "eeg_raw_wo_avg"
    elif "eeg_after_preproc" in ephys_type:
        preproc_dir = "eeg_after_preproc_wo_avg"
    elif "eeg_wo_adapt_filt" in ephys_type:
        preproc_dir = "eeg_wo_adapt_filt_wo_avg"
    elif "emg" in ephys_type:
        preproc_dir = "emg_after_preproc_wo_avg"
        if "highpass30" in ephys_type:
            preproc_dir = "emg_highpass30_after_preproc_wo_avg"
        elif "highpass60" in ephys_type:
            preproc_dir = "emg_highpass60_after_preproc_wo_avg"
    else:
        raise ValueError("ephys_type must be eeg or emg")
    with open(str(data_root / subject / DATE / "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    metadata = {key: metadata[key] for key in metadata.keys() if key != "subject"}
    exp_names = list(metadata.keys())
    tasks = [metadata[key]["task"] for key in exp_names]
    calib_with = [
        (
            metadata[key]["calibrated with"]
            if "calibrated with" in metadata[key].keys()
            else ""
        )
        for key in exp_names
    ]
    online_idx = [
        i
        for i, value in enumerate(calib_with)
        if value == f"backup_calibrated_{sub_idx}"
    ]
    offline_idx = [
        i for i, key in enumerate(exp_names) if key == f"backup_calibrated_{sub_idx}"
    ]
    assert len(offline_idx) == 1
    offline_idx = offline_idx[0]
    onoff_names = [exp_names[idx] for idx in online_idx]
    onoff_names.append(f"backup_calibrated_{sub_idx}")
    tasks_onoff = [tasks[idx] for idx in online_idx]
    tasks_onoff.append(tasks[offline_idx])

    for task, exp_name in zip(tasks_onoff, onoff_names):
        csv_path = str(data_root / subject / DATE / f"word_list_{exp_name}.csv")
        words = np.loadtxt(csv_path, delimiter=",", dtype=int)
        # cprint(
        #     f"searching:\n{str(data_root / subject / DATE / preproc_dir / f'{DATE}_{exp_name}')}",
        #     "green",
        # )
        eeg_paths = natsorted(
            (data_root / subject / DATE / preproc_dir / f"{DATE}_{exp_name}").glob(
                "*.npy"
            )
        )
        eegs = np.stack([np.load(str(path)) for path in eeg_paths])
        n_eegs = len(eegs)
        eegs_table = pd.concat(
            [
                eegs_table,
                pd.DataFrame(
                    {
                        "date": [DATE] * n_eegs,
                        "subject": [subject] * n_eegs,
                        "task": [task] * n_eegs,
                        "label": words,
                        "exp_name": [exp_name] * n_eegs,
                    }
                ),
            ]
        )
        eegs_all.append(np.squeeze(eegs))
    eegs_all = np.concatenate(eegs_all)
    # cprint(f"eegs_all.shape: {eegs_all.shape}", "green")
    return eegs_all, eegs_table


def compute_voice_envelope_avg(eeg: np.ndarray) -> np.ndarray:
    """compute voice envelope

    Args:
        eeg (np.ndarray): eeg data

    Returns:
        np.ndarray: speech envelope
    """
    speech = eeg[130, :] - eeg[131, :]
    speech = filter_data(speech, 256, 100, 127)
    speech_env = np.abs(signal.hilbert(speech))
    # 5分割して平均
    speech_env = np.mean(np.array_split(speech_env, 5), axis=0)
    return speech_env
