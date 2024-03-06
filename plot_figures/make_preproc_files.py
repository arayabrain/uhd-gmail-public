import json
from pathlib import Path
from typing import Tuple

import mne
import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from scipy.stats import zscore
from termcolor import cprint

from uhd_eeg.preprocess.adaptive_filter import (
    NLMS,
    bipolar_np,
    get_ch_type_after_resample,
)


class Args:
    n_ch = 139
    n_ch_eeg = 128
    n_ch_noise = 3
    unit_coeff = 1.0e-6
    preamp_gain = 10
    eps = 1.0e-20
    th = 5
    dura_sec = 1.25
    Fs = 256
    Fs_after_resample = 256
    bandpass_low = 2.0
    bandpass_high = 118.0
    num_trial_avg = 5
    nlms_mu = 0.1
    nlms_w = "random"
    emg_highpass = None


args = Args()
data_root = Path("data/")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(0)
n_ch_to_use = args.n_ch_eeg + args.n_ch_noise
info = mne.create_info(
    ch_names=n_ch_to_use,
    sfreq=args.Fs,
    ch_types=[get_ch_type_after_resample(i, n_ch_to_use) for i in range(n_ch_to_use)],
    verbose=False,
)
notch_freqs = np.arange(50, args.Fs_after_resample / 2, 50)
filter_length = min(
    int(round(6.6 * args.Fs_after_resample)),
    round(args.Fs_after_resample * args.dura_sec * args.num_trial_avg - 1),
)
data_ch_idx = np.arange(args.n_ch_eeg)
noise_ch_idx = np.arange(args.n_ch_noise) + args.n_ch_eeg
adapt_filt = NLMS(data_ch_idx, noise_ch_idx, mu=args.nlms_mu, w=args.nlms_w)


def preproc(
    eeg: np.ndarray, wo_adapt_filt: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """preprocess EEG and EMG data

    Args:
        eeg (np.ndarray): EEG and EMG data (n_ch, n_time)
        wo_adapt_filt (bool, optional): apply adaptive filter or not. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: preprocessed EEG and EMG data
    """
    eeg = bipolar_np(eeg)  # (n_ch_eeg + n_ch_noise, n_samp)
    eeg *= args.unit_coeff
    eeg[data_ch_idx] /= args.preamp_gain
    raw = mne.io.RawArray(eeg, info, verbose=False)
    raw.notch_filter(
        notch_freqs,
        filter_length=filter_length,
        fir_design="firwin",
        trans_bandwidth=1.5,
        verbose=False,
    )
    raw.set_eeg_reference("average", verbose=False)
    raw.filter(args.bandpass_low, args.bandpass_high, picks="all", verbose=False)
    emg = raw.get_data()[args.n_ch_eeg :]
    if args.emg_highpass is not None:
        raw.filter(
            args.emg_highpass,
            args.bandpass_high,
            picks="all",
            filter_length=filter_length,
            fir_design="firwin",
            verbose=False,
        )
    if not wo_adapt_filt:
        eeg_norm = adapt_filt(raw.get_data(), normalize="zscore")[: args.n_ch_eeg]
    else:
        eeg_norm = zscore(raw.get_data()[: args.n_ch_eeg], axis=1)
    emg_norm = zscore(emg, axis=1)
    return eeg_norm, emg_norm


def load_save_ephys(
    DATE: str, sub_idx: int, subject: str
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """load and save ephys data

    Args:
        DATE (str): Date of the experiment
        sub_idx (int): sub-index
        subject (str): subject name

    Returns:
        Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]: eegs_table, eegs, eegs_norm, emgs_norm
    """
    dura = int(1.25 * 5 * 256)
    eegs_table = pd.DataFrame(columns=("date", "subject", "task", "label"))
    eegs = []
    preproc_dir = "eeg_margin_before_preproc"
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
    cprint(f"onoff_names: {onoff_names}", "green")
    cprint(f"tasks: {tasks}", "green")
    cprint(f"offline_idx: {offline_idx}", "green")
    tasks_onoff = [tasks[idx] for idx in online_idx]
    tasks_onoff.append(tasks[offline_idx])
    for online_name, task in zip(onoff_names, tasks):
        csv_path = str(data_root / subject / DATE / f"word_list_{online_name}.csv")
        words = np.loadtxt(csv_path, delimiter=",", dtype=int)
        eeg_paths = natsorted(
            (data_root / subject / DATE / preproc_dir / f"{DATE}_{online_name}").glob(
                "*.npy"
            )
        )
        for eeg_path in eeg_paths:
            # cprint(f"loading {eeg_path}", "cyan")
            eeg = np.load(eeg_path)
            eegs.append(eeg)
        eegs_table = pd.concat(
            [
                eegs_table,
                pd.DataFrame(
                    {
                        "date": [DATE] * len(eeg_paths),
                        "subject": [subject] * len(eeg_paths),
                        "task": [task] * len(eeg_paths),
                        "label": words,
                    }
                ),
            ]
        )
        eegs_raw = []
        eegs_norm = []
        emgs_norm = []
        for eeg, eeg_path in zip(eegs, eeg_paths):
            eeg_norm, emg_norm = preproc(eeg, wo_adapt_filt=False)
            eeg_wo_adapt_filt_norm = preproc(eeg, wo_adapt_filt=True)[0]
            eeg_norm = eeg_norm[:, -dura:]
            emg_norm = emg_norm[:, -dura:]
            eeg_raw = eeg[:128, -dura:]
            eeg_wo_adapt_filt = eeg_wo_adapt_filt_norm[:, -dura:]
            eegs_raw.append(eeg_raw)
            eegs_norm.append(eeg_norm)
            emgs_norm.append(emg_norm)
            eeg_raw_path = Path(
                str(eeg_path).replace("eeg_margin_before_preproc", "eeg_raw_wo_avg")
            )
            eeg_wo_adapt_filt_path = Path(
                str(eeg_path).replace(
                    "eeg_margin_before_preproc", "eeg_wo_adapt_filt_wo_avg"
                )
            )
            eeg_after_preproc_path = Path(
                str(eeg_path).replace(
                    "eeg_margin_before_preproc", "eeg_after_preproc_wo_avg"
                )
            )
            if args.emg_highpass is None:
                emg_after_preproc_path = Path(
                    str(eeg_path).replace(
                        "eeg_margin_before_preproc", "emg_after_preproc_wo_avg"
                    )
                )
            else:
                emg_after_preproc_path = Path(
                    str(eeg_path).replace(
                        "eeg_margin_before_preproc",
                        f"emg_highpass{int(args.emg_highpass)}_after_preproc_wo_avg",
                    )
                )
            eeg_raw_path.parent.mkdir(parents=True, exist_ok=True)
            eeg_wo_adapt_filt_path.parent.mkdir(parents=True, exist_ok=True)
            eeg_after_preproc_path.parent.mkdir(parents=True, exist_ok=True)
            emg_after_preproc_path.parent.mkdir(parents=True, exist_ok=True)
            cprint(f"saving {eeg_raw_path}", "cyan")
            cprint(f"saving {eeg_wo_adapt_filt_path}", "cyan")
            cprint(f"saving {eeg_after_preproc_path}", "cyan")
            cprint(f"saving {emg_after_preproc_path}", "cyan")
            np.save(eeg_raw_path, eeg_raw)
            np.save(eeg_wo_adapt_filt_path, eeg_wo_adapt_filt)
            np.save(eeg_after_preproc_path, eeg_norm)
            np.save(emg_after_preproc_path, emg_norm)
        cprint(f"len table {len(eegs_table)}", "cyan")
    eegs = np.array(eegs)
    eegs_norm = np.array(eegs_norm)
    emgs_norm = np.array(emgs_norm)
    return eegs_table, eegs, eegs_norm, emgs_norm


def main():
    data_tuple = [
        ("20230511", 1, "subject1"),
        ("20230529", 1, "subject1"),
        ("20230529", 2, "subject1"),
        ("20230512", 1, "subject2"),
        ("20230512", 2, "subject2"),
        ("20230516", 1, "subject2"),
        ("20230523", 1, "subject3"),
        ("20230523", 2, "subject3"),
        ("20230524", 1, "subject3"),
        ("20230524", 2, "subject3"),
    ]
    for DATE, sub_idx, subject in data_tuple:
        eegs_table, eegs, eegs_norm, emgs_norm = load_save_ephys(DATE, sub_idx, subject)
        cprint(f"len table {len(eegs_table)}", "green")
        cprint(f"eegs.shape {eegs.shape}", "green")
        cprint(f"eegs_norm.shape {eegs_norm.shape}", "green")
        cprint(f"emgs_norm.shape {emgs_norm.shape}", "green")


if __name__ == "__main__":
    main()
