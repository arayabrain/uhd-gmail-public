import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from termcolor import cprint

data_root = Path("data/")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_cv = 10


def load_igs(DATE, sub_idx, subject, Model):
    igs_table = pd.DataFrame(
        columns=("date", "subject", "model", "task", "cv", "label")
    )
    igs_all = []
    if Model in ["LSTM", "GRU"]:
        is_rnn = True
        torch.backends.cudnn.enabled = False
    else:
        is_rnn = False
    if Model == "CovTanSVM":
        is_svm = True
        cprint("skipped CovTanSVM", "cyan")
        return None
    else:
        is_svm = False
    if "EMG" in Model:
        preproc_dir = "emg_after_preproc"
        if "highpass30" in Model:
            preproc_dir += "_highpass30"
        elif "highpass60" in Model:
            preproc_dir += "_highpass60"
        else:
            pass
    else:  # EEG
        if "wo_adapt_filt" in Model:
            preproc_dir = "eeg_after_preproc_wo_adapt_filt"
        else:
            preproc_dir = "eeg_after_preproc"
    with open(str(data_root / subject / DATE / "metadata.json")) as f:
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

    for cv in range(n_cv):
        if is_svm:
            model_path = (
                data_root
                / subject
                / DATE
                / f"config_color_backup_{DATE[2:]}_{sub_idx}_{Model}"
                / f"CovTanSVM_config_color_N5_cv{cv}.dill"
            )
        else:
            model_path = (
                data_root
                / subject
                / DATE
                / f"config_color_backup_{DATE[2:]}_{sub_idx}_{Model}"
                / f"model_weight_config_color_N5_cv{cv}.pth"
            )
        for task, online_name in zip(tasks, online_names):
            csv_path = str(data_root / subject / DATE / f"word_list_{online_name}.csv")
            words = np.loadtxt(csv_path, delimiter=",", dtype=int)
            save_path = model_path.parent / f"ig_{online_name}"
            igs = torch.load(str(save_path / f"igs_cv{cv}.pt")).cpu().detach().numpy()
            likelihoods = (
                torch.load(
                    str(
                        model_path.parent
                        / f"inference_{online_name}"
                        / f"likelihoods_cv{cv}.pt"
                    )
                )
                .cpu()
                .detach()
                .numpy()
            )
            pred_labels = (
                torch.load(
                    str(
                        model_path.parent
                        / f"inference_{online_name}"
                        / f"pred_labels_cv{cv}.pt"
                    )
                )
                .cpu()
                .detach()
                .numpy()
            )
            n_igs = len(igs)
            igs_table = pd.concat(
                [
                    igs_table,
                    pd.DataFrame(
                        {
                            "date": [DATE] * n_igs,
                            "subject": [subject] * n_igs,
                            "model": [Model] * n_igs,
                            "task": [task] * n_igs,
                            "cv": [cv] * n_igs,
                            "label": words,
                            "likelihood_green": likelihoods[:, 0],
                            "likelihood_magenta": likelihoods[:, 1],
                            "likelihood_orange": likelihoods[:, 2],
                            "likelihood_violet": likelihoods[:, 3],
                            "likelihood_yellow": likelihoods[:, 4],
                            "pred_label": pred_labels,
                            "correct_decode": pred_labels == words,
                        }
                    ),
                ]
            )
            igs_all.append(igs)
    igs_all = np.concatenate(igs_all)
    return igs_all, igs_table
