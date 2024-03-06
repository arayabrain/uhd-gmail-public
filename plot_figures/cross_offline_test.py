import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from termcolor import cprint

from plot_figures.src.eval_accs import eval_ensemble

DATA_LIST = [
    ["subject1-1", "subject1", "min-overt", "20230511", "1"],
    ["subject1-2", "subject1", "overt", "20230529", "1"],
    ["subject1-3", "subject1", "covert", "20230529", "2"],
    ["subject2-1", "subject2", "min-overt", "20230512", "1"],
    ["subject2-2", "subject2", "overt", "20230512", "2"],
    ["subject2-3", "subject2", "covert", "20230516", "1"],
    ["subject3-1", "subject3", "overt", "20230523", "1"],
    ["subject3-2", "subject3", "min-overt", "20230523", "2"],
    ["subject3-3", "subject3", "covert", "20230524", "1"],
    ["subject3-4", "subject3", "min-overt", "20230524", "2"],
]
DATA_LIST = pd.DataFrame(
    DATA_LIST, columns=["dataname", "subject", "task", "date", "sub_idx"]
)
N_CV = 9
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def copy_files():
    dir_root = Path("outputs/20xx-xx-xx/xx-xx-xx")
    dst_root = Path("data/")
    datanames = DATA_LIST["dataname"].values
    subjects = DATA_LIST["subject"].values
    tasks = DATA_LIST["task"].values
    dates = DATA_LIST["date"].values
    sub_indices = DATA_LIST["sub_idx"].values
    for dataname, subject, task, date, sub_idx in zip(
        datanames, subjects, tasks, dates, sub_indices
    ):
        dir_src = dir_root / f"parallel_sets={dataname}"
        dir_dst = (
            dst_root
            / subject
            / date
            / f"config_color_backup_{date[2:]}_{sub_idx}_EEGNet_within_offline_split_{task}"
        )
        dir_dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(dir_src, dir_dst, dirs_exist_ok=True)


def cross_offline_test():
    tasks_type = ["overt", "min-overt", "covert"]
    methods = [
        "mean",
        "max",
        "zscore_mean",
        "zscore_max",
        "entropy_weighted",
        "inverse_entropy_weighted",
        "majority",
    ]
    dir_root = Path("/mnt/tsukuyomi/uhd-eeg")
    datanames = DATA_LIST["dataname"].values
    subjects = DATA_LIST["subject"].values
    tasks = DATA_LIST["task"].values
    dates = DATA_LIST["date"].values
    sub_indices = DATA_LIST["sub_idx"].values
    for dataname, subject, task, date, sub_idx in zip(
        datanames, subjects, tasks, dates, sub_indices
    ):
        tasks_extracted = [t for t in tasks_type if t != task]
        dir_model = (
            dir_root
            / subject
            / date
            / f"config_color_backup_{date[2:]}_{sub_idx}_EEGNet_within_offline_split_{task}"
        )
        assert len(tasks_extracted) == 2
        for task_target in tasks_extracted:
            paths_label = list(
                (dir_root / subject).glob(
                    f"*/eeg_after_preproc_test/*/word_list_{subject}_{task_target}.csv"
                )
            )
            # cprint(len(paths_label), "cyan")
            # cprint(paths_label, "cyan")
            # get loss_val
            offline_losses = np.load(str(dir_model / "loss_val.npy"))
            offline_accs_epoch = np.load(str(dir_model / "acc_val.npy"))
            ind_best_epoch = [np.argmin(offline_losses[cv, :]) for cv in range(N_CV)]
            best_accs = []
            for cv in range(N_CV):
                best_accs.append(offline_accs_epoch[cv, ind_best_epoch[cv]])
            offline_accs = np.array(best_accs)
            # Get the most accurate model (cv)
            sort_top = np.argsort(-offline_accs)
            accs_top = []
            balanced_accs_top = []
            for k in range(N_CV):
                cvs = sort_top[: k + 1]
                model_paths = [str(dir_model / f"weight_cv{cv}.pth") for cv in cvs]
                accs = []
                balanced_accs = []
                for method in methods:
                    words_all = []
                    preds_all = []
                    for path_label in paths_label:
                        words = np.loadtxt(path_label, delimiter=",").astype(np.int64)
                        dir_pt = path_label.parent
                        preds_label = eval_ensemble(
                            model_paths=model_paths,
                            pt_dir=dir_pt,
                            device=DEVICE,
                            method=method,
                            is_rnn=False,
                            is_svm=False,
                            surrogate=None,
                        )
                        words_all += words.tolist()
                        preds_all += preds_label.tolist()
                    words_all = np.array(words_all)
                    preds_all = np.array(preds_all)
                    acc = (preds_all == words_all).sum() / len(words_all)
                    balanced_acc = balanced_accuracy_score(words_all, preds_all)
                    accs.append(acc)
                    balanced_accs.append(balanced_acc)
                    cprint(
                        f"ensemble method = {method}, balanced_accs = {balanced_accs}",
                        "cyan",
                    )
                accs = np.array(accs)
                balanced_accs = np.array(balanced_accs)
                accs_top.append(accs)
                balanced_accs_top.append(balanced_accs)
            accs_top = np.array(accs_top)
            balanced_accs_top = np.array(balanced_accs_top)
            cprint(f"shape accs_top = {accs_top.shape}", "cyan")
            cprint(f"shape balanced_accs_top = {balanced_accs_top.shape}", "cyan")
            # save as pkl
            accs_dict = {}
            balanced_accs_dict = {}
            for i, method in enumerate(methods):
                accs_dict[method] = accs_top[:, i]
                balanced_accs_dict[method] = balanced_accs_top[:, i]
            # save accs_dict with pickle
            with open(
                dir_model
                / f"cross_within_offline_split_accs_dict_{task}_to_{task_target}.pkl",
                "wb",
            ) as f:
                pickle.dump(accs_dict, f)
            with open(
                dir_model
                / f"cross_within_offline_split_balanced_accs_dict_{task}_to_{task_target}.pkl",
                "wb",
            ) as f:
                pickle.dump(balanced_accs_dict, f)


def within_offline_test():
    tasks_type = ["overt", "min-overt", "covert"]
    methods = [
        "mean",
        "max",
        "zscore_mean",
        "zscore_max",
        "entropy_weighted",
        "inverse_entropy_weighted",
        "majority",
    ]
    dir_root = Path("data/")
    datanames = DATA_LIST["dataname"].values
    subjects = DATA_LIST["subject"].values
    tasks = DATA_LIST["task"].values
    dates = DATA_LIST["date"].values
    sub_indices = DATA_LIST["sub_idx"].values
    for dataname, subject, task, date, sub_idx in zip(
        datanames, subjects, tasks, dates, sub_indices
    ):
        dir_model = (
            dir_root
            / subject
            / date
            / f"config_color_backup_{date[2:]}_{sub_idx}_EEGNet_within_offline_split_{task}"
        )
        path_label = (
            dir_root
            / subject
            / date
            / "eeg_after_preproc_test"
            / f"{date}_backup_calibrated_{sub_idx}"
            / f"word_list_{subject}_{task}.csv"
        )
        # cprint(len(paths_label), "cyan")
        # cprint(paths_label, "cyan")
        # get loss_val
        offline_losses = np.load(str(dir_model / "loss_val.npy"))
        offline_accs_epoch = np.load(str(dir_model / "acc_val.npy"))
        ind_best_epoch = [np.argmin(offline_losses[cv, :]) for cv in range(N_CV)]
        best_accs = []
        for cv in range(N_CV):
            best_accs.append(offline_accs_epoch[cv, ind_best_epoch[cv]])
        offline_accs = np.array(best_accs)
        # Get the most accurate model (cv)
        sort_top = np.argsort(-offline_accs)
        accs_top = []
        balanced_accs_top = []
        for k in range(N_CV):
            cvs = sort_top[: k + 1]
            model_paths = [str(dir_model / f"weight_cv{cv}.pth") for cv in cvs]
            accs = []
            balanced_accs = []
            for method in methods:
                words = np.loadtxt(path_label, delimiter=",").astype(np.int64)
                dir_pt = path_label.parent
                preds_label = eval_ensemble(
                    model_paths=model_paths,
                    pt_dir=dir_pt,
                    device=DEVICE,
                    method=method,
                    is_rnn=False,
                    is_svm=False,
                    surrogate=None,
                )
                acc = (preds_label == words).sum() / len(words)
                balanced_acc = balanced_accuracy_score(words, preds_label)
                accs.append(acc)
                balanced_accs.append(balanced_acc)
                cprint(
                    f"ensemble method = {method}, balanced_accs = {balanced_accs}",
                    "cyan",
                )
            accs = np.array(accs)
            balanced_accs = np.array(balanced_accs)
            accs_top.append(accs)
            balanced_accs_top.append(balanced_accs)
        accs_top = np.array(accs_top)
        balanced_accs_top = np.array(balanced_accs_top)
        cprint(f"shape accs_top = {accs_top.shape}", "cyan")
        cprint(f"shape balanced_accs_top = {balanced_accs_top.shape}", "cyan")
        # Save as pkl
        accs_dict = {}
        balanced_accs_dict = {}
        for i, method in enumerate(methods):
            accs_dict[method] = accs_top[:, i]
            balanced_accs_dict[method] = balanced_accs_top[:, i]
        # save accs_dict with pickle
        with open(
            dir_model / f"within_offline_split_accs_dict_{task}.pkl",
            "wb",
        ) as f:
            pickle.dump(accs_dict, f)
        with open(
            dir_model / f"within_offline_split_balanced_accs_dict_{task}.pkl",
            "wb",
        ) as f:
            pickle.dump(balanced_accs_dict, f)


if __name__ == "__main__":
    # copy_files() # If trianed using hydra, comment on the proposal and copy it to "data/".
    cross_offline_test()
    within_offline_test()
