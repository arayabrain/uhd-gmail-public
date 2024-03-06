"""plot accuracy of different models (Table1, 2)"""

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from termcolor import cprint
from tqdm import tqdm

from plot_figures.src.default_plt import (
    cm_to_inch,
    dark_blue,
    light_blue,
    medium_blue,
    plt,
)

DATA_TUPLE = [
    ("20230511", 1, "subject1", "minimally overt"),
    ("20230529", 1, "subject1", "overt"),
    ("20230529", 2, "subject1", "covert"),
    ("20230512", 1, "subject2", "minimally overt"),
    ("20230512", 2, "subject2", "overt"),
    ("20230516", 1, "subject2", "covert"),
    ("20230523", 1, "subject3", "overt"),
    ("20230523", 2, "subject3", "minimally overt"),
    ("20230524", 1, "subject3", "covert"),
    ("20230524", 2, "subject3", "minimally overt"),
]


def get_on_off_accs(data_tuple: List[Tuple], models: List[str]):
    topk = 4
    n_cv = 10
    data_root = Path("data/")
    columns = (
        [
            "model",
            "subject",
            "date",
            "sub_idx",
            "task",
            "online_acc",
            "online_balanced_acc",
        ]
        + [f"offline_acc_{cv}" for cv in range(n_cv)]
        + [f"offline_balanced_acc_{cv}" for cv in range(n_cv)]
    )
    accs = pd.DataFrame(columns=columns)
    for input_tuple in data_tuple:
        for model in tqdm(models):
            date, sub_idx, subject, task = input_tuple
            path_online_acc = (
                data_root
                / f"{subject}/{date}/config_color_backup_{date[2:]}_{sub_idx}_{model}/accs_dict.pkl"
            )
            path_online_balanced_acc = (
                data_root
                / f"{subject}/{date}/config_color_backup_{date[2:]}_{sub_idx}_{model}/online_balanced_accs_dict.pkl"
            )
            path_offline_acc = (
                data_root
                / f"{subject}/{date}/config_color_backup_{date[2:]}_{sub_idx}_{model}/offline_accs.npy"
            )
            path_offline_balanced_acc = (
                data_root
                / f"{subject}/{date}/config_color_backup_{date[2:]}_{sub_idx}_{model}/offline_balanced_accs.npy"
            )
            if (not path_online_acc.exists()) or (not path_offline_acc.exists()):
                cprint(f"{path_online_acc} or {path_offline_acc} does not exist", "red")
            else:
                offline_acc = np.load(path_offline_acc)
                offline_balanced_acc = np.load(path_offline_balanced_acc)
                with open(path_online_acc, "rb") as f:
                    online_accs_dict = pickle.load(f)
                with open(path_online_balanced_acc, "rb") as f:
                    online_balanced_accs_dict = pickle.load(f)
                accs = pd.concat(
                    [
                        accs,
                        pd.DataFrame(
                            {
                                "model": model,
                                "subject": subject,
                                "date": date,
                                "sub_idx": sub_idx,
                                "task": task,
                                "online_acc": online_accs_dict["zscore_mean"][topk - 1],
                                "online_balanced_acc": online_balanced_accs_dict[
                                    "mean"
                                ][topk - 1],
                                **{
                                    f"offline_acc_{cv}": offline_acc[cv]
                                    for cv in range(n_cv)
                                },
                                **{
                                    f"offline_balanced_acc_{cv}": offline_balanced_acc[
                                        cv
                                    ]
                                    for cv in range(n_cv)
                                },
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )
    return accs


def plot_accs_save(
    accs: pd.DataFrame,
    save_path: Path,
    acc_type: str,
    online: bool,
    offline: bool,
    additional_text: str = "",
    subjects: List[str] = ["shun", "yasu", "rousslan"],
):
    n_cv = 10
    save_path.mkdir(exist_ok=True, parents=True)
    tasks = ["overt", "minimally overt", "covert"]
    models = accs["model"].unique()
    fontsize = 12
    width_onoff = 0.25 if online and offline else 0
    onoff_text = ""
    onoff_text += "online_" if online else ""
    onoff_text += "offline_" if offline else ""
    x = 0
    tick = False
    legend = None
    legend_first = [True, True, True]
    xticks = []
    xticks_labels = []
    fig, ax = plt.subplots(figsize=(18 * cm_to_inch, 9 * cm_to_inch))
    for i, task in enumerate(tasks):
        task_data = accs[(accs["task"] == task) & (accs["subject"].isin(subjects))]
        online_means = task_data.groupby("model")[f"online_{acc_type}"].mean()
        offline_means = task_data.groupby("model")[
            [f"offline_{acc_type}_{cv}" for cv in range(n_cv)]
        ].mean()
        offline_means = offline_means.mean(axis=1)
        model_names = offline_means.index
        if task == "overt":
            color = dark_blue
        elif task == "minimally overt":
            color = medium_blue
        elif task == "covert":
            color = light_blue
        for j, model_name in enumerate(model_names):
            if "EEGNet" in model_name:
                x_model = 0
                label = "EEGNet"
            elif "LSTM" in model_name:
                x_model = 1
                label = "LSTM"
            elif "CovTanSVM" in model_name:
                x_model = 2
                label = "SVM"
            if "EMG" in model_name:
                marker = "^"
                x_marker = 0.2
                if legend_first[0]:
                    legend = "EMG"
                    legend_first[0] = False
            elif "wo_adapt_filt" in model_name:
                marker = ","
                x_marker = -0.2
                if legend_first[1]:
                    legend = "min-preprocessed"
                    legend_first[1] = False
            else:
                marker = "o"
                x_marker = 0
                tick = True
                if legend_first[2]:
                    legend = "denoised"
                    legend_first[2] = False
            x = x_model + i * 3 * 1.5 + x_marker
            if tick:
                xticks.append(x)
                xticks_labels.append(label)
                tick = False
            # scatter offline means without filling
            if offline:
                ax.scatter(
                    x - width_onoff,
                    offline_means[model_name],
                    marker=marker,
                    color="none",
                    edgecolors=color,
                    label=f"off {legend}" if legend is not None else None,
                )
            if online:
                ax.scatter(
                    x + width_onoff,
                    online_means[model_name],
                    marker=marker,
                    color=color,
                    label=f"on {legend}" if legend is not None else None,
                )
            legend = None
    # ax.set_ylabel("Accuracy")
    # ax.set_title("Accuracy by task and model")
    # hlines at y=0.2
    # get xrange
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax)
    ax.hlines(0.2, xmin, xmax, linestyles="dashed", color="black", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.5))
    ax.set_yticklabels(np.arange(0, 1.1, 0.5), fontsize=fontsize)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels, rotation="vertical", fontsize=fontsize)
    # ax.legend()

    fig.tight_layout()

    # plt.show()
    subjects_text = "".join(subjects)
    plt.savefig(
        save_path / f"{onoff_text}{acc_type}{additional_text}_{subjects_text}.png"
    )
    plt.clf()
    plt.close()


def plot_accs_decimation_save(
    accs: pd.DataFrame,
    save_path: Path,
    acc_type: str,
    online: bool,
    offline: bool,
    show_ratio: bool,
    yticks_auto: bool = False,
    ylim: List[float] = [0, 1],
    yticks: List[float] = [0, 0.5, 1.0],
    baseline: float = 0.2,
    additional_text: str = "",
    subjects: List[str] = ["shun", "yasu", "rousslan"],
):
    n_cv = 10
    save_path.mkdir(exist_ok=True, parents=True)
    tasks = ["overt", "minimally overt", "covert"]
    models = accs["model"].unique()
    fontsize = 12
    marker = "o"
    width_onoff = 0.25 if online and offline else 0
    onoff_text = ""
    onoff_text += "online_" if online else ""
    onoff_text += "offline_" if offline else ""
    x = 0
    legend = True
    xticks = []
    xticks_labels = []
    fig, ax = plt.subplots(figsize=(18 * cm_to_inch, 9 * cm_to_inch))
    for i, task in enumerate(tasks):
        task_data = accs[(accs["task"] == task) & (accs["subject"].isin(subjects))]
        online_means = task_data.groupby("model")[f"online_{acc_type}"].mean()
        offline_means = task_data.groupby("model")[
            [f"offline_{acc_type}_{cv}" for cv in range(n_cv)]
        ].mean()
        offline_means = offline_means.mean(axis=1)
        model_names = offline_means.index
        online_full = online_means["EEGNet"] if show_ratio else 1
        offline_full = offline_means["EEGNet"] if show_ratio else 1
        if task == "overt":
            color = dark_blue
        elif task == "minimally overt":
            color = medium_blue
        elif task == "covert":
            color = light_blue
        for j, model_name in enumerate(model_names):
            if "4ch" in model_name:
                x_model = 0
                label = "4"
            elif "8ch" in model_name:
                x_model = 1
                label = "8"
            elif "16ch" in model_name:
                x_model = 2
                label = "16"
            elif "32ch" in model_name:
                x_model = 3
                label = "32"
            else:
                x_model = 4
                label = "128"
            x = x_model + i * 4 * 1.5
            xticks.append(x)
            xticks_labels.append(label)
            # scatter offline means without filling
            if offline:
                ax.scatter(
                    x - width_onoff,
                    offline_means[model_name] / offline_full,
                    marker=marker,
                    color="none",
                    edgecolors=color,
                    label=f"off" if legend else None,
                )
            if online:
                ax.scatter(
                    x + width_onoff,
                    online_means[model_name] / online_full,
                    marker=marker,
                    color=color,
                    label=f"on" if legend else None,
                )
            legend = False
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax)
    ax.hlines(baseline, xmin, xmax, linestyles="dashed", color="black", linewidth=1)
    if not yticks_auto:
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=fontsize)
    else:
        # set yticks fontsize
        yticks = ax.get_yticks()
        # get yticks label (:2f)
        yticks_labels = [f"{ytick:.2f}" for ytick in yticks]
        ax.set_yticklabels(yticks_labels, fontsize=fontsize)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels, rotation="horizontal", fontsize=fontsize)
    # ax.legend()

    fig.tight_layout()

    # plt.show()
    ratio_text = "ratio_" if show_ratio else ""
    subjects_text = "".join(subjects)
    plt.savefig(
        save_path
        / f"channel_decimation_{ratio_text}{onoff_text}{acc_type}{additional_text}_{subjects_text}.png"
    )
    plt.clf()
    plt.close()


def save_accs_table(
    accs: pd.DataFrame,
    save_path: Path,
    acc_type: str,
    offline: bool,
    models: List[str],
    additional_text: str = "",
):
    n_cv = 10
    save_path.mkdir(exist_ok=True, parents=True)
    conditions = []
    for task in ["overt", "minimally overt", "covert"]:
        for model in models:
            conditions.append(f"{task}_{model}")
    subjects_all = [["shun"], ["yasu"], ["rousslan"], ["shun", "yasu", "rousslan"]]
    models = accs["model"].unique()
    onoff_text = "_online" if not offline else "_offline"
    table_data = np.zeros((len(subjects_all), len(conditions)))
    for i, subjects in enumerate(subjects_all):
        for j, condition in enumerate(conditions):
            task = condition.split("_")[0]
            model_name = condition.split(f"{task}_")[-1]
            task_data = accs[(accs["task"] == task) & (accs["subject"].isin(subjects))]
            online_means = task_data.groupby("model")[f"online_{acc_type}"].mean()
            offline_means = task_data.groupby("model")[
                [f"offline_{acc_type}_{cv}" for cv in range(n_cv)]
            ].mean()
            offline_means = offline_means.mean(axis=1)
            if offline:
                table_data[i, j] = offline_means[model_name]
            else:
                table_data[i, j] = online_means[model_name]
    table = pd.DataFrame(
        table_data,
        columns=conditions,
        index=["shun", "yasu", "rousslan", "all"],
    )
    table.to_csv(save_path / f"{acc_type}{onoff_text}{additional_text}.csv")
    return table
