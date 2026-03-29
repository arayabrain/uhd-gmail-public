"""plot mutual information (Fig. 2B, 4C)"""

import time
from itertools import combinations, product
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.image import imread
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.multitest import multipletests
from termcolor import cprint
from tqdm import tqdm

from plot_figures.src.default_plt import (
    cm_to_inch,
    dark_blue,
    light_blue,
    medium_blue,
    plt,
)
from plot_figures.src.load_ephys import load_ephys_before_avg
from plot_figures.src.surrogates import aaft, ft, iaaft


def ephys_load(
    data_tuple: List[Tuple],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], pd.DataFrame]:
    """load ephys data

    Args:
        data_tuple (List[Tuple]): information of ephys data

    Returns:
        Tuple: ephys data, ephys label, eegs table
    """
    eegs_all = []
    eegs_wo_adapt_filt_all = []
    eegs_after_preproc_all = []
    emgs_all = []
    emgs_highpass30_all = []
    emgs_highpass60_all = []
    eegs_norm_all = []
    eegs_wo_adapt_filt_norm_all = []
    eegs_after_preproc_norm_all = []
    emgs_norm_all = []
    emgs_highpass30_norm_all = []
    emgs_highpass60_norm_all = []
    eegs_table_all = pd.DataFrame(columns=("date", "subject", "task", "label"))
    for input_tuple in tqdm(data_tuple):
        date, sub_idx, subject = input_tuple
        eegs, eegs_table = load_ephys_before_avg(date, sub_idx, subject, "eeg_raw")
        eegs_wo_adapt_filt, _ = load_ephys_before_avg(
            date, sub_idx, subject, "eeg_wo_adapt_filt"
        )
        eegs_after_preproc, _ = load_ephys_before_avg(
            date, sub_idx, subject, "eeg_after_preproc"
        )
        emgs, _ = load_ephys_before_avg(date, sub_idx, subject, "emg")
        emgs_highpass30, _ = load_ephys_before_avg(
            date, sub_idx, subject, "emg_highpass30"
        )
        emgs_highpass60, _ = load_ephys_before_avg(
            date, sub_idx, subject, "emg_highpass60"
        )
        eegs_norm = (eegs - np.mean(eegs, axis=2, keepdims=True)) / np.std(
            eegs, axis=2, keepdims=True
        )
        eegs_wo_adapt_filt_norm = (
            eegs_wo_adapt_filt - np.mean(eegs_wo_adapt_filt, axis=2, keepdims=True)
        ) / np.std(eegs_wo_adapt_filt, axis=2, keepdims=True)
        eegs_after_preproc_norm = (
            eegs_after_preproc - np.mean(eegs_after_preproc, axis=2, keepdims=True)
        ) / np.std(eegs_after_preproc, axis=2, keepdims=True)
        emgs_norm = (emgs - np.mean(emgs, axis=2, keepdims=True)) / np.std(
            emgs, axis=2, keepdims=True
        )
        emgs_highpass30_norm = (
            emgs_highpass30 - np.mean(emgs_highpass30, axis=2, keepdims=True)
        ) / np.std(emgs_highpass30, axis=2, keepdims=True)
        emgs_highpass60_norm = (
            emgs_highpass60 - np.mean(emgs_highpass60, axis=2, keepdims=True)
        ) / np.std(emgs_highpass60, axis=2, keepdims=True)
        eegs_all.extend(list(eegs))
        eegs_wo_adapt_filt_all.extend(list(eegs_wo_adapt_filt))
        eegs_after_preproc_all.extend(list(eegs_after_preproc))
        emgs_all.extend(list(emgs))
        emgs_highpass30_all.extend(list(emgs_highpass30))
        emgs_highpass60_all.extend(list(emgs_highpass60))
        eegs_norm_all.extend(list(eegs_norm))
        eegs_wo_adapt_filt_norm_all.extend(list(eegs_wo_adapt_filt_norm))
        eegs_after_preproc_norm_all.extend(list(eegs_after_preproc_norm))
        emgs_norm_all.extend(list(emgs_norm))
        emgs_highpass30_norm_all.extend(list(emgs_highpass30_norm))
        emgs_highpass60_norm_all.extend(list(emgs_highpass60_norm))
        eegs_table_all = pd.concat([eegs_table_all, eegs_table])
    cprint(f"len eegs_all: {len(eegs_all)}", "cyan")
    cprint(
        f"len eegs_after_preproc_norm_all: {len(eegs_after_preproc_norm_all)}", "cyan"
    )
    cprint(f"len emgs_all: {len(emgs_all)}", "cyan")
    eegs_all = np.array(eegs_all)
    eegs_wo_adapt_filt_all = np.array(eegs_wo_adapt_filt_all)
    eegs_after_preproc_all = np.array(eegs_after_preproc_all)
    emgs_all = np.array(emgs_all)
    emgs_highpass30_all = np.array(emgs_highpass30_all)
    emgs_highpass60_all = np.array(emgs_highpass60_all)
    eegs_norm_all = np.array(eegs_norm_all)
    eegs_wo_adapt_filt_norm_all = np.array(eegs_wo_adapt_filt_norm_all)
    eegs_after_preproc_norm_all = np.array(eegs_after_preproc_norm_all)
    emgs_norm_all = np.array(emgs_norm_all)
    emgs_highpass30_norm_all = np.array(emgs_highpass30_norm_all)
    emgs_highpass60_norm_all = np.array(emgs_highpass60_norm_all)
    eegs = np.stack(
        [
            eegs_all,
            eegs_wo_adapt_filt_all,
            eegs_after_preproc_all,
            eegs_norm_all,
            eegs_wo_adapt_filt_norm_all,
            eegs_after_preproc_norm_all,
        ]
    )
    emgs = np.stack(
        [
            emgs_all,
            emgs_highpass30_all,
            emgs_highpass60_all,
            emgs_norm_all,
            emgs_highpass30_norm_all,
            emgs_highpass60_norm_all,
        ]
    )
    eegs_label = [
        "raw",
        "wo_adapt_filt",
        "after_preproc",
        "norm",
        "wo_adapt_filt_norm",
        "after_preproc_norm",
    ]
    emgs_label = [
        "raw",
        "highpass30",
        "highpass60",
        "norm",
        "highpass30_norm",
        "highpass60_norm",
    ]
    return (
        eegs,
        emgs,
        eegs_label,
        emgs_label,
        eegs_table_all,
    )


def calc_mi(
    i: int,
    eegs: np.ndarray,
    emgs: np.ndarray,
    eegs_label: List[str],
    emgs_label: List[str],
    eeg_type: str,
    emg_type: str,
    emg_index: int,
    k: int,
    seed: int,
):
    """calc mutual information"""
    # cprint(f"i: {i}", "cyan")
    x = eegs[eegs_label.index(eeg_type), i, :, :].T
    y = emgs[emgs_label.index(emg_type), i, emg_index, :]
    return (
        mutual_info_regression(
            x,
            y,
            discrete_features=False,
            n_neighbors=k,
            random_state=seed,
        ),
        i,
    )


def get_mis(
    eegs: np.ndarray,
    emgs: np.ndarray,
    eegs_label: List[str],
    emgs_label: List[str],
    eeg_type: str,
    emg_type: str,
    emg_index: int,
    k: int,
    seed: int,
) -> np.ndarray:
    n_data = eegs.shape[1]
    n_ch = eegs.shape[2]
    mis = np.zeros((n_data, n_ch))
    global calc_mi_

    def calc_mi_(i):
        return calc_mi(
            i,
            eegs=eegs,
            emgs=emgs,
            eegs_label=eegs_label,
            emgs_label=emgs_label,
            eeg_type=eeg_type,
            emg_type=emg_type,
            emg_index=emg_index,
            k=k,
            seed=seed,
        )

    with Pool() as p:
        for mi, i in tqdm(p.imap_unordered(calc_mi_, range(n_data))):
            mis[i] = mi
    return mis


def show_montage(
    color: np.ndarray, save_path: Path, show_colorbar: bool = False, vmax: float = None
) -> None:
    """show montage

    Args:
        color (np.ndarray): color for each electrode
        save_path (Path): save path
        show_colorbar (bool, optional): whether to show colorbar. Defaults to False.
        vmax (float, optional): max value for colorbar. Defaults to None.
    """
    img = imread("plot_figures/montage_colorless.png")
    coordinates = np.load("plot_figures/coordinates_colorless.npy")
    plt.figure(figsize=(7.5 * cm_to_inch, 7.5 * cm_to_inch))
    plt.imshow(img)
    plt.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=18.0,
        c=color,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        linewidths=0,
    )
    if show_colorbar:
        plt.colorbar()
    plt.axis("off")
    plt.savefig(save_path)
    plt.clf()
    plt.close()


def save_montage_mi(
    mis: np.ndarray, save_path: Path, show_colorbar: bool = False, vmax: float = None
):
    """save montage mi

    Args:
        mis (np.ndarray): mis (n_trials, n_ch)
        save_path (Path): save path
        show_colorbar (bool, optional): show colorbar. Defaults to False.
        vmax (float, optional): max value for colorbar. Defaults to None.
    """
    mean_mis = np.mean(mis, axis=0)
    show_montage(mean_mis, save_path, show_colorbar, vmax)


def emg_contamination_montage(
    mis: np.ndarray,
    mis_table: pd.DataFrame,
    show_colorbar: bool = False,
    vmax: float = None,
):
    """emg contamination montage

    Args:
        mis (np.ndarray): mis (n_condition, n_trials, n_ch)
        mis_table (pd.DataFrame): mis table
        show_colorbar (bool, optional): show colorbar. Defaults to False.
        vmax (float, optional): max value for colorbar. Defaults to None.
    """
    save_dir = Path("figures/fig4")
    eeg_types = ["raw", "wo_adapt_filt", "after_preproc", "norm", "wo_adapt_filt_norm"]
    emg_names = ["EOG", "EMG_upper", "EMG_lower"]
    emg_type = "raw"
    surrogate_type = "none"
    for eeg_type in eeg_types:
        for emg_name in emg_names:
            target_index = np.where(
                (mis_table["eeg_type"] == eeg_type)
                & (mis_table["emg_type"] == emg_type)
                & (mis_table["emg_name"] == emg_name)
                & (mis_table["surrogate_type"] == surrogate_type)
            )[0]
            assert len(target_index) == 1
            target_index = target_index[0]
            show_colorbar_text = "_with_colorbar" if show_colorbar else ""
            (save_dir / "emg_contamination_montage").mkdir(exist_ok=True, parents=True)
            save_path = (
                save_dir
                / "emg_contamination_montage"
                / f"{eeg_type}_{emg_name}_{surrogate_type}{show_colorbar_text}.png"
            )
            save_path.parent.mkdir(exist_ok=True, parents=True)
            save_montage_mi(
                mis=mis[target_index],
                save_path=save_path,
                show_colorbar=show_colorbar,
                vmax=vmax,
            )


def plot_mi(
    stats_mi: pd.DataFrame, eeg_types: List[str], emg_types: List[str], save_path: Path
) -> None:
    means = stats_mi["mean_mi"].values
    sems = stats_mi["sem_mi"].values
    colors = [dark_blue, medium_blue, light_blue]
    n_per_group = len(eeg_types)
    width = (1 - 0.2) / n_per_group
    ratio = 0.4

    for emg_type in emg_types:
        for i, task in enumerate(["overt", "minimally overt", "covert"]):
            title_txt = ""
            for j, eeg_type in enumerate(eeg_types):
                target_indices = np.where(
                    (stats_mi["task"] == task)
                    & (stats_mi["eeg_type"] == eeg_type)
                    & (stats_mi["emg_type"] == emg_type)
                )[0]
                plt.bar(
                    i + width * (1 - ratio) * j,
                    means[target_indices],
                    width=width * ratio,
                    yerr=sems[target_indices],
                    color=colors[i],
                )
                title_txt += f"{eeg_type} "
            title_txt += f"vs {emg_type}"
            plt.title(title_txt)
        plt.savefig(save_path)
        plt.clf()
        plt.close()


def get_stats_mi(
    mis: np.ndarray,
    eegs_table_all: pd.DataFrame,
    eeg_types: List[str],
    emg_types: List[str],
) -> pd.DataFrame:
    """get stats mi"""
    stats_mi = pd.DataFrame(
        columns=("task", "eeg_type", "emg_type", "mean_mi", "sem_mi")
    )
    for eeg_type in eeg_types:
        for emg_type in emg_types:
            for task in ["overt", "minimally overt", "covert"]:
                target_indices = np.where(
                    (eegs_table_all["task"] == task)
                    & (eegs_table_all["eeg_type"] == eeg_type)
                    & (eegs_table_all["emg_type"] == emg_type)
                )[0]
                mean_mi = np.mean(mis[target_indices])
                sem_mi = np.std(mis[target_indices]) / np.sqrt(len(target_indices))
                stats_mi = stats_mi.append(
                    {
                        "task": task,
                        "eeg_type": eeg_type,
                        "emg_type": emg_type,
                        "mean_mi": mean_mi,
                        "sem_mi": sem_mi,
                    },
                    ignore_index=True,
                )
    return stats_mi


def get_surrogates(
    eegs: np.ndarray,
    surrogate_type: str,
    eeg_type: str,
    eegs_label: List[str],
    within_session: bool = True,
    eegs_table_all: pd.DataFrame = None,
) -> np.ndarray:
    """get surrogates

    Args:
        eegs (np.ndarray): eegs (n_eeg_type, n_trial, n_electrode, n_time)
        surrogate_type (str): specify surrogate type
        eeg_type (str): eeg type
        eegs_label (List[str]): eegs label
        eegs_table_all (pd.DataFrame): eeg table

    Raises:
        ValueError: _description_

    Returns:
        np.ndarray: _description_
    """
    x = eegs[eegs_label.index(eeg_type), :, :, :]
    n_data, n_ch = x.shape[:2]
    combs = list(product(range(n_data), range(n_ch)))
    surrogates = np.zeros_like(x)
    global calc_surrogate
    if surrogate_type == "ft":

        def calc_surrogate(i: int) -> Tuple[np.ndarray, int]:
            """calc surrogate with ft

            Args:
                i (int): index of data

            Returns:
                Tuple[np.ndarray, int]: surrogate, index of data
            """
            return ft(x[i]), i

        with Pool() as p:
            for surrogate, i in tqdm(p.imap_unordered(calc_surrogate, range(len(x)))):
                surrogates[i] = surrogate
    elif surrogate_type == "aaft":

        def calc_surrogate(k: int) -> Tuple[np.ndarray, int, int]:
            """calc surrogate with aaft

            Args:
                k (int): index of combination

            Returns:
                Tuple[np.ndarray, int, int]: surrogate, index of data, index of channel
            """
            i, j = combs[k]
            return aaft(x[i, j]), i, j

        with Pool() as p:
            for surrogate, i, j in tqdm(
                p.imap_unordered(calc_surrogate, range(len(combs)))
            ):
                surrogates[i, j] = surrogate

    elif surrogate_type == "iaaft":

        def calc_surrogate(k: int) -> Tuple[np.ndarray, int, int]:
            """calc surrogate with iaaft

            Args:
                k (int): index of combination

            Returns:
                Tuple[np.ndarray, int, int]: surrogate, index of data, index of channel
            """
            i, j = combs[k]
            return iaaft(x[i, j], ns=1, verbose=False), i, j

        with Pool() as p:
            for surrogate, i, j in tqdm(
                p.imap_unordered(calc_surrogate, range(len(combs)))
            ):
                surrogates[i, j] = surrogate
    elif surrogate_type == "rs":
        surrogates = np.random.permutation(x.transpose(2, 0, 1)).transpose(1, 2, 0)
    elif surrogate_type == "trial_shuffle":
        if within_session:
            eegs_table_all = eegs_table_all.assign(
                session="none",
            )
            for i in range(len(eegs_table_all)):
                # cprint(f"i: {i}", "cyan")
                # cprint(eegs_table_all.loc[i, "date"], "cyan")
                # cprint(eegs_table_all.loc[i, "subject"], "cyan")
                # cprint(eegs_table_all.loc[i, "task"], "cyan")
                eegs_table_all.loc[i, "session"] = (
                    f"{eegs_table_all.loc[i, 'date']}"
                    + f"_{eegs_table_all.loc[i, 'subject']}"
                    + f"_{eegs_table_all.loc[i, 'task']}"
                )
            # eegs_table_all.to_csv("eegs_table_all.csv")
            for session in eegs_table_all["session"].unique():
                target_indices = np.where(eegs_table_all["session"] == session)[0]
                surrogates[target_indices] = np.random.permutation(x[target_indices])
        else:
            surrogates = np.random.permutation(x)
    else:
        raise ValueError(f"surrogate_type: {surrogate_type}")
    return surrogates


def collect_surrogates(
    eegs: np.ndarray,
    surrogate_types: List[str],
    eeg_types_for_surrogate: List[str],
    eegs_label: List[str],
    within_session: bool = True,
    eegs_table_all: pd.DataFrame = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """collect surrogates

    Args:
        eegs (np.ndarray): eegs (n_eeg_type, n_trial, n_electrode, n_time)
        surrogate_types (List[str]): surrogate types
        eeg_types_for_surrogate (List[str]): eeg types for surrogate
        eegs_label (List[str]): eeg label
        within_session (bool, optional): whether to shuffle within session.
        eegs_table_all (pd.DataFrame): eeg table

    Returns:
        Tuple[np.ndarray, pd.DataFrame]: surrogates, surrogate table
    """
    surrogate_table = pd.DataFrame(columns=("surrogate_type", "eeg_type"))
    surrogates = []
    for eeg_type in eeg_types_for_surrogate:
        for surrogate_type in surrogate_types:
            surrogates.append(
                get_surrogates(
                    eegs,
                    surrogate_type,
                    eeg_type,
                    eegs_label,
                    within_session,
                    eegs_table_all,
                )
            )
            surrogate_table = pd.concat(
                [
                    surrogate_table,
                    pd.DataFrame(
                        {
                            "surrogate_type": surrogate_type,
                            "eeg_type": eeg_type,
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )
    surrogates = np.array(surrogates)
    return surrogates, surrogate_table


def collect_mis(
    eegs: np.ndarray,
    emgs: np.ndarray,
    surrogates: np.ndarray,
    eeg_types_for_mi: List[str],
    eeg_types_for_surrogate: List[str],
    emg_types: List[str],
    emg_names: List[str],
    eegs_label: List[str],
    emgs_label: List[str],
    surrogate_types: List[str],
    surrogate_table: pd.DataFrame,
    k: int,
    seed: int,
):
    mis = []
    mis_table = pd.DataFrame(
        columns=("eeg_type", "emg_type", "emg_index", "emg_name", "surrogate_type")
    )
    for eeg_type in eeg_types_for_mi:
        for emg_type in emg_types:
            for emg_index, emg_name in enumerate(emg_names):
                mis.append(
                    get_mis(
                        eegs,
                        emgs,
                        eegs_label,
                        emgs_label,
                        eeg_type,
                        emg_type,
                        emg_index,
                        k,
                        seed,
                    )
                )
                mis_table = pd.concat(
                    [
                        mis_table,
                        pd.DataFrame(
                            {
                                "eeg_type": eeg_type,
                                "emg_type": emg_type,
                                "emg_index": emg_index,
                                "emg_name": emg_name,
                                "surrogate_type": "none",
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )
    surrogate_labels = [
        "_".join(pair)
        for pair in zip(
            surrogate_table["surrogate_type"].values, surrogate_table["eeg_type"].values
        )
    ]
    for surrogate_type in surrogate_types:
        for eeg_type in eeg_types_for_surrogate:
            for emg_type in emg_types:
                for emg_index, emg_name in enumerate(emg_names):
                    mis.append(
                        get_mis(
                            surrogates,
                            emgs,
                            surrogate_labels,
                            emgs_label,
                            f"{surrogate_type}_{eeg_type}",
                            emg_type,
                            emg_index,
                            k,
                            seed,
                        )
                    )
                    mis_table = pd.concat(
                        [
                            mis_table,
                            pd.DataFrame(
                                {
                                    "eeg_type": eeg_type,
                                    "emg_type": emg_type,
                                    "emg_index": emg_index,
                                    "emg_name": emg_name,
                                    "surrogate_type": surrogate_type,
                                },
                                index=[0],
                            ),
                        ],
                        ignore_index=True,
                    )
    mis = np.array(mis)
    return mis, mis_table


def plot_mis_for_target_conditions(
    mis: np.ndarray,
    mis_table: pd.DataFrame,
    eegs_table_all: pd.DataFrame,
    target_conditions: List[List[str]],
    save_dir: Path,
    tasks: Tuple[str] = ("overt", "minimally overt", "covert"),
    fontsize: int = 12,
    show_base: bool = False,
    show_shuffle: bool = False,
):
    """plot mis

    Args:
        mis (np.ndarray): all mis
        mis_table (pd.DataFrame): mis table
        target_conditions (List[List[str]]): target conditions
            e.g. [["raw", "after_preproc", "rs", "raw", "EOG"],
                  ["raw", "after_preproc", "ft", "raw", "EOG"]]
            the List contains [eeg_types, emg_type, emg_name]
        save_dir (Path): save directory
        tasks (Tuple[str], optional): tasks. Defaults to
            ("overt", "minimally overt", "covert").
        fontsize (int, optional): fontsize. Defaults to 12.
        show_base (bool, optional): whether to show base. Defaults to False.
        show_shuffle (bool, optional): whether to show shuffle. Defaults to False.
    """
    n_per_group = len(tasks)
    width = (1 - 0.2) / n_per_group if show_shuffle else (1 - 0.2) / (n_per_group - 1)
    ratio = 0.4
    colors = [dark_blue, medium_blue, light_blue]
    xticks = []
    X = []
    task_labels = []
    means = []
    sems = []
    title_txt = ""
    xticks_label = []
    xmins_base = []
    xmaxs_base = []
    ys_base = []
    for i_eeg, target_condition in enumerate(target_conditions):
        eeg_type, emg_type, emg_name, surrogate_type = target_condition
        title_txt += f"{eeg_type}_{emg_type}_{emg_name}_{surrogate_type}"
        target_indices = np.where(
            (mis_table["eeg_type"] == eeg_type)
            & (mis_table["emg_type"] == emg_type)
            & (mis_table["emg_name"] == emg_name)
            & (mis_table["surrogate_type"] == surrogate_type)
        )[0]
        # cprint(target_indices, "cyan")
        assert len(target_indices) == 1
        target_index = target_indices[0]
        mis_tmp = np.mean(mis[target_index], axis=1)  # axis=1: average over channels
        for i_task, task in enumerate(tasks):
            target_indices_trial = np.where((eegs_table_all["task"] == task))[0]
            X.append(mis_tmp[target_indices_trial])
            mean = np.mean(mis_tmp[target_indices_trial], axis=0)
            sem = np.std(mis_tmp[target_indices_trial], axis=0) / np.sqrt(
                len(target_indices_trial)
            )
            if i_eeg == 0:
                eeg_text = "raw"
            elif i_eeg == 1:
                eeg_text = "min-preprocessed"
            elif i_eeg == 2:
                eeg_text = "denoised"
            elif i_eeg == 3:
                eeg_text = "shuffle"
                xmins_base.append(i_task - width * (1 - ratio))
                (
                    xmaxs_base.append(i_task + width * (1 - ratio) * (i_eeg + 1))
                    if show_shuffle
                    else xmaxs_base.append(i_task + width * (1 - ratio) * i_eeg)
                )
                ys_base.append(mean)
            if (not show_shuffle) and (i_eeg == 3):
                continue
            plt.bar(
                i_task + width * (1 - ratio) * i_eeg,
                mean,
                width=width * ratio,
                # yerr=sem,
                color=colors[i_task],
            )
            # error bar
            plt.errorbar(
                i_task + width * (1 - ratio) * i_eeg,
                mean,
                yerr=sem,
                fmt="none",
                ecolor="black",
                capsize=0,
                elinewidth=1,
            )
            xticks.append(i_task + width * (1 - ratio) * i_eeg)
            means.append(mean)
            sems.append(sem)
            task_labels.append(task)
            xticks_label.append(eeg_text)
    if show_base:
        # hline
        for xmin_base, xmax_base, y_base in zip(xmins_base, xmaxs_base, ys_base):
            plt.hlines(
                y_base,
                xmin_base,
                xmax_base,
                color="black",
                linewidth=1,
                linestyles="dashed",
            )
    # xticks_label = ["raw", "w/o adapt. filt.", "w/ adapt. filt."] * 3
    # xticks_label = ["raw", "w/o adapt. filt.", "w/ adapt. filt.", "aaft"] * 3
    plt.xticks(xticks, xticks_label, fontsize=fontsize, rotation=90)
    # plt.tick_params(labelsize=fontsize, xticksrotation=90)
    plt.ylim(0, 0.13)
    plt.yticks([0, 0.05, 0.1], fontsize=fontsize)
    # plt.title(title_txt)
    plt.tight_layout()
    plt.savefig(save_dir / f"mis_{title_txt}.png")
    plt.clf()
    plt.close()


def statistical_test_target(
    mis: np.ndarray,
    mis_table: pd.DataFrame,
    eegs_table_all: pd.DataFrame,
    target_conditions: List[List[str]],
    save_dir: Path,
    tasks: Tuple[str] = ("overt", "minimally overt", "covert"),
):
    """plot mis

    Args:
        mis (np.ndarray): all mis
        mis_table (pd.DataFrame): mis table
        target_conditions (List[List[str]]): target conditions
            e.g. [["raw", "after_preproc", "rs", "raw", "EOG"],
                  ["raw", "after_preproc", "ft", "raw", "EOG"]]
            the List contains [eeg_types, emg_type, emg_name]
        save_dir (Path): save directory
        tasks (Tuple[str], optional): tasks. Defaults to
            ("overt", "minimally overt", "covert").
    """
    alpha = 0.05
    X_all = []
    labels_all = []
    for i_eeg, target_condition in enumerate(target_conditions):
        eeg_type, emg_type, emg_name, surrogate_type = target_condition
        target_indices = np.where(
            (mis_table["eeg_type"] == eeg_type)
            & (mis_table["emg_type"] == emg_type)
            & (mis_table["emg_name"] == emg_name)
            & (mis_table["surrogate_type"] == surrogate_type)
        )[0]
        assert len(target_indices) == 1
        target_index = target_indices[0]
        mis_tmp = np.mean(mis[target_index], axis=1)  # axis=1: average over channels
        for i_task, task in enumerate(tasks):
            target_indices_trial = np.where((eegs_table_all["task"] == task))[0]
            X_all.append(mis_tmp[target_indices_trial])
            if i_eeg == 0:
                eeg_text = "raw"
            elif i_eeg == 1:
                eeg_text = "min-preprocessed"
            elif i_eeg == 2:
                eeg_text = "denoised"
            elif i_eeg == 3:
                eeg_text = "shuffle"
            labels_all.append(f"{eeg_text}_{task}")
    for task in tasks:
        X = []
        labels = []
        for i, label in enumerate(labels_all):
            # cprint(label, "red")
            # cprint(task, "red")
            if f"_{task}" in label:
                X.append(X_all[i])
                labels.append(label)
        # for x in X:
        #     cprint(x.shape, "yellow")
        friedman_res = friedmanchisquare(*X)
        if friedman_res.pvalue < alpha:
            cprint(f"{task}", "cyan")
        else:
            cprint(f"{eeg_text} {emg_type} {emg_name} {surrogate_type}", "green")
        results_friedman = pd.DataFrame(
            {
                "statistic": [friedman_res.statistic],
                "p_value": [friedman_res.pvalue],
            }
        )
        results_friedman.to_csv(
            save_dir / f"friedman_{task}_{emg_name}.csv", index=False
        )

        p_values_before_corrected = []
        combs = list(combinations(range(len(X)), 2))
        stats = []
        for i, j in combs:
            wilcoxon_res = wilcoxon(X[i], X[j], alternative="two-sided")
            p_value = wilcoxon_res.pvalue
            stat = wilcoxon_res.statistic
            p_values_before_corrected.append(p_value)
            stats.append(stat)
        rejects, p_values_corrected, _, _ = multipletests(
            p_values_before_corrected, alpha=alpha, method="bonferroni"
        )
        for i_iter, (i, j) in enumerate(combs):
            if rejects[i_iter]:
                cprint(
                    f"{labels[i]} vs {labels[j]} p {p_values_corrected[i_iter]}", "cyan"
                )
            else:
                cprint(
                    f"{labels[i]} vs {labels[j]} p {p_values_corrected[i_iter]}",
                    "green",
                )
        results_wilcoxon = pd.DataFrame(
            {
                "label1": [labels[i] for i, _ in combs],
                "label2": [labels[j] for _, j in combs],
                "n_label1": [len(X[i]) for i, _ in combs],
                "n_label2": [len(X[j]) for _, j in combs],
                "statistic": stats,
                "p_value": p_values_before_corrected,
                "p_value_corrected": p_values_corrected,
                "reject": rejects,
            }
        )
        results_wilcoxon.to_csv(
            save_dir / f"wilcoxon_{task}_{emg_name}.csv", index=False
        )


def plot_mi_corr_between_speechs(mis, mis_table):
    target_index = np.where(
        (mis_table["eeg_type"] == "raw")
        & (mis_table["emg_type"] == "raw")
        & (mis_table["surrogate_type"] == "none")
    )[0]
    mi_roi = np.mean(mis[target_index], axis=1)


def main() -> None:
    """main"""
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
    k = 3
    seed = 0
    save_dir = Path("figures/fig2")
    eeg_type = "raw"
    emg_type = "raw"
    emg_names = ["EOG", "EMG_upper", "EMG_lower"]
    emg_index = 0
    eeg_types_for_mi = [
        "raw",
        "wo_adapt_filt",
        "after_preproc",
        "norm",
        "wo_adapt_filt_norm",
        "after_preproc_norm",
    ]
    eeg_types_for_before_after = [
        ["raw", "after_preproc"],
        ["wo_adapt_filt", "after_preproc"],
        ["norm", "after_preproc_norm"],
        ["wo_adapt_filt_norm", "after_preproc_norm"],
    ]
    emg_types_for_mi = [
        "raw",
        "highpass30",
        "highpass60",
        "norm",
        "highpass30_norm",
        "highpass60_norm",
    ]
    eeg_types_for_surrogate = ["raw", "wo_adapt_filt", "norm", "wo_adapt_filt_norm"]
    surrogate_types = ["trial_shuffle", "rs", "ft", "aaft", "iaaft"]
    within_session = True

    # target_conditions = list(product(eeg_types_for_before_after, surrogate_types))
    # target_conditions = [
    #     target_condition[0] + [target_condition[1]] + [emg_type] + [emg_name]
    #     for target_condition in target_conditions
    #     for emg_type in emg_types_for_mi
    #     for emg_name in emg_names
    #     if ((not "norm" in target_condition[0][0]) and (not "norm" in emg_type))
    #     or (("norm" in target_condition[0][0]) and ("norm" in emg_type))
    # ]
    target_conditions_all = [
        [
            ["raw", "raw", "EOG", "none"],
            ["wo_adapt_filt", "raw", "EOG", "none"],
            ["after_preproc", "raw", "EOG", "none"],
            ["raw", "raw", "EOG", "trial_shuffle"],
        ],
        [
            ["raw", "raw", "EMG_upper", "none"],
            ["wo_adapt_filt", "raw", "EMG_upper", "none"],
            ["after_preproc", "raw", "EMG_upper", "none"],
            ["raw", "raw", "EMG_upper", "trial_shuffle"],
        ],
        [
            ["raw", "raw", "EMG_lower", "none"],
            ["wo_adapt_filt", "raw", "EMG_lower", "none"],
            ["after_preproc", "raw", "EMG_lower", "none"],
            ["raw", "raw", "EMG_lower", "trial_shuffle"],
        ],
    ]

    st = time.time()
    (save_dir / "data").mkdir(exist_ok=True)
    (
        eegs,
        emgs,
        eegs_label,
        emgs_label,
        eegs_table_all,
    ) = ephys_load(data_tuple)
    np.save(save_dir / "data" / "eegs.npy", eegs)
    np.save(save_dir / "data" / "emgs.npy", emgs)
    np.save(save_dir / "data" / "eegs_label.npy", eegs_label)
    np.save(save_dir / "data" / "emgs_label.npy", emgs_label)
    eegs_table_all.to_csv(save_dir / "data" / "eegs_table_all.csv")
    # eegs = np.load(save_dir / "data" / "eegs.npy")
    # emgs = np.load(save_dir / "data" / "emgs.npy")
    # eegs_label = np.load(save_dir / "data" / "eegs_label.npy")
    # emgs_label = np.load(save_dir / "data" / "emgs_label.npy")
    # eegs_label = eegs_label.tolist()
    # emgs_label = emgs_label.tolist()
    # eegs_table_all = pd.read_csv(save_dir / "data" / "eegs_table_all.csv")
    cprint(f"eegs.shape: {eegs.shape}", "green")
    cprint(f"emgs.shape: {emgs.shape}", "green")
    cprint(f"len eegs_label: {len(eegs_label)}", "green")
    cprint(f"len emgs_label: {len(emgs_label)}", "green")

    surrogates, surrogate_table = collect_surrogates(
        eegs,
        surrogate_types,
        eeg_types_for_surrogate,
        eegs_label,
        within_session,
        eegs_table_all,
    )
    cprint(f"surrogates.shape: {surrogates.shape}", "cyan")
    np.save(save_dir / "data" / "surrogates.npy", surrogates)
    surrogate_table.to_csv(save_dir / "data" / "surrogate_table.csv")
    # surrogates = np.load(save_dir / "data" / "surrogates.npy")
    # surrogate_table = pd.read_csv(save_dir / "data" / "surrogate_table.csv")

    cprint("start calc mi", "cyan")
    mis, mis_table = collect_mis(
        eegs,
        emgs,
        surrogates,
        eeg_types_for_mi,
        eeg_types_for_surrogate,
        emg_types_for_mi,
        emg_names,
        eegs_label,
        emgs_label,
        surrogate_types,
        surrogate_table,
        k,
        seed,
    )
    cprint(f"mis.shape: {mis.shape}", "cyan")
    cprint(f"len mis_table: {len(mis_table)}", "cyan")
    np.save(save_dir / "data" / "mis.npy", mis)
    mis_table.to_csv(save_dir / "data" / "mis_table.csv")
    # mis = np.load(save_dir / "data" / "mis.npy")
    # mis_table = pd.read_csv(save_dir / "data" / "mis_table.csv")
    cprint(f"mis.shape: {mis.shape}", "cyan")
    cprint(f"len mis_table: {len(mis_table)}", "cyan")

    save_dir = save_dir / "mis"
    save_dir.mkdir(exist_ok=True)
    for target_conditions in target_conditions_all:
        plot_mis_for_target_conditions(
            mis,
            mis_table,
            eegs_table_all,
            target_conditions,
            save_dir,
            fontsize=12,
            show_base=True,
            show_shuffle=False,
        )

    save_dir = save_dir / "mis_statistical_test"
    save_dir.mkdir(exist_ok=True)
    for target_conditions in target_conditions_all:
        statistical_test_target(
            mis,
            mis_table,
            eegs_table_all,
            target_conditions,
            save_dir,
        )

    emg_contamination_montage(mis, mis_table, show_colorbar=True, vmax=0.115)

    elapsed_time = time.time() - st
    cprint(f"elapsed_time: {elapsed_time} [sec]", "cyan")


if __name__ == "__main__":
    main()
