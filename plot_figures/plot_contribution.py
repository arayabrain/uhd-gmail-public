"""plot spatio temporal contribution (Fig. 3-5, Fig. S2)"""

import itertools
import pickle
from copy import deepcopy
from operator import itemgetter
from pathlib import Path
from typing import List, Optional

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.image import imread
from scipy.special import softmax
from scipy.stats import pearsonr, zscore
from statsmodels.stats.multitest import multipletests
from termcolor import cprint
from tqdm import tqdm

from plot_figures.src.default_plt import (
    cm_to_inch,
    green,
    magenta,
    orange,
    plt,
    violet,
    yellow,
)
from plot_figures.src.load_ephys import load_ephys
from plot_figures.src.load_ig import load_igs
from plot_figures.src.weighted_corr import WeightedCorr


def imshow_each_ig_montage_representative(
    tasks: List[str],
    colors: List[str],
    igs_all: List[np.ndarray],
    igs_table_all: pd.DataFrame,
    eegs_all: List[np.ndarray],
    correct_decode: bool,
    clip: float,
    model: str = "EEGNet",
    cmax: float = 2.0,
    logscale: bool = False,
) -> None:
    """imshow each ig map

    Args:
        tasks (List[str]): tasks
        colors (List[str]): colors
        igs_all (List[np.ndarray]): ig
        igs_table_all (pd.DataFrame): ig tabele
        eegs_all (List[np.ndarray]): eeg
        correct_decode (bool): correct decode
        clip (float): clip
        model (str, optional): model name. Defaults to "EEGNet".
        cmax (float, optional): max value of colorbar. Defaults to 2.0.
        logscale (bool, optional): logscale or not. Defaults to False.
    """
    img = imread("plot_figures/montage_colorless.png")
    coordinates = np.load("plot_figures/coordinates_colorless.npy")
    if logscale:
        norm = LogNorm()
        kwargs = {
            "cmap": "viridis",
            "linewidths": 0,
            "norm": norm,
        }
    else:
        kwargs = {
            "cmap": "viridis",
            "linewidths": 0,
            "vmin": -cmax,
            "vmax": cmax,
        }
    igs_table_all = igs_table_all.assign(eeg_data_index=0)
    models = ["EEGNet", "EMG_EEGNet", "EEGNet_wo_adapt_filt"]
    for model_ in models:
        for cv in range(10):
            idx_target = (
                (igs_table_all["model"] == model_)
                # & (igs_table_all["task"] == task)
                # & (igs_table_all["label"] == label)
                # & (igs_table_all["correct_decode"] == correct_decode)
                & (igs_table_all["cv"] == cv)
            )
            igs_table_all.loc[idx_target, "eeg_data_index"] = np.arange(sum(idx_target))
    for task in tasks:
        for label, color in enumerate(tqdm(colors)):
            idx_target = (
                (igs_table_all["model"] == model)
                & (igs_table_all["task"] == task)
                & (igs_table_all["label"] == label)
                & (igs_table_all["correct_decode"] == correct_decode)
            )
            if idx_target.sum() == 0:
                cprint(
                    (f"no data for label: {label}, " f"task: {task}, correct_decode"),
                    "yellow",
                )
                continue
            cprint(
                (f"label: {label}, task: {task}" f" sum {idx_target.sum()}"),
                "cyan",
            )
            igs_tmp = np.array(itemgetter(*np.where(idx_target)[0])(igs_all))
            eeg_data_indices = igs_table_all.loc[idx_target, "eeg_data_index"].values
            mean_igs_tmp = igs_tmp.mean(axis=2)
            mean_ig = zscore(mean_igs_tmp.mean(axis=0))
            sd_ig = zscore(-mean_igs_tmp.std(axis=0, ddof=1))
            corrs = np.corrcoef(mean_ig, mean_igs_tmp)[0, 1:]
            argmax = np.argmax(corrs)
            cprint(f"argmax: {argmax}", "cyan")
            representative_ig_mat = igs_tmp[argmax, :, :]
            representative_ig = zscore(mean_igs_tmp[argmax])
            representative_eeg = eegs_all[eeg_data_indices[argmax]]
            idx_target = (
                (igs_table_all["model"] == "EEGNet_wo_adapt_filt")
                & (igs_table_all["task"] == task)
                & (igs_table_all["label"] == label)
                & (igs_table_all["correct_decode"] == correct_decode)
            )
            # cprint(representative_eeg.shape, "green")
            if logscale:
                representative_ig -= representative_ig.min() + 1e-5
                representative_ig -= representative_ig.min() + 1e-5

            plt.figure(figsize=(3 * cm_to_inch, 3 * cm_to_inch))
            plt.imshow(img)
            plt.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                s=5,
                c=representative_ig,
                **kwargs,
            )
            # plt.colorbar()
            plt.axis("off")
            for save_path in [Path("figures/fig3"), Path("figures/fig4")]:
                (save_path / "representative").mkdir(exist_ok=True, parents=True)
                plt.savefig(
                    save_path
                    / "representative"
                    / f"mean_montage_{task}_{color}_{model}.png"
                )
            plt.clf()
            plt.close()

            plt.figure(figsize=(3 * cm_to_inch, 3 * cm_to_inch))
            n_ch = representative_eeg.shape[0]
            representative_eeg = zscore(representative_eeg, axis=1)
            representative_eeg = np.clip(representative_eeg, -clip, clip)
            for ch in range(n_ch):
                plt.plot(
                    representative_eeg[ch] - ch * 5,
                    color="black",
                    linewidth=0.1,
                )
            plt.axis("off")
            for save_path in [Path("figures/fig3"), Path("figures/fig4")]:
                plt.savefig(save_path / "representative" / f"eeg_{task}_{color}.png")
            plt.clf()
            plt.close()

            plt.figure(figsize=(3 * cm_to_inch, 3 * cm_to_inch))
            plt.imshow(
                representative_ig_mat, cmap="viridis", aspect="auto", vmin=-5, vmax=5
            )
            plt.axis("off")
            for save_path in [Path("figures/fig3"), Path("figures/fig4")]:
                plt.savefig(save_path / "representative" / f"ig_mat_{task}_{color}.png")
            plt.clf()
            plt.close()

            representative_ig_mat_time_avg = representative_ig_mat.mean(
                axis=1, keepdims=True
            )
            plt.figure(figsize=(0.2 * cm_to_inch, 3 * cm_to_inch))
            plt.imshow(
                representative_ig_mat_time_avg,
                cmap="viridis",
                aspect="auto",
                vmin=-5,
                vmax=5,
            )
            plt.axis("off")
            for save_path in [Path("figures/fig3"), Path("figures/fig4")]:
                plt.savefig(
                    save_path / "representative" / f"ig_timeavg_{task}_{color}.png"
                )
            plt.clf()
            plt.close()

            representative_ig_mat_spatial_avg = representative_ig_mat.mean(
                axis=0, keepdims=True
            )
            plt.figure(figsize=(3 * cm_to_inch, 0.2 * cm_to_inch))
            plt.imshow(
                representative_ig_mat_spatial_avg,
                cmap="viridis",
                aspect="auto",
                vmin=-5,
                vmax=5,
            )
            plt.axis("off")
            for save_path in [Path("figures/fig3"), Path("figures/fig4")]:
                plt.savefig(
                    save_path / "representative" / f"ig_spatialavg_{task}_{color}.png"
                )
            plt.clf()
            plt.close()


def imshow_diff_ig_montage(
    tasks: List[str],
    colors: List[str],
    igs_all: List[np.ndarray],
    igs_table_all: pd.DataFrame,
    correct_decode: bool,
    save_path: Path,
    model1: str = "EEGNet",
    model2: str = "EEGNet_wo_adapt_filt",
    cmax: float = 2.0,
    logscale: bool = False,
    cmap: str = "viridis",
    w_size: float = 3.0,
    h_size: float = 3.0,
    show_each: bool = False,
    show_avg: bool = False,
    trial_based: bool = False,
    compare_with_emgs: bool = True,
    mi_emgs: List[np.ndarray] = None,
    corr_type: str = "pearson",
    mis: Optional[np.ndarray] = None,
) -> None:
    """imshow each ig map

    Args:
        tasks (List[str]): tasks
        colors (List[str]): colors
        igs_all (List[np.ndarray]): ig
        igs_table_all (pd.DataFrame): ig tabele
        correct_decode (bool): correct decode
        save_path (Path): save path
        model1 (str, optional): model name. Defaults to "EEGNet".
        model2 (str, optional): model name. Defaults to "EEGNet_wo_adapt_filt".
        cmax (float, optional): max value of colorbar. Defaults to 2.0.
        logscale (bool, optional): logscale or not. Defaults to False.
        cmap (str, optional): cmap. Defaults to "viridis".
        w_size (float, optional): width size. Defaults to 3.0.
        h_size (float, optional): height size. Defaults to 3.0.
        show_each (bool, optional): show each or not. Defaults to False.
        show_avg (bool, optional): show avg or not. Defaults to False.
        trial_based (bool, optional): trial based or not. Defaults to False.
        compare_with_emgs (bool, optional): compare with emgs or not. Defaults to True.
        mi_emgs (List[np.ndarray], optional): mi emgs. Defaults to None.
        corr_type (str, optional): correlation type. Defaults to "pearson".
        mis (Optional[np.ndarray], optional): mis. Defaults to None.
    """
    img = imread("plot_figures/montage_colorless.png")
    coordinates = np.load("plot_figures/coordinates_colorless.npy")
    if logscale:
        norm = LogNorm()
        kwargs = {
            "cmap": cmap,
            "linewidths": 0,
            "norm": norm,
        }
    else:
        kwargs = {
            "cmap": cmap,
            "linewidths": 0,
            "vmin": -cmax,
            "vmax": cmax,
        }
    igs_table_all = igs_table_all.assign(EEGNet_correct_decode=0)
    eegnet_correct_decode = igs_table_all["correct_decode"].values[
        igs_table_all["model"] == "EEGNet"
    ]
    models = igs_table_all["model"].unique()
    for model in models:
        igs_table_all["EEGNet_correct_decode"].values[
            igs_table_all["model"] == model
        ] = eegnet_correct_decode
    if not show_each:
        fig, axes = plt.subplots(
            len(tasks),
            len(colors) if not show_avg else len(colors) + 1,
            tight_layout=True,
            figsize=(
                len(colors) * w_size * cm_to_inch,
                len(tasks) * h_size * cm_to_inch,
            ),
        )
    avg_diffs = []
    color_diffs = []
    idx_target0s = []
    idx_target0_labels = []
    for task in tasks:
        color_diff = []
        for label, color in enumerate(tqdm(colors)):
            idx_target0_all = np.where(
                (
                    (igs_table_all["model"] == model1)
                    & (igs_table_all["cv"] == 0)
                    # & (igs_table_all["task"] == task)
                    # & (igs_table_all["label"] == label)
                    # & (igs_table_all["correct_decode"] == correct_decode)
                )
            )[0]
            idx_target0_target = np.where(
                (
                    (igs_table_all["model"] == model1)
                    & (igs_table_all["cv"] == 0)
                    & (igs_table_all["task"] == task)
                    & (igs_table_all["label"] == label)
                    & (igs_table_all["correct_decode"] == correct_decode)
                )
            )[0]
            idx_target0_values = np.intersect1d(idx_target0_all, idx_target0_target)
            idx_target0 = np.where(np.isin(idx_target0_all, idx_target0_values))[0]
            idx_target0s.append(idx_target0)
            idx_target0_labels.append(f"{task}_{color}")
            # cprint(f"idx_target0_values: {idx_target0_values}", "cyan")
            # cprint(f"idx_target0: {idx_target0}", "cyan")
            # cprint(f"len idx_target0: {len(idx_target0)}", "cyan")
            # cprint(f"len idx_target0_all: {len(idx_target0_all)}", "cyan")

            idx_target1 = (
                (igs_table_all["model"] == model1)
                & (igs_table_all["task"] == task)
                & (igs_table_all["label"] == label)
                & (igs_table_all["correct_decode"] == correct_decode)
            )
            idx_target1_2 = (
                (igs_table_all["model"] == model2)
                & (igs_table_all["task"] == task)
                & (igs_table_all["label"] == label)
                & (igs_table_all["EEGNet_correct_decode"] == correct_decode)
            )
            idx_target2 = (
                (igs_table_all["model"] == model2)
                & (igs_table_all["task"] == task)
                & (igs_table_all["label"] == label)
                & (igs_table_all["correct_decode"] == correct_decode)
            )
            if (idx_target1.sum() == 0) or (idx_target2.sum() == 0):
                cprint(
                    (f"no data for label: {label}, " f"task: {task}, correct_decode"),
                    "yellow",
                )
                continue
            cprint(
                (
                    f"label: {label}, task: {task}"
                    f" sum1 {idx_target1.sum()} sum2 {idx_target2.sum()}"
                ),
                "cyan",
            )
            igs_tmp1 = np.array(itemgetter(*np.where(idx_target1)[0])(igs_all))
            if trial_based:
                igs_tmp2 = np.array(
                    itemgetter(*np.where(idx_target1_2)[0])(igs_all)
                )  # (n_trials, n_ch, n_times)
            else:
                igs_tmp2 = np.array(
                    itemgetter(*np.where(idx_target2)[0])(igs_all)
                )  # (n_trials, n_ch, n_times)
            mean_igs_tmp1 = igs_tmp1.mean(axis=2)
            mean_igs_tmp2 = igs_tmp2.mean(axis=2)  # (n_trials, n_ch)
            mean_ig1 = zscore(mean_igs_tmp1.mean(axis=0))
            mean_ig2 = zscore(mean_igs_tmp2.mean(axis=0))
            sd_ig1 = zscore(-mean_igs_tmp1.std(axis=0, ddof=1))
            sd_ig2 = zscore(-mean_igs_tmp2.std(axis=0, ddof=1))
            if logscale:
                mean_ig1 -= mean_ig1.min() + 1e-5
                mean_ig2 -= mean_ig2.min() + 1e-5
                sd_ig1 -= sd_ig1.min() + 1e-5
                sd_ig2 -= sd_ig2.min() + 1e-5
            diffs = mean_igs_tmp1 - mean_igs_tmp2  # (n_trials, n_ch)
            mean_diff = np.mean(diffs, axis=0)
            corrs_diff = np.corrcoef(mean_diff, diffs)[0, 1:]
            argmax_corr_diff = np.argmax(corrs_diff)
            representative_ig1 = mean_igs_tmp1[argmax_corr_diff]
            representative_ig2 = mean_igs_tmp2[argmax_corr_diff]
            representative_diff = representative_ig1 - representative_ig2
            diff = mean_ig1 - mean_ig2
            for i, tmp in enumerate(
                [representative_ig1, representative_ig2, representative_diff]
            ):
                if i == 0:
                    representative_text = f"representative_{model1}"
                    kwargs = {
                        "cmap": "viridis",
                        "linewidths": 0,
                        "vmin": -cmax,
                        "vmax": cmax,
                    }
                elif i == 1:
                    representative_text = f"representative_{model2}"
                    kwargs = {
                        "cmap": "viridis",
                        "linewidths": 0,
                        "vmin": -cmax,
                        "vmax": cmax,
                    }
                elif i == 2:
                    representative_text = f"representative_diff"
                    kwargs = {
                        "cmap": cmap,
                        "linewidths": 0,
                        "vmin": -cmax,
                        "vmax": cmax,
                    }
                else:
                    pass
                plt.figure(figsize=(3 * cm_to_inch, 3 * cm_to_inch))
                plt.imshow(img)
                plt.scatter(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    s=5,
                    c=tmp,
                    **kwargs,
                )
                # plt.colorbar()
                plt.axis("off")
                (save_path / "representative").mkdir(exist_ok=True, parents=True)
                plt.savefig(
                    save_path
                    / "representative"
                    / f"{representative_text}_montage_{task}_{color}_{cmap}_sd{cmax}.png"
                )
                plt.clf()
                plt.close()
            if not show_each:
                axes[tasks.index(task), label].imshow(img)
                axes[tasks.index(task), label].scatter(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    s=5,
                    c=diff if not trial_based else zscore(diffs.mean(axis=0)),
                    **kwargs,
                )
                # plt.colorbar()
                axes[tasks.index(task), label].axis("off")
                # axes[tasks.index(task), label].set_title(f"{task} {color}")
                color_diff.append(zscore(diffs.mean(axis=0)))
            else:
                plt.figure(figsize=(3 * cm_to_inch, 3 * cm_to_inch))
                plt.imshow(img)
                plt.scatter(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    s=5,
                    c=diff if not trial_based else zscore(diff.mean(axis=0)),
                    **kwargs,
                )
                # plt.colorbar()
                plt.axis("off")
                (save_path / "montage").mkdir(exist_ok=True, parents=True)
                plt.savefig(
                    save_path
                    / "montage"
                    / f"diff_montage_{task}_{color}_{cmap}_sd{cmax}.png"
                )
                plt.clf()
                plt.close()
        if show_avg:
            label = 5
            color = "avg"
            idx_target0_all = np.where(
                (
                    (igs_table_all["model"] == model1)
                    & (igs_table_all["cv"] == 0)
                    # & (igs_table_all["task"] == task)
                    # & (igs_table_all["correct_decode"] == correct_decode)
                )
            )[0]
            idx_target0_target = np.where(
                (
                    (igs_table_all["model"] == model1)
                    & (igs_table_all["cv"] == 0)
                    & (igs_table_all["task"] == task)
                    & (igs_table_all["correct_decode"] == correct_decode)
                )
            )[0]
            idx_target0_values = np.intersect1d(idx_target0_all, idx_target0_target)
            idx_target0 = np.where(np.isin(idx_target0_all, idx_target0_values))[0]
            idx_target0s.append(idx_target0)
            idx_target0_labels.append(f"{task}_{color}")
            idx_target1 = (
                (igs_table_all["model"] == model1)
                & (igs_table_all["task"] == task)
                & (igs_table_all["correct_decode"] == correct_decode)
            )
            idx_target1_2 = (
                (igs_table_all["model"] == model2)
                & (igs_table_all["task"] == task)
                & (igs_table_all["EEGNet_correct_decode"] == correct_decode)
            )
            idx_target2 = (
                (igs_table_all["model"] == model2)
                & (igs_table_all["task"] == task)
                & (igs_table_all["correct_decode"] == correct_decode)
            )
            igs_tmp1 = np.array(itemgetter(*np.where(idx_target1)[0])(igs_all))
            if trial_based:
                igs_tmp2 = np.array(itemgetter(*np.where(idx_target1_2)[0])(igs_all))
            else:
                igs_tmp2 = np.array(itemgetter(*np.where(idx_target2)[0])(igs_all))
            mean_igs_tmp1 = igs_tmp1.mean(axis=2)
            mean_igs_tmp2 = igs_tmp2.mean(axis=2)
            mean_ig1 = zscore(mean_igs_tmp1.mean(axis=0))
            mean_ig2 = zscore(mean_igs_tmp2.mean(axis=0))
            sd_ig1 = zscore(-mean_igs_tmp1.std(axis=0, ddof=1))
            sd_ig2 = zscore(-mean_igs_tmp2.std(axis=0, ddof=1))
            if logscale:
                mean_ig1 -= mean_ig1.min() + 1e-5
                mean_ig2 -= mean_ig2.min() + 1e-5
                sd_ig1 -= sd_ig1.min() + 1e-5
                sd_ig2 -= sd_ig2.min() + 1e-5
            diffs = mean_igs_tmp1 - mean_igs_tmp2
            diff = mean_ig1 - mean_ig2
            if not show_each:
                axes[tasks.index(task), label].imshow(img)
                axes[tasks.index(task), label].scatter(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    s=5,
                    c=diff if not trial_based else zscore(diffs.mean(axis=0)),
                    **kwargs,
                )
                # plt.colorbar()
                axes[tasks.index(task), label].axis("off")
                # axes[tasks.index(task), label].set_title(f"{task} {color}")
                color_diff.append(zscore(diffs.mean(axis=0)))
            else:
                plt.figure(figsize=(3 * cm_to_inch, 3 * cm_to_inch))
                plt.imshow(img)
                plt.scatter(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    s=5,
                    c=diff if not trial_based else zscore(diffs.mean(axis=0)),
                    **kwargs,
                )
                # plt.colorbar()
                plt.axis("off")
                trial_based_text = "_trial_based" if trial_based else ""
                (save_path / "montage").mkdir(exist_ok=True, parents=True)
                plt.savefig(
                    save_path
                    / "montage"
                    / f"diff_montage_{task}_{color}_{cmap}_sd{cmax}{trial_based_text}.png"
                )
                plt.clf()
                plt.close()
            avg_diffs.append(zscore(diffs.mean(axis=0)))
        color_diff = np.array(color_diff)
        color_diffs.append(color_diff)
    color_diffs = np.array(color_diffs)  # (n_tasks, n_colors, n_times)
    if not show_each:
        show_avg_text = "_withavg" if show_avg else ""
        trial_based_text = "_trial_based" if trial_based else ""
        (save_path / "montage").mkdir(exist_ok=True, parents=True)
        plt.savefig(
            save_path
            / "montage"
            / f"diff_montage_combine_{cmap}_sd{cmax}{show_avg_text}{trial_based_text}.png"
        )
        plt.clf()
        plt.close()

    if compare_with_emgs:
        avg_diffs += mi_emgs
    # corr between speeches
    alpha = 0.05
    weights_all = []
    if corr_type == "pearson":
        corr_func = pearsonr
    elif corr_type == "weighted_pearson":
        mean_mi_emgs = np.array(mi_emgs).mean(axis=0)  # (n_channels, )
        inv_mean_mi_emgs = 1 / mean_mi_emgs
        # normalize to 0-1
        weights = inv_mean_mi_emgs / inv_mean_mi_emgs.sum()
        # weights = softmax(inv_mean_mi_emgs)
        corr_func = WeightedCorr(w=weights)
    elif corr_type == "weight_multiplied_pearson":
        corr_func = pearsonr
        for idx_target0, idx_target0_label in zip(idx_target0s, idx_target0_labels):
            mean_mi_emgs = np.mean(
                mis[:, idx_target0, :], axis=(0, 1)
            )  # (n_channels, )
            inv_mean_mi_emgs = 1 / mean_mi_emgs
            # normalize to 0-1
            weights = inv_mean_mi_emgs / np.sum(inv_mean_mi_emgs)
            # weights = softmax(inv_mean_mi_emgs)
            weights_all.append(weights)
            fig_weight, ax_weight = plt.subplots()
            ax_weight.bar(np.arange(len(weights)), weights)
            ax_weight.set_title(f"weights_{idx_target0_label}")
            ax_weight.set_xlabel("channel")
            ax_weight.set_ylabel("weight")
            save_path_s2 = Path("figures/figS2")
            (save_path_s2 / "weights").mkdir(exist_ok=True, parents=True)
            fig_weight.savefig(
                save_path_s2 / "weights" / f"weights_{idx_target0_label}.png"
            )
        # save weights_all
        # np.save(save_path / "weights" / "weights_all.npy", weights_all)
    else:
        raise ValueError("corr_type must be pearson or weighted_pearson")
    avg_diffs = np.array(avg_diffs)
    cprint(f"avg_diffs.shape: {avg_diffs.shape}", "cyan")
    corr_mat = np.corrcoef(avg_diffs)
    corr_mat[np.diag_indices_from(corr_mat)] = np.nan
    p_value = np.zeros((len(corr_mat), len(corr_mat)))
    p_value[np.diag_indices_from(p_value)] = np.nan
    for i in range(len(corr_mat)):
        for j in range(len(corr_mat)):
            if i <= j:
                continue
            corr_mat[i, j], p_value[i, j] = corr_func(
                avg_diffs[i],
                avg_diffs[j],
            )
            corr_mat[j, i] = corr_mat[i, j]
            p_value[j, i] = p_value[i, j]
    p_vec_before_corrected = p_value[np.triu_indices_from(p_value, k=1)]
    rejects, p_vec_after_corrected, _, _ = multipletests(
        p_vec_before_corrected, method="bonferroni", alpha=alpha
    )
    # plot correlation matrix
    cmax_corr = 1.0
    plt.figure(figsize=(7 * cm_to_inch, 7 * cm_to_inch))
    plt.imshow(corr_mat, cmap="jet", vmin=-cmax_corr, vmax=cmax_corr)
    plt.colorbar()
    # mark significant correlation
    for i_reject, reject in enumerate(rejects):
        if reject:
            i, j = (
                np.triu_indices_from(p_value, k=1)[0][i_reject],
                np.triu_indices_from(p_value, k=1)[1][i_reject],
            )
            if p_vec_after_corrected[i_reject] < 0.001:
                marker = "***"
            elif p_vec_after_corrected[i_reject] < 0.01:
                marker = "**"
            elif p_vec_after_corrected[i_reject] < 0.05:
                marker = "*"
            else:
                raise ValueError("p value must be < 0.05")
            plt.text(
                j,
                i,
                marker,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=10,
                color="white",
            )
            plt.text(
                i,
                j,
                marker,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=10,
                color="white",
            )
    # top, right axis on
    plt.gca().spines["top"].set_visible(True)
    plt.gca().spines["right"].set_visible(True)
    # tick off
    plt.tick_params(
        bottom=False,
        left=False,
        right=False,
        top=False,
        labelbottom=False,
        labelleft=False,
    )
    plt.tight_layout()
    compare_with_emgs_text = "_compare_with_emgs" if compare_with_emgs else ""
    (save_path / "corr").mkdir(exist_ok=True, parents=True)
    if corr_type == "pearson":
        plt.savefig(
            save_path
            / "corr"
            / f"montage_diff_corr_between_speech_{model1}_{model2}{compare_with_emgs_text}.png"
        )
    else:
        save_path_s2 = Path("figures/figS2")
        (save_path_s2 / "corr").mkdir(exist_ok=True, parents=True)
        plt.savefig(
            save_path_s2
            / "corr"
            / f"montage_diff_corr_between_speech_{corr_type}_{model1}_{model2}{compare_with_emgs_text}.png"
        )
    plt.clf()
    plt.close()

    # plot combine correlation matrix for each color_diffs
    fig, axes = plt.subplots(
        1,
        len(colors) + 1,
        tight_layout=True,
        figsize=(len(colors) * 3 * cm_to_inch, 3 * cm_to_inch),
    )
    color_diffs = np.transpose(color_diffs, (1, 0, 2))  # (n_colors, n_tasks, n_times)
    for color_diff, color, ax in zip(color_diffs, colors + ["avg"], axes):
        # cprint(f"color_diff.shape: {color_diff.shape}", "cyan") # (n_tasks, n_times)
        corr_mat = np.zeros((len(color_diff), len(color_diff)))
        corr_mat[np.diag_indices_from(corr_mat)] = np.nan
        p_value = np.zeros((len(corr_mat), len(corr_mat)))
        p_value[np.diag_indices_from(p_value)] = np.nan
        for i in range(len(corr_mat)):
            for j in range(len(corr_mat)):
                if i <= j:
                    continue
                task_i = tasks[i]
                task_j = tasks[j]
                idx_target0_labels_i = f"{task_i}_{color}"
                idx_target0_labels_j = f"{task_j}_{color}"
                if corr_type == "weight_multiplied_pearson":
                    weight_i = weights_all[
                        idx_target0_labels.index(idx_target0_labels_i)
                    ]
                    weight_j = weights_all[
                        idx_target0_labels.index(idx_target0_labels_j)
                    ]
                corr_mat[i, j], p_value[i, j] = corr_func(
                    (
                        color_diff[i]
                        if corr_type != "weight_multiplied_pearson"
                        else color_diff[i] * weight_i
                    ),
                    (
                        color_diff[j]
                        if corr_type != "weight_multiplied_pearson"
                        else color_diff[j] * weight_j
                    ),
                )
                corr_mat[j, i] = corr_mat[i, j]
                p_value[j, i] = p_value[i, j]
        p_vec_before_corrected = p_value[np.triu_indices_from(p_value, k=1)]
        rejects, p_vec_after_corrected, _, _ = multipletests(
            p_vec_before_corrected, method="bonferroni", alpha=alpha
        )
        # plot correlation matrix
        cmax_corr = 1.0
        ax.imshow(corr_mat, cmap="jet", vmin=-cmax_corr, vmax=cmax_corr)
        # ax.colorbar()
        # mark significant correlation
        for i_reject, reject in enumerate(rejects):
            if reject:
                i, j = (
                    np.triu_indices_from(p_value, k=1)[0][i_reject],
                    np.triu_indices_from(p_value, k=1)[1][i_reject],
                )
                if p_vec_after_corrected[i_reject] < 0.001:
                    marker = "***"
                elif p_vec_after_corrected[i_reject] < 0.01:
                    marker = "**"
                elif p_vec_after_corrected[i_reject] < 0.05:
                    marker = "*"
                else:
                    raise ValueError("p value must be < 0.05")
                ax.text(
                    j,
                    i,
                    marker,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=10,
                    color="white",
                )
                ax.text(
                    i,
                    j,
                    marker,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=10,
                    color="white",
                )
        # top, right axis on
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        # tick off
        ax.tick_params(
            bottom=False,
            left=False,
            right=False,
            top=False,
            labelbottom=False,
            labelleft=False,
        )
        # ax.set_title(color)
    plt.tight_layout()
    if corr_type == "pearson":
        plt.savefig(
            save_path
            / "corr"
            / f"montage_diff_corr_between_speech_combine_colors_{model1}_{model2}.png"
        )
    else:
        save_path_s2 = Path("figures/figS2")
        plt.savefig(
            save_path_s2
            / "corr"
            / f"montage_diff_corr_between_speech_combine_colors_{corr_type}_{model1}_{model2}.png"
        )
    plt.clf()
    plt.close()

    # matrix of corr between each diff and each emg
    fig, axes = plt.subplots(
        1,
        len(colors) + 1,
        tight_layout=True,
        figsize=(len(colors) * 3 * cm_to_inch, 3 * cm_to_inch),
    )
    for color_diff, color, ax in zip(color_diffs, colors + ["avg."], axes):
        corr_mat = np.zeros((len(color_diff), len(mi_emgs)))
        corr_mat[np.diag_indices_from(corr_mat)] = np.nan
        p_value = np.zeros((len(corr_mat), len(corr_mat)))
        p_value[np.diag_indices_from(p_value)] = np.nan
        for i in range(len(corr_mat)):
            for j in range(len(mi_emgs)):
                corr_mat[i, j], p_value[i, j] = corr_func(
                    color_diff[i],
                    mi_emgs[j],
                )
        # plot correlation matrix
        cmax_corr = 1.0
        ax.imshow(corr_mat, cmap="jet", vmin=-cmax_corr, vmax=cmax_corr)
        # ax.colorbar()
        # mark significant correlation
        # correction for p_value
        p_vec_before_corrected = p_value.flatten()
        rejects, p_vec_after_corrected, _, _ = multipletests(
            p_vec_before_corrected, method="bonferroni", alpha=alpha
        )
        p_value = p_vec_after_corrected.reshape(p_value.shape)
        for i in range(len(corr_mat)):
            for j in range(len(mi_emgs)):
                if p_value[i, j] < 0.001:
                    marker = "***"
                elif p_value[i, j] < 0.01:
                    marker = "**"
                elif p_value[i, j] < 0.05:
                    marker = "*"
                else:
                    continue
                ax.text(
                    j,
                    i,
                    marker,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=10,
                    color="white",
                )
        # top, right axis on
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        # tick off
        ax.tick_params(
            bottom=False,
            left=False,
            right=False,
            top=False,
            labelbottom=False,
            labelleft=False,
        )
        # ax.set_title(color)
    plt.tight_layout()
    if corr_type == "pearson":
        plt.savefig(
            save_path
            / "corr"
            / f"montage_diff_corr_between_igeeg_and_miemg_{model1}_{model2}.png"
        )
    else:
        save_path_s2 = Path("figures/figS2")
        plt.savefig(
            save_path_s2
            / "corr"
            / f"montage_diff_corr_between_igeeg_and_miemg_{corr_type}_{model1}_{model2}.png"
        )
    plt.clf()
    plt.close()

    # matrix of corr between emg of different tasks
    mi_emgs_avgs = []
    for idx_target0, idx_target0_label in zip(idx_target0s, idx_target0_labels):
        mi_emgs_avgs.append(
            np.mean(mis[:, idx_target0, :], axis=1)
        )  # List[(n_emgs, n_channels)]
    fig, axes = plt.subplots(
        3,
        len(colors) + 1,
        tight_layout=True,
        figsize=(len(colors) * 3 * cm_to_inch, 3 * cm_to_inch * 3),
    )
    for emg_idx, emg_label in enumerate(["EOG", "EMG_upper", "EMG_lower"]):
        for color in colors + ["avg"]:
            ax = axes[emg_idx, colors.index(color) if color != "avg" else -1]
            corr_mat = np.zeros((len(tasks), len(tasks)))
            corr_mat[np.diag_indices_from(corr_mat)] = np.nan
            p_value = np.zeros((len(corr_mat), len(corr_mat)))
            p_value[np.diag_indices_from(p_value)] = np.nan
            for i in range(len(corr_mat)):
                for j in range(len(corr_mat)):
                    if i <= j:
                        continue
                    task_i = tasks[i]
                    task_j = tasks[j]
                    idx_target0_labels_i = f"{task_i}_{color}"
                    idx_target0_labels_j = f"{task_j}_{color}"
                    corr_mat[i, j], p_value[i, j] = pearsonr(
                        mi_emgs_avgs[idx_target0_labels.index(idx_target0_labels_i)][
                            emg_idx
                        ],
                        mi_emgs_avgs[idx_target0_labels.index(idx_target0_labels_j)][
                            emg_idx
                        ],
                    )
                    corr_mat[j, i] = corr_mat[i, j]
                    p_value[j, i] = p_value[i, j]
            p_vec_before_corrected = p_value[np.triu_indices_from(p_value, k=1)]
            rejects, p_vec_after_corrected, _, _ = multipletests(
                p_vec_before_corrected, method="bonferroni", alpha=alpha
            )
            # plot correlation matrix
            cmax_corr = 1.0
            ax.imshow(corr_mat, cmap="jet", vmin=-cmax_corr, vmax=cmax_corr)
            # ax.colorbar()
            # mark significant correlation
            for i_reject, reject in enumerate(rejects):
                if reject:
                    i, j = (
                        np.triu_indices_from(p_value, k=1)[0][i_reject],
                        np.triu_indices_from(p_value, k=1)[1][i_reject],
                    )
                    if p_vec_after_corrected[i_reject] < 0.001:
                        marker = "***"
                    elif p_vec_after_corrected[i_reject] < 0.01:
                        marker = "**"
                    elif p_vec_after_corrected[i_reject] < 0.05:
                        marker = "*"
                    else:
                        raise ValueError("p value must be < 0.05")
                    ax.text(
                        j,
                        i,
                        marker,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=10,
                        color="white",
                    )
                    ax.text(
                        i,
                        j,
                        marker,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=10,
                        color="white",
                    )
            # top, right axis on
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            # tick off
            ax.tick_params(
                bottom=False,
                left=False,
                right=False,
                top=False,
                labelbottom=False,
                labelleft=False,
            )
            # ax.set_title(color)
    plt.tight_layout()
    save_path_s2 = Path("figures/figS2")
    (save_path_s2 / "corr").mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path_s2 / "corr" / f"mi_emgs_pearsoncorr_between_tasks.png")
    plt.clf()
    plt.close()


def imshow_combine_ig_montage(
    tasks: List[str],
    colors: List[str],
    igs_all: List[np.ndarray],
    igs_table_all: pd.DataFrame,
    correct_decode: bool,
    save_path: Path,
    model: str = "EEGNet",
    cmax: float = 2.0,
    logscale: bool = False,
    cmap: str = "viridis",
    show_sd: bool = False,
    show_avg: bool = False,
    compare_with_emgs: bool = True,
    mi_emgs: List[np.ndarray] = None,
) -> None:
    """imshow each ig map

    Args:
        tasks (List[str]): tasks
        colors (List[str]): colors
        igs_all (List[np.ndarray]): ig
        igs_table_all (pd.DataFrame): ig tabele
        correct_decode (bool): correct decode
        save_path (Path): save path
        model (str, optional): model name. Defaults to "EEGNet".
        cmax (float, optional): max value of colorbar. Defaults to 2.0.
        logscale (bool, optional): logscale or not. Defaults to False.
        cmap (str, optional): cmap. Defaults to "viridis".
        show_sd (bool, optional): show sd or not. Defaults to False.
        show_avg (bool, optional): show avg or not. Defaults to False.
        compare_with_emgs (bool, optional): compare with emgs or not. Defaults to True.
        mi_emgs (List[np.ndarray], optional): mi emgs. Defaults to None.
    """
    img = imread("plot_figures/montage_colorless.png")
    coordinates = np.load("plot_figures/coordinates_colorless.npy")
    if logscale:
        norm = LogNorm()
        kwargs = {
            "cmap": cmap,
            "linewidths": 0,
            "norm": norm,
        }
    else:
        kwargs = {
            "cmap": cmap,
            "linewidths": 0,
            "vmin": -cmax,
            "vmax": cmax,
        }
    color_igs = []
    avg_igs = []
    fig, axes = plt.subplots(
        len(tasks),
        len(colors) if not show_avg else len(colors) + 1,
        tight_layout=True,
        figsize=(len(colors) * 3 * cm_to_inch, len(tasks) * 3 * cm_to_inch),
    )
    for task in tasks:
        color_ig = []
        for label, color in enumerate(tqdm(colors)):
            idx_target = (
                (igs_table_all["model"] == model)
                & (igs_table_all["task"] == task)
                & (igs_table_all["label"] == label)
                & (igs_table_all["correct_decode"] == correct_decode)
            )
            if idx_target.sum() == 0:
                cprint(
                    (f"no data for label: {label}, " f"task: {task}, correct_decode"),
                    "yellow",
                )
                continue
            cprint(
                (f"label: {label}, task: {task}" f" sum {idx_target.sum()}"),
                "cyan",
            )
            igs_tmp = np.array(itemgetter(*np.where(idx_target)[0])(igs_all))
            mean_igs_tmp = igs_tmp.mean(axis=2)
            mean_ig = zscore(mean_igs_tmp.mean(axis=0))
            sd_ig = zscore(-mean_igs_tmp.std(axis=0, ddof=1))
            if logscale:
                mean_ig -= mean_ig.min() + 1e-5
                sd_ig -= sd_ig.min() + 1e-5

            axes[tasks.index(task), label].imshow(img)
            axes[tasks.index(task), label].scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                s=5,
                c=mean_ig if not show_sd else sd_ig,
                **kwargs,
            )
            axes[tasks.index(task), label].axis("off")
            # axes[tasks.index(task), label].set_title(f"{task} {color}")
            # axes[tasks.index(task), label].colorbar()
            color_ig.append(mean_ig if not show_sd else sd_ig)
        if show_avg:
            label = 5
            idx_target = (
                (igs_table_all["model"] == model)
                & (igs_table_all["task"] == task)
                & (igs_table_all["correct_decode"] == correct_decode)
            )
            igs_tmp = np.array(itemgetter(*np.where(idx_target)[0])(igs_all))
            mean_igs_tmp = igs_tmp.mean(axis=2)
            mean_ig = zscore(mean_igs_tmp.mean(axis=0))
            sd_ig = zscore(-mean_igs_tmp.std(axis=0, ddof=1))
            if logscale:
                mean_ig -= mean_ig.min() + 1e-5
                sd_ig -= sd_ig.min() + 1e-5

            axes[tasks.index(task), label].imshow(img)
            axes[tasks.index(task), label].scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                s=5,
                c=mean_ig if not show_sd else sd_ig,
                **kwargs,
            )
            axes[tasks.index(task), label].axis("off")
            # axes[tasks.index(task), label].set_title(f"{task} {color}")
            # axes[tasks.index(task), label].colorbar()
            avg_igs.append(mean_ig if not show_sd else sd_ig)
            color_ig.append(mean_ig if not show_sd else sd_ig)
        color_ig = np.array(color_ig)  # (6, 128)
        color_igs.append(color_ig)
    (save_path / "montage").mkdir(exist_ok=True, parents=True)
    eval_text = "sd" if show_sd else "mean"
    show_avg_text = "_with_avg" if show_avg else ""
    plt.savefig(
        save_path
        / "montage"
        / f"{eval_text}_montage_combine_sd{cmax}_{model}{show_avg_text}.png"
    )
    plt.clf()
    plt.close()

    alpha = 0.05
    avg_igs = np.array(avg_igs)
    corr_mat = np.corrcoef(avg_igs)
    corr_mat[np.diag_indices_from(corr_mat)] = np.nan
    p_value = np.zeros((len(tasks), len(tasks)))
    p_value[np.diag_indices_from(p_value)] = np.nan
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            if i <= j:
                continue
            print(f"{tasks[i]} vs {tasks[j]}: {corr_mat[i, j]}")
            corr_mat[i, j], p_value[i, j] = pearsonr(
                avg_igs[i],
                avg_igs[j],
            )
            corr_mat[j, i] = corr_mat[i, j]
            p_value[j, i] = p_value[i, j]
    p_vec_before_corrected = p_value[np.triu_indices_from(p_value, k=1)]
    rejects, p_vec_after_corrected, _, _ = multipletests(
        p_vec_before_corrected, method="bonferroni", alpha=alpha
    )
    # plot correlation matrix
    cmax_corr = 1.0
    plt.figure(figsize=(7 * cm_to_inch, 7 * cm_to_inch))
    plt.imshow(corr_mat, cmap="jet", vmin=-cmax_corr, vmax=cmax_corr)
    plt.colorbar()
    # mark significant correlation
    for i_reject, reject in enumerate(rejects):
        if reject:
            i, j = (
                np.triu_indices_from(p_value, k=1)[0][i_reject],
                np.triu_indices_from(p_value, k=1)[1][i_reject],
            )
            if p_vec_after_corrected[i_reject] < 0.001:
                marker = "***"
            elif p_vec_after_corrected[i_reject] < 0.01:
                marker = "**"
            elif p_vec_after_corrected[i_reject] < 0.05:
                marker = "*"
            else:
                raise ValueError("p value must be < 0.05")
            plt.text(
                j,
                i,
                marker,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=10,
                color="white",
            )
            plt.text(
                i,
                j,
                marker,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=10,
                color="white",
            )
    # top, right axis on
    plt.gca().spines["top"].set_visible(True)
    plt.gca().spines["right"].set_visible(True)
    # tick off
    plt.tick_params(
        bottom=False,
        left=False,
        right=False,
        top=False,
        labelbottom=False,
        labelleft=False,
    )
    plt.tight_layout()
    plt.savefig(
        save_path
        / "montage"
        / f"montage_corr_between_speech_{eval_text}_{model}_only_avg.png"
    )
    plt.clf()
    plt.close()

    color_igs = np.array(color_igs)  # (3, 6, 128)
    corr_mats = []
    p_values = []
    for i in range(color_igs.shape[1]):
        corr_mat = np.zeros((len(tasks), len(tasks)))
        corr_mat[np.diag_indices_from(corr_mat)] = np.nan
        p_value = np.zeros((len(tasks), len(tasks)))
        p_value[np.diag_indices_from(p_value)] = np.nan
        for j in range(len(tasks)):
            for k in range(len(tasks)):
                if j <= k:
                    continue
                corr_mat[j, k], p_value[j, k] = pearsonr(
                    color_igs[j, i, :],
                    color_igs[k, i, :],
                )
                corr_mat[k, j] = corr_mat[j, k]
                p_value[k, j] = p_value[j, k]
        corr_mats.append(corr_mat)
        p_values.append(p_value)
    corr_mats = np.array(corr_mats)  # (6, 3, 3)
    p_values = np.array(p_values)  # (6, 3, 3)
    p_values_corrected = []
    rejects = []
    for p_value in p_values:
        p_vec_before_corrected = p_value[np.triu_indices_from(p_value, k=1)]
        reject, p_vec_after_corrected, _, _ = multipletests(
            p_vec_before_corrected, method="bonferroni", alpha=alpha
        )
        rejects.append(reject)
        p_value_corrected = np.zeros((len(tasks), len(tasks)))
        p_value_corrected[np.triu_indices_from(p_value_corrected, k=1)] = (
            p_vec_after_corrected
        )
        p_value_corrected[np.tril_indices_from(p_value_corrected, k=-1)] = (
            p_vec_after_corrected
        )
        p_values_corrected.append(p_value_corrected)
    p_values_corrected = np.array(p_values_corrected)  # (6, 3, 3)
    rejects = np.array(rejects)  # (6, 3)
    # plot correlation matrix
    labels = colors + ["avg."]
    fig, axes = plt.subplots(
        1,
        p_values_corrected.shape[0],
        tight_layout=True,
        figsize=(p_values_corrected.shape[0] * 3 * cm_to_inch, 3 * cm_to_inch),
    )
    for i, (corr_mat, reject, p_value_corrected) in enumerate(
        zip(corr_mats, rejects, p_values_corrected)
    ):
        axes[i].imshow(corr_mat, cmap="jet", vmin=-1, vmax=1)
        # axes[i].set_title(f"{labels[i]}")
        # mark significant correlation
        for i_reject, reject in enumerate(reject):
            if reject:
                j, k = (
                    np.triu_indices_from(p_value_corrected, k=1)[0][i_reject],
                    np.triu_indices_from(p_value_corrected, k=1)[1][i_reject],
                )
                if p_value_corrected[j, k] < 0.001:
                    marker = "***"
                elif p_value_corrected[j, k] < 0.01:
                    marker = "**"
                elif p_value_corrected[j, k] < 0.05:
                    marker = "*"
                else:
                    raise ValueError("p value must be < 0.05")
                axes[i].text(
                    k,
                    j,
                    marker,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=10,
                    color="white",
                )
                axes[i].text(
                    j,
                    k,
                    marker,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=10,
                    color="white",
                )
        # top, right axis on
        axes[i].spines["top"].set_visible(True)
        axes[i].spines["right"].set_visible(True)
        # tick off
        axes[i].tick_params(
            bottom=False,
            left=False,
            right=False,
            top=False,
            labelbottom=False,
            labelleft=False,
        )
    plt.savefig(
        save_path / "montage" / f"montage_corr_between_speech_{eval_text}_{model}.png"
    )
    plt.clf()
    plt.close()

    if compare_with_emgs:
        corr_mats = np.zeros((color_igs.shape[1], len(tasks), len(mi_emgs)))
        p_mats = np.zeros((color_igs.shape[1], len(tasks), len(mi_emgs)))
        for i in range(color_igs.shape[1]):
            for j in range(len(tasks)):
                for k in range(len(mi_emgs)):
                    corr_mats[i, j, k], p_mats[i, j, k] = pearsonr(
                        color_igs[j, i, :],
                        mi_emgs[k],
                    )
        # plot correlation matrix
        labels = colors + ["avg."]
        fig, axes = plt.subplots(
            1,
            p_mats.shape[0],
            tight_layout=True,
            figsize=(p_mats.shape[0] * 3 * cm_to_inch, 3 * cm_to_inch),
        )
        for i, (corr_mat, p_mat) in enumerate(zip(corr_mats, p_mats)):
            axes[i].imshow(corr_mat, cmap="jet", vmin=-1, vmax=1)
            p_1d = p_mat.flatten()
            rejects, p_vec_after_corrected, _, _ = multipletests(
                p_1d, method="bonferroni", alpha=alpha
            )
            p_mat = p_vec_after_corrected.reshape(p_mat.shape)
            # axes[i].set_title(f"{labels[i]}")
            # mark significant correlation
            for j in range(len(tasks)):
                for k in range(len(mi_emgs)):
                    if p_mat[j, k] < 0.001:
                        marker = "***"
                    elif p_mat[j, k] < 0.01:
                        marker = "**"
                    elif p_mat[j, k] < 0.05:
                        marker = "*"
                    else:
                        marker = None
                    axes[i].text(
                        k,
                        j,
                        marker,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=10,
                        color="white",
                    )
            # top, right axis on
            axes[i].spines["top"].set_visible(True)
            axes[i].spines["right"].set_visible(True)
            # tick off
            axes[i].tick_params(
                bottom=False,
                left=False,
                right=False,
                top=False,
                labelbottom=False,
                labelleft=False,
            )
        plt.savefig(
            save_path / "montage" / f"montage_corr_emgs_{eval_text}_{model}.png"
        )
        plt.clf()
        plt.close()


def imshow_jaccard_indices_and_statistical_results(
    time_series: np.ndarray,
    time_series_all: List[np.ndarray],
    labels: List[str],
    corr_type: str,
    corr_trial_base: bool,
    threshold: float,
    ax: mpl.axes.Axes,
    fontsize_star: float,
):
    """imshow function for jaccard indices and statistical results used in imshow_temporal_jaccard_indices

    Args:
        time_series np.ndarray: time series
        time_series_all (List[np.ndarray]): time series all
        labels (List[str]): labels
        corr_type (str): corr type
        corr_trial_base (bool): corr trial base
        threshold (float): threshold
        ax (mpl.axes.Axes): ax
        fontsize_star (float): fontsize star

    Raises:
        ValueError: [description]
    """
    num_iter = 10000
    alpha = 0.05
    reorder_names = [
        "ig_eeg w/ adapt filt",
        "ig_eeg w/o adapt filt",
        "ig_EOG",
        "ig_EMG_upper",
        "ig_EMG_lower",
    ]
    reorder_indices = []
    for name in reorder_names:
        if name not in labels:
            raise ValueError(f"{name} is not in labels")
        else:
            reorder_indices.append(labels.index(name))
    (
        time_series[0],
        time_series[1],
        time_series[2],
        time_series[3],
        time_series[4],
    ) = (
        time_series[reorder_indices[0]],
        time_series[reorder_indices[1]],
        time_series[reorder_indices[2]],
        time_series[reorder_indices[3]],
        time_series[reorder_indices[4]],
    )
    (
        time_series_all[0],
        time_series_all[1],
        time_series_all[2],
        time_series_all[3],
        time_series_all[4],
    ) = (
        time_series_all[reorder_indices[0]],
        time_series_all[reorder_indices[1]],
        time_series_all[reorder_indices[2]],
        time_series_all[reorder_indices[3]],
        time_series_all[reorder_indices[4]],
    )
    (
        labels[0],
        labels[1],
        labels[2],
        labels[3],
        labels[4],
    ) = (
        labels[reorder_indices[0]],
        labels[reorder_indices[1]],
        labels[reorder_indices[2]],
        labels[reorder_indices[3]],
        labels[reorder_indices[4]],
    )
    time_series_surrogate = np.zeros(
        (num_iter, time_series.shape[0], time_series.shape[1])
    )
    num_repeat = 50
    time_series_all_surrogate = [
        np.array(
            [
                np.random.permutation(tmp.transpose(1, 0)).transpose(1, 0)
                for _ in range(num_repeat)
            ]
        )
        for tmp in time_series_all
    ]  # tmp: shape (n_trial, 320)
    for i in range(num_iter):
        time_series_surrogate[i] = np.random.permutation(
            time_series.transpose(1, 0)
        ).transpose(1, 0)
    jaccard_indices = np.ones((len(time_series), len(time_series)))
    jaccard_indices[:, :] = np.nan
    jaccard_indices_all = np.ones(
        (time_series_all[0].shape[0], len(time_series), len(time_series))
    )
    jaccard_indices_all[:, :, :] = np.nan
    for i, j in itertools.combinations(np.arange(len(time_series)), 2):
        if corr_type == "jaccard":
            jaccard = jaccard_index(
                np.where(time_series[i] > threshold)[0],
                np.where(time_series[j] > threshold)[0],
            )
            jaccard_all = np.array(
                [
                    jaccard_index(
                        np.where(time_series_all[i][k, :] > threshold)[0],
                        np.where(time_series_all[j][k, :] > threshold)[0],
                    )
                    for k in range(time_series_all[i].shape[0])
                ]
            )  # (n_trial,)
        elif corr_type == "pearson":
            jaccard = np.corrcoef(time_series[i], time_series[j])[0, 1]
            jaccard_all = np.array(
                [
                    np.corrcoef(time_series_all[i][k, :], time_series_all[j][k, :])[
                        0, 1
                    ]
                    for k in range(time_series_all[i].shape[0])
                ]
            )
        jaccard_indices[i, j] = jaccard
        jaccard_indices[j, i] = jaccard
        jaccard_indices_all[:, i, j] = jaccard_all
        jaccard_indices_all[:, j, i] = jaccard_all
    jaccard_indices_all_mean = np.nanmean(jaccard_indices_all, axis=0)
    vmin = 0 if corr_type == "jaccard" else -1
    im = ax.imshow(
        jaccard_indices_all_mean if corr_trial_base else jaccard_indices,
        cmap="jet",
        vmin=vmin,
        vmax=1,
    )
    # jaccard for surrogate
    jaccard_indices_surrogate = np.ones((num_iter, len(time_series), len(time_series)))
    jaccard_indices_surrogate[:, :, :] = np.nan
    jaccard_indices_all_surrogate = np.ones(
        (
            num_repeat,
            time_series_all[0].shape[0],
            len(time_series),
            len(time_series),
        )
    )
    cprint("start compute surrogate jaccard", "cyan")
    for i, j in itertools.combinations(np.arange(len(time_series)), 2):
        for k in tqdm(range(num_iter)):
            if corr_type == "jaccard":
                jaccard = jaccard_index(
                    np.where(time_series_surrogate[k, i] > threshold)[0],
                    np.where(time_series[j] > threshold)[0],
                )
            elif corr_type == "pearson":
                jaccard = np.corrcoef(time_series_surrogate[k, i], time_series[j])[0, 1]
            jaccard_indices_surrogate[k, i, j] = jaccard
            jaccard_indices_surrogate[k, j, i] = jaccard
        for k in range(num_repeat):
            if corr_type == "jaccard":
                jaccard_all = np.array(
                    [
                        jaccard_index(
                            np.where(time_series_all_surrogate[i][k, l, :] > threshold)[
                                0
                            ],
                            np.where(time_series_all[j][l, :] > threshold)[0],
                        )
                        for l in tqdm(range(time_series_all[0].shape[0]))
                    ]
                )
            elif corr_type == "pearson":
                jaccard_all = np.array(
                    [
                        np.corrcoef(
                            time_series_all_surrogate[i][k, l, :],
                            time_series_all[j][l, :],
                        )[0, 1]
                        for l in tqdm(range(time_series_all[0].shape[0]))
                    ]
                )
            jaccard_indices_all_surrogate[k, :, i, j] = jaccard_all
            jaccard_indices_all_surrogate[k, :, j, i] = jaccard_all
    jaccard_indices_all_surrogate = np.concatenate(
        jaccard_indices_all_surrogate, axis=0
    )  # n_surrogate, n_timeseries, n_timeseries
    cprint(jaccard_indices_all_surrogate.shape, "green")
    # compute p value
    p_values = np.ones((len(time_series), len(time_series)))
    p_values[:, :] = np.nan
    p_values_before_corrected = []
    p_values_all_before_corrected = []
    for i, j in itertools.combinations(np.arange(len(time_series)), 2):
        p_value = (
            np.sum(jaccard_indices_surrogate[:, i, j] > jaccard_indices[i, j]) + 1
        ) / (num_iter + 1)
        p_value_all = (
            np.sum(
                jaccard_indices_all_surrogate[:, i, j] > jaccard_indices_all_mean[i, j]
            )
        ) / (len(jaccard_indices_all_surrogate) + 1)
        p_values_before_corrected.append(p_value)
        p_values_all_before_corrected.append(p_value_all)
    p_values_before_corrected = np.array(p_values_before_corrected)
    rejects, p_values_corrected, _, _ = multipletests(
        p_values_before_corrected, method="bonferroni", alpha=alpha
    )
    p_values_all_before_corrected = np.array(p_values_all_before_corrected)
    rejects_all, p_values_all_corrected, _, _ = multipletests(
        p_values_all_before_corrected, method="bonferroni", alpha=alpha
    )
    significant = np.ones((len(time_series), len(time_series)))
    significant[:, :] = np.nan
    for i_iter, (i, j) in enumerate(
        itertools.combinations(np.arange(len(time_series)), 2)
    ):
        p_values[i, j] = (
            p_values_all_corrected[i_iter]
            if corr_trial_base
            else p_values_corrected[i_iter]
        )
        p_values[j, i] = (
            p_values_all_corrected[i_iter]
            if corr_trial_base
            else p_values_corrected[i_iter]
        )
        significant[i, j] = rejects_all[i_iter] if corr_trial_base else rejects[i_iter]
        significant[j, i] = rejects_all[i_iter] if corr_trial_base else rejects[i_iter]
    significant = np.where(significant == True)

    for i, j in zip(*significant):
        ax.text(
            j,
            i,
            "*",
            ha="center",
            va="center",
            color="k",
            fontsize=fontsize_star,
            fontweight="bold",
        )
    # ax.set_xticks(np.arange(len(time_series)))
    # ax.set_yticks(np.arange(len(time_series)))
    # ax.set_xticklabels(labels, fontsize=fontsize_tick)
    # ax.set_yticklabels(labels, fontsize=fontsize_tick)
    # ax.tick_params(axis="x", rotation=90)
    # ax.tick_params(axis="y", rotation=0)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    # ax.set_title(f"{task}_{color}", fontsize=fontsize_tick)
    # fig.colorbar(im, ax=ax, orientation="vertical")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    # ax.axis("off")


def imshow_temporal_jaccard_indices(
    igs_norm_all: List[np.ndarray],
    igs_table_all: pd.DataFrame,
    eegs_norm_all: List[np.ndarray],
    eegs_table_all: pd.DataFrame,
    emgs_norm_all: List[np.ndarray],
    speeches_norm_all: List[np.ndarray],
    colors: List[tuple],
    models: List[str],
    tasks: List[str],
    save_path: Path,
    ephys_norm_type: str = "mean",
    top_k: int = 10,
    w_size: float = 6.0,
    h_size: float = 3.5,
    fontsize_legend: float = 3.5,
    fontsize_tick: float = 3.5,
    fontsize_star: float = 5.0,
    threshold: float = 0,
    only_ig: bool = False,
    corr_type: str = "jaccard",
    corr_trial_base: bool = True,
) -> None:
    """plot ig temporal

    Args:
        igs_norm_all (List[np.ndarray]): ig norm
        igs_table_all (pd.DataFrame): ig table
        eegs_norm_all (List[np.ndarray]): eeg norm
        eegs_table_all (pd.DataFrame): eeg table
        emgs_norm_all (List[np.ndarray]): emg norm
        speeches_norm_all (List[np.ndarray]): speech norm
        colors (List[tuple]): color
        models (List[str]): models
        tasks (List[str]): tasks
        save_path (Path): save path
        ephys_norm_type (str, optional): ephys norm type. Defaults to "mean".
        top_k (int, optional): top k. Defaults to 10.
        w_size (float, optional): width size. Defaults to 6.0.
        h_size (float, optional): height size. Defaults to 3.5.
        fontsize_legend (float, optional): fontsize legend. Defaults to 3.5.
        fontsize_tick (float, optional): fontsize tick. Defaults to 3.5.
        fontsize_star (float, optional): fontsize star. Defaults to 5.0.
        threshold (float, optional): threshold. Defaults to 0.5.
        only_ig (bool, optional): only ig or not. Defaults to False.
        corr_type (str, optional): corr type. Defaults to "jaccard".
        corr_trial_base (bool, optional): corr trial base or not. Defaults to False.
    """
    # T = 320
    # jaccards = []
    # cprint("start compute chance level for jaccard", "cyan")
    # for _ in tqdm(range(num_iter)):
    #     a = np.random.randn(T)
    #     b = np.random.randn(T)
    #     jaccard = jaccard_index(
    #         np.where(a > threshold)[0],
    #         np.where(b > threshold)[0],
    #     )
    #     jaccards.append(jaccard)
    # jaccards = np.array(jaccards)
    # upper = np.percentile(jaccards, 97.5)
    # lower = np.percentile(jaccards, 2.5)
    T = igs_norm_all[0].shape[-1]
    n_timeseries = 5  # EEGNet, EEGNet_wo_adapt_filt, EMG (3 ch)
    if not only_ig:
        n_timeseries += 5
    igs_table_all = igs_table_all.assign(EEGNet_correct_decode=0)
    eegnet_correct_decode = igs_table_all["correct_decode"].values[
        igs_table_all["model"] == "EEGNet"
    ]
    models = igs_table_all["model"].unique()
    for model in models:
        igs_table_all["EEGNet_correct_decode"].values[
            igs_table_all["model"] == model
        ] = eegnet_correct_decode

    fig, axes = plt.subplots(
        len(tasks),
        len(colors) + 1,
        tight_layout=True,
        figsize=(len(colors) * w_size * cm_to_inch, len(tasks) * h_size * cm_to_inch),
    )
    for task in tasks:
        mean_time_series_all = np.empty((n_timeseries, 0, T))
        for label, color in enumerate(colors):
            ax = axes[tasks.index(task), label]
            time_series = []
            time_series_all = []
            labels = []
            mean_time_series_all_tmp = []
            for model in models:
                # idx_target = (
                #     (igs_table_all["model"] == model)
                #     & (igs_table_all["label"] == label)
                #     & (igs_table_all["task"] == task)
                #     # & (igs_table_all["correct_decode"] == True)
                # )
                # idx_target_correct = (
                #     (igs_table_all["model"] == model)
                #     & (igs_table_all["label"] == label)
                #     & (igs_table_all["task"] == task)
                #     & (igs_table_all["correct_decode"] == True)
                # )
                idx_target = (
                    (igs_table_all["model"] == model)
                    & (igs_table_all["label"] == label)
                    & (igs_table_all["task"] == task)
                    & (igs_table_all["EEGNet_correct_decode"] == True)
                )
                idx_ephys_target = (eegs_table_all["label"] == label) & (
                    eegs_table_all["task"] == task
                )
                if idx_target.sum() == 0:
                    cprint(
                        (
                            f"no data for model: {model}, color: {color}, "
                            f"task: {task}"
                        ),
                        "yellow",
                    )
                    continue
                cprint(
                    (
                        f"model: {model}, color: {color}, task: {task} "
                        f"sum {idx_target.sum()}"
                    ),
                    "cyan",
                )
                igs_tmp = np.array(itemgetter(*np.where(idx_target)[0])(igs_norm_all))
                igs_tmp_spatial = np.mean(igs_tmp, axis=(0, 2))
                igs_tmp_all = deepcopy(igs_tmp)
                igs_tmp = np.mean(igs_tmp, axis=0)  # (128, 320)
                eegs_tmp = np.array(
                    itemgetter(*np.where(idx_ephys_target)[0])(eegs_norm_all)
                )
                eegs_tmp_all = deepcopy(eegs_tmp)
                eegs_tmp = np.mean(eegs_tmp, axis=0)  # (128, 320)
                emgs_tmp = np.array(
                    itemgetter(*np.where(idx_ephys_target)[0])(emgs_norm_all)
                )
                emgs_tmp_all = deepcopy(emgs_tmp)
                emgs_tmp = np.mean(emgs_tmp, axis=0)  # (3, 320)
                if "EMG" in model:
                    type_ = "emg"
                elif "wo_adapt_filt" in model:
                    type_ = "eeg w/o adapt filt"
                else:
                    type_ = "eeg w/ adapt filt"
                if type_ == "emg":
                    for idx_emg, (emg, ig_emg) in enumerate(zip(emgs_tmp, igs_tmp)):
                        if idx_emg == 0:
                            emg_name = "EOG"
                        elif idx_emg == 1:
                            emg_name = "EMG_upper"
                        elif idx_emg == 2:
                            emg_name = "EMG_lower"
                        else:
                            raise ValueError("idx_emg must be 0, 1 or 2")
                        if not only_ig:
                            time_series.append(emg)
                            time_series_all.append(emgs_tmp_all[:, idx_emg, :])
                            labels.append(f"{emg_name}")
                            mean_time_series_all_tmp.append(emgs_tmp_all[:, idx_emg, :])
                        # time_series.append(ig_emg)
                        time_series.append(zscore(ig_emg))
                        time_series_all.append(
                            zscore(igs_tmp_all[:, idx_emg, :], axis=1)
                        )
                        labels.append(f"ig_{emg_name}")
                        mean_time_series_all_tmp.append(
                            zscore(igs_tmp_all[:, idx_emg, :], axis=1)
                        )
                else:
                    if ephys_norm_type == "mean":
                        eeg_tmp = zscore(eegs_tmp.mean(axis=0))
                        eeg_tmp_all = zscore(eegs_tmp_all.mean(axis=1), axis=1)
                        ig_tmp = zscore(igs_tmp.mean(axis=0))
                        ig_tmp_all = zscore(igs_tmp_all.mean(axis=1), axis=1)
                    elif ephys_norm_type == "weighted":
                        eeg_tmp = zscore(
                            np.average(eegs_tmp, axis=0, weights=igs_tmp_spatial)
                        )
                        eeg_tmp_all = zscore(
                            np.average(eegs_tmp_all, axis=1, weights=igs_tmp_spatial),
                            axis=1,
                        )
                        ig_tmp = zscore(
                            np.average(igs_tmp, axis=0, weights=igs_tmp_spatial)
                        )
                        ig_tmp_all = zscore(
                            np.average(igs_tmp_all, axis=1, weights=igs_tmp_spatial),
                            axis=1,
                        )
                    elif ephys_norm_type == "top_k":
                        topk_indices = np.argsort(igs_tmp_spatial)[::-1][:top_k]
                        eeg_tmp = zscore(
                            np.mean(
                                eegs_tmp[topk_indices],
                                axis=0,
                            )
                        )
                        eeg_tmp_all = zscore(
                            np.mean(
                                eegs_tmp_all[:, topk_indices, :],
                                axis=1,
                            ),
                            axis=1,
                        )
                        ig_tmp = zscore(
                            np.mean(
                                igs_tmp[topk_indices],
                                axis=0,
                            )
                        )
                        ig_tmp_all = zscore(
                            np.mean(
                                igs_tmp_all[:, topk_indices, :],
                                axis=1,
                            ),
                            axis=1,
                        )
                    else:
                        raise ValueError("ephys_norm_type must be mean or weighted")
                    if not only_ig:
                        time_series.append(eeg_tmp)
                        time_series_all.append(eeg_tmp_all)
                        labels.append(f"{type_}")
                        mean_time_series_all_tmp.append(eeg_tmp_all)
                    time_series.append(ig_tmp)
                    time_series_all.append(ig_tmp_all)
                    labels.append(f"ig_{type_}")
                    mean_time_series_all_tmp.append(ig_tmp_all)
            speech = np.array(
                itemgetter(*np.where(idx_ephys_target)[0])(speeches_norm_all)
            )
            speech_all = deepcopy(speech)
            speech = np.mean(speech, axis=0)
            if not only_ig:
                time_series.append(speech)
                time_series_all.append(speech_all)
                labels.append("speech")
                mean_time_series_all_tmp.append(speech_all)
            time_series = np.array(time_series)
            imshow_jaccard_indices_and_statistical_results(
                time_series,
                time_series_all,
                labels,
                corr_type,
                corr_trial_base,
                threshold,
                ax,
                fontsize_star,
            )
            mean_time_series_all_tmp = np.array(
                mean_time_series_all_tmp
            )  # (n_timeseries, n_trial, T)
            mean_time_series_all = np.concatenate(
                [mean_time_series_all, mean_time_series_all_tmp], axis=1
            )
        mean_time_series = np.mean(mean_time_series_all, axis=1)  # (n_timeseries, T)
        ax = axes[tasks.index(task), -1]
        imshow_jaccard_indices_and_statistical_results(
            mean_time_series,
            mean_time_series_all,
            labels,
            corr_type,
            corr_trial_base,
            threshold,
            ax,
            fontsize_star,
        )

    if only_ig:
        only_ig_txt = "_only_ig"
    else:
        only_ig_txt = ""
    if corr_trial_base:
        corr_trial_base_txt = "_trial_base"
    else:
        corr_trial_base_txt = ""
    fig.savefig(
        save_path
        / f"combined_{corr_type}_indices_{ephys_norm_type}{only_ig_txt}{corr_trial_base_txt}.png"
    )
    plt.clf()
    plt.close()


def imshow_jaccard_indices_between_tasks_and_statistical_results(
    time_series: np.ndarray,
    ax: mpl.axes.Axes,
    num_iter: int,
    alpha: float,
    corr_type: str,
    threshold: float,
    fontsize_star: float,
    fig: Optional[mpl.figure.Figure] = None,
):
    """imshow function for jaccard indices and statistical results used in imshow_temporal_jaccard_indices_between_tasks

    Args:
        time_series (np.ndarray): time series
        ax (mpl.axes.Axes): ax
        num_iter (int): num iter
        alpha (float): alpha
        corr_type (str): corr type
        threshold (float): threshold
        fontsize_star (float): fontsize star
        fig (Optional[mpl.figure.Figure], optional): fig. Defaults to None.
    """
    jaccard_indices = np.ones((len(time_series), len(time_series)))
    jaccard_indices[:, :] = np.nan
    jaccard_indices_surrogate = np.ones((num_iter, len(time_series), len(time_series)))
    jaccard_indices_surrogate[:, :, :] = np.nan
    for i, j in itertools.combinations(np.arange(len(time_series)), 2):
        if corr_type == "jaccard":
            jaccard = jaccard_index(
                np.where(time_series[i] > threshold)[0],
                np.where(time_series[j] > threshold)[0],
            )
            for i_iter in range(num_iter):
                jaccard_surrogate = jaccard_index(
                    np.where(np.random.permutation(time_series[i]) > threshold)[0],
                    np.where(time_series[j] > threshold)[0],
                )
                jaccard_indices_surrogate[i_iter, i, j] = jaccard_surrogate
                jaccard_indices_surrogate[i_iter, j, i] = jaccard_surrogate
        elif corr_type == "pearson":
            jaccard = np.corrcoef(time_series[i], time_series[j])[0, 1]
            for i_iter in range(num_iter):
                jaccard_surrogate = np.corrcoef(
                    np.random.permutation(time_series[i]),
                    time_series[j],
                )[0, 1]
                jaccard_indices_surrogate[i_iter, i, j] = jaccard_surrogate
                jaccard_indices_surrogate[i_iter, j, i] = jaccard_surrogate
        jaccard_indices[i, j] = jaccard
        jaccard_indices[j, i] = jaccard
    vmin = 0 if corr_type == "jaccard" else -1
    im = ax.imshow(
        jaccard_indices,
        cmap="jet",
        vmin=vmin,
        vmax=1,
    )
    # fig.colorbar(im, ax=ax, orientation="vertical")
    # compute p value
    p_values = np.ones((len(time_series), len(time_series)))
    p_values[:, :] = np.nan
    p_values_before_corrected = []
    for i, j in itertools.combinations(np.arange(len(time_series)), 2):
        p_value = (
            np.sum(jaccard_indices_surrogate[:, i, j] > jaccard_indices[i, j]) + 1
        ) / (num_iter + 1)
        p_values_before_corrected.append(p_value)
    p_values_before_corrected = np.array(p_values_before_corrected)
    rejects, p_values_corrected, _, _ = multipletests(
        p_values_before_corrected, method="bonferroni", alpha=alpha
    )
    significant = np.ones((len(time_series), len(time_series)))
    significant[:, :] = np.nan
    for i_iter, (i, j) in enumerate(
        itertools.combinations(np.arange(len(time_series)), 2)
    ):
        p_values[i, j] = p_values_corrected[i_iter]
        p_values[j, i] = p_values_corrected[i_iter]
        significant[i, j] = rejects[i_iter]
        significant[j, i] = rejects[i_iter]
    significant = np.where(significant == True)

    for i, j in zip(*significant):
        if p_values[i, j] < 0.001:
            marker = "***"
        elif p_values[i, j] < 0.01:
            marker = "**"
        elif p_values[i, j] < 0.05:
            marker = "*"
        ax.text(
            j,
            i,
            marker,
            ha="center",
            va="center",
            color="k",
            fontsize=fontsize_star,
            fontweight="bold",
        )
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


def imshow_temporal_jaccard_indices_between_tasks(
    igs_norm_all: List[np.ndarray],
    igs_table_all: pd.DataFrame,
    colors: List[tuple],
    model: str,
    tasks: List[str],
    save_path: Path,
    norm_type: str = "mean",
    top_k: int = 10,
    w_size: float = 6.0,
    h_size: float = 3.5,
    fontsize_legend: float = 3.5,
    fontsize_tick: float = 3.5,
    fontsize_star: float = 5.0,
    threshold: float = 0,
    corr_type: str = "jaccard",
) -> None:
    """plot ig temporal

    Args:
        igs_norm_all (List[np.ndarray]): ig norm
        igs_table_all (pd.DataFrame): ig table
        colors (List[tuple]): color
        models (List[str]): models
        tasks (List[str]): tasks
        save_path (Path): save path
        norm_type (str, optional): norm type. Defaults to "mean".
        top_k (int, optional): top k. Defaults to 10.
        w_size (float, optional): width size. Defaults to 6.0.
        h_size (float, optional): height size. Defaults to 3.5.
        fontsize_legend (float, optional): fontsize legend. Defaults to 3.5.
        fontsize_tick (float, optional): fontsize tick. Defaults to 3.5.
        fontsize_star (float, optional): fontsize star. Defaults to 5.0.
        threshold (float, optional): threshold. Defaults to 0.5.
        only_ig (bool, optional): only ig or not. Defaults to False.
        corr_type (str, optional): corr type. Defaults to "jaccard".
        corr_trial_base (bool, optional): corr trial base or not. Defaults to False.
    """
    # T = 320
    num_iter = 9999
    alpha = 0.05
    fig, axes = plt.subplots(
        1,
        len(colors) + 1,
        tight_layout=True,
        figsize=(len(colors) * w_size * cm_to_inch, len(tasks) * h_size * cm_to_inch),
    )

    for label, color in enumerate(colors):
        ax = axes[label]
        time_series = []
        labels = []
        for task in tasks:
            idx_target = (
                (igs_table_all["model"] == model)
                & (igs_table_all["label"] == label)
                & (igs_table_all["task"] == task)
                & (igs_table_all["correct_decode"] == True)
            )
            cprint(
                (
                    f"model: {model}, color: {color}, task: {task} "
                    f"sum {idx_target.sum()}"
                ),
                "cyan",
            )
            igs_tmp = np.array(itemgetter(*np.where(idx_target)[0])(igs_norm_all))
            igs_tmp_spatial = np.mean(igs_tmp, axis=(0, 2))
            igs_tmp = np.mean(igs_tmp, axis=0)  # (128, 320)
            if norm_type == "mean":
                ig_tmp = zscore(igs_tmp.mean(axis=0))
            elif norm_type == "weighted":
                ig_tmp = zscore(np.average(igs_tmp, axis=0, weights=igs_tmp_spatial))
            elif norm_type == "top_k":
                topk_indices = np.argsort(igs_tmp_spatial)[::-1][:top_k]
                ig_tmp = zscore(
                    np.mean(
                        igs_tmp[topk_indices],
                        axis=0,
                    )
                )
            time_series.append(ig_tmp)
            labels.append(f"{task}")

        imshow_jaccard_indices_between_tasks_and_statistical_results(
            np.array(time_series),
            ax,
            num_iter,
            alpha,
            corr_type,
            threshold,
            fontsize_star,
            # fig,
        )

        # mean
        ax = axes[-1]
        time_series = []
        labels = []
        for task in tasks:
            idx_target = (
                (igs_table_all["model"] == model)
                & (igs_table_all["task"] == task)
                & (igs_table_all["correct_decode"] == True)
            )
            cprint((f"model: {model}, task: {task} " f"sum {idx_target.sum()}"), "cyan")
            igs_tmp = np.array(itemgetter(*np.where(idx_target)[0])(igs_norm_all))
            igs_tmp_spatial = np.mean(igs_tmp, axis=(0, 2))
            igs_tmp = np.mean(igs_tmp, axis=0)  # (128, 320)
            if norm_type == "mean":
                ig_tmp = zscore(igs_tmp.mean(axis=0))
            elif norm_type == "weighted":
                ig_tmp = zscore(np.average(igs_tmp, axis=0, weights=igs_tmp_spatial))
            elif norm_type == "top_k":
                topk_indices = np.argsort(igs_tmp_spatial)[::-1][:top_k]
                ig_tmp = zscore(
                    np.mean(
                        igs_tmp[topk_indices],
                        axis=0,
                    )
                )
            time_series.append(ig_tmp)
            labels.append(f"{task}")
        imshow_jaccard_indices_between_tasks_and_statistical_results(
            np.array(time_series),
            ax,
            num_iter,
            alpha,
            corr_type,
            threshold,
            fontsize_star,
            # fig,
        )

    fig.savefig(
        save_path / f"combined_between_speech_{corr_type}_indices_{norm_type}.png"
    )
    plt.clf()
    plt.close()


def plot_temporal_representative(
    igs_norm_all: List[np.ndarray],
    igs_table_all: pd.DataFrame,
    eegs_norm_all: List[np.ndarray],
    eegs_table_all: pd.DataFrame,
    emgs_norm_all: List[np.ndarray],
    speeches_norm_all: List[np.ndarray],
    colors: List[tuple],
    models: List[str],
    tasks: List[str],
    save_path: Path,
    ephys_norm_type: str = "mean",
    top_k: int = 10,
    w_size: float = 6.0,
    h_size: float = 3.5,
    fontsize_legend: float = 3.5,
    fontsize_tick: float = 3.5,
    lw: float = 0.8,
    only_ig: bool = False,
) -> None:
    """plot ig temporal

    Args:
        igs_norm_all (List[np.ndarray]): ig norm
        igs_table_all (pd.DataFrame): ig table
        eegs_norm_all (List[np.ndarray]): eeg norm
        eegs_table_all (pd.DataFrame): eeg table
        emgs_norm_all (List[np.ndarray]): emg norm
        speeches_norm_all (List[np.ndarray]): speech norm
        colors (List[tuple]): color
        models (List[str]): models
        tasks (List[str]): tasks
        save_path (Path): save path
        ephys_norm_type (str, optional): ephys norm type. Defaults to "mean".
        top_k (int, optional): top k. Defaults to 10.
        w_size (float, optional): width size. Defaults to 6.0.
        h_size (float, optional): height size. Defaults to 3.5.
        fontsize_legend (float, optional): fontsize legend. Defaults to 3.5.
        fontsize_tick (float, optional): fontsize tick. Defaults to 3.5.
        lw (float, optional): linewidth. Defaults to 0.8.
        only_ig (bool, optional): only ig or not. Defaults to False.
    """
    igs_table_all = igs_table_all.assign(EEGNet_correct_decode=0)
    eegnet_correct_decode = igs_table_all["correct_decode"].values[
        igs_table_all["model"] == "EEGNet"
    ]
    models = igs_table_all["model"].unique()
    for model in models:
        igs_table_all["EEGNet_correct_decode"].values[
            igs_table_all["model"] == model
        ] = eegnet_correct_decode

    for task in tasks:
        for label, color in enumerate(colors):
            time_series = []
            labels = []
            for model in models:
                idx_target = (
                    (igs_table_all["model"] == model)
                    & (igs_table_all["label"] == label)
                    & (igs_table_all["task"] == task)
                    & (igs_table_all["EEGNet_correct_decode"] == True)
                )
                # idx_ephys_target = (eegs_table_all["label"] == label) & (
                #     eegs_table_all["task"] == task
                # )
                if idx_target.sum() == 0:
                    cprint(
                        (
                            f"no data for model: {model}, color: {color}, "
                            f"task: {task}"
                        ),
                        "yellow",
                    )
                    continue
                cprint(
                    (
                        f"model: {model}, color: {color}, task: {task} "
                        f"sum {idx_target.sum()}"
                    ),
                    "cyan",
                )
                igs_tmp = np.array(itemgetter(*np.where(idx_target)[0])(igs_norm_all))
                igs_tmp_spatial = np.mean(igs_tmp, axis=(0, 2))
                # eegs_tmp = np.array(
                #     itemgetter(*np.where(idx_ephys_target)[0])(eegs_norm_all)
                # )
                # emgs_tmp = np.array(
                #     itemgetter(*np.where(idx_ephys_target)[0])(emgs_norm_all)
                # )
                if "EMG" in model:
                    type_ = "emg"
                elif "wo_adapt_filt" in model:
                    type_ = "eeg w/o adapt filt"
                else:
                    type_ = "eeg w/ adapt filt"
                if type_ == "emg":
                    for idx_emg in range(igs_tmp.shape[1]):
                        # emg = emgs_tmp[:, idx_emg, :]
                        ig_emg = igs_tmp[:, idx_emg, :]
                        if idx_emg == 0:
                            emg_name = "EOG"
                        elif idx_emg == 1:
                            emg_name = "EMG_upper"
                        elif idx_emg == 2:
                            emg_name = "EMG_lower"
                        else:
                            raise ValueError("idx_emg must be 0, 1 or 2")
                        # if not only_ig:
                        # time_series.append(emg)
                        # labels.append(f"{emg_name}")
                        time_series.append(zscore(ig_emg, axis=1))
                        labels.append(f"ig_{emg_name}")
                else:
                    topk_indices = np.argsort(igs_tmp_spatial)[::-1][:top_k]
                    # eeg_tmp = zscore(eegs_tmp[:, topk_indices[0], :], axis=1)
                    ig_tmp = zscore(igs_tmp[:, topk_indices[0], :], axis=1)
                    # if not only_ig:
                    #     time_series.append(eeg_tmp)
                    #     labels.append(f"{type_}")
                    time_series.append(ig_tmp)
                    labels.append(f"ig_{type_}")
            # speech = np.array(
            #     itemgetter(*np.where(idx_ephys_target)[0])(speeches_norm_all)
            # )
            # if not only_ig:
            # time_series.append(speech)
            # labels.append("speech")
            # reorder the labels into ["ig_eeg w/ adapt filt", "ig_eeg w/o adapt filt", "ig_EOG", "ig_EMG_upper", "ig_EMG_lower"]
            reorder_names = [
                "ig_eeg w/ adapt filt",
                "ig_eeg w/o adapt filt",
                "ig_EOG",
                "ig_EMG_upper",
                "ig_EMG_lower",
            ]
            reorder_indices = []
            for name in reorder_names:
                if name not in labels:
                    raise ValueError(f"{name} is not in labels")
                else:
                    reorder_indices.append(labels.index(name))
            (
                time_series[0],
                time_series[1],
                time_series[2],
                time_series[3],
                time_series[4],
            ) = (
                time_series[reorder_indices[0]],
                time_series[reorder_indices[1]],
                time_series[reorder_indices[2]],
                time_series[reorder_indices[3]],
                time_series[reorder_indices[4]],
            )
            (
                labels[0],
                labels[1],
                labels[2],
                labels[3],
                labels[4],
            ) = (
                labels[reorder_indices[0]],
                labels[reorder_indices[1]],
                labels[reorder_indices[2]],
                labels[reorder_indices[3]],
                labels[reorder_indices[4]],
            )
            # index of "ig_eeg w/ adapt filt" in labels list
            idx = labels.index("ig_eeg w/ adapt filt")
            ig_eeg = time_series[idx]  # (n_trial, 320)
            ig_eeg_avg = np.mean(ig_eeg, axis=0)  # (320,)
            # extract representative trial (the most similar to average))
            corr = np.array(
                [
                    np.corrcoef(ig_eeg_avg, ig_eeg[i, :])[0, 1]
                    for i in range(ig_eeg.shape[0])
                ]
            )
            idx_representative = np.argmax(corr)
            # plot representative trial
            fig, ax = plt.subplots(
                1,
                1,
                tight_layout=True,
                figsize=(w_size * cm_to_inch, h_size * cm_to_inch),
            )
            representatives = []
            for i, (time_series_tmp, label) in enumerate(zip(time_series, labels)):
                ax.plot(
                    np.arange(len(time_series_tmp[idx_representative])) / 256,
                    time_series_tmp[idx_representative],
                    label=f"{label}",
                    linewidth=lw,
                    c=mpl.cm.tab10(i),
                )
                representatives.append(time_series_tmp[idx_representative])
            representatives = np.array(representatives)
            ax.legend(fontsize=fontsize_legend)
            ax.set_xlim(0, 1.25)
            ax.set_xticks([0, 0.5, 1])
            # ax.axis("off")
            ax.xaxis.set_tick_params(labelsize=fontsize_tick, length=1.3, width=0.3)
            ax.yaxis.set_tick_params(labelsize=fontsize_tick, length=1.3, width=0.3)
            # set xy axis width
            width_axis = 0.3
            ax.spines["bottom"].set_linewidth(width_axis)
            ax.spines["left"].set_linewidth(width_axis)
            # ax.set_xlabel("Time (s)", fontsize=fontsize_tick)
            # ax.set_ylabel("Z-scored Integrated Gradients", fontsize=fontsize_tick)
            # ax.ylim(-0.75, 0.75)
            # ax.axis("off")
            # Create scale bar
            fig.savefig(
                save_path / f"representative_trial_{task}_{color}_{ephys_norm_type}.png"
            )
            plt.clf()
            plt.close()

            jaccard_indices = np.ones((len(time_series), len(time_series)))
            jaccard_indices[:, :] = np.nan
            for i, j in itertools.combinations(np.arange(len(time_series)), 2):
                jaccard = jaccard_index(
                    np.where(representatives[i] > 0)[0],
                    np.where(representatives[j] > 0)[0],
                )
                jaccard_indices[i, j] = jaccard
                jaccard_indices[j, i] = jaccard
            fig, ax = plt.subplots(
                1,
                1,
                tight_layout=True,
                figsize=(3.35 * cm_to_inch, 3.35 * cm_to_inch),
            )
            im = ax.imshow(
                jaccard_indices,
                cmap="jet",
                vmin=0,
                vmax=1,
            )
            # set xy axis width
            width_axis = 0.3
            ax.spines["top"].set_linewidth(width_axis)
            ax.spines["right"].set_linewidth(width_axis)
            ax.spines["bottom"].set_linewidth(width_axis)
            ax.spines["left"].set_linewidth(width_axis)
            # ax.set_title(f"{task}_{color}", fontsize=fontsize_tick)
            # ax.set_xticks(np.arange(len(time_series)))
            # ax.set_yticks(np.arange(len(time_series)))
            # ax.set_xticklabels(labels, fontsize=fontsize_tick)
            # ax.set_yticklabels(labels, fontsize=fontsize_tick)
            # ax.tick_params(axis="x", rotation=90, width=width_axis, length=1.3)
            # ax.tick_params(axis="y", rotation=0, width=width_axis, length=1.3)
            ax.tick_params(labelbottom=False, labelleft=False, length=0)
            # plt.colorbar(im, ax=ax, orientation="vertical")
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            fig.savefig(
                save_path
                / f"representative_trial_{task}_{color}_{ephys_norm_type}_corr.png"
            )
            plt.clf()
            plt.close()


def jaccard_index(set1, set2):
    """
    Calculate Jaccard Index between two sets.

    Parameters:
    - set1: Numpy array or iterable representing the first set.
    - set2: Numpy array or iterable representing the second set.

    Returns:
    - Jaccard Index as a float value.
    """

    # Convert sets to Numpy arrays
    set1 = np.array(set1)
    set2 = np.array(set2)

    # Calculate the intersection and union of the sets
    intersection = len(np.intersect1d(set1, set2))
    union = len(np.union1d(set1, set2))

    # Calculate Jaccard Index
    jaccard_index = intersection / union if union != 0 else 0.0

    return jaccard_index


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
    models = ["EEGNet", "EMG_EEGNet", "EEGNet_wo_adapt_filt"]
    colors = ["green", "magenta", "orange", "violet", "yellow"]
    tasks = ["overt", "minimally overt", "covert"]
    correct_decode = True
    clip = 5

    igs_all = []
    igs_norm_all = []
    igs_table_all = pd.DataFrame(
        columns=("date", "subject", "model", "task", "cv", "label")
    )
    for input_tuple in tqdm(data_tuple):
        for model in models:
            DATE, sub_idx, subject = input_tuple
            igs, igs_table = load_igs(DATE, sub_idx, subject, model)
            igs_norm = np.abs(igs)
            igs_norm = (
                igs_norm - np.mean(igs_norm, axis=(2, 3), keepdims=True)
            ) / np.std(igs_norm, axis=(2, 3), keepdims=True)
            igs_norm = np.clip(igs_norm, -clip, clip)
            igs_norm_all.extend(list(igs_norm[:, 0]))
            igs_all.extend(list(igs[:, 0]))
            igs_table_all = pd.concat([igs_table_all, igs_table])
    # igs_all = np.concatenate(igs_all, axis=0)
    cprint(f"len igs_all: {len(igs_all)}", "cyan")
    cprint(f"len igs_norm_all: {len(igs_norm_all)}", "cyan")
    cprint(f"len igs_table_all: {len(igs_table_all)}", "cyan")
    # save igs with pickle
    # save_path = Path("figures/fig3/data/")
    # save_path.mkdir(exist_ok=True, parents=True)
    # with open(save_path / "igs_all.pkl", "wb") as f:
    #     pickle.dump(igs_all, f)
    # with open(save_path / "igs_norm_all.pkl", "wb") as f:
    #     pickle.dump(igs_norm_all, f)
    # igs_table_all.to_csv(save_path / "igs_table_all.csv", index=False)

    eegs_all = []
    eegs_wo_adapt_filt_all = []
    emgs_all = []
    speeches_all = []
    eegs_norm_all = []
    eegs_wo_adapt_filt_norm_all = []
    emgs_norm_all = []
    speeches_norm_all = []
    eegs_table_all = pd.DataFrame(columns=("date", "subject", "task", "label"))
    for input_tuple in tqdm(data_tuple):
        DATE, sub_idx, subject = input_tuple
        eegs, eegs_table = load_ephys(DATE, sub_idx, subject, "eeg")
        eegs_wo_adapt_filt, _ = load_ephys(DATE, sub_idx, subject, "eeg_wo_adapt_filt")
        emgs, _ = load_ephys(DATE, sub_idx, subject, "emg")
        speeches, _ = load_ephys(DATE, sub_idx, subject, "speech")
        eegs_norm = (eegs - np.mean(eegs, axis=2, keepdims=True)) / np.std(
            eegs, axis=2, keepdims=True
        )
        eegs_wo_adapt_filt_norm = (
            eegs_wo_adapt_filt - np.mean(eegs_wo_adapt_filt, axis=2, keepdims=True)
        ) / np.std(eegs_wo_adapt_filt, axis=2, keepdims=True)
        emgs_norm = (emgs - np.mean(emgs, axis=2, keepdims=True)) / np.std(
            emgs, axis=2, keepdims=True
        )
        speeches_norm = (speeches - np.mean(speeches, axis=1, keepdims=True)) / np.std(
            speeches, axis=1, keepdims=True
        )
        eegs_norm_all.extend(list(eegs_norm))
        eegs_wo_adapt_filt_norm_all.extend(list(eegs_wo_adapt_filt_norm))
        emgs_norm_all.extend(list(emgs_norm))
        speeches_norm_all.extend(list(speeches_norm))
        eegs_all.extend(list(eegs))
        eegs_wo_adapt_filt_all.extend(list(eegs_wo_adapt_filt))
        emgs_all.extend(list(emgs))
        speeches_all.extend(list(speeches))
        eegs_table_all = pd.concat([eegs_table_all, eegs_table])
    cprint(f"len eegs_all: {len(eegs_all)}", "cyan")
    cprint(f"len eegs_wo_adapt_filt_all: {len(eegs_wo_adapt_filt_all)}", "cyan")
    cprint(f"len emgs_all: {len(emgs_all)}", "cyan")

    imshow_each_ig_montage_representative(
        tasks,
        colors,
        igs_norm_all,
        igs_table_all,
        eegs_norm_all,
        correct_decode,
        clip,
        model="EEGNet",
        cmax=1.0,
        logscale=False,
    )

    save_path = Path("figures/fig3/temporal_representative")
    save_path.mkdir(exist_ok=True, parents=True)
    plot_temporal_representative(
        igs_norm_all,
        igs_table_all,
        eegs_norm_all,
        eegs_table_all,
        emgs_norm_all,
        speeches_norm_all,
        colors,
        models,
        tasks,
        save_path,
        ephys_norm_type="top_k",
        top_k=10,
        w_size=6.0,
        h_size=3.5,
        fontsize_legend=1.5,
        fontsize_tick=4.0,
        lw=0.4,
        only_ig=False,
    )

    save_path = Path("figures/fig3")
    save_path.mkdir(exist_ok=True, parents=True)
    # imshow_temporal_jaccard_indices
    imshow_temporal_jaccard_indices(
        igs_norm_all,
        igs_table_all,
        eegs_norm_all,
        eegs_table_all,
        emgs_norm_all,
        speeches_norm_all,
        colors,
        models,
        tasks,
        save_path,
        ephys_norm_type="top_k",
        top_k=10,
        w_size=6.0,
        h_size=5.5,
        fontsize_legend=3.5,
        fontsize_tick=3.5,
        fontsize_star=5.0,
        threshold=0,
        only_ig=True,
        corr_type="jaccard",
        corr_trial_base=True,
    )

    save_path = Path("figures/fig3")
    save_path.mkdir(exist_ok=True, parents=True)
    imshow_temporal_jaccard_indices_between_tasks(
        igs_norm_all,
        igs_table_all,
        colors,
        model="EEGNet",
        tasks=tasks,
        save_path=save_path,
        norm_type="mean",
        top_k=10,
        w_size=6.0,
        h_size=5.5,
        fontsize_legend=3.5,
        fontsize_tick=3.5,
        fontsize_star=9.0,
        threshold=0,
        corr_type="jaccard",
    )

    mis = np.load("figures/fig2/data/mis.npy")
    mis_table = pd.read_csv("figures/fig2/data/mis_table.csv")
    target_index = np.where(
        (mis_table["eeg_type"] == "raw")
        & (mis_table["emg_type"] == "raw")
        & (mis_table["surrogate_type"] == "none")
    )[0]
    mi_roi = np.mean(mis[target_index], axis=1)
    mi_roi = [mi for mi in mi_roi]
    eegs_table_all_onoff = pd.read_csv("figures/fig2/data/eegs_table_all.csv")
    target_indices_mi = np.where(
        eegs_table_all_onoff["exp_name"].str.contains("online").values
    )[0]
    save_path = Path("figures/fig4")
    save_path.mkdir(exist_ok=True, parents=True)
    imshow_combine_ig_montage(
        tasks,
        colors,
        igs_norm_all,
        igs_table_all,
        correct_decode,
        save_path,
        model="EEGNet",
        cmax=1.0,
        logscale=False,
        cmap="viridis",
        show_sd=False,
        show_avg=True,
        compare_with_emgs=True,
        mi_emgs=mi_roi,
    )
    save_path = Path("figures/fig5")
    save_path.mkdir(exist_ok=True, parents=True)
    for corr_type in ["pearson", "weight_multiplied_pearson"]:
        imshow_diff_ig_montage(
            tasks,
            colors,
            igs_norm_all,
            igs_table_all,
            correct_decode,
            save_path,
            model1="EEGNet",
            model2="EEGNet_wo_adapt_filt",
            cmax=1.0,
            logscale=False,
            cmap="bwr",
            show_each=False,
            show_avg=True,
            trial_based=True,
            compare_with_emgs=True,
            mi_emgs=mi_roi,
            corr_type=corr_type,
            mis=mis[target_index][:, target_indices_mi, :],
        )


if __name__ == "__main__":
    main()
