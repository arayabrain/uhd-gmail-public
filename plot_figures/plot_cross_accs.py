"""plot accuracy of different models"""

import pandas as pd
import torch
from termcolor import cprint
from tqdm import tqdm

from plot_figures.src.eval_accs import src_dist_specify_save

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_dist_tuple(src_list, src, dist_list):

    subject, task = src[2], src[3]
    src_list_df = pd.DataFrame(src_list, columns=["date", "sub_idx", "subject", "task"])
    dist_list_df = pd.DataFrame(
        dist_list, columns=["date", "subject", "online_name", "task"]
    )
    # remove "on-" and "off" from "task" column values of dist_list_df
    dist_list_df["task"] = dist_list_df["task"].str.replace("on-", "")
    dist_list_df["task"] = dist_list_df["task"].str.replace("off-", "")

    # get indices of different tasks within same subject
    dist_idx = dist_list_df[
        (dist_list_df["subject"] == subject) & (dist_list_df["task"] != task)
    ].index
    dist_tuple = [dist_list[i] for i in dist_idx]
    return dist_tuple


def main() -> None:
    """main function"""
    src_list = [
        ("20230511", 1, "shun", "min-overt"),
        ("20230529", 1, "shun", "overt"),
        ("20230529", 2, "shun", "covert"),
        ("20230512", 1, "yasu", "min-overt"),
        ("20230512", 2, "yasu", "overt"),
        ("20230516", 1, "yasu", "covert"),
        ("20230523", 1, "rousslan", "overt"),
        ("20230523", 2, "rousslan", "min-overt"),
        ("20230524", 1, "rousslan", "covert"),
        ("20230524", 2, "rousslan", "min-overt"),
    ]
    dist_list = [
        # ("20230511", "shun", ["backup_calibrated_1"], "off-min-overt"),
        # ("20230511", "shun", ["backup_online_1", "backup_online_2"], "on-min-overt"),
        # ("20230529", "shun", ["backup_calibrated_1"], "off-overt"),
        # ("20230529", "shun", ["backup_calibrated_2"], "off-covert"),
        # ("20230529", "shun", ["backup_online_1"], "on-overt"),
        # ("20230529", "shun", ["backup_online_2"], "on-covert"),
        # ("20230512", "yasu", ["backup_calibrated_1"], "off-min-overt"),
        # ("20230512", "yasu", ["backup_calibrated_2"], "off-overt"),
        # ("20230512", "yasu", ["backup_online_3"], "on-min-overt"),
        # ("20230512", "yasu", ["backup_online_4"], "on-overt"),
        # ("20230516", "yasu", ["backup_calibrated_1"], "off-covert"),
        # ("20230516", "yasu", ["backup_online_1"], "on-covert"),
        # ("20230523", "rousslan", ["backup_calibrated_1"], "off-overt"),
        # ("20230523", "rousslan", ["backup_calibrated_2"], "off-min-overt"),
        # ("20230523", "rousslan", ["backup_online_2"], "on-overt"),
        # ("20230523", "rousslan", ["backup_online_3"], "on-min-overt"),
        # ("20230524", "rousslan", ["backup_calibrated_1"], "off-covert"),
        # ("20230524", "rousslan", ["backup_calibrated_2"], "off-min-overt"),
        # ("20230524", "rousslan", ["backup_online_1"], "on-covert"),
        # ("20230524", "rousslan", ["backup_online_2"], "on-min-overt"),
    ]

    # models = ["EEGNet", "LSTM", "GRU", "EEGNet_with_mask"]
    models = [
        # "CovTanSVM",
        # "CovTanSVM2",
        # "CovTanSVM2_wo_adapt_filt",
        # "EEGNet",
        # "EEGNet_with_mask_4ch",
        # "EEGNet_with_mask_8ch",
        # "EEGNet_with_mask_16ch",
        # "EEGNet_with_mask_32ch",
        # "EEGNet_wo_adapt_filt",
        # "EEGNetSpaceConv_wo_adapt_filt",
        # "EMG_EEGNet",
        # "EMG_EEGNet_highpass30",
        # "EMG_EEGNet_highpass60",
        # "EMG_CovTanSVM2",
        # "EMG_LSTM2",
        # "LSTM2",
        # "LSTM2_wo_adapt_filt",
    ]
    for src in src_list:
        dist_tuple = get_dist_tuple(src_list, src, dist_list)
        cprint(f"src: {src}", "green")
        cprint(f"dist_tuple: {dist_tuple}", "green")
        for dist in dist_tuple:
            for model in tqdm(models):
                src_dist_specify_save(src, dist, model, device=DEVICE)


if __name__ == "__main__":
    main()
