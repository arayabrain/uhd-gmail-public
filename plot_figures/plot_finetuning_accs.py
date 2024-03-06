"""plot accuracy of different models"""

from pathlib import Path

import torch
from tqdm import tqdm

from plot_figures.src.eval_accs import finetuning_onlinetest_save

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main() -> None:
    """main function"""
    target_dirs = [
        "/mnt/tsukuyomi/uhd-eeg/shun/min-overt_covert",
        "/mnt/tsukuyomi/uhd-eeg/shun/overt_covert",
        "/mnt/tsukuyomi/uhd-eeg/shun/overt_min-overt_covert",
        "/mnt/tsukuyomi/uhd-eeg/yasu/min-overt_covert",
        "/mnt/tsukuyomi/uhd-eeg/yasu/overt_covert",
        "/mnt/tsukuyomi/uhd-eeg/yasu/overt_min-overt_covert",
        "/mnt/tsukuyomi/uhd-eeg/rousslan/min-overt_covert",
        "/mnt/tsukuyomi/uhd-eeg/rousslan/overt_covert",
        "/mnt/tsukuyomi/uhd-eeg/rousslan/overt_min-overt_covert",
    ]

    models = [
        "EEGNet",
        # "EEGNet_finetuning_all",
    ]
    for target_dir in target_dirs:
        target_dir = Path(target_dir)
        for model in tqdm(models):
            finetuning_onlinetest_save(target_dir, model, device=DEVICE)


if __name__ == "__main__":
    main()
