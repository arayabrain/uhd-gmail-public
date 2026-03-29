"""plot accuracy of different models"""

from pathlib import Path

import torch
from tqdm import tqdm

from plot_figures.src.eval_accs import finetuning_onlinetest_save

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main() -> None:
    """main function"""
    data_root = Path("data/weights")
    subjects = ["subject1", "subject2", "subject3"]
    conditions = ["min-overt_covert", "overt_covert", "overt_min-overt_covert"]
    target_dirs = [
        data_root / subject / condition
        for subject in subjects
        for condition in conditions
    ]

    models = [
        "EEGNet",
        # "EEGNet_finetuning_all",
    ]
    for target_dir in target_dirs:
        for model in tqdm(models):
            finetuning_onlinetest_save(target_dir, model, device=DEVICE)


if __name__ == "__main__":
    main()
