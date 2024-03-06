"""plot accuracy of different models (Table1, 2)"""

import torch
from tqdm import tqdm

from plot_figures.src.eval_accs import eval_ensemble_save

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main() -> None:
    """main function"""
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
    models = [
        "EEGNet_with_mask_4ch",
        "EEGNet_with_mask_8ch",
        "EEGNet_with_mask_16ch",
        "EEGNet_with_mask_32ch",
        "EEGNet",  # TODO fix with state_dict
        "EEGNet_wo_adapt_filt",  # TODO fix with state_dict
        "EMG_EEGNet",  # TODO fix with state_dict
        "LSTM",  # TODO fix with state_dict
        "LSTM_wo_adapt_filt",  # TODO fix with state_dict
        "EMG_LSTM",  # TODO fix with state_dict
        "CovTanSVM",
        "CovTanSVM_wo_adapt_filt",
        "EMG_CovTanSVM",
    ]
    for input_tuple in data_tuple:
        for model in tqdm(models):
            date, sub_idx, subject = input_tuple
            eval_ensemble_save(date, sub_idx, subject, model, device=DEVICE)


if __name__ == "__main__":
    main()
