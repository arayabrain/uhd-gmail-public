"""plot accuracy of different models (Table1, 2)"""

import torch
from tqdm import tqdm

from scripts.figures._lib.eval_accs import eval_ensemble_save
from scripts.figures._bids_runs import OFFLINE_RUNS

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main() -> None:
    """main function"""
    # BIDS offline runs (sub-N / ses-* / task / acq); calib index is within-session order.
    data_tuple = []
    for run in OFFLINE_RUNS:
        same_day = [
            r for r in OFFLINE_RUNS if r.subject == run.subject and r.session == run.session
        ]
        data_tuple.append(
            (run.session.removeprefix("ses-"), same_day.index(run) + 1, run.subject)
        )
    models = [
        "EEGNet_with_mask_4ch",
        "EEGNet_with_mask_8ch",
        "EEGNet_with_mask_16ch",
        "EEGNet_with_mask_32ch",
        "EEGNet",
        "EEGNet_wo_adapt_filt",
        "EMG_EEGNet",
        "LSTM",
        "LSTM_wo_adapt_filt",
        "EMG_LSTM",
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
