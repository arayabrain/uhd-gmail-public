"""function to evaluate accuracy of a model"""

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import dill
import numpy as np
import torch
from hydra import compose, initialize
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from scipy.stats import zscore
from sklearn.metrics import balanced_accuracy_score
from termcolor import cprint

from plot_figures.src.surrogates import aaft, ft, iaaft
from uhd_eeg.datasets.DatasetUHD import EEGDataset, EMGDataset
from uhd_eeg.models.CNN.EEGNet import EEGNet, EEGNet_with_mask
from uhd_eeg.models.RNN.RNN import MultiLayerRNN

N_CV = 10
DATA_ROOT = Path("data/")


def eval_net(
    model_path: str,
    pt_dir: str,
    device: torch.device,
    is_rnn: bool = False,
    is_svm: bool = False,
) -> np.ndarray:
    """eval score of a model

    Args:
        model_path (str): model path
        pt_dir (str): directory of test data
        device (torch.device): device
        is_rnn (bool, optional): the model is rnn or not. Defaults to False.
        is_svm (bool, optional): the model is svm or not. Defaults to False.

    Returns:
        np.ndarray: predicted labels
    """
    if "EMG" in model_path:
        eval_emg = True
    else:
        eval_emg = False
    if is_svm:
        net = dill.load(open(model_path, "rb"))
    else:
        with initialize(config_path="../../configs/trainer"):
            args = compose("config_color_combine.yaml")
        if eval_emg:
            OmegaConf.set_struct(args, False)
            OmegaConf.update(args, "decode_from", "emg", merge=True)
            OmegaConf.update(args, "num_channels", 3, merge=True)
            OmegaConf.set_struct(args, True)
        duration = 320
        if "EEGNet_with_mask" in model_path:
            net = EEGNet_with_mask(args, T=duration)
        elif "LSTM" in model_path:
            rnn_type = "LSTM"
            net = MultiLayerRNN(
                input_size=args.num_channels,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                output_size=args.n_class,
                rnn_type=rnn_type,
                bidirectional=args.bidirectional,
                dropout_rate=args.dropout_rate,
                last_activation=args.last_activation,
            )
        elif "GRU" in model_path:
            rnn_type = "GRU"
            net = MultiLayerRNN(
                input_size=args.num_channels,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                output_size=args.n_class,
                rnn_type=rnn_type,
                bidirectional=args.bidirectional,
                dropout_rate=args.dropout_rate,
                last_activation=args.last_activation,
            )
        else:
            net = EEGNet(args, T=duration)
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)
        net.eval()

    preds = []
    for _, pt_path in enumerate(natsorted(list(Path(pt_dir).glob("*.pt")))):
        # print(pt_path)
        eeg = torch.load(pt_path, map_location=device)
        if is_rnn:
            eeg = eeg[:, 0, :, :].permute(0, 2, 1)
        if is_svm:
            eeg = eeg.squeeze(1)
            eeg = eeg.cpu().detach().numpy()
            pred = net.predict_proba(eeg)
        else:
            pred = net(eeg).cpu().detach().numpy()[0]
        # print(pred.shape)
        # print(idx, pred.argmax(axis=-1))
        preds.append(pred.argmax(axis=-1))
    preds = np.array(preds)
    return preds


def softmax(x: np.ndarray, axis: int = 1, beta: float = 1.0) -> np.ndarray:
    """softmax function with temperature

    Args:
        x (np.ndarray): input tensor
        axis (int, optional): axis. Defaults to 1.
        beta (float, optional): inverse temperature. Defaults to 1.0.

    Returns:
        np.ndarray: softmaxed tensor
    """
    c = np.max(beta * x, axis=axis, keepdims=True)
    ex = np.exp(beta * x - c)
    sum_ex = np.sum(ex, axis=axis, keepdims=True)
    return ex / sum_ex


def eval_ensemble(
    model_paths: List[str],
    pt_dir: str,
    device: torch.device,
    method: str = "mean",
    is_rnn: bool = False,
    is_svm: bool = False,
    surrogate: str = None,
    n_surrogate: int = 100,
) -> np.ndarray:
    """eval score of ensemble models

    Args:
        model_paths (List[str]): model paths
        pt_dir (str): direcotry of test data
        device (torch.device): device
        method (str, optional): ensemble method. Defaults to "mean".
        is_rnn (bool, optional): the model is rnn or not. Defaults to False.
        is_svm (bool, optional): the model is svm or not. Defaults to False.
        surrogate (str, optional): surrogate method. Defaults to None.
        n_surrogate (int, optional): number of surrogate data. Defaults to 100.

    Returns:
        np.ndarray: predicted labels
    """
    if "EMG" in model_paths[0]:
        eval_emg = True
    else:
        eval_emg = False
    nets = []
    if is_svm:
        for model_path in model_paths:
            net = dill.load(open(model_path, "rb"))
            nets.append(net)
    else:
        for model_path in model_paths:
            with initialize(config_path="../../configs/trainer"):
                args = compose("config_color_combine.yaml")
            if eval_emg:
                OmegaConf.set_struct(args, False)
                OmegaConf.update(args, "decode_from", "emg", merge=True)
                OmegaConf.update(args, "num_channels", 3, merge=True)
                OmegaConf.set_struct(args, True)
            duration = 320
            if "EEGNet_with_mask" in model_paths[0]:
                net = EEGNet_with_mask(args, T=duration)
            elif "LSTM" in model_paths[0]:
                rnn_type = "LSTM"
                net = MultiLayerRNN(
                    input_size=args.num_channels,
                    hidden_size=args.hidden_size,
                    num_layers=args.num_layers,
                    output_size=args.n_class,
                    rnn_type=rnn_type,
                    bidirectional=args.bidirectional,
                    dropout_rate=args.dropout_rate,
                    last_activation=args.last_activation,
                )
            elif "GRU" in model_paths[0]:
                rnn_type = "GRU"
                net = MultiLayerRNN(
                    input_size=args.num_channels,
                    hidden_size=args.hidden_size,
                    num_layers=args.num_layers,
                    output_size=args.n_class,
                    rnn_type=rnn_type,
                    bidirectional=args.bidirectional,
                    dropout_rate=args.dropout_rate,
                    last_activation=args.last_activation,
                )
            else:
                net = EEGNet(args, T=duration)
            net.load_state_dict(torch.load(model_path, map_location=device))
            net.to(device)
            net.eval()
            nets.append(net)

    # print(f'num models = {len(nets)}')
    pred_labels = []
    with torch.no_grad():
        for _, pt_path in enumerate(natsorted(list(Path(pt_dir).glob("*.pt")))):
            # print(pt_path)
            eeg = torch.load(pt_path, map_location=device)
            if surrogate is not None:
                eeg = eeg.cpu().detach().numpy()
                if surrogate == "iaaft":
                    eeg = np.array(
                        [
                            iaaft(eeg[0, 0, ch, :], n_surrogate)
                            for ch in range(eeg.shape[2])
                        ]
                    )  # (n_ch, n_surrogate, n_time)
                    eeg = eeg.transpose(1, 0, 2)[
                        :, np.newaxis, :, :
                    ]  # (n_surrogate, 1, n_ch, n_time)
                elif surrogate == "aaft":
                    eeg = np.array(
                        [aaft(eeg[0, 0, ch, :]) for ch in range(eeg.shape[2])]
                    )  # (n_ch, n_surrogate, n_time)
                    eeg = eeg.transpose(1, 0, 2)[
                        :, np.newaxis, :, :
                    ]  # (n_surrogate, 1, n_ch, n_time)
                elif surrogate == "ft":
                    eeg = np.array([ft(eeg) for _ in range(n_surrogate)])[
                        :, np.newaxis, :, :
                    ]  # (n_surrogate, n_ch, n_time)
                elif surrogate == "rs":
                    permutation_indices = [
                        np.random.permutation(eeg.shape[2]) for _ in range(n_surrogate)
                    ]
                    eeg = np.array(
                        [
                            eeg[:, 0, :, permutation_idx]
                            for permutation_idx in permutation_indices
                        ]
                    )  # (n_surrogate, 1, n_ch, n_time)
                else:
                    ValueError(f"surrogate method {surrogate} is not supported.")
                eeg = torch.from_numpy(eeg).float().to(device)
            if is_rnn:
                eeg = eeg[:, 0, :, :].permute(0, 2, 1)
            if is_svm:
                eeg = eeg.squeeze(1)
                eeg = eeg.cpu().detach().numpy()
            preds = []
            for net in nets:
                if is_svm:
                    pred = net.predict_proba(eeg)[0]
                    # pred = net.predict_proba(eeg) # (n_surrogate, n_classes)
                else:
                    pred = net(eeg).cpu().detach().numpy()[0]
                    # pred = net(eeg).cpu().detach().numpy() # (n_surrogate, n_classes)
                preds.append(pred)
            preds = np.array(
                preds
            )  # (n_models, n_classes) | (n_models, n_surrogate, n_classes)
            # cprint(preds.shape, "green")
            if method == "mean":
                preds = np.mean(preds, axis=0)  # (n_classes) | (n_surrogate, n_classes)
            elif method == "max":
                preds = np.max(preds, axis=0)  # (n_classes) | (n_surrogate, n_classes)
            elif method == "zscore_mean":
                preds = zscore(
                    preds, axis=-1
                )  # (n_models, n_classes) | (n_models, n_surrogate, n_classes)
                preds = np.mean(preds, axis=0)  # (n_classes) | (n_surrogate, n_classes)
            elif method == "zscore_max":
                preds = zscore(
                    preds, axis=-1
                )  # (n_models, n_classes) | (n_models, n_surrogate, n_classes)
                preds = np.max(preds, axis=0)  # (n_classes) | (n_surrogate, n_classes)
            elif method == "entropy_weighted":
                preds = softmax(
                    preds, axis=-1
                )  # (n_models, n_classes) | (n_models, n_surrogate, n_classes)
                entropy = -np.sum(
                    preds * np.log(preds), axis=-1
                )  # (n_models) | (n_surrogate, n_models)
                preds = np.dot(preds.T, entropy)  # (n_classes)
            elif method == "inverse_entropy_weighted":
                preds = softmax(preds, axis=-1)
                entropy = -np.sum(preds * np.log(preds), axis=-1)  # (n_models)
                preds = np.dot(preds.T, 1 / entropy)  # (n_classes)
            elif method == "majority":
                preds = np.argmax(preds, axis=1)
                preds = np.bincount(preds, minlength=5)
            pred_labels.append(np.argmax(preds))
            # print(pt_path)
            # print(preds)
    pred_labels = np.array(pred_labels)
    return pred_labels


def eval_ensemble_from_dataset(
    args: DictConfig,
    model_paths: List[str],
    dataset_test: EEGDataset | EMGDataset,
    device: torch.device,
    method: str = "mean",
    n_worekers: int = 0,
) -> np.ndarray:
    """eval score of ensemble models

    Args:
        model_paths (List[str]): model paths
        pt_dir (str): direcotry of test data
        device (torch.device): device
        method (str, optional): ensemble method. Defaults to "mean".
        n_worekers (int, optional): number of workers. Defaults to 0.

    Returns:
        np.ndarray: predicted labels
    """
    nets = []
    for model_path in model_paths:
        net = EEGNet(args, T=320)
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)
        net.eval()
        nets.append(net)
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        num_workers=n_worekers,
        pin_memory=False,
        shuffle=True,
    )

    pred_labels = []
    words = np.empty((0,))
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            eeg, labels = data
            eeg = eeg.to(device)
            labels = labels.to(device)
            words = np.hstack((words, labels.cpu().detach().numpy()))
            preds = []
            for net in nets:
                pred = net(eeg).cpu().detach().numpy()[0]
                preds.append(pred)
            preds = np.array(
                preds
            )  # (n_models, n_classes) | (n_models, n_surrogate, n_classes)
            # cprint(preds.shape, "green")
            if method == "mean":
                preds = np.mean(preds, axis=0)  # (n_classes) | (n_surrogate, n_classes)
            elif method == "max":
                preds = np.max(preds, axis=0)  # (n_classes) | (n_surrogate, n_classes)
            elif method == "zscore_mean":
                preds = zscore(
                    preds, axis=-1
                )  # (n_models, n_classes) | (n_models, n_surrogate, n_classes)
                preds = np.mean(preds, axis=0)  # (n_classes) | (n_surrogate, n_classes)
            elif method == "zscore_max":
                preds = zscore(
                    preds, axis=-1
                )  # (n_models, n_classes) | (n_models, n_surrogate, n_classes)
                preds = np.max(preds, axis=0)  # (n_classes) | (n_surrogate, n_classes)
            elif method == "entropy_weighted":
                preds = softmax(
                    preds, axis=-1
                )  # (n_models, n_classes) | (n_models, n_surrogate, n_classes)
                entropy = -np.sum(
                    preds * np.log(preds), axis=-1
                )  # (n_models) | (n_surrogate, n_models)
                preds = np.dot(preds.T, entropy)  # (n_classes)
            elif method == "inverse_entropy_weighted":
                preds = softmax(preds, axis=-1)
                entropy = -np.sum(preds * np.log(preds), axis=-1)  # (n_models)
                preds = np.dot(preds.T, 1 / entropy)  # (n_classes)
            elif method == "majority":
                preds = np.argmax(preds, axis=1)
                preds = np.bincount(preds, minlength=5)
            pred_labels.append(np.argmax(preds))
            # print(pt_path)
            # print(preds)
    pred_labels = np.array(pred_labels)
    return pred_labels, words


# evaluate and save func
def eval_ensemble_save(
    date: str, sub_idx: int, subject: str, model: str, device: torch.device
) -> None:
    """evaluate and save accuracy of ensemble models

    Args:
        date (str): date
        sub_idx (int): try number
        subject (str): subject name
        model (str): model name
        device (torch.device): device
    """
    if ("LSTM" in model) or ("GRU" in model):
        is_rnn = True
    else:
        is_rnn = False
    if "CovTanSVM" in model:
        is_svm = True
    else:
        is_svm = False
    if "EMG" in model:
        preproc_dir = "emg_after_preproc"
        if "highpass30" in model:
            preproc_dir += "_highpass30"
        elif "highpass60" in model:
            preproc_dir += "_highpass60"
        else:
            pass
    else:  # EEG
        if "wo_adapt_filt" in model:
            preproc_dir = "eeg_after_preproc_wo_adapt_filt"
        else:
            preproc_dir = "eeg_after_preproc"
    with open(str(DATA_ROOT / subject / date / "metadata.json"), encoding="utf8") as f:
        metadata = json.load(f)
    metadata = {key: metadata[key] for key in metadata.keys() if key != "subject"}
    exp_names = list(metadata.keys())
    # tasks = [metadata[key]["task"] for key in exp_names]
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

    online_accs = []
    balanced_accs = []
    for cv in range(N_CV):
        if is_svm:
            model_path = (
                DATA_ROOT
                / subject
                / date
                / f"config_color_backup_{date[2:]}_{sub_idx}_{model}"
                / f"CovTanSVM_config_color_N5_cv{cv}.dill"
            )
        else:
            model_path = (
                DATA_ROOT
                / subject
                / date
                / f"config_color_backup_{date[2:]}_{sub_idx}_{model}"
                / f"model_weight_config_color_N5_cv{cv}.pth"
            )
        words_all = []
        preds_all = []
        for online_name in online_names:
            pt_dir = str(
                DATA_ROOT / subject / date / preproc_dir / f"{date}_{online_name}"
            )
            csv_path = str(DATA_ROOT / subject / date / f"word_list_{online_name}.csv")
            words = np.loadtxt(csv_path, delimiter=",", dtype=int)
            preds = eval_net(
                str(model_path), pt_dir, device, is_rnn=is_rnn, is_svm=is_svm
            )
            words_all.append(words)
            preds_all.append(preds)
        words_all = np.concatenate(words_all)
        preds_all = np.concatenate(preds_all)
        # print(online_names)
        # print(pt_dir)
        # print(words_all.shape, preds_all.shape)
        # print(type(words_all), type(preds_all))
        acc = (preds_all == words_all).sum() / len(words_all)
        balanced_acc = balanced_accuracy_score(words_all, preds_all)
        # print(cv, acc)
        online_accs.append(acc)
        balanced_accs.append(balanced_acc)
    online_accs = np.array(online_accs)
    balanced_accs = np.array(balanced_accs)
    cprint(f"single balanced_accs = {balanced_accs}", "cyan")

    if is_svm:
        offline_accs_epoch = np.load(
            str(model_path.parent / "acc_val_config_color_avg5.npy")
        )
        offline_accs = offline_accs_epoch[:]
        offline_accs_last = offline_accs_epoch[:]
        # ind_best_epoch = [np.argmax(offline_accs_epoch[cv, :]) for cv in range(10)]
    else:
        offline_losses = np.load(
            str(model_path.parent / "loss_val_config_color_avg5.npy")
        )
        offline_accs_epoch = np.load(
            str(model_path.parent / "acc_val_config_color_avg5.npy")
        )
        ind_best_epoch = [np.argmin(offline_losses[cv, :]) for cv in range(10)]
        best_accs = []
        for cv in range(10):
            best_accs.append(offline_accs_epoch[cv, ind_best_epoch[cv]])
        offline_accs = np.array(best_accs)
        last_accs = []
        for cv in range(10):
            last_accs.append(offline_accs_epoch[cv, -1])
        offline_accs_last = np.array(last_accs)
        # sort_top_online = np.argsort(-online_accs)
    np.save(str(model_path.parent / "offline_accs.npy"), offline_accs)  # (N_CV,)
    np.save(
        str(model_path.parent / "offline_accs_last.npy"), offline_accs_last
    )  # (N_CV,)
    np.save(str(model_path.parent / "online_accs.npy"), online_accs)  # (N_CV,)
    np.save(
        str(model_path.parent / "online_balanced_accs.npy"), balanced_accs
    )  # (N_CV,)

    methods = [
        "mean",
        "max",
        "zscore_mean",
        "zscore_max",
        "entropy_weighted",
        "inverse_entropy_weighted",
        "majority",
    ]
    sort_top = np.argsort(-offline_accs)
    accs_top = []
    balanced_accs_top = []
    for k in range(N_CV):
        cvs = sort_top[: k + 1]
        if is_svm:
            model_paths = [
                str(
                    DATA_ROOT
                    / subject
                    / date
                    / f"config_color_backup_{date[2:]}_{sub_idx}_{model}"
                    / f"CovTanSVM_config_color_N5_cv{cv}.dill"
                )
                for cv in cvs
            ]
        else:
            model_paths = [
                str(
                    DATA_ROOT
                    / subject
                    / date
                    / f"config_color_backup_{date[2:]}_{sub_idx}_{model}"
                    / f"model_weight_config_color_N5_cv{cv}.pth"
                )
                for cv in cvs
            ]
        accs = []
        balanced_accs = []
        for method in methods:
            words_all = []
            preds_all = []
            for online_name in online_names:
                pt_dir = str(
                    DATA_ROOT / subject / date / preproc_dir / f"{date}_{online_name}"
                )
                csv_path = str(
                    DATA_ROOT / subject / date / f"word_list_{online_name}.csv"
                )
                words = np.loadtxt(csv_path, delimiter=",", dtype=int)
                pred_labels = eval_ensemble(
                    model_paths,
                    pt_dir,
                    device,
                    method=method,
                    is_rnn=is_rnn,
                    is_svm=is_svm,
                )
                words_all.append(words)
                preds_all.append(pred_labels)
            words_all = np.concatenate(words_all)
            preds_all = np.concatenate(preds_all)
            acc = (preds_all == words_all).sum() / len(words_all)
            balanced_acc = balanced_accuracy_score(words_all, preds_all)
            accs.append(acc)
            balanced_accs.append(balanced_acc)
            cprint(
                f"ensemble method = {method}, balanced_accs = {balanced_accs}", "cyan"
            )
        accs = np.array(accs)
        balanced_accs = np.array(balanced_accs)
        accs_top.append(accs)
        balanced_accs_top.append(balanced_accs)
    accs_top = np.array(accs_top)
    balanced_accs_top = np.array(balanced_accs_top)
    # make methods and accs_top dict
    accs_dict = {}
    balanced_accs_dict = {}
    for i, method in enumerate(methods):
        accs_dict[method] = accs_top[:, i]
        balanced_accs_dict[method] = balanced_accs_top[:, i]
    # save accs_dict with pickle
    with open(str(model_path.parent / "accs_dict.pkl"), "wb") as f:
        pickle.dump(accs_dict, f)
    with open(str(model_path.parent / "online_balanced_accs_dict.pkl"), "wb") as f:
        pickle.dump(balanced_accs_dict, f)


# src dist specify func
def src_dist_specify_save(
    src: Tuple[str, int, str, str],
    dist: Tuple[str, str, List[str], str],
    model: str,
    device: torch.device,
) -> None:
    """evaluate and save accuracy of ensemble models

    Args:
        src(List[str, int, str]): source data
        dist(List[str, int, List[str]]): dist data
        model (str): model name
        device (torch.device): device
    """
    date_src, sub_idx_src, subject_src, task_src = src
    date_dist, subject_dist, online_names, task_dist = dist

    if ("LSTM" in model) or ("GRU" in model):
        is_rnn = True
    else:
        is_rnn = False
    if "CovTanSVM" in model:
        is_svm = True
    else:
        is_svm = False
    if "EMG" in model:
        preproc_dir = "emg_after_preproc"
        if "highpass30" in model:
            preproc_dir += "_highpass30"
        elif "highpass60" in model:
            preproc_dir += "_highpass60"
        else:
            pass
    else:  # EEG
        if "wo_adapt_filt" in model:
            preproc_dir = "eeg_after_preproc_wo_adapt_filt"
        else:
            preproc_dir = "eeg_after_preproc"
    with open(
        str(DATA_ROOT / subject_src / date_src / "metadata.json"), encoding="utf8"
    ) as f:
        metadata = json.load(f)
    metadata = {key: metadata[key] for key in metadata.keys() if key != "subject"}
    exp_names = list(metadata.keys())

    online_accs = []
    balanced_accs = []
    for cv in range(N_CV):
        if is_svm:
            model_path = (
                DATA_ROOT
                / subject_src
                / date_src
                / f"config_color_backup_{date_src[2:]}_{sub_idx_src}_{model}"
                / f"CovTanSVM_config_color_N5_cv{cv}.dill"
            )
        else:
            model_path = (
                DATA_ROOT
                / subject_src
                / date_src
                / f"config_color_backup_{date_src[2:]}_{sub_idx_src}_{model}"
                / f"model_weight_config_color_N5_cv{cv}.pth"
            )
        words_all = []
        preds_all = []
        for online_name in online_names:
            pt_dir = str(
                DATA_ROOT
                / subject_dist
                / date_dist
                / preproc_dir
                / f"{date_dist}_{online_name}"
            )
            csv_path = str(
                DATA_ROOT / subject_dist / date_dist / f"word_list_{online_name}.csv"
            )
            words = np.loadtxt(csv_path, delimiter=",", dtype=int)
            preds = eval_net(
                str(model_path), pt_dir, device, is_rnn=is_rnn, is_svm=is_svm
            )
            words_all.append(words)
            preds_all.append(preds)
        words_all = np.concatenate(words_all)
        preds_all = np.concatenate(preds_all)
        # print(online_names)
        # print(pt_dir)
        # print(words_all.shape, preds_all.shape)
        # print(type(words_all), type(preds_all))
        acc = (preds_all == words_all).sum() / len(words_all)
        balanced_acc = balanced_accuracy_score(words_all, preds_all)
        # print(cv, acc)
        online_accs.append(acc)
        balanced_accs.append(balanced_acc)
    online_accs = np.array(online_accs)
    balanced_accs = np.array(balanced_accs)
    cprint(f"single balanced_accs = {balanced_accs}", "cyan")

    if is_svm:
        offline_accs_epoch = np.load(
            str(model_path.parent / "acc_val_config_color_avg5.npy")
        )
        offline_accs = offline_accs_epoch[:]
        offline_accs_last = offline_accs_epoch[:]
        # ind_best_epoch = [np.argmax(offline_accs_epoch[cv, :]) for cv in range(10)]
    else:
        offline_losses = np.load(
            str(model_path.parent / "loss_val_config_color_avg5.npy")
        )
        offline_accs_epoch = np.load(
            str(model_path.parent / "acc_val_config_color_avg5.npy")
        )
        ind_best_epoch = [np.argmin(offline_losses[cv, :]) for cv in range(10)]
        best_accs = []
        for cv in range(10):
            best_accs.append(offline_accs_epoch[cv, ind_best_epoch[cv]])
        offline_accs = np.array(best_accs)
        last_accs = []
        for cv in range(10):
            last_accs.append(offline_accs_epoch[cv, -1])
        offline_accs_last = np.array(last_accs)
        # sort_top_online = np.argsort(-online_accs)

    np.save(
        str(model_path.parent / f"cross_accs_{task_src}_{task_dist}.npy"), online_accs
    )  # (N_CV,)
    np.save(
        str(model_path.parent / f"cross_balanced_accs_{task_src}_{task_dist}.npy"),
        balanced_accs,
    )  # (N_CV,)

    methods = [
        "mean",
        "max",
        "zscore_mean",
        "zscore_max",
        "entropy_weighted",
        "inverse_entropy_weighted",
        "majority",
    ]
    sort_top = np.argsort(-offline_accs)
    accs_top = []
    balanced_accs_top = []
    for k in range(N_CV):
        cvs = sort_top[: k + 1]
        if is_svm:
            model_paths = [
                str(
                    DATA_ROOT
                    / subject_src
                    / date_src
                    / f"config_color_backup_{date_src[2:]}_{sub_idx_src}_{model}"
                    / f"CovTanSVM_config_color_N5_cv{cv}.dill"
                )
                for cv in cvs
            ]
        else:
            model_paths = [
                str(
                    DATA_ROOT
                    / subject_src
                    / date_src
                    / f"config_color_backup_{date_src[2:]}_{sub_idx_src}_{model}"
                    / f"model_weight_config_color_N5_cv{cv}.pth"
                )
                for cv in cvs
            ]
        accs = []
        balanced_accs = []
        for method in methods:
            words_all = []
            preds_all = []
            for online_name in online_names:
                pt_dir = str(
                    DATA_ROOT
                    / subject_dist
                    / date_dist
                    / preproc_dir
                    / f"{date_dist}_{online_name}"
                )
                csv_path = str(
                    DATA_ROOT
                    / subject_dist
                    / date_dist
                    / f"word_list_{online_name}.csv"
                )
                words = np.loadtxt(csv_path, delimiter=",", dtype=int)
                pred_labels = eval_ensemble(
                    model_paths,
                    pt_dir,
                    device,
                    method=method,
                    is_rnn=is_rnn,
                    is_svm=is_svm,
                )
                words_all.append(words)
                preds_all.append(pred_labels)
            words_all = np.concatenate(words_all)
            preds_all = np.concatenate(preds_all)
            acc = (preds_all == words_all).sum() / len(words_all)
            balanced_acc = balanced_accuracy_score(words_all, preds_all)
            accs.append(acc)
            balanced_accs.append(balanced_acc)
            cprint(
                f"ensemble method = {method}, balanced_accs = {balanced_accs}", "cyan"
            )
        accs = np.array(accs)
        balanced_accs = np.array(balanced_accs)
        accs_top.append(accs)
        balanced_accs_top.append(balanced_accs)
    accs_top = np.array(accs_top)
    balanced_accs_top = np.array(balanced_accs_top)
    # make methods and accs_top dict
    accs_dict = {}
    balanced_accs_dict = {}
    for i, method in enumerate(methods):
        accs_dict[method] = accs_top[:, i]
        balanced_accs_dict[method] = balanced_accs_top[:, i]
    # save accs_dict with pickle
    with open(
        str(model_path.parent / f"cross_accs_dict_{task_src}_{task_dist}.pkl"), "wb"
    ) as f:
        pickle.dump(accs_dict, f)
    with open(
        str(
            model_path.parent
            / f"cross_online_balanced_accs_dict_{task_src}_{task_dist}.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(balanced_accs_dict, f)


def finetuning_onlinetest_save(
    target_dir: Path,
    model: str,
    device: torch.device,
) -> None:
    """evaluate and save accuracy of ensemble models

    Args:
        target_dir (Path): target directory
        model (str): model name
        device (torch.device): device
    """
    # target_dir = Path("/mnt/tsukuyomi/uhd-eeg/shun/min-overt_covert/EEGNet_finetuning_all")
    tasks = target_dir.name
    cprint(tasks, "cyan")
    if ("LSTM" in model) or ("GRU" in model):
        is_rnn = True
    else:
        is_rnn = False
    if "CovTanSVM" in model:
        is_svm = True
    else:
        is_svm = False
    if "EMG" in model:
        preproc_dir = "emg_after_preproc"
        if "highpass30" in model:
            preproc_dir += "_highpass30"
        elif "highpass60" in model:
            preproc_dir += "_highpass60"
        else:
            pass
    else:  # EEG
        if "wo_adapt_filt" in model:
            preproc_dir = "eeg_after_preproc_wo_adapt_filt"
        else:
            preproc_dir = "eeg_after_preproc"

    online_accs = []
    balanced_accs = []
    for cv in range(N_CV):
        if is_svm:
            model_path = target_dir / f"{model}" / f"{model}_cv{cv}.dill"
        else:
            model_path = target_dir / f"{model}" / f"weight_covert_cv{cv}.pth"
            if not model_path.exists():
                model_path = (
                    target_dir / f"{model}" / f"model_weight_config_color_N5_cv{cv}.pth"
                )
        cprint(model_path, "cyan")
        words_all = []
        preds_all = []
        pt_dir = str(target_dir / preproc_dir / "test")
        csv_path = str(target_dir / f"word_list_{tasks}_test.csv")
        words = np.loadtxt(csv_path, delimiter=",", dtype=int)
        preds = eval_net(str(model_path), pt_dir, device, is_rnn=is_rnn, is_svm=is_svm)
        words_all.append(words)
        preds_all.append(preds)
        words_all = np.concatenate(words_all)
        preds_all = np.concatenate(preds_all)
        # print(online_names)
        # print(pt_dir)
        # print(words_all.shape, preds_all.shape)
        # print(type(words_all), type(preds_all))
        acc = (preds_all == words_all).sum() / len(words_all)
        balanced_acc = balanced_accuracy_score(words_all, preds_all)
        # print(cv, acc)
        online_accs.append(acc)
        balanced_accs.append(balanced_acc)
    online_accs = np.array(online_accs)
    balanced_accs = np.array(balanced_accs)
    cprint(f"single balanced_accs = {balanced_accs}", "cyan")

    if is_svm:
        offline_accs_epoch = np.load(str(model_path.parent / "acc_val_covert.npy"))
        offline_accs = offline_accs_epoch[:]
        offline_accs_last = offline_accs_epoch[:]
        # ind_best_epoch = [np.argmax(offline_accs_epoch[cv, :]) for cv in range(10)]
    else:
        try:
            offline_losses = np.load(str(model_path.parent / "loss_val_covert.npy"))
            offline_accs_epoch = np.load(str(model_path.parent / "acc_val_covert.npy"))
        except FileNotFoundError:
            offline_losses = np.load(
                str(model_path.parent / "loss_val_config_color_avg5.npy")
            )
            offline_accs_epoch = np.load(
                str(model_path.parent / "acc_val_config_color_avg5.npy")
            )
        ind_best_epoch = [np.argmin(offline_losses[cv, :]) for cv in range(10)]
        best_accs = []
        for cv in range(10):
            best_accs.append(offline_accs_epoch[cv, ind_best_epoch[cv]])
        offline_accs = np.array(best_accs)
        last_accs = []
        for cv in range(10):
            last_accs.append(offline_accs_epoch[cv, -1])
        offline_accs_last = np.array(last_accs)
        # sort_top_online = np.argsort(-online_accs)

    np.save(
        str(model_path.parent / f"finetuning_accs_{tasks}.npy"), online_accs
    )  # (N_CV,)
    np.save(
        str(model_path.parent / f"finetuning_balanced_accs_{tasks}.npy"),
        balanced_accs,
    )  # (N_CV,)

    methods = [
        "mean",
        "max",
        "zscore_mean",
        "zscore_max",
        "entropy_weighted",
        "inverse_entropy_weighted",
        "majority",
    ]
    sort_top = np.argsort(-offline_accs)
    accs_top = []
    balanced_accs_top = []
    for k in range(N_CV):
        cvs = sort_top[: k + 1]
        if is_svm:
            model_paths = [
                str(target_dir / model / f"weight_covert_cv{cv}.dill") for cv in cvs
            ]
        else:
            model_paths = [
                str(target_dir / model / f"weight_covert_cv{cv}.pth") for cv in cvs
            ]
            if not Path(model_paths[0]).exists():
                model_paths = [
                    str(target_dir / model / f"model_weight_config_color_N5_cv{cv}.pth")
                    for cv in cvs
                ]
        accs = []
        balanced_accs = []
        for method in methods:
            words_all = []
            preds_all = []
            pt_dir = str(target_dir / preproc_dir / "test")
            csv_path = str(target_dir / f"word_list_{tasks}_test.csv")
            words = np.loadtxt(csv_path, delimiter=",", dtype=int)
            pred_labels = eval_ensemble(
                model_paths,
                pt_dir,
                device,
                method=method,
                is_rnn=is_rnn,
                is_svm=is_svm,
            )
            words_all.append(words)
            preds_all.append(pred_labels)
            words_all = np.concatenate(words_all)
            preds_all = np.concatenate(preds_all)
            acc = (preds_all == words_all).sum() / len(words_all)
            balanced_acc = balanced_accuracy_score(words_all, preds_all)
            accs.append(acc)
            balanced_accs.append(balanced_acc)
            cprint(
                f"ensemble method = {method}, balanced_accs = {balanced_accs}", "cyan"
            )
        accs = np.array(accs)
        balanced_accs = np.array(balanced_accs)
        accs_top.append(accs)
        balanced_accs_top.append(balanced_accs)
    accs_top = np.array(accs_top)
    balanced_accs_top = np.array(balanced_accs_top)
    # make methods and accs_top dict
    accs_dict = {}
    balanced_accs_dict = {}
    for i, method in enumerate(methods):
        accs_dict[method] = accs_top[:, i]
        balanced_accs_dict[method] = balanced_accs_top[:, i]
    # save accs_dict with pickle
    with open(
        str(model_path.parent / f"finetuning_onine_accs_dict_{tasks}.pkl"), "wb"
    ) as f:
        pickle.dump(accs_dict, f)
    with open(
        str(model_path.parent / f"finetuning_online_balanced_accs_dict_{tasks}.pkl"),
        "wb",
    ) as f:
        pickle.dump(balanced_accs_dict, f)


def within_offline_split_test_save(
    args: DictConfig,
    dataset_test: EEGDataset | EMGDataset,
    n_worekers: int,
    device: torch.device,
) -> None:
    """evaluate and save accuracy of ensemble models

    Args:
        dataset_test (EEGDataset | EMGDataset): test dataset
        n_worekers (int): number of workers
        device (torch.device): device
    """
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=len(dataset_test),
        num_workers=n_worekers,
        pin_memory=False,
        shuffle=True,
    )
    online_accs = []
    balanced_accs = []
    for cv in range(9):
        model_path = f"weight_cv{cv}.pth"
        cprint(model_path, "cyan")
        net = EEGNet(args, T=320)
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)
        net.eval()

        preds_all = np.empty((0, 5))
        words_all = np.empty((0))
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds_all = np.vstack((preds_all, net(inputs).cpu().detach().numpy()))
            words_all = np.hstack((words_all, labels.cpu().detach().numpy()))
        preds_all = np.argmax(preds_all, axis=1)
        acc = (preds_all == words_all).sum() / len(words_all)
        balanced_acc = balanced_accuracy_score(words_all, preds_all)
        online_accs.append(acc)
        balanced_accs.append(balanced_acc)
    online_accs = np.array(online_accs)
    balanced_accs = np.array(balanced_accs)
    cprint(f"single balanced_accs = {balanced_accs}", "cyan")

    offline_losses = np.load("loss_val.npy")
    offline_accs_epoch = np.load("acc_val.npy")
    ind_best_epoch = [np.argmin(offline_losses[cv, :]) for cv in range(9)]
    best_accs = []
    for cv in range(9):
        best_accs.append(offline_accs_epoch[cv, ind_best_epoch[cv]])
    offline_accs = np.array(best_accs)
    last_accs = []
    for cv in range(9):
        last_accs.append(offline_accs_epoch[cv, -1])
    offline_accs_last = np.array(last_accs)
    np.save(f"within_offline_split_accs.npy", online_accs)  # (N_CV,)
    np.save(f"within_offline_split_balanced_accs.npy", balanced_accs)  # (N_CV,)

    methods = [
        "mean",
        "max",
        "zscore_mean",
        "zscore_max",
        "entropy_weighted",
        "inverse_entropy_weighted",
        "majority",
    ]
    sort_top = np.argsort(-offline_accs)
    accs_top = []
    balanced_accs_top = []
    for k in range(9):
        cvs = sort_top[: k + 1]
        model_paths = [f"weight_cv{cv}.pth" for cv in cvs]
        accs = []
        balanced_accs = []
        for method in methods:
            words_all = []
            preds_all = []
            pred_labels, words = eval_ensemble_from_dataset(
                args,
                model_paths,
                dataset_test,
                device,
                method,
            )
            words_all.append(words)
            preds_all.append(pred_labels)
            words_all = np.concatenate(words_all)
            preds_all = np.concatenate(preds_all)
            acc = (preds_all == words_all).sum() / len(words_all)
            balanced_acc = balanced_accuracy_score(words_all, preds_all)
            accs.append(acc)
            balanced_accs.append(balanced_acc)
            cprint(
                f"ensemble method = {method}, balanced_accs = {balanced_accs}", "cyan"
            )
        accs = np.array(accs)
        balanced_accs = np.array(balanced_accs)
        accs_top.append(accs)
        balanced_accs_top.append(balanced_accs)
    accs_top = np.array(accs_top)
    balanced_accs_top = np.array(balanced_accs_top)
    # make methods and accs_top dict
    accs_dict = {}
    balanced_accs_dict = {}
    for i, method in enumerate(methods):
        accs_dict[method] = accs_top[:, i]
        balanced_accs_dict[method] = balanced_accs_top[:, i]
    # save accs_dict with pickle
    with open("within_offline_split_accs_dict.pkl", "wb") as f:
        pickle.dump(accs_dict, f)
    with open("within_offline_split_balanced_accs_dict.pkl", "wb") as f:
        pickle.dump(balanced_accs_dict, f)
