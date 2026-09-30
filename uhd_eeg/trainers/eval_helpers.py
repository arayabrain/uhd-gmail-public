"""Shared training and evaluation helpers for offline CV trainers."""

from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.stats import zscore
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from termcolor import cprint
from torchinfo import summary

from uhd_eeg.models.CNN.EEGNet import EEGNet, EEGNet_with_mask
from uhd_eeg.models.RNN.RNN import MultiLayerRNN
from uhd_eeg.models.Transformer.cBraMod.cBraMod import CBraModClassifier


def get_save_dir(args: DictConfig) -> str:
    if args.use_hydra_savedir:
        return "."
    dir_save = f"{args.saved_data_root}/{args.config_name}"
    os.makedirs(dir_save, exist_ok=True)
    return dir_save


def model_filename(args: DictConfig, cv: int) -> str:
    if args.model_name == "CovTanSVM":
        return f"CovTanSVM_{args.config_name}_N{args.n_trial_avg}_cv{cv}.dill"
    return f"model_weight_{args.config_name}_N{args.n_trial_avg}_cv{cv}.pth"


def model_path(args: DictConfig, cv: int) -> str:
    return os.path.join(get_save_dir(args), model_filename(args, cv))


def get_behavior(args: DictConfig) -> str:
    gmail = args.get("gmail")
    if gmail is not None and "task" in gmail:
        task = str(gmail.task)
        if task == "minimallyovert":
            return "minimally_overt"
        return task
    if gmail is not None and "behavior" in gmail:
        return str(gmail.behavior)
    run_key = str(args.parallel_sets)
    if "_task-" in run_key:
        task = run_key.split("_task-", 1)[1].split("_", 1)[0]
        if task == "minimallyovert":
            return "minimally_overt"
        return task
    return "unknown"


def get_subject_id(args: DictConfig) -> str:
    gmail = args.get("gmail")
    if gmail is not None and "subject" in gmail:
        return str(gmail.subject)
    run_key = str(args.parallel_sets)
    if run_key.startswith("sub-"):
        return run_key.split("_", 1)[0]
    raise ValueError(f"Could not resolve subject id from parallel_sets={run_key!r}")


def build_model(args: DictConfig, duration: int):
    if args.model_name == "EEGNet":
        return EEGNet(args, T=duration)
    if args.model_name == "EEGNet_with_mask":
        return EEGNet_with_mask(args, T=duration)
    if args.model_name == "cBraMod":
        return CBraModClassifier(args.cBraMod)
    if args.model_name == "RNN":
        return MultiLayerRNN(
            input_size=args.num_channels,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=args.n_class,
            rnn_type=args.rnn_type,
            bidirectional=args.bidirectional,
            dropout_rate=args.dropout_rate,
            last_activation=args.last_activation,
        )
    if args.model_name == "CovTanSVM":
        clf_ = SVC(kernel="rbf", probability=True, class_weight="balanced")
        covest = Covariances("oas")
        ts = TangentSpace()
        return make_pipeline(covest, ts, clf_)
    raise NotImplementedError(f"Unsupported model_name={args.model_name}")


def build_optimizer(args: DictConfig, net: nn.Module):
    separate_cbramod_optimizer = (
        args.model_name == "cBraMod"
        and args.cBraMod.encoder_lr != args.cBraMod.decoder_lr
    )
    if separate_cbramod_optimizer:
        optimizer_cls = optim.AdamW if args.optimizer == "AdamW" else optim.Adam
        encoder_optimizer_kwargs = {"lr": args.cBraMod.encoder_lr, "eps": args.eps}
        decoder_optimizer_kwargs = {"lr": args.cBraMod.decoder_lr, "eps": args.eps}
        if args.optimizer == "AdamW":
            encoder_optimizer_kwargs["weight_decay"] = args.weight_decay
            decoder_optimizer_kwargs["weight_decay"] = args.weight_decay
        return [
            optimizer_cls(net.backbone.parameters(), **encoder_optimizer_kwargs),
            optimizer_cls(net.classifier.parameters(), **decoder_optimizer_kwargs),
        ]
    if args.optimizer == "AdamW":
        return optim.AdamW(
            net.parameters(),
            lr=args.learning_rate,
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    return optim.Adam(net.parameters(), lr=args.learning_rate, eps=args.eps)


def build_scheduler(args: DictConfig, optimizer):
    if not args.scheduler.apply:
        return None
    optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
    if args.scheduler.name == "StepLR":
        scheduler = [
            optim.lr_scheduler.StepLR(
                opt,
                step_size=args.scheduler.step_size,
                gamma=args.scheduler.gamma,
            )
            for opt in optimizers
        ]
    elif args.scheduler.name == "ReduceLROnPlateau":
        scheduler = [
            optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=args.scheduler.factor,
                patience=args.scheduler.patience,
                verbose=True,
            )
            for opt in optimizers
        ]
    elif args.scheduler.name == "CosineAnnealingLR":
        scheduler = [
            optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.scheduler.Tmax)
            for opt in optimizers
        ]
    else:
        raise NotImplementedError
    return scheduler if isinstance(optimizer, list) else scheduler[0]


def optimizer_zero_grad(optimizer) -> None:
    optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
    for opt in optimizers:
        opt.zero_grad()


def optimizer_step(optimizer) -> None:
    optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
    for opt in optimizers:
        opt.step()


def scheduler_step(args: DictConfig, scheduler, loss_val: float) -> None:
    if scheduler is None:
        return
    schedulers = scheduler if isinstance(scheduler, list) else [scheduler]
    for sch in schedulers:
        if args.scheduler.name == "ReduceLROnPlateau":
            sch.step(loss_val)
        else:
            sch.step()


def prepare_inputs(args: DictConfig, inputs: torch.Tensor, labels: torch.Tensor):
    if args.no_avg:
        inputs = torch.concat([inp for inp in inputs])
        labels = labels.repeat_interleave(args.n_trial_avg)
    if args.model_name == "RNN":
        inputs = inputs[:, 0, :, :].permute(0, 2, 1)
    return inputs, labels


def summarize_model(args: DictConfig, model, duration: int) -> None:
    if args.model_name == "RNN":
        summary(model, input_size=(args.batch_size, duration, args.num_channels))
    elif args.model_name == "CovTanSVM":
        print(model)
    else:
        summary(model, input_size=(1, 1, args.num_channels, duration))


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    c = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - c)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def combine_predictions(preds: np.ndarray, method: str, n_class: int) -> np.ndarray:
    if method == "mean":
        return np.mean(preds, axis=0)
    if method == "max":
        return np.max(preds, axis=0)
    if method == "zscore_mean":
        return np.mean(zscore(preds, axis=-1), axis=0)
    if method == "zscore_max":
        return np.max(zscore(preds, axis=-1), axis=0)
    if method == "entropy_weighted":
        probs = softmax(preds, axis=-1)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
        return np.dot(probs.T, entropy)
    if method == "inverse_entropy_weighted":
        probs = softmax(preds, axis=-1)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
        return np.dot(probs.T, 1 / (entropy + 1e-12))
    if method == "majority":
        return np.bincount(np.argmax(preds, axis=1), minlength=n_class)
    raise ValueError(f"ensemble method {method} is not supported.")


def append_test_history(args: DictConfig, rows: list[dict]) -> None:
    if not rows:
        return
    record_history_filepath = getattr(
        args,
        "test_record_history_filepath",
        "outputs/offline/baseline/history_color_rotating_test_fold_test.csv",
    )
    record_history_filepath = os.fspath(record_history_filepath)
    df_new = pd.DataFrame(rows)
    if os.path.exists(record_history_filepath):
        df = pd.read_csv(record_history_filepath)
        for col in df_new.columns:
            if col not in df.columns:
                df[col] = "None"
        for col in df.columns:
            if col not in df_new.columns:
                df_new[col] = "None"
        df = pd.concat([df, df_new[df.columns]], ignore_index=True)
    else:
        os.makedirs(os.path.dirname(record_history_filepath) or ".", exist_ok=True)
        df = df_new
    df.to_csv(record_history_filepath, index=False)
    cprint(f"Saved test history to {record_history_filepath}", "green")


def update_best_model(
    args: DictConfig,
    epoch: int,
    cv: int,
    net: nn.Module,
    acc_tr: np.ndarray,
    acc_val: np.ndarray,
    loss_tr_log: np.ndarray,
    loss_val_log: np.ndarray,
    current_best,
):
    metric = args.save_best.monitor
    if metric == "acc_val":
        score = acc_val[cv, epoch]
        is_better = current_best is None or current_best < score
    elif metric == "loss_val":
        score = loss_val_log[cv, epoch]
        is_better = current_best is None or current_best > score
    elif metric == "acc_tr":
        score = acc_tr[cv, epoch]
        is_better = current_best is None or current_best < score
    elif metric == "loss_tr":
        score = loss_tr_log[cv, epoch]
        is_better = current_best is None or current_best > score
    else:
        raise ValueError(f"save_best.monitor {metric} is not supported.")
    if is_better:
        return score, copy.deepcopy(net), True
    return current_best, None, False


def collect_svm_data(loader):
    inputs_all = []
    labels_all = []
    for inputs, labels in loader:
        inputs_all.append(inputs.cpu().detach().numpy())
        labels_all.append(labels.cpu().detach().numpy())
    inputs_all = np.concatenate(inputs_all, axis=0)[:, 0, :, :]
    labels_all = np.concatenate(labels_all, axis=0)
    return inputs_all, labels_all
