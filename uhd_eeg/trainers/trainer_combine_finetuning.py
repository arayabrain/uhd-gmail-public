import copy
import os
from typing import Optional

import dill
import hydra
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from torch.utils.data.dataset import Subset
from torchinfo import summary

from uhd_eeg.datasets.DatasetUHD import EEGDataset, EMGDataset
from uhd_eeg.models.CNN.EEGNet import EEGNet, EEGNet_with_mask
from uhd_eeg.models.RNN.RNN import MultiLayerRNN

if multiprocessing.get_start_method() == "fork":
    multiprocessing.set_start_method("spawn", force=True)
    print("{} setup done".format(multiprocessing.get_start_method()))


def fit_decoder_CV(
    args: DictConfig,
    dataset: EEGDataset | EMGDataset,
    labels_train: np.ndarray,
    task: str,
    duration: int,
    device: torch.device,
    pretrained_path: Optional[str] = None,
    lr_divide: float = 1.0,
):
    """fit EEGNet with cross-validation

    Args:
        args (DictConfig): config
        dataset (EEGDataset | EMGDataset): dataset
        labels_train (np.ndarray): labels
        task (str): task name ("overt", "min-overt", "covert")
        duration (int): duration
        device (torch.device): device
        pretrained_path (Optional[str], optional): pretrained model path. Defaults to None.
        lr_divide (float, optional): divide learning rate. Defaults to 1.0.

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_
    """
    print("dataset size: ", len(dataset))
    # np.random.seed(0)  # fix split
    inds = np.random.permutation(np.arange(len(dataset)))
    n_cv = args.n_splits
    acc_tr = np.zeros((n_cv, args.n_epochs))
    acc_val = np.zeros((n_cv, args.n_epochs))
    loss_tr_log = np.zeros((n_cv, args.n_epochs))
    loss_val_log = np.zeros((n_cv, args.n_epochs))
    skf = StratifiedKFold(n_splits=n_cv)
    for cv, (ind_tr, ind_val) in enumerate(skf.split(inds, labels_train)):
        # dataset_tr, dataset_val = torch.utils.data.random_split(dataset, [train_size, val_size])
        dataset_tr = Subset(dataset, ind_tr)
        dataset_val = Subset(dataset, ind_val)

        train_loader = torch.utils.data.DataLoader(
            dataset_tr,
            batch_size=args.batch_size,
            num_workers=args.n_worekers,
            pin_memory=False,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=args.n_worekers,
            pin_memory=False,
        )

        if args.model_name == "EEGNet":
            net = EEGNet(args, T=duration)
        elif args.model_name == "EEGNet_with_mask":
            net = EEGNet_with_mask(args, T=duration)
        elif args.model_name == "RNN":
            net = MultiLayerRNN(
                input_size=args.num_channels,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                output_size=args.n_class,
                rnn_type=args.rnn_type,
                bidirectional=args.bidirectional,
                dropout_rate=args.dropout_rate,
                last_activation=args.last_activation,
            )
        elif args.model_name == "CovTanSVM":
            clf_ = SVC(kernel="rbf", probability=True, class_weight="balanced")
            covest = Covariances("oas")
            ts = TangentSpace()
            clf = make_pipeline(covest, ts, clf_)
        else:
            raise NotImplementedError
        if args.model_name != "CovTanSVM":
            net.to(device)
            if pretrained_path is not None:
                net.load_state_dict(torch.load(f"{pretrained_path}_cv{cv}.pth"))
                print(f"pretrained model loaded from {pretrained_path}", flush=True)
                # Fixed weights except for the last layer
                for param in net.parameters():
                    param.requires_grad = False
                last_layer = list(net.children())[-1]
                for param in last_layer.parameters():
                    param.requires_grad = True
        else:
            if pretrained_path is not None:
                with open(pretrained_path, "rb") as f:
                    clf = dill.load(f)
        # ---------------
        #      Loss
        # ---------------
        if args.model_name != "CovTanSVM":
            criterion = nn.CrossEntropyLoss()
            if args.optimizer == "AdamW":
                optimizer = optim.AdamW(
                    net.parameters(),
                    lr=args.learning_rate / lr_divide,
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                )
            else:
                optimizer = optim.Adam(
                    net.parameters(), lr=args.learning_rate / lr_divide, eps=args.eps
                )
            if args.scheduler.apply:
                if args.scheduler.name == "StepLR":
                    scheduler = optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=args.scheduler.step_size,
                        gamma=args.scheduler.gamma,
                    )
                elif args.scheduler.name == "ReduceLROnPlateau":
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=args.scheduler.factor,
                        patience=args.scheduler.patience,
                        verbose=True,
                    )
                elif args.scheduler.name == "CosineAnnealingLR":
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=args.scheduler.Tmax
                    )
                else:
                    raise NotImplementedError
        if args.model_name == "RNN":
            summary(
                net,
                input_size=(args.batch_size, duration, args.num_channels),
            )
        elif args.model_name == "CovTanSVM":
            print(clf)
        else:
            summary(net, input_size=(1, 1, args.num_channels, duration))

        if args.model_name == "CovTanSVM":
            X_tr = []
            y_tr = []
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                X_tr.append(inputs)
                y_tr.append(labels)
            X_tr = np.concatenate(X_tr, axis=0)[:, 0, :, :]
            y_tr = np.concatenate(y_tr, axis=0)

            X_val = []
            y_val = []
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs = inputs.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                X_val.append(inputs)
                y_val.append(labels)
            X_val = np.concatenate(X_val, axis=0)[:, 0, :, :]
            y_val = np.concatenate(y_val, axis=0)

            clf.fit(X_tr, y_tr)
            pred_tr = clf.predict(X_tr)
            pred_val = clf.predict(X_val)
            if cv == 0:
                acc_tr_all = []
                acc_val_all = []
            acc_tr = accuracy_score(y_tr, pred_tr)
            acc_val = accuracy_score(y_val, pred_val)
            acc_tr_all.append(acc_tr)
            acc_val_all.append(acc_val)
            if args.save_results:
                filename = f"CovTanSVM_{task}_cv{cv}.dill"
                if args.use_hydra_savedir:
                    with open(filename, "wb") as f:
                        dill.dump(clf, f)
                    print(f"model saved at {filename}", flush=True)
                else:
                    if cv == 0:
                        dir_save = f"{args.saved_data_root}/{args.config_name}"
                        os.makedirs(dir_save, exist_ok=True)
                    with open(f"{dir_save}/{filename}", "wb") as f:
                        dill.dump(clf, f)
                    print(f"model saved at {dir_save}/{filename}", flush=True)

        else:
            for epoch in range(args.n_epochs):  # loop over the dataset multiple times
                loss_tr = 0.0
                n_data_tr = 0
                n_correct_tr = 0
                for i, data in enumerate(train_loader, 0):
                    net.train()
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    if args.no_avg:
                        inputs = torch.concat([inp for inp in inputs])
                        labels = labels.repeat_interleave(args.n_trial_avg)
                    if args.model_name == "RNN":
                        inputs = inputs[:, 0, :, :].permute(0, 2, 1)

                    optimizer.zero_grad()

                    pred_tr = net(inputs)
                    loss = criterion(pred_tr, labels.long())
                    loss.backward()
                    optimizer.step()
                    n_correct_tr += torch.sum(pred_tr.argmax(axis=-1) == labels).item()
                    n_data_tr += len(labels)
                    loss_tr += loss.item()

                # test data
                loss_val = 0.0
                n_data_val = 0
                n_correct_val = 0
                net.eval()
                for i, data in enumerate(val_loader, 0):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    if args.no_avg:
                        inputs = torch.concat([inp for inp in inputs])
                        labels = labels.repeat_interleave(args.n_trial_avg)
                    if args.model_name == "RNN":
                        inputs = inputs[:, 0, :, :].permute(0, 2, 1)

                    pred_val = net(inputs)
                    loss_val = criterion(pred_val, labels.long()).item()
                    n_correct_val += torch.sum(
                        pred_val.argmax(axis=-1) == labels
                    ).item()
                    n_data_val += len(labels)
                    loss_val += loss.item()
                if args.scheduler.apply:
                    if args.scheduler.name == "ReduceLROnPlateau":
                        scheduler.step(loss_val)
                    else:
                        scheduler.step()

                # epoch end
                acc_tr[cv, epoch] = n_correct_tr / n_data_tr
                acc_val[cv, epoch] = n_correct_val / n_data_val
                loss_tr_log[cv, epoch] = loss_tr / n_data_tr
                loss_val_log[cv, epoch] = loss_val / n_data_val
                if args.save_best.apply:
                    if args.save_best.monitor == "acc_val":
                        if epoch == 0:
                            current_best = acc_val[cv, epoch]
                            net_best = copy.deepcopy(net)
                        elif current_best < acc_val[cv, epoch]:
                            current_best = acc_val[cv, epoch]
                            net_best = copy.deepcopy(net)
                        else:
                            pass
                    elif args.save_best.monitor == "loss_val":
                        if epoch == 0:
                            current_best = loss_val_log[cv, epoch]
                            net_best = copy.deepcopy(net)
                        elif current_best > loss_val_log[cv, epoch]:
                            current_best = loss_val_log[cv, epoch]
                            net_best = copy.deepcopy(net)
                        else:
                            pass
                    elif args.save_best.monitor == "acc_tr":
                        if epoch == 0:
                            current_best = acc_tr[cv, epoch]
                            net_best = copy.deepcopy(net)
                        elif current_best < acc_tr[cv, epoch]:
                            current_best = acc_tr[cv, epoch]
                            net_best = copy.deepcopy(net)
                        else:
                            pass
                    elif args.save_best.monitor == "loss_tr":
                        if epoch == 0:
                            current_best = loss_tr_log[cv, epoch]
                            net_best = copy.deepcopy(net)
                        elif current_best > loss_tr_log[cv, epoch]:
                            current_best = loss_tr[cv, epoch]
                            net_best = copy.deepcopy(net)
                        else:
                            pass
                # print stats every 50 epochs
                n_epoch_per_report = 50
                if epoch % n_epoch_per_report == n_epoch_per_report - 1:
                    print(
                        f"[cv{cv}-{epoch + 1}]",
                        f"loss_tr: {loss_tr_log[cv, epoch-(n_epoch_per_report-1):epoch+1].mean(axis=-1):.2f}",
                        f"loss_val: {loss_val_log[cv, epoch-(n_epoch_per_report-1):epoch+1].mean(axis=-1):.2f}",
                        f"acc_tr: {acc_tr[cv, epoch-(n_epoch_per_report-1):epoch+1].mean(axis=-1):.2f}",
                        f"acc_val: {acc_val[cv, epoch-(n_epoch_per_report-1):epoch+1].mean(axis=-1):.2f}",
                    )
            if args.save_results:
                if cv == 0:
                    if not args.use_hydra_savedir:
                        dir_save = f"{args.saved_data_root}/{args.config_name}"
                        os.makedirs(dir_save, exist_ok=True)
                if args.save_best.apply:
                    net = copy.deepcopy(net_best)
                if args.use_hydra_savedir:
                    torch.save(net.state_dict(), f"weight_{task}_cv{cv}.pth")
                else:
                    torch.save(net.state_dict(), f"{dir_save}/weight_{task}_cv{cv}.pth")
    if args.model_name == "CovTanSVM":
        if args.save_results:
            acc_tr_all = np.array(acc_tr_all)
            acc_val_all = np.array(acc_val_all)
            if args.use_hydra_savedir:
                np.save(f"acc_tr_{task}.npy", acc_tr_all)
                np.save(f"acc_val_{task}.npy", acc_val_all)
            else:
                np.save(f"{dir_save}/acc_tr_{task}.npy", acc_tr_all)
                np.save(f"{dir_save}/acc_val_{task}.npy", acc_val_all)
        print("Finished Training", flush=True)
        print(f"acc_tr_all: {acc_tr_all}", flush=True)
        print(f"acc_val_all: {acc_val_all}", flush=True)
    else:
        if args.use_hydra_savedir:
            np.save(f"loss_tr_{task}.npy", loss_tr_log)
            np.save(f"loss_val_{task}.npy", loss_val_log)
            np.save(f"acc_tr_{task}.npy", acc_tr)
            np.save(f"acc_val_{task}.npy", acc_val)
        else:
            np.save(f"{dir_save}/loss_tr_{task}.npy", loss_tr_log)
            np.save(f"{dir_save}/loss_val_{task}.npy", loss_val_log)
            np.save(f"{dir_save}/acc_tr_{task}.npy", acc_tr)
            np.save(f"{dir_save}/acc_val_{task}.npy", acc_val)
        print("dataset size: ", len(dataset))
        print(
            f" loss_tr: {loss_tr_log[:, -1].mean(axis=0):.2f}, loss_val: {loss_val_log[:, -1].mean(axis=0):.2f}, acc_tr: {acc_tr[:, -1].mean(axis=0):.2f}, acc_val: {acc_val[:, -1].mean(axis=0):.2f}"
        )
        print(
            f"avg acc_val for last 5 epochs for each CV: {acc_val[:, -5:].mean(axis=-1)}"
        )


@hydra.main(
    version_base=None,
    config_path="../../configs/trainer",
    config_name="config_color_combine.yaml",
)
def run(args: DictConfig) -> None:
    OmegaConf.set_struct(args, False)
    OmegaConf.update(args, "gpu", args[args.parallel_sets]["gpu"], merge=True)
    OmegaConf.update(args, "gmail", args[args.parallel_sets], merge=True)
    OmegaConf.set_struct(args, True)
    if args.decode_from == "eeg":
        dataset = EEGDataset(args)
    elif args.decode_from == "emg":
        dataset = EMGDataset(args)
    else:
        raise ValueError("decode_from must be eeg or emg")
    subject = args.parallel_sets.split("-")[0]
    tasks_name = args.parallel_sets.split(f"{subject}-")[-1]
    tasks = tasks_name.split("_")
    csv_dir = args.gmail.csv_dir
    n_files_cum = np.loadtxt(
        f"{csv_dir}/n_files_cum_{tasks_name}_train.csv", delimiter=","
    )
    cum = 0
    task_old = None
    duration = dataset.window_eegnet
    device = dataset.device
    for task, n_files in zip(tasks, n_files_cum):
        print(f"task: {task}")
        indices = np.arange(cum, n_files, dtype=int)
        dataset_train = Subset(dataset, indices)
        labels_all = dataset.labels_all
        labels_train = np.array([labels_all[ind] for ind in indices])

        if task_old is not None:
            if args.use_hydra_savedir:
                pretrained_path = f"weight_{task_old}"
            else:
                pretrained_path = (
                    f"{args.saved_data_root}/{args.config_name}/weight_{task_old}"
                )
            lr_divide = args.lr_divide
        else:
            pretrained_path = None
            lr_divide = 1.0
        fit_decoder_CV(
            args,
            dataset_train,
            labels_train,
            task,
            duration,
            device,
            pretrained_path,
            lr_divide=lr_divide,
        )
        task_old = task
        cum = n_files


if __name__ == "__main__":
    run()
