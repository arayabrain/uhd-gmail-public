import copy
import os

import dill
import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from termcolor import cprint
from torch.utils.data.dataset import Subset

from uhd_eeg.datasets.DatasetUHD import EEGDataset, EMGDataset
from uhd_eeg.trainers.trainer import CalculateAndRecordStats
from uhd_eeg.trainers.trainer_within_offline_split import (
    append_test_history,
    build_model,
    build_optimizer,
    build_scheduler,
    collect_svm_data,
    get_behavior,
    get_save_dir,
    get_subject_id,
    model_path,
    optimizer_step,
    optimizer_zero_grad,
    prepare_inputs,
    scheduler_step,
    summarize_model,
    update_best_model,
)
from uhd_eeg.utils.seeding import seed_everything


def evaluate_model(
    args: DictConfig, model, loader, device: torch.device
) -> tuple[float, float]:
    pred_scores = []
    labels_all = []
    with torch.no_grad():
        for inputs, labels in loader:
            if args.model_name == "CovTanSVM":
                x = inputs.cpu().detach().numpy()[:, 0, :, :]
                scores = model.predict_proba(x)[0]
                labels_np = labels.cpu().detach().numpy()
            else:
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs, labels = prepare_inputs(args, inputs, labels)
                scores = model(inputs).cpu().detach().numpy()
                scores = scores.mean(axis=0) if scores.shape[0] > 1 else scores[0]
                labels_np = labels.cpu().detach().numpy()
                labels_np = labels_np[:1] if labels_np.shape[0] > 1 else labels_np
            pred_scores.append(scores)
            labels_all.append(labels_np[0])

    pred_labels = np.argmax(np.array(pred_scores), axis=1)
    labels_all = np.array(labels_all)
    return (
        accuracy_score(labels_all, pred_labels),
        balanced_accuracy_score(labels_all, pred_labels),
    )


def fit_rotating_test_fold(
    args: DictConfig,
    dataset: EEGDataset | EMGDataset,
    duration: int,
    device: torch.device,
) -> None:
    """10-fold train/validation/test with following-fold validation."""
    n_cv = args.n_splits
    labels_all = dataset.labels_all
    skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=args.seed)
    folds = [val_idx for _, val_idx in skf.split(np.arange(len(dataset)), labels_all)]

    acc_tr = np.zeros((n_cv, args.n_epochs))
    acc_val = np.zeros((n_cv, args.n_epochs))
    balanced_acc_tr = np.zeros((n_cv, args.n_epochs))
    balanced_acc_val = np.zeros((n_cv, args.n_epochs))
    loss_tr_log = np.zeros((n_cv, args.n_epochs))
    loss_val_log = np.zeros((n_cv, args.n_epochs))
    test_history_rows = []
    hydra_file_path = os.path.join(os.getcwd(), ".hydra", "config.yaml")
    subject_id = get_subject_id(args)

    for test_fold in range(n_cv):
        val_fold = (test_fold + 1) % n_cv
        train_folds = [fold for fold in range(n_cv) if fold not in (test_fold, val_fold)]
        ind_test = folds[test_fold]
        ind_val = folds[val_fold]
        ind_tr = np.concatenate([folds[fold] for fold in train_folds])

        cprint(
            f"test_fold={test_fold}, val_fold={val_fold}, train_folds={train_folds}",
            "cyan",
        )
        recorder = CalculateAndRecordStats(
            args.record_history_filepath,
            test_fold,
            get_behavior(args),
            args.model_name,
            subject_id,
        )

        train_loader = torch.utils.data.DataLoader(
            Subset(dataset, ind_tr),
            batch_size=args.batch_size,
            num_workers=args.n_worekers,
            pin_memory=False,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            Subset(dataset, ind_val),
            batch_size=args.batch_size,
            num_workers=args.n_worekers,
            pin_memory=False,
        )
        test_loader = torch.utils.data.DataLoader(
            Subset(dataset, ind_test),
            batch_size=1,
            num_workers=args.n_worekers,
            pin_memory=False,
            shuffle=False,
        )

        model = build_model(args, duration)
        summarize_model(args, model, duration)

        if args.model_name == "CovTanSVM":
            x_tr, y_tr = collect_svm_data(train_loader)
            x_val, y_val = collect_svm_data(val_loader)
            model.fit(x_tr, y_tr)
            pred_tr = model.predict(x_tr)
            pred_val = model.predict(x_val)
            acc_tr_cv = accuracy_score(y_tr, pred_tr)
            acc_val_cv = accuracy_score(y_val, pred_val)
            balanced_acc_tr_cv = balanced_accuracy_score(y_tr, pred_tr)
            balanced_acc_val_cv = balanced_accuracy_score(y_val, pred_val)
            recorder.update(
                loss_tr=0.0,
                loss_val=0.0,
                acc_tr=acc_tr_cv,
                acc_val=acc_val_cv,
                balanced_acc_tr=balanced_acc_tr_cv,
                balanced_acc_val=balanced_acc_val_cv,
                epoch=0,
            )
            if args.save_results:
                with open(model_path(args, test_fold), "wb") as f:
                    dill.dump(model, f)
        else:
            model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = build_optimizer(args, model)
            scheduler = build_scheduler(args, optimizer)
            current_best = None
            model_best = None

            for epoch in range(args.n_epochs):
                loss_tr = 0.0
                n_data_tr = 0
                n_correct_tr = 0
                pred_tr_array = []
                label_tr_array = []
                model.train()
                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    inputs, labels = prepare_inputs(args, inputs, labels)
                    optimizer_zero_grad(optimizer)
                    pred_tr = model(inputs)
                    loss = criterion(pred_tr, labels.long())
                    loss.backward()
                    optimizer_step(optimizer)
                    n_correct_tr += torch.sum(pred_tr.argmax(axis=-1) == labels).item()
                    n_data_tr += len(labels)
                    loss_tr += loss.item()
                    pred_tr_array.append(pred_tr.argmax(axis=-1).cpu().detach())
                    label_tr_array.append(labels.cpu().detach())

                pred_tr_array = torch.cat(pred_tr_array).numpy()
                label_tr_array = torch.cat(label_tr_array).numpy()
                balanced_acc_tr[test_fold, epoch] = balanced_accuracy_score(
                    label_tr_array, pred_tr_array
                )

                loss_val = 0.0
                n_data_val = 0
                n_correct_val = 0
                pred_val_array = []
                label_val_array = []
                model.eval()
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        inputs, labels = prepare_inputs(args, inputs, labels)
                        pred_val = model(inputs)
                        loss = criterion(pred_val, labels.long())
                        n_correct_val += torch.sum(
                            pred_val.argmax(axis=-1) == labels
                        ).item()
                        n_data_val += len(labels)
                        loss_val += loss.item()
                        pred_val_array.append(pred_val.argmax(axis=-1).cpu().detach())
                        label_val_array.append(labels.cpu().detach())

                pred_val_array = torch.cat(pred_val_array).numpy()
                label_val_array = torch.cat(label_val_array).numpy()
                balanced_acc_val[test_fold, epoch] = balanced_accuracy_score(
                    label_val_array, pred_val_array
                )
                acc_tr[test_fold, epoch] = n_correct_tr / n_data_tr
                acc_val[test_fold, epoch] = n_correct_val / n_data_val
                loss_tr_log[test_fold, epoch] = loss_tr / n_data_tr
                loss_val_log[test_fold, epoch] = loss_val / n_data_val
                scheduler_step(args, scheduler, loss_val_log[test_fold, epoch])

                if args.save_best.apply:
                    current_best, new_model_best, get_best = update_best_model(
                        args,
                        epoch,
                        test_fold,
                        model,
                        acc_tr,
                        acc_val,
                        loss_tr_log,
                        loss_val_log,
                        current_best,
                    )
                    if get_best:
                        model_best = new_model_best
                        recorder.update(
                            loss_tr=loss_tr_log[test_fold, epoch],
                            loss_val=loss_val_log[test_fold, epoch],
                            acc_tr=acc_tr[test_fold, epoch],
                            acc_val=acc_val[test_fold, epoch],
                            balanced_acc_tr=balanced_acc_tr[test_fold, epoch],
                            balanced_acc_val=balanced_acc_val[test_fold, epoch],
                            epoch=epoch + 1,
                        )
                elif epoch == args.n_epochs - 1:
                    recorder.update(
                        loss_tr=loss_tr_log[test_fold, epoch],
                        loss_val=loss_val_log[test_fold, epoch],
                        acc_tr=acc_tr[test_fold, epoch],
                        acc_val=acc_val[test_fold, epoch],
                        balanced_acc_tr=balanced_acc_tr[test_fold, epoch],
                        balanced_acc_val=balanced_acc_val[test_fold, epoch],
                        epoch=epoch + 1,
                    )

            if args.save_best.apply and model_best is not None:
                model = copy.deepcopy(model_best)
            if args.save_results:
                torch.save(model.state_dict(), model_path(args, test_fold))

        recorder.save()
        acc_test, balanced_acc_test = evaluate_model(args, model, test_loader, device)
        test_history_rows.append(
            {
                "sbj": subject_id,
                "behavior": get_behavior(args),
                "model_name": args.model_name,
                "eval_type": "single",
                "CV": test_fold,
                "test_fold": test_fold,
                "val_fold": val_fold,
                "train_folds": ";".join(map(str, train_folds)),
                "ensemble_method": "None",
                "n_models": 1,
                "acc_test": round(acc_test, 3),
                "balanced_acc_test": round(balanced_acc_test, 3),
                "config": hydra_file_path,
            }
        )

    dir_save = get_save_dir(args)
    np.save(f"{dir_save}/rotating_test_fold_loss_tr.npy", loss_tr_log)
    np.save(f"{dir_save}/rotating_test_fold_loss_val.npy", loss_val_log)
    np.save(f"{dir_save}/rotating_test_fold_acc_tr.npy", acc_tr)
    np.save(f"{dir_save}/rotating_test_fold_acc_val.npy", acc_val)
    np.save(f"{dir_save}/rotating_test_fold_balanced_acc_tr.npy", balanced_acc_tr)
    np.save(f"{dir_save}/rotating_test_fold_balanced_acc_val.npy", balanced_acc_val)
    append_test_history(args, test_history_rows)


@hydra.main(
    version_base=None,
    config_path="../../configs/trainer",
    config_name="config_color_within_offline_split.yaml",
)
def run(args: DictConfig) -> None:
    seed_everything(int(args.seed))
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

    seed_everything(int(args.seed))
    fit_rotating_test_fold(args, dataset, dataset.window_eegnet, dataset.device)


if __name__ == "__main__":
    run()
