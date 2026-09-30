"""Smoke: EMGDataset + 3-ch EEGNet path used for Fig. 3 / Table S1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from uhd_eeg.datasets.DatasetUHD import EMGDataset
from uhd_eeg.models.CNN.EEGNet import EEGNet


def _write_synthetic_emg_run(root: Path, n_trials: int = 10, n_samp: int = 2880) -> None:
    """Layout expected by EMGDataset: flat ``{i}.npy`` + ``word_list.csv``."""
    npy_dir = root / "npy"
    csv_dir = root / "csv"
    npy_dir.mkdir(parents=True)
    csv_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    labels = []
    for i in range(n_trials):
        epoch = rng.normal(scale=30.0, size=(139, n_samp)).astype(np.float64)
        # Non-zero bipolar EMG/EOG differentials (ch 132–137).
        for ch, freq in ((132, 40), (134, 55), (136, 70)):
            epoch[ch] += 50.0 * np.sin(np.linspace(0, freq * np.pi, n_samp))
        np.save(npy_dir / f"{i}.npy", epoch)
        labels.append(i % 5)
    np.savetxt(csv_dir / "word_list.csv", np.asarray(labels, dtype=int), delimiter=",", fmt="%d")


def test_emg_dataset_and_eegnet_forward(tmp_path: Path):
    """EMGDataset (decode_from=emg) yields 3-ch tensors; EEGNet(num_channels=3) runs."""
    _write_synthetic_emg_run(tmp_path)
    args = OmegaConf.create(
        {
            "fs": 256,
            "gpu": 0,
            "dura_unit": 1.25,
            "n_trial_avg": 5,
            "jitter": 0.0,
            "n_ch_eeg": 128,
            "n_ch_noise": 3,
            "unit_coeff": 1.0e-6,
            "preamp_gain": 10,
            "nlms": {"mu": 0.1, "w": "random"},
            "bandpass": {"low": 2.0, "high": 118.0},
            "emg_highpass": {"apply": False, "low": 60, "high": 127},
            "wo_adapt_filt": False,
            "use_hydra_savedir": False,
            "gmail": {
                "npy_dir": str(tmp_path / "npy"),
                "csv_dir": str(tmp_path / "csv"),
                "csv_header": "",
            },
            "model_name": "EEGNet",
            "num_channels": 3,
            "n_class": 5,
            "k1": 30,
            "k2": 4,
            "F1": 16,
            "F2": 32,
            "D": 2,
            "p1": 2,
            "p2": 4,
            "dr1": 0.5,
            "dr2": 0.75,
            "hilbert_transform": False,
        }
    )

    dataset = EMGDataset(args)
    assert len(dataset) == 10
    x, y = dataset[0]
    # Averaged window: (1, n_emg_ch, T) with channels EOG / EMG upper / EMG lower.
    assert x.shape == (1, 3, dataset.window_eegnet)
    assert int(y) in range(5)

    device = torch.device("cpu")
    model = EEGNet(args, T=dataset.window_eegnet).to(device)
    logits = model(x.unsqueeze(0).to(device))  # (1, 1, 3, T)
    assert logits.shape == (1, args.n_class)
