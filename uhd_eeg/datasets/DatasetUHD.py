"Datasets for decoder training and evaluating"
import glob
import os
from pathlib import Path
from typing import Tuple

import mne
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils.data import Dataset

from uhd_eeg.preprocess.adaptive_filter import (
    NLMS,
    bipolar_np,
    get_ch_type_after_resample,
)


class EEGDataset(Dataset):
    """EEG Dataset"""

    def __init__(self, args: DictConfig) -> None:
        """__init__

        Args:
            args (DictConfig): config
        """
        super().__init__()

        mne.set_log_level("ERROR")
        self.args = args
        self.labels_all = self.parse_word_list(args)
        self.fs = args.fs
        n_ch_to_use = args.n_ch_eeg + args.n_ch_noise
        # Only the first trial in args.n_trial_avg cannot get negative jitter due to buffer size.
        duration_with_jitter = int(
            np.round(self.fs * (args.dura_unit * args.n_trial_avg + 1 * args.jitter))
        )
        self.dura = int(args.dura_unit * self.fs * args.n_trial_avg)
        self.window_eegnet = round(self.dura / args.n_trial_avg)
        np.random.seed(0)
        self.device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        )
        n_ch_to_use = args.n_ch_eeg + args.n_ch_noise
        self.info = mne.create_info(
            ch_names=n_ch_to_use,
            sfreq=self.fs,
            ch_types=[
                get_ch_type_after_resample(i, n_ch_to_use) for i in range(n_ch_to_use)
            ],
            verbose=False,
        )
        self.notch_freqs = np.arange(50, self.fs / 2, 50)
        self.filter_length = min(
            int(round(6.6 * args.fs)),
            duration_with_jitter - 1,
        )
        print("notch filter length: ", self.filter_length)

        self.data_ch_idx = np.arange(args.n_ch_eeg)
        self.noise_ch_idx = np.arange(args.n_ch_noise) + args.n_ch_eeg
        self.adapt_filt = NLMS(
            self.data_ch_idx, self.noise_ch_idx, mu=args.nlms.mu, w=args.nlms.w
        )
        self.inner_trial_onset_list = [
            int(args.jitter * self.fs + args.dura_unit * self.fs * i)
            for i in range(args.n_trial_avg + 1)
        ]

        # epoching
        eegs = self.parse_npy_files(args, duration_with_jitter)
        self.preprocessed_data = []
        for st_eeg in eegs:
            st_preproc = self.preproc(st_eeg, self.args)
            self.preprocessed_data.append(
                st_preproc
            )  # [(n_ch, 6.25+jitter)] * n_events

    def __len__(self) -> int:
        """__len__

        Returns:
            int: number of data
        """
        return len(self.labels_all)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """getitem

        Args:
            index (int): data index

        Returns:
            torch.Tensor: EEG data, label
        """
        X = self.preprocessed_data[index]  # (n_ch, n_timesteps)
        newX = []
        for i in range(0, self.args.n_trial_avg):
            if (i != self.args.n_trial_avg - 1) and (self.args.jitter > 0):
                jitter = np.floor(
                    (
                        np.random.randint(
                            -self.args.jitter * self.fs, self.args.jitter * self.fs
                        )
                    )
                ).astype(int)
            else:
                jitter = 0
            newX.append(
                X[
                    :,
                    self.inner_trial_onset_list[i]
                    + jitter : self.inner_trial_onset_list[i + 1]
                    + jitter,
                ]
            )
        newX = torch.stack(newX, dim=0)
        if self.args.no_avg:
            X = newX.unsqueeze(1)  # (n_trial_avg, 1, n_ch_eeg, dura)
        else:
            X = torch.mean(newX, dim=0, keepdim=True)  # (1, n_ch_eeg, dura)
        y = self.labels_all[index]
        # DEBUG
        assert X.shape[-1] == 320, "{} shoud be 320".format(X.shape[-1])

        return X, y

    def parse_npy_files(
        self, args: DictConfig, duration_with_jitter: int
    ) -> np.ndarray:
        """get EEG and EMG data from npy files

        Args:
            args (DictConfig): config
            duration_with_jitter (int): duration with jitter

        Returns:
            np.ndarray: EEG data
        """
        npy_dir = Path(args.gmail.npy_dir)
        if args.use_hydra_savedir:
            npy_dir = Path(get_original_cwd()) / npy_dir
        npy_files = list(npy_dir.glob("*.npy"))
        n_trials = len(npy_files)  # 0 - n_trials
        eeg_emg_trig_data = [None] * n_trials

        for npy_file in npy_files:
            print("loading {}".format(npy_file))
            single_trial_eeg = np.load(npy_file)
            single_trial_eeg = single_trial_eeg[:, -duration_with_jitter:]
            index = int(npy_file.name.replace(".npy", ""))
            eeg_emg_trig_data[index] = single_trial_eeg
        assert np.all(
            [eeg_emg_trig_data[i] is not None for i in range(n_trials)]
        ), "some data is missing"
        eeg_emg_trig_data = np.stack(
            eeg_emg_trig_data, axis=0
        )  # (n_trials, n_ch, n_times)
        return eeg_emg_trig_data

    def parse_word_list(self, args: DictConfig) -> np.ndarray:
        """get word list from csv file

        Args:
            args (DictConfig): config

        Returns:
            np.ndarray: word list
        """
        csv_dir = Path(args.gmail.csv_dir)
        if args.use_hydra_savedir:
            csv_dir = Path(get_original_cwd()) / csv_dir
        words = np.loadtxt(
            csv_dir / f"word_list{args.gmail.csv_header}.csv",
            delimiter=",",
            dtype=int,
        )
        return words

    def preproc(self, eeg: np.ndarray, args: DictConfig) -> torch.Tensor:
        """preprocess EEG data

        Args:
            eeg (np.ndarray): EEG and EEG data
            args (DictConfig): config

        Returns:
            torch.Tensor: preprocessed EEG data
        """
        eeg = bipolar_np(eeg)  # (n_ch_eeg + n_ch_noise, n_samp)
        eeg *= args.unit_coeff
        eeg[self.data_ch_idx] /= args.preamp_gain
        raw = mne.io.RawArray(eeg, self.info, verbose=False)
        raw.notch_filter(
            self.notch_freqs,
            filter_length=self.filter_length,
            fir_design="firwin",
            trans_bandwidth=1.5,
            verbose=False,
        )
        raw.set_eeg_reference("average", verbose=False)
        raw.filter(args.bandpass.low, args.bandpass.high, picks="all", verbose=False)
        if args.wo_adapt_filt:
            eeg_norm = raw.get_data()[: args.n_ch_eeg]
        else:
            eeg_norm = self.adapt_filt(raw.get_data(), normalize="zscore")[
                : args.n_ch_eeg
            ]
        eeg_norm = torch.from_numpy(eeg_norm).float().to(self.device)
        return eeg_norm  # (n_ch_eeg, dura)


class EMGDataset(Dataset):
    """EMG Dataset"""

    def __init__(self, args: DictConfig) -> None:
        """__init__

        Args:
            args (DictConfig): config
        """
        super().__init__()

        mne.set_log_level("ERROR")
        self.args = args
        self.labels_all = self.parse_word_list(args)
        self.fs = args.fs
        n_ch_to_use = args.n_ch_eeg + args.n_ch_noise
        duration_with_jitter = int(
            np.round(self.fs * (args.dura_unit * args.n_trial_avg + 1 * args.jitter))
        )
        self.dura = int(args.dura_unit * self.fs * args.n_trial_avg)
        self.window_eegnet = round(self.dura / args.n_trial_avg)
        np.random.seed(0)
        self.device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        )
        n_ch_to_use = args.n_ch_eeg + args.n_ch_noise
        self.info = mne.create_info(
            ch_names=n_ch_to_use,
            sfreq=self.fs,
            ch_types=[
                get_ch_type_after_resample(i, n_ch_to_use) for i in range(n_ch_to_use)
            ],
            verbose=False,
        )
        self.notch_freqs = np.arange(50, self.fs / 2, 50)
        self.filter_length = min(
            int(round(6.6 * args.fs)),  # 1585,
            duration_with_jitter - 1,
        )
        print("notch filter length: ", self.filter_length)
        self.data_ch_idx = np.arange(args.n_ch_eeg)
        self.noise_ch_idx = np.arange(args.n_ch_noise) + args.n_ch_eeg
        self.adapt_filt = NLMS(
            self.data_ch_idx, self.noise_ch_idx, mu=args.nlms.mu, w=args.nlms.w
        )
        self.inner_trial_onset_list = [
            int(args.jitter * self.fs + args.dura_unit * self.fs * i)
            for i in range(args.n_trial_avg + 1)
        ]
        # epoching
        eegs = self.parse_npy_files(args, duration_with_jitter)
        self.preprocessed_data = []
        for st_eeg in eegs:
            st_preproc = self.preproc(st_eeg, self.args)
            self.preprocessed_data.append(
                st_preproc
            )  # [(n_ch, 6.25+jitter)] * n_events

    def __len__(self) -> int:
        """__len__

        Returns:
            int: number of data
        """
        return len(self.labels_all)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """getitem

        Args:
            index (int): data index

        Returns:
            Tuple[torch.Tensor, int]: EMG data, label
        """
        X = self.preprocessed_data[index]  # (n_ch, n_timesteps)
        newX = []
        for i in range(0, self.args.n_trial_avg):
            if (i != self.args.n_trial_avg - 1) and (self.args.jitter > 0):
                jitter = np.floor(
                    (
                        np.random.randint(
                            -self.args.jitter * self.fs, self.args.jitter * self.fs
                        )
                    )
                ).astype(int)
            else:
                jitter = 0
            newX.append(
                X[
                    :,
                    self.inner_trial_onset_list[i]
                    + jitter : self.inner_trial_onset_list[i + 1]
                    + jitter,
                ]
            )
        newX = torch.stack(newX, dim=0)
        X = torch.mean(newX, dim=0, keepdim=True)  # (1, n_ch_eeg, dura)
        y = self.labels_all[index]
        assert X.shape[-1] == 320, "{} shoud be 320".format(X.shape[-1])
        return X, y

    def parse_npy_files(
        self, args: DictConfig, duration_with_jitter: int
    ) -> np.ndarray:
        """get EEG  and EMG data from npy files

        Args:
            args (DictConfig): config
            duration_with_jitter (int): duration with jitter

        Returns:
            np.ndarray: EEG  and EMG data
        """
        npy_dir = Path(args.gmail.npy_dir)
        if args.use_hydra_savedir:
            npy_dir = Path(get_original_cwd()) / npy_dir
        npy_files = list(npy_dir.glob("*.npy"))
        n_trials = len(npy_files)  # 0 - n_trials
        eeg_emg_trig_data = [None] * n_trials
        for npy_file in npy_files:
            print("loading {}".format(npy_file))
            single_trial_eeg = np.load(npy_file)
            single_trial_eeg = single_trial_eeg[:, -duration_with_jitter:]
            index = int(npy_file.name.replace(".npy", ""))
            eeg_emg_trig_data[index] = single_trial_eeg
        assert np.all(
            [eeg_emg_trig_data[i] is not None for i in range(n_trials)]
        ), "some data is missing"
        eeg_emg_trig_data = np.stack(
            eeg_emg_trig_data, axis=0
        )  # (n_trials, n_ch, n_times)
        return eeg_emg_trig_data

    def parse_word_list(self, args: DictConfig) -> np.ndarray:
        """get word list from csv file

        Args:
            args (DictConfig): config

        Returns:
            np.ndarray: word list
        """
        csv_dir = Path(args.gmail.csv_dir)
        if args.use_hydra_savedir:
            csv_dir = Path(get_original_cwd()) / csv_dir
        words = np.loadtxt(
            csv_dir / f"word_list{args.gmail.csv_header}.csv",
            delimiter=",",
            dtype=int,
        )
        return words

    def preproc(self, eeg: np.ndarray, args: DictConfig) -> torch.Tensor:
        """preprocess EMG data

        Args:
            eeg (np.ndarray): EEG and EMG data
            args (DictConfig): config

        Returns:
            torch.Tensor: preprocessed EMG data
        """
        eeg = bipolar_np(eeg)  # (n_ch_eeg + n_ch_noise, n_samp)
        eeg *= args.unit_coeff
        eeg[self.data_ch_idx] /= args.preamp_gain
        raw = mne.io.RawArray(eeg, self.info, verbose=False)
        raw.notch_filter(
            self.notch_freqs,
            filter_length=self.filter_length,
            fir_design="firwin",
            trans_bandwidth=1.5,
            verbose=False,
        )
        raw.set_eeg_reference("average", verbose=False)
        raw.filter(args.bandpass.low, args.bandpass.high, picks="all", verbose=False)
        emg = raw.get_data()[self.args.n_ch_eeg :, :]
        if self.args.emg_highpass.apply:
            emg = mne.filter.filter_data(
                emg,
                sfreq=self.fs,
                l_freq=self.args.emg_highpass.low,
                h_freq=self.args.emg_highpass.high,
            )
        return torch.from_numpy(emg).float().to(self.device)  # (n_ch_eeg, dura)
