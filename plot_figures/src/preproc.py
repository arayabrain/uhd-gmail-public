"""preprocess EEG data"""

import mne
import numpy as np

from uhd_eeg.preprocess.adaptive_filter import (
    NLMS,
    bipolar_np,
    get_ch_type_after_resample,
)


class Args:
    """preproc args"""

    n_ch = 139
    n_ch_eeg = 128
    n_ch_noise = 3
    unit_coeff = 1.0e-6
    preamp_gain = 10
    eps = 1.0e-20
    th = 5
    dura_sec = 1.25
    Fs = 256
    Fs_after_resample = 256
    bandpass_low = 2.0
    bandpass_high = 118.0
    num_trial_avg = 5
    nlms_mu = 0.1
    nlms_w = "random"
    normalize = "zscore"  # None | "min_max" | "zscore"


class Preprocesser:
    """preprocesser class"""

    def __init__(self, args: Args):
        np.random.seed(0)
        # prepare filters
        self.unit_coeff = args.unit_coeff
        self.preamp_gain = args.preamp_gain
        self.bandpass_low = args.bandpass_low
        self.bandpass_high = args.bandpass_high
        self.n_ch_eeg = args.n_ch_eeg
        self.normalize = args.normalize

        n_ch_to_use = args.n_ch_eeg + args.n_ch_noise
        self.info = mne.create_info(
            ch_names=n_ch_to_use,
            sfreq=args.Fs,
            ch_types=[
                get_ch_type_after_resample(i, n_ch_to_use) for i in range(n_ch_to_use)
            ],
            verbose=False,
        )
        self.notch_freqs = np.arange(50, args.Fs_after_resample / 2, 50)
        self.filter_length = min(
            int(round(6.6 * args.Fs_after_resample)),
            round(args.Fs_after_resample * args.dura_sec * args.num_trial_avg - 1),
        )
        self.data_ch_idx = np.arange(args.n_ch_eeg)
        noise_ch_idx = np.arange(args.n_ch_noise) + args.n_ch_eeg
        self.adapt_filt = NLMS(
            self.data_ch_idx, noise_ch_idx, mu=args.nlms_mu, w=args.nlms_w
        )

    def preproc(self, eeg: np.ndarray) -> np.ndarray:
        """preproc pipeline w/o avg

        Args:
            eeg (np.ndarray): eeg

        Returns:
            np.ndarray: preprocessed data
        """
        eeg = bipolar_np(eeg)  # (n_ch_eeg + n_ch_noise, n_samp)
        eeg *= self.unit_coeff
        eeg[self.data_ch_idx] /= self.preamp_gain
        raw = mne.io.RawArray(eeg, self.info, verbose=False)
        raw.notch_filter(
            self.notch_freqs,
            filter_length=self.filter_length,
            fir_design="firwin",
            trans_bandwidth=1.5,
            verbose=False,
        )
        raw.set_eeg_reference("average", verbose=False)
        raw.filter(self.bandpass_low, self.bandpass_high, picks="all", verbose=False)
        eeg_norm = self.adapt_filt(raw.get_data(), normalize=self.normalize)[
            : self.n_ch_eeg
        ]
        return eeg_norm
