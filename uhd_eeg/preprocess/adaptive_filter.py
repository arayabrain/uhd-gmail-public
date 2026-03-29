import mne
import numpy as np
from tqdm import tqdm

from uhd_eeg.preprocess.padasip_mod import FilterNLMS


class NLMS:
    """Normalized Least Mean Squares filter for EEG data"""

    def __init__(self, data_ch_idx, noise_ch_idx, mu=0.1, w="random"):
        self.data_ch_idx = data_ch_idx
        self.noise_ch_idx = noise_ch_idx
        self.filter = FilterNLMS(
            in_ch=len(noise_ch_idx), out_ch=len(data_ch_idx), mu=mu, w=w
        )

    def __call__(self, x, normalize=None, replace_noise=False):
        """x: (n_ch, n_samp)"""
        data = x[self.data_ch_idx]  # (n_data_ch, n_samp)
        noise = x[self.noise_ch_idx]  # (n_noise_ch, n_samp)
        if normalize == "zscore":
            data = normalize_chwise_zscore(data, ddof=1, axis=-1)
            noise = normalize_chwise_zscore(noise, ddof=1, axis=-1)
        elif normalize == "min_max":
            data = normalize_chwise_min_max(data, axis=-1)
            noise = normalize_chwise_min_max(noise, axis=-1)
        elif normalize is not None:
            raise ValueError(
                f"Normalization method should be 'zscore' or 'min_max', not {normalize}"
            )

        _, filt_data, _ = self.filter.run(data.T, noise.T)  # (n_samp, n_data_ch)
        x[self.data_ch_idx] = filt_data.T
        if replace_noise:
            x[self.noise_ch_idx] = noise

        return x


def filt_whole_raw(
    raw,
    data_types=("eeg",),
    noise_types=("eog", "emg"),
    epoch_sec=0.5,
    mu=0.1,
    use_bipolar=True,
    normalize: str = None,
    replace_noise=False,
    w="random",
):
    """apply adaptive filter to whole raw data."""
    if use_bipolar:
        raw = bipolar(raw)

    data_ch_idx = np.where(np.isin(raw.get_channel_types(), data_types))[0]
    noise_ch_idx = np.where(np.isin(raw.get_channel_types(), noise_types))[0]
    data = raw.get_data()  # (n_ch, n_samp)

    filt = NLMS(data_ch_idx, noise_ch_idx, mu=mu, w=w)

    n_samp = data.shape[1]
    epoch_samp = round(epoch_sec * raw.info["sfreq"])
    has_extra = n_samp % epoch_samp != 0
    num_epochs = n_samp // epoch_samp + int(has_extra)

    filt_data = []
    for epoch in tqdm(range(num_epochs), desc="filtering"):
        start = epoch * epoch_samp
        end = min(n_samp, start + epoch_samp)
        filt_data.append(
            filt(data[:, start:end], normalize=normalize, replace_noise=replace_noise)
        )
        if end == n_samp:
            break

    raw._data = np.concatenate(filt_data, axis=-1)
    return raw


def normalize_chwise_min_max(x, axis=None):
    min_ = x.min(axis=axis, keepdims=True)
    max_ = x.max(axis=axis, keepdims=True)
    return (x - min_) / (max_ - min_) - 0.5


def normalize_chwise_zscore(x, ddof=0, eps=None, axis=None):
    eps = eps or np.finfo(float).eps
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, ddof=ddof, keepdims=True)
    return (x - mean) / (std + eps)


def bipolar(raw, anodes=None, cathodes=None, ch_names=None):
    if anodes is None:
        anodes = ["128", "130", "132", "134", "136"]
    if cathodes is None:
        cathodes = ["129", "131", "133", "135", "137"]
    if ch_names is None:
        ch_names = ["DISPLAY", "MIC", "EOG", "EMG_UPPER_LIP", "EMG_LOWER_LIP"]
    raw = mne.set_bipolar_reference(raw, anodes, cathodes, ch_names, copy=False)
    return raw


def bipolar_np(data, anodes=None, cathodes=None, n_ch_eeg=128):
    """
    Bipolarize EEG data.
    Return only EEG channels and bipolarized channels.
    TODO: Currently only works for bipolarizing non-EEG channels.
    """
    if anodes is None:
        anodes = [132, 134, 136]
    if cathodes is None:
        cathodes = [133, 135, 137]

    bipoled_data = np.concatenate(
        [data[:n_ch_eeg], data[anodes] - data[cathodes]], axis=0
    )
    return bipoled_data


def get_ch_type_after_resample(ch_idx, num_channel):
    assert num_channel == 131
    if ch_idx < 128:
        return "eeg"
    elif ch_idx in [128]:
        return "eog"
    elif ch_idx in [129, 130]:
        return "emg"
