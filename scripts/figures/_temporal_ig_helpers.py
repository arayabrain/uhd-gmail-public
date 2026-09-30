"""Minimal helpers for Fig. 3b temporal IG paper-trace figures.

Supports loading precomputed per-window IG tensors so the manuscript layout can
be produced without model checkpoints. Optional checkpoint recompute is gated
behind ``--checkpoint-root`` and is not required for the default path.
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mne.filter import filter_data

from uhd_eeg.preprocess.adaptive_filter import (
    NLMS,
    bipolar_np,
    get_ch_type_after_resample,
)

try:
    from scripts.figures._ig_common import (
        CM_TO_INCH,
        LABEL_COLORS,
        load_ig_tensor,
        normalize_condition,
        normalize_subject_id,
    )
except ImportError:  # pragma: no cover
    from _ig_common import (  # type: ignore[no-redef]
        CM_TO_INCH,
        LABEL_COLORS,
        load_ig_tensor,
        normalize_condition,
        normalize_subject_id,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES_CSV = (
    REPO_ROOT / "outputs" / "temporal_ig_trial_examples" / "selected_trial_candidates.csv"
)

TRACE_COLORS = {
    "Denoised EEG": "#1f77b4",
    "min preprocessed EEG": "#d9a441",
    "EOG": "#2ca02c",
    "EMG upper": "#d62728",
    "EMG lower": "#9467bd",
}

MODEL_OUTPUT_RE = re.compile(
    r"^(?P<condition>.+?)_"
    r"(?P<model>EEGNet_wo_adapt_filt|EMG_EEGNet|EEGNet)_"
    r"(?P<run_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<run_time>\d{2}-\d{2}-\d{2})_"
    r"(?P<task>.+)_cv(?P<cv>\d+)_(?P<kind>igs|trial_predictions)\.(?P<ext>pt|csv)$"
)

__all__ = [
    "CM_TO_INCH",
    "DEFAULT_CANDIDATES_CSV",
    "TRACE_COLORS",
    "CandidateRecord",
    "ModelOutputRecord",
    "PreprocessArgs",
    "TrialPreprocessor",
    "compute_model_window_igs",
    "discover_model_output_records",
    "load_candidate_records",
    "load_precomputed_window_igs",
    "load_trial_averaged_ig_as_windows",
    "preprocess_emg",
    "split_online_inner_trials",
    "temporal_ig_trace_set",
    "top_channel_sets",
    "trim_to_online_duration",
]


@dataclass(frozen=True)
class PreprocessArgs:
    n_ch_eeg: int = 128
    n_ch_noise: int = 3
    unit_coeff: float = 1.0e-6
    preamp_gain: float = 10.0
    fs: int = 256
    fs_after_resample: int = 256
    bandpass_low: float = 2.0
    bandpass_high: float = 118.0
    duration_sec: float = 1.25
    num_trial_avg: int = 5
    jitter_buffer_sec: float = 0.1
    nlms_mu: float = 0.1
    nlms_w: str = "random"


@dataclass(frozen=True)
class CandidateRecord:
    rank: int
    selection_type: str
    condition: str
    subject: str
    task: str
    label: int
    color: str
    source_trial: str
    trial_file: Path
    cv: int
    pred_probability: float
    run_date: str
    run_time: str


@dataclass(frozen=True)
class ModelOutputRecord:
    condition: str
    model: str
    task: str
    cv: int
    run_date: str
    run_time: str
    ig_path: Path | None = None
    predictions_path: Path | None = None


class FilterNLMS:
    def __init__(self, in_ch: int, out_ch: int, mu: float = 0.1, w: str = "random"):
        self.w = self.init_weights(w, in_ch, out_ch)
        self.mu = mu
        self.eps = 0.001

    @staticmethod
    def init_weights(w: str | np.ndarray, in_ch: int, out_ch: int) -> np.ndarray:
        shape = (in_ch, out_ch)
        if isinstance(w, str):
            if w == "random":
                return np.random.normal(0, 0.5, shape)
            if w == "zeros":
                return np.zeros(shape)
            raise ValueError(f"unknown NLMS weight initializer: {w}")
        if w.shape != shape:
            raise ValueError(f"NLMS weights must have shape {shape}, got {w.shape}")
        return np.asarray(w, dtype="float64")

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w

    def run(self, d: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = np.empty_like(d)
        e = np.empty_like(d)
        for k in range(len(d)):
            y[k] = self.predict(x[k])
            e[k] = d[k] - y[k]
            self.w += self.mu / (self.eps + x[k] @ x[k]) * x[k][:, np.newaxis] @ e[k][
                np.newaxis, :
            ]
        return y, e, self.w


class LocalNLMS:
    def __init__(
        self,
        data_ch_idx: np.ndarray,
        noise_ch_idx: np.ndarray,
        mu: float = 0.1,
        w: str = "random",
    ) -> None:
        self.data_ch_idx = data_ch_idx
        self.noise_ch_idx = noise_ch_idx
        self.filter = FilterNLMS(
            in_ch=len(noise_ch_idx), out_ch=len(data_ch_idx), mu=mu, w=w
        )

    def __call__(self, x: np.ndarray, normalize: str | None = None) -> np.ndarray:
        data = x[self.data_ch_idx]
        noise = x[self.noise_ch_idx]
        if normalize == "zscore":
            data = zscore_chwise(data)
            noise = zscore_chwise(noise)
        elif normalize is not None:
            raise ValueError(f"unsupported normalization: {normalize}")
        _, filt_data, _ = self.filter.run(data.T, noise.T)
        x[self.data_ch_idx] = filt_data.T
        return x


class TrialPreprocessor:
    def __init__(self, args: PreprocessArgs) -> None:
        import mne

        np.random.seed(0)
        self.args = args
        n_ch_to_use = args.n_ch_eeg + args.n_ch_noise
        self.info = mne.create_info(
            ch_names=n_ch_to_use,
            sfreq=args.fs,
            ch_types=[
                get_ch_type_after_resample(i, n_ch_to_use) for i in range(n_ch_to_use)
            ],
            verbose=False,
        )
        self.notch_freqs = np.arange(50, args.fs_after_resample / 2, 50)
        self.filter_length = min(
            int(round(6.6 * args.fs_after_resample)),
            round(args.fs_after_resample * args.duration_sec * args.num_trial_avg - 1),
        )
        self.data_ch_idx = np.arange(args.n_ch_eeg)
        noise_ch_idx = np.arange(args.n_ch_noise) + args.n_ch_eeg
        self.adapt_filt = NLMS(
            self.data_ch_idx, noise_ch_idx, mu=args.nlms_mu, w=args.nlms_w
        )

    def preprocess(self, eeg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import mne

        args = self.args
        data = bipolar_np(eeg.copy())
        data *= args.unit_coeff
        data[self.data_ch_idx] /= args.preamp_gain

        raw = mne.io.RawArray(data, self.info, verbose=False)
        raw.notch_filter(
            self.notch_freqs,
            filter_length=self.filter_length,
            fir_design="firwin",
            trans_bandwidth=1.5,
            verbose=False,
        )
        raw.set_eeg_reference("average", verbose=False)
        raw.filter(args.bandpass_low, args.bandpass_high, picks="all", verbose=False)

        minimally_preprocessed = zscore_chwise(raw.get_data()[: args.n_ch_eeg])
        denoised = self.adapt_filt(raw.get_data().copy(), normalize="zscore")[
            : args.n_ch_eeg
        ]
        return minimally_preprocessed, denoised


def zscore_chwise(x: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, ddof=1, keepdims=True)
    return (x - mean) / np.maximum(std, eps)


def trim_to_online_duration(eeg: np.ndarray, args: PreprocessArgs) -> np.ndarray:
    duration = int(round(args.fs * args.duration_sec * args.num_trial_avg))
    return eeg[:, -duration:]


def preprocess_emg(eeg: np.ndarray, args: PreprocessArgs, emg_highpass: float) -> np.ndarray:
    emg = np.vstack(
        (
            eeg[132, :] - eeg[133, :],
            eeg[134, :] - eeg[135, :],
            eeg[136, :] - eeg[137, :],
        )
    )
    emg = filter_data(
        emg,
        args.fs,
        emg_highpass,
        args.fs / 2 - 1,
        verbose=False,
    )
    return zscore_chwise(emg)


def split_online_inner_trials(x: np.ndarray, args: PreprocessArgs) -> np.ndarray:
    window = int(round(args.duration_sec * args.fs))
    chunks = []
    for i in range(args.num_trial_avg):
        start = i * window
        chunks.append(x[:, start : start + window])
    return np.stack(chunks, axis=0)


def top_channel_sets(igs: np.ndarray) -> dict[str, np.ndarray]:
    if igs.ndim == 4 and igs.shape[1] == 1:
        igs = igs[:, 0]
    if igs.ndim != 3:
        raise ValueError(f"expected IG shape (window, channel, time), got {igs.shape}")
    scores = np.abs(igs).mean(axis=(0, 2))
    order = np.argsort(scores)[::-1]
    return {"top1": order[:1], "top10": order[:10]}


def _ig_no_batch_channel(igs: np.ndarray) -> np.ndarray:
    if igs.ndim == 4 and igs.shape[1] == 1:
        igs = igs[:, 0]
    if igs.ndim != 3:
        raise ValueError(f"expected IG shape (window, channel, time), got {igs.shape}")
    return igs


def _temporal_ig_trace(
    igs: np.ndarray,
    channels: np.ndarray,
    mode: str,
    inner_index: int | None = None,
    trace_transform: str = "abs",
) -> np.ndarray:
    abs_igs = np.abs(_ig_no_batch_channel(igs))
    channels = np.asarray(channels, dtype=int)
    if mode == "full":
        trace = np.concatenate(
            [abs_igs[i, channels, :].mean(axis=0) for i in range(abs_igs.shape[0])]
        )
    elif mode == "inner":
        if inner_index is None:
            raise ValueError("inner_index is required for mode='inner'")
        trace = abs_igs[inner_index, channels, :].mean(axis=0)
    elif mode == "averaged":
        trace = abs_igs[:, channels, :].mean(axis=(0, 1))
    else:
        raise ValueError(f"unknown temporal IG mode: {mode}")

    if trace_transform == "abs":
        return trace
    if trace_transform == "abs-zscore":
        std = float(np.nanstd(trace))
        if std == 0.0 or np.isnan(std):
            return np.zeros_like(trace)
        return (trace - float(np.nanmean(trace))) / std
    raise ValueError(f"unknown temporal IG trace transform: {trace_transform}")


def temporal_ig_trace_set(
    eegnet_igs: np.ndarray,
    wo_igs: np.ndarray,
    emg_igs: np.ndarray,
    eegnet_channels: np.ndarray,
    wo_channels: np.ndarray,
    mode: str,
    inner_index: int | None = None,
    trace_transform: str = "abs",
) -> dict[str, np.ndarray]:
    return {
        "Denoised EEG": _temporal_ig_trace(
            eegnet_igs, eegnet_channels, mode, inner_index, trace_transform
        ),
        "min preprocessed EEG": _temporal_ig_trace(
            wo_igs, wo_channels, mode, inner_index, trace_transform
        ),
        "EOG": _temporal_ig_trace(emg_igs, np.array([0]), mode, inner_index, trace_transform),
        "EMG upper": _temporal_ig_trace(
            emg_igs, np.array([1]), mode, inner_index, trace_transform
        ),
        "EMG lower": _temporal_ig_trace(
            emg_igs, np.array([2]), mode, inner_index, trace_transform
        ),
    }


def load_candidate_records(path: Path, rank: int) -> list[CandidateRecord]:
    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["rank"]) != rank:
                continue
            condition = normalize_condition(row["condition"])
            subject = normalize_subject_id(row["subject"])
            records.append(
                CandidateRecord(
                    rank=int(row["rank"]),
                    selection_type=row["selection_type"],
                    condition=condition,
                    subject=subject,
                    task=row["task"],
                    label=int(row["label"]),
                    color=row["color"],
                    source_trial=row["source_trial"],
                    trial_file=Path(row["trial_file"]),
                    cv=int(row["cv"]),
                    pred_probability=float(row["pred_probability"]),
                    run_date=row["run_date"],
                    run_time=row["run_time"],
                )
            )
    records.sort(key=lambda r: (r.task, r.selection_type, r.label, r.condition))
    return records


def candidate_from_cli(
    *,
    condition: str,
    task: str,
    selection_type: str,
    color: str,
    source_trial: str,
    cv: int,
    trial_file: Path | None,
    label: int | None,
    rank: int = 1,
) -> CandidateRecord:
    condition = normalize_condition(condition)
    subject = normalize_subject_id(condition.split("-")[0])
    if label is None:
        label = next(
            (idx for idx, name in LABEL_COLORS.items() if name == color),
            0,
        )
    return CandidateRecord(
        rank=rank,
        selection_type=selection_type,
        condition=condition,
        subject=subject,
        task=task,
        label=label,
        color=color,
        source_trial=str(source_trial),
        trial_file=Path(trial_file) if trial_file is not None else Path(""),
        cv=cv,
        pred_probability=float("nan"),
        run_date="",
        run_time="",
    )


def discover_model_output_records(
    prediction_dir: Path,
) -> dict[tuple[str, str, int, str], ModelOutputRecord]:
    seen_kinds: dict[tuple[str, str, int, str], set[str]] = {}
    paths: dict[tuple[str, str, int, str], dict[str, Path]] = {}
    records: dict[tuple[str, str, int, str], ModelOutputRecord] = {}
    if not prediction_dir.exists():
        return {}
    for path in prediction_dir.iterdir():
        match = MODEL_OUTPUT_RE.match(path.name)
        if match is None:
            continue
        condition = normalize_condition(match.group("condition"))
        model = match.group("model")
        task = match.group("task")
        cv = int(match.group("cv"))
        key = (condition, task, cv, model)
        seen_kinds.setdefault(key, set()).add(match.group("kind"))
        paths.setdefault(key, {})[match.group("kind")] = path
        records[key] = ModelOutputRecord(
            condition=condition,
            model=model,
            task=task,
            cv=cv,
            run_date=match.group("run_date"),
            run_time=match.group("run_time"),
            ig_path=paths[key].get("igs"),
            predictions_path=paths[key].get("trial_predictions"),
        )
    return {
        key: record
        for key, record in records.items()
        if {"igs", "trial_predictions"}.issubset(seen_kinds.get(key, set()))
    }


def _window_ig_candidates(
    ig_dir: Path,
    record: CandidateRecord,
    model: str,
) -> list[Path]:
    stem_bits = [
        f"{record.condition}_{model}",
        f"task-{record.task}" if False else record.task,
        f"cv{record.cv}",
        f"trial{record.source_trial}",
    ]
    patterns = [
        f"{record.condition}_{model}_*_{record.task}_cv{record.cv}_trial{record.source_trial}_windows.npy",
        f"{record.condition}_{model}_*_{record.task}_cv{record.cv}_trial{record.source_trial}_windows.npz",
        f"{record.condition}_{model}_{record.task}_cv{record.cv}_trial{record.source_trial}_windows.npy",
        f"{model}_cv{record.cv}_trial{record.source_trial}_windows.npy",
    ]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(ig_dir.glob(pattern)))
    # Also accept a flat layout: <model>.npy next to a metadata sidecar.
    direct = ig_dir / model / f"trial{record.source_trial}_cv{record.cv}_windows.npy"
    if direct.exists():
        found.append(direct)
    _ = stem_bits  # reserved for future naming schemes
    return found


def load_precomputed_window_igs(
    ig_dir: Path,
    record: CandidateRecord,
    model: str,
) -> np.ndarray | None:
    """Load per-window IG array shaped ``(5, channel, time)`` if present."""
    for path in _window_ig_candidates(ig_dir, record, model):
        if path.suffix == ".npz":
            with np.load(path) as payload:
                key = "igs" if "igs" in payload else payload.files[0]
                igs = np.asarray(payload[key])
        else:
            igs = np.load(path)
        if igs.ndim == 4 and igs.shape[1] == 1:
            igs = igs[:, 0]
        if igs.ndim != 3:
            raise ValueError(f"expected window IG shape (window, channel, time) in {path}")
        return igs
    return None


def load_trial_averaged_ig_as_windows(
    prediction_dir: Path,
    record: CandidateRecord,
    model: str,
) -> np.ndarray | None:
    """Fallback: expand a single averaged trial IG to five identical windows.

    Saved manuscript ``*_igs.pt`` tensors are attributions for the *averaged*
    1.25 s input. Expanding them is enough to draw the manuscript ``averaged``
    paper-trace layout when per-window recomputes are unavailable.
    """
    key = (record.condition, record.task, record.cv, model)
    model_records = discover_model_output_records(prediction_dir)
    model_record = model_records.get(key)
    if model_record is None or model_record.ig_path is None:
        # Tolerate legacy subjectN condition tokens in filenames.
        for alt_key, alt in model_records.items():
            if (
                alt.task == record.task
                and alt.cv == record.cv
                and alt.model == model
                and normalize_condition(alt.condition) == record.condition
            ):
                model_record = alt
                break
    if model_record is None or model_record.ig_path is None or model_record.predictions_path is None:
        return None

    igs = load_ig_tensor(model_record.ig_path)
    ig_index = None
    with model_record.predictions_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = row.get("source_trial") or row.get("dataset_index") or ""
            if str(source) == str(record.source_trial) and int(row["label"]) == record.label:
                ig_index = int(row["ig_index"])
                break
            if str(row.get("ig_index", "")) == str(record.source_trial):
                ig_index = int(row["ig_index"])
                break
    if ig_index is None:
        return None
    trial_ig = igs[ig_index]
    return np.stack([trial_ig for _ in range(5)], axis=0)


class AttrDict(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        self[name] = value


def to_attr_dict(value):
    if isinstance(value, dict):
        return AttrDict({key: to_attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_attr_dict(item) for item in value]
    return value


def integrated_gradients(model, inputs, labels, baseline, n_steps: int):
    import torch

    if n_steps < 1:
        raise ValueError("--ig-steps must be >= 1")
    total_grad = torch.zeros_like(inputs)
    diff = inputs - baseline
    alphas = torch.linspace(1.0 / n_steps, 1.0, n_steps, device=inputs.device)
    for alpha in alphas:
        scaled = (baseline + alpha * diff).detach().requires_grad_(True)
        outputs = model(scaled)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        target_scores = outputs.gather(1, labels[:, None]).sum()
        model.zero_grad(set_to_none=True)
        if scaled.grad is not None:
            scaled.grad.zero_()
        target_scores.backward()
        total_grad += scaled.grad.detach()
    return diff * total_grad / n_steps


def compute_model_window_igs(
    record: ModelOutputRecord,
    windows: np.ndarray,
    label: int,
    ig_steps: int,
    device_name: str,
    checkpoint_root: Path | None = None,
) -> np.ndarray:
    """Recompute per-window IG when checkpoints are available."""
    import torch
    import yaml

    if checkpoint_root is None:
        raise FileNotFoundError(
            "Per-window IG recompute requires --checkpoint-root pointing at "
            "Hydra run directories with model weights. Prefer --ig-dir / "
            "--precomputed-window-ig-dir with saved attributions instead."
        )
    if windows.ndim != 3:
        raise ValueError(f"expected windows shape (window, channel, time), got {windows.shape}")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from uhd_eeg.models.CNN.EEGNet import EEGNet

    run_dir = checkpoint_root / record.run_date / record.run_time
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"missing Hydra config: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        model_args = to_attr_dict(yaml.safe_load(handle))
    model_args.gpu = 0
    model_args.batch_size = windows.shape[0]
    model_path = (
        run_dir
        / f"model_weight_{model_args.config_name}_N{model_args.n_trial_avg}_cv{record.cv}.pth"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"missing model checkpoint: {model_path}")
    if int(model_args.num_channels) != windows.shape[1]:
        raise ValueError(
            f"{record.model} expects {model_args.num_channels} channels, "
            f"got {windows.shape[1]}"
        )
    if model_args.model_name != "EEGNet":
        raise ValueError(f"expected EEGNet architecture, got {model_args.model_name}")

    device = torch.device(
        device_name
        if device_name != "auto"
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    model = EEGNet(model_args, T=windows.shape[-1])
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    inputs = torch.tensor(windows[:, np.newaxis, :, :], dtype=torch.float32, device=device)
    labels = torch.full((inputs.shape[0],), label, dtype=torch.long, device=device)
    baseline = inputs.mean(dim=-1, keepdim=True).expand(inputs.shape)
    attributions = integrated_gradients(model, inputs, labels, baseline, ig_steps)
    return attributions.detach().cpu().numpy()
