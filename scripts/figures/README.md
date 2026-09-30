# Figure scripts (revision)

Manuscript figure → entry-point mapping for `scripts/figures/`. Paths assume the
repo root as the working directory. See the top-level [README](../../README.md)
for install, OpenNeuro download, `configs/paths.yaml`, full run order, GPU
flags, tests, citation, and license.

These scripts generate the **data panels** of each figure. Final manuscript
figures were assembled in presentation software; some layout elements (panel
arrangement, labels, and scale bars) were drawn or adjusted by hand. For
example, the published Fig. 3b 100 ms scale bar was drawn manually while
`--time-bar-sec 0.1` produces the closest script output.

| Manuscript panel | Script / command | GPU |
|---|---|---|
| Fig. 1 preprocessing | `plot_fig1_preprocessing.py` | no |
| Fig. 1b voice volume | `plot_fig1_fig2_rms.py` (from `data/voice_volume/voice_volume_subject.csv`; `*` = P_adj < 0.05) | no |
| Fig. 2a EMG RMS | `plot_fig1_fig2_rms.py` | no |
| Fig. 2 MI | `plot_fig2_mutual_information.py` | no |
| Fig. S1 | `plot_adaptive_filter_waveforms_trace.py` | no |
| Fig. 3b | `make_temporal_ig_paper_trace.py` | no* |
| Fig. 3 temporal tables / Jaccard | `plot_temporal_contribution.py`, `plot_temporal_jaccard_trial_based.py` | no* |
| Fig. 4 spatial IG / Pearson | `plot_eegnet_spatial_contribution.py`, `plot_spatial_correlation_between_tasks.py`, `plot_spatial_contribution_mi_correlation.py` | no* |
| Fig. 5 adapt-filter shift | `plot_eegnet_wo_adapt_diff_spatial_contribution.py`, `plot_spatial_shift_correlation_between_tasks.py` | no* |
| Fig. S2a k-medoids montage | `plot_fig_s2a_electrode_montage.py` | no |
| Fig. S2b offline density | `plot_fig_s2b_channel_decimation_offline.py` (train: `scripts/offline/run_channel_decimation_EEGNet.py`) | train **yes** |
| Fig. S2b online density | `plot_fig_s2b_channel_decimation_online.py` (eval: `scripts/pseudo_online_test/run_channel_decimation_pseudo_online.py`) | eval **yes** |
| Fig. S3 / S4 | `plot_intersubject_variability_publication.py` (+ `scripts/analysis/make_intersubject_tables.py`) | no |
| Fig. S5 wo-adapt Pearson | `plot_spatial_correlation_between_tasks_wo_adapt_filt.py` | no* |
| Fig. S6 weighted / shift | `weighted_corr.py` + shift scripts | no* |
| Fig. S7 jitter (online) | `plot_fig_s7_jitter_ablation.py` (train/eval under `scripts/offline/` + `scripts/pseudo_online_test/`) | train/eval **yes** |
| Tables 1–2 | `evaluate_decoding_accs.py`, `plot_table_decoding_accs.py` | no* |
| ITR | `scripts/analysis/compute_manuscript_itr.py` | no |
| Tables S1–S2 | `scripts/leave_test/` (`evaluate_cross_modal_controls.py`, `train_eeg3_mi_artifact_controls.py`, `summarize_supplementary_tables_s1_s2.py`) | train/eval **yes** |
| Tables S3–S4 | `scripts/analysis/count_bids_trials.py --verify-manuscript` | no |
| IG upstream | `scripts/analysis/compute_ig.py` | **yes** |

\*CPU once IG / history products exist.

Shared helpers: `_ig_common.py`, `_temporal_ig_helpers.py`, `_adaptive_filter_waveforms.py`,
`_jitter_ablation_helpers.py`, `_channel_decimation_helpers.py`, `condition_colors.py`,
`_bids_runs.py`.

`mixed_effects.py` is intentionally not ported.

## Fig. 3b defaults

```bash
uv run python scripts/figures/make_temporal_ig_paper_trace.py \
  --ig-dir outputs/integrated_gradients \
  --precomputed-window-ig-dir outputs/temporal_ig_windows
```

Author-confirmed defaults (published panel):

- `--condition sub-6-overt`
- `--task overt`
- `--selection-type dissimilar`
- `--color orange`
- `--source-trial 33`
- `--cv 0`
- `--top-name top1`
- `--mode averaged`
- `--emg-highpass 60`
- `--ig-steps 32`
- `--dpi 600`
- `--amp-bar-z 2`
- `--time-bar-sec 0.1` (closest script match to the hand-drawn 100 ms bar)
- `--spacing-scale 2` / `--height-scale 2`

### Expected top1 channels (verification)

| Signal | 0-based index | BIDS `channels.tsv` name |
|---|---|---|
| Denoised EEG (EEGNet) | **35** | **EEG036** |
| Minimally preprocessed EEG (`EEGNet_wo_adapt_filt`) | **78** | **EEG079** |
