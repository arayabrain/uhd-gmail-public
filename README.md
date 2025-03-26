<p align="center">
  <img src="docs/logo.png" width="1000">
<br />

# Delineating neural contributions to electroencephalogram-based speech decoding

Public repository for Gmail interface papers using uhd-EEG<br> 
Author: Motoshige Sato<sup>1</sup>, Yasuo Kabe<sup>1</sup>, Sensho Nobe<sup>1</sup>, Akito Yoshida<sup>1</sup>, Masakazu Inoue<sup>1</sup>, Mayumi Shimizu<sup>1</sup>, Kenichi Tomeoka<sup>1</sup>, Shuntaro Sasai<sup>1*</sup><br> 
<sup>1</sup>[Araya Inc.](https://www.araya.org/en/)

## preparation
1. Install requirements package.
   ```bash
   uv sync
   ```
## usage
1. Save preprocesed EEG/EMG.
   ```bash
   uv run python plot_figures/make_preproc_files.py
   ```
2. Visualization of preprocessing pipeline (Fig. 1)
   ```bash
   uv run python plot_figures/plot_preprocesssing.py
   ```
3. Visualization of volume of speech (Fig. 1) and RMS of EMGs (Fig. 2)
   ```bash
   uv run python plot_figures/plot_rms.py
   ```
4. Quantify the contamination level of EMG to EEG (mutual information, Fig. 2)
   ```bash
   uv run python plot_figures/plot_mis.py
   ```
5. Train decoders. You can specify in `parallel_sets` which subjects and which sessions' data to train.
   ```bash
   uv run python uhd_eeg/trainers/trainer.py -m hydra/launcher=joblib parallel_sets=subject1-1,subject1-2,subject1-3
   ```
6. Copy the trained models and metrics to `data/`
7. Run the inference for online data and evaluate metrics (Table 1, 2, Fig. S1)
   ```bash
   uv run python plot_figures/evaluate_accs.py
   ```
8. Visualization of electrodes used when hypothetically reducing electrode density (Fig. S1)
   ```bash
   uv run python plot_figures/show_montage_decimation.py
   ```
9. Analysis on decoding contributions (integrated gradients, Fig.3-5, Fig.S2)
   ```bash
   uv run python plot_figures/plot_contribution.py
   ```
