<p align="center">
  <img src="docs/logo.png" width="1000">
<br />

# Delineating neural contributions to electroencephalogram-based speech decoding

Public repository for Gmail interface papers using uhd-EEG

## preparation
1. Install requirements package.
   ```bash
   pip install -r requirements.txt
   ```
## usage
1. Save EEG/EMG of various pretreatment processes.
   ```bash
   python uhd_eeg/plot_figures/make_preproc_files.py
   ```
2. Visualization of pre-processing pipeline (Fig. 1)
   ```bash
   python uhd_eeg/plot_figures/plot_preprocesssing.py
   ```
3. Visualization of volume of speech (Fig. 1) and RMS of EMGs (Fig. 2)
   ```bash
   python uhd_eeg/plot_figures/plot_rms.py
   ```
4. Quantify the contamination level of EMG to EEG (mutual information, Fig. 2)
   ```bash
   python uhd_eeg/plot_figures/plot_mis.py
   ```
5. Train decoders. You can specify in `parallel_sets` which subjects and which sessions' data to train.
   ```bash
   python uhd_eeg/trainers/trainer.py -m hydra/launcher=joblib parallel_sets=subject1-1,subject1-2,subject1-3
   ```
6. Copy the trained models and metrics to `data/`
7. Run the inference for online data and evaluate metrics (Table 1, 2, Fig. S1)
   ```bash
   python uhd_eeg/plot_figures/evaluate_accs.py
   ```
8. Visualization of electrodes used when hypothetically reducing electrode density (Fig. S1)
   ```bash
   python uhd_eeg/plot_figures/show_montage_decimation.py
   ```
9. Analysis on decoding contributions (integrated gradients, Fig.3-5, Fig.S2)
   ```bash
   python uhd_eeg/plot_figures/plot_contribution.py
   ```

## TODOs

- [x] add usage for Fig. 3-5
- [x] check excesses and deficiencies for source codes
- [x] add conceptual picutre
