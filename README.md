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
1. Train decoders. You can specify in `parallel_sets` which subjects and which sessions' data to train.
   ```bash
   python uhd_eeg/trainers/trainer.py -m hydra/launcher=joblib parallel_sets=subject1-1,subject1-2,subject1-3
   ```

## TODOs

- [ ] add usage
- [ ] check excesses and deficiencies for source codes
- [x] add conceptual picutre
