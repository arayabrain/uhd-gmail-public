# g.Pangolin UHD speech decoding project

## Status

Just transfered basic directory structure and usable modules from [MetaAI reimplementation repo](https://github.com/SeanNobel/speech-decoding/).

## preparation
1. install this repository as a pip package in editable mode.
`pip install -e .`

## Data Version Management Policy (Proposal by inoue)
Since the data is expected to be very large, DVC is not suitable.
Then, preprocess pipeline is managed by github and ClearML.
Your approach of preprocessing is required to be written in `uhd_eeg/preprocess/pipline-<approach name>.py`

## Experiment Management Policy
* Use [ClearML](https://clear.ml/docs/latest/docs/) set up on apollo5(http://192.168.1.72:8080) in Nakameguro Studio.
* Tutorial is put in [examples/clearml_tutorials](examples/clearml_tutorials)

## Formatter Policy (Proposal by inoue)
* Use [Black](https://black.readthedocs.io/en/stable/)
    ```
    pip install black
    black .
    ```


## TODOs

- [ ] Convert from W&B to ClearML
- [ ] Check data form and build dataclass & preprocessing
- [ ] etc