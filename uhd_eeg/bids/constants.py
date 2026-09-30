"""Constants shared by BIDS ingest."""

SFREQ = 256
N_CH_TOTAL = 139
N_CH_EEG = 128
EPOCH_SAMPLES = 2880  # 6.25 s analysis window at 256 Hz (packet-aligned)
UNIT_COEFF = 1.0e-6  # raw ADC → Volts used when writing OpenNeuro files

TASK_LABELS = {
    "overt": "overt",
    "minimallyovert": "minimally overt",
    "covert": "covert",
}

SUBJECT_IDS = tuple(f"sub-{i}" for i in range(1, 10))
