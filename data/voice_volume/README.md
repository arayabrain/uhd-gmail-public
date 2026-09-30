# Voice volume (Fig. 1b)

Derived per-trial and subject×condition voice-volume tables used for Fig. 1b.

## Files

- `voice_volume_trial.csv` — one row per trial
- `voice_volume_subject.csv` — one row per subject × condition

## Columns

| Column | Description |
|---|---|
| `volume_db` | Mean of `20 log10(frame RMS)` after band-pass filtering (absolute scale) |
| `volume_db_subject_centered` | Subject-centered volume (subject file only): subtract the subject’s mean over the three condition means |
| `n_trials` | Number of trials averaged into the subject×condition row (subject file only) |
| `source` | Recording source used for that row (see values in the CSVs) |

Absolute `volume_db` depends on the recording source, so compare subjects only
with the subject-centered values.
