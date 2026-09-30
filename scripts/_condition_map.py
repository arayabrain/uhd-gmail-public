"""Shared BIDS condition keys used by leave-test / pseudo-online / IG scripts.

Conditions are addressed by OpenNeuro entities only (``sub-N``, ``task``, ``acq``).
"""

from __future__ import annotations

# (public_key, speech_condition, subject_id)
BIDS_CONDITIONS: list[tuple[str, str, str]] = [
    ("sub-1_task-minimallyovert_acq-calibration", "minimally_overt", "sub-1"),
    ("sub-1_task-overt_acq-calibration", "overt", "sub-1"),
    ("sub-1_task-covert_acq-calibration", "covert", "sub-1"),
    ("sub-2_task-minimallyovert_acq-calibration", "minimally_overt", "sub-2"),
    ("sub-2_task-overt_acq-calibration", "overt", "sub-2"),
    ("sub-2_task-covert_acq-calibration", "covert", "sub-2"),
    ("sub-3_task-overt_acq-calibration", "overt", "sub-3"),
    ("sub-3_task-minimallyovert_acq-calibration", "minimally_overt", "sub-3"),
    ("sub-3_task-covert_acq-calibration", "covert", "sub-3"),
    ("sub-4_task-overt_acq-calibration", "overt", "sub-4"),
    ("sub-4_task-minimallyovert_acq-calibration", "minimally_overt", "sub-4"),
    ("sub-4_task-covert_acq-calibration", "covert", "sub-4"),
    ("sub-5_task-overt_acq-calibration", "overt", "sub-5"),
    ("sub-5_task-minimallyovert_acq-calibration", "minimally_overt", "sub-5"),
    ("sub-5_task-covert_acq-calibration", "covert", "sub-5"),
    ("sub-6_task-overt_acq-calibration", "overt", "sub-6"),
    ("sub-6_task-minimallyovert_acq-calibration", "minimally_overt", "sub-6"),
    ("sub-6_task-covert_acq-calibration", "covert", "sub-6"),
    ("sub-7_task-overt_acq-calibration", "overt", "sub-7"),
    ("sub-7_task-minimallyovert_acq-calibration", "minimally_overt", "sub-7"),
    ("sub-7_task-covert_acq-calibration", "covert", "sub-7"),
    ("sub-8_task-overt_acq-calibration", "overt", "sub-8"),
    ("sub-8_task-minimallyovert_acq-calibration", "minimally_overt", "sub-8"),
    ("sub-8_task-covert_acq-calibration", "covert", "sub-8"),
    ("sub-9_task-overt_acq-calibration", "overt", "sub-9"),
    ("sub-9_task-minimallyovert_acq-calibration", "minimally_overt", "sub-9"),
    ("sub-9_task-covert_acq-calibration", "covert", "sub-9"),
]

CONDITION_KEYS = [key for key, _, _ in BIDS_CONDITIONS]
CONDITION_TO_BEHAVIOR = {key: behavior for key, behavior, _ in BIDS_CONDITIONS}
CONDITION_TO_SUBJECT = {key: subject for key, _, subject in BIDS_CONDITIONS}
