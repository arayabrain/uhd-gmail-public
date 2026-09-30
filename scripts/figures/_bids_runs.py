"""Published OpenNeuro ds007591 runs addressed by BIDS entities only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BidsRun:
    subject: str  # sub-N
    session: str  # ses-YYYYMMDD
    task: str  # overt | minimallyovert | covert
    acq: str  # calibration | online
    run: str = "01"

    @property
    def key(self) -> str:
        return f"{self.subject}_task-{self.task}_acq-{self.acq}_run-{self.run}"

    @property
    def parallel_set(self) -> str:
        """Hydra ``parallel_sets`` key (no ``_run-`` segment)."""
        return f"{self.subject}_task-{self.task}_acq-{self.acq}"

    @property
    def condition(self) -> str:
        if self.task == "minimallyovert":
            return "minimally_overt"
        return self.task

    @property
    def derived_csv_dir(self) -> str:
        return f"data/derived/{self.subject}/{self.session}"

    @property
    def derived_npy_dir(self) -> str:
        return (
            f"data/derived/{self.subject}/{self.session}/"
            f"task-{self.task}_acq-{self.acq}_run-{self.run}"
        )


def _runs(*specs: tuple[str, str, str, str]) -> tuple[BidsRun, ...]:
    return tuple(BidsRun(subject, session, task, acq) for subject, session, task, acq in specs)


# Offline (acq-calibration) runs used for rotating CV / within-session decoding.
OFFLINE_RUNS: tuple[BidsRun, ...] = _runs(
    ("sub-1", "ses-20230511", "minimallyovert", "calibration"),
    ("sub-1", "ses-20230529", "overt", "calibration"),
    ("sub-1", "ses-20230529", "covert", "calibration"),
    ("sub-2", "ses-20230512", "minimallyovert", "calibration"),
    ("sub-2", "ses-20230512", "overt", "calibration"),
    ("sub-2", "ses-20230516", "covert", "calibration"),
    ("sub-3", "ses-20230523", "overt", "calibration"),
    ("sub-3", "ses-20230524", "minimallyovert", "calibration"),
    ("sub-3", "ses-20230524", "covert", "calibration"),
    ("sub-4", "ses-20260601", "overt", "calibration"),
    ("sub-4", "ses-20260604", "minimallyovert", "calibration"),
    ("sub-4", "ses-20260604", "covert", "calibration"),
    ("sub-5", "ses-20260602", "overt", "calibration"),
    ("sub-5", "ses-20260602", "minimallyovert", "calibration"),
    ("sub-5", "ses-20260602", "covert", "calibration"),
    ("sub-6", "ses-20260518", "overt", "calibration"),
    ("sub-6", "ses-20260605", "minimallyovert", "calibration"),
    ("sub-6", "ses-20260525", "covert", "calibration"),
    ("sub-7", "ses-20260520", "overt", "calibration"),
    ("sub-7", "ses-20260527", "minimallyovert", "calibration"),
    ("sub-7", "ses-20260527", "covert", "calibration"),
    ("sub-8", "ses-20260522", "overt", "calibration"),
    ("sub-8", "ses-20260529", "minimallyovert", "calibration"),
    ("sub-8", "ses-20260529", "covert", "calibration"),
    ("sub-9", "ses-20260523", "overt", "calibration"),
    ("sub-9", "ses-20260530", "minimallyovert", "calibration"),
    ("sub-9", "ses-20260530", "covert", "calibration"),
)

# Post-hoc online (acq-online) runs used for Tables 1–2 and Fig. S7.
# sub-1 minimally overt uses run-01 only (run-02 exists on OpenNeuro but is unused).
ONLINE_RUNS: tuple[BidsRun, ...] = _runs(
    ("sub-1", "ses-20230511", "minimallyovert", "online"),
    ("sub-1", "ses-20230529", "overt", "online"),
    ("sub-1", "ses-20230529", "covert", "online"),
    ("sub-2", "ses-20230512", "minimallyovert", "online"),
    ("sub-2", "ses-20230512", "overt", "online"),
    ("sub-2", "ses-20230516", "covert", "online"),
    ("sub-3", "ses-20230523", "overt", "online"),
    ("sub-3", "ses-20230524", "minimallyovert", "online"),
    ("sub-3", "ses-20230524", "covert", "online"),
    ("sub-4", "ses-20260601", "overt", "online"),
    ("sub-4", "ses-20260604", "minimallyovert", "online"),
    ("sub-4", "ses-20260604", "covert", "online"),
    ("sub-5", "ses-20260602", "overt", "online"),
    ("sub-5", "ses-20260602", "minimallyovert", "online"),
    ("sub-5", "ses-20260602", "covert", "online"),
    ("sub-6", "ses-20260518", "overt", "online"),
    ("sub-6", "ses-20260605", "minimallyovert", "online"),
    ("sub-6", "ses-20260525", "covert", "online"),
    ("sub-7", "ses-20260520", "overt", "online"),
    ("sub-7", "ses-20260527", "minimallyovert", "online"),
    ("sub-7", "ses-20260527", "covert", "online"),
    ("sub-8", "ses-20260522", "overt", "online"),
    ("sub-8", "ses-20260529", "minimallyovert", "online"),
    ("sub-8", "ses-20260529", "covert", "online"),
    ("sub-9", "ses-20260523", "overt", "online"),
    ("sub-9", "ses-20260530", "minimallyovert", "online"),
    ("sub-9", "ses-20260530", "covert", "online"),
)
