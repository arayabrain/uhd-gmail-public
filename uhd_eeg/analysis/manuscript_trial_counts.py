"""Expected trial counts from Supplementary Tables S3 and S4 (supplementary.tex)."""

from __future__ import annotations

from dataclasses import dataclass

WORD_COLUMNS = ("green", "magenta", "orange", "violet", "yellow")


@dataclass(frozen=True)
class TrialCountRow:
    subject: str  # sub-N
    recording_type: str  # offline | online
    condition: str  # overt | minimally_overt | covert
    n_total: int
    green: int
    magenta: int
    orange: int
    violet: int
    yellow: int

    def word_counts(self) -> dict[str, int]:
        return {
            "green": self.green,
            "magenta": self.magenta,
            "orange": self.orange,
            "violet": self.violet,
            "yellow": self.yellow,
        }


def _row(
    subject_num: int,
    recording_type: str,
    condition: str,
    n: int,
    green: int,
    magenta: int,
    orange: int,
    violet: int,
    yellow: int,
) -> TrialCountRow:
    return TrialCountRow(
        subject=f"sub-{subject_num}",
        recording_type=recording_type,
        condition=condition,
        n_total=n,
        green=green,
        magenta=magenta,
        orange=orange,
        violet=violet,
        yellow=yellow,
    )


# Supplementary Table S3 (per subject); transcription from supplementary.tex.
SUPPLEMENTARY_TABLE_S3: tuple[TrialCountRow, ...] = (
    _row(1, "offline", "covert", 100, 24, 23, 24, 17, 12),
    _row(1, "online", "covert", 50, 11, 17, 10, 9, 3),
    _row(1, "offline", "minimally_overt", 100, 22, 22, 20, 20, 16),
    _row(1, "online", "minimally_overt", 50, 14, 18, 9, 5, 4),
    _row(1, "offline", "overt", 100, 24, 23, 17, 20, 16),
    _row(1, "online", "overt", 50, 7, 14, 13, 8, 8),
    _row(2, "offline", "covert", 100, 29, 26, 18, 14, 13),
    _row(2, "online", "covert", 50, 10, 15, 13, 6, 6),
    _row(2, "offline", "minimally_overt", 100, 30, 26, 18, 15, 11),
    _row(2, "online", "minimally_overt", 50, 18, 9, 11, 6, 6),
    _row(2, "offline", "overt", 100, 24, 26, 20, 15, 15),
    _row(2, "online", "overt", 50, 17, 14, 6, 8, 5),
    _row(3, "offline", "covert", 100, 37, 28, 13, 8, 14),
    _row(3, "online", "covert", 50, 13, 14, 9, 7, 7),
    _row(3, "offline", "minimally_overt", 100, 29, 27, 18, 12, 14),
    _row(3, "online", "minimally_overt", 50, 19, 10, 7, 6, 8),
    _row(3, "offline", "overt", 100, 31, 25, 18, 13, 13),
    _row(3, "online", "overt", 50, 12, 14, 8, 6, 10),
    _row(4, "offline", "covert", 100, 20, 20, 20, 20, 20),
    _row(4, "online", "covert", 50, 10, 10, 10, 10, 10),
    _row(4, "offline", "minimally_overt", 100, 20, 20, 20, 20, 20),
    _row(4, "online", "minimally_overt", 50, 10, 10, 10, 10, 10),
    _row(4, "offline", "overt", 100, 20, 20, 20, 20, 20),
    _row(4, "online", "overt", 50, 10, 10, 10, 10, 10),
    _row(5, "offline", "covert", 100, 20, 20, 20, 20, 20),
    _row(5, "online", "covert", 50, 12, 8, 12, 9, 9),
    _row(5, "offline", "minimally_overt", 100, 20, 20, 20, 20, 20),
    _row(5, "online", "minimally_overt", 50, 9, 11, 10, 9, 11),
    _row(5, "offline", "overt", 100, 20, 20, 20, 20, 20),
    _row(5, "online", "overt", 50, 10, 12, 10, 8, 10),
    _row(6, "offline", "covert", 100, 20, 20, 20, 20, 20),
    _row(6, "online", "covert", 50, 10, 11, 7, 9, 13),
    _row(6, "offline", "minimally_overt", 100, 20, 20, 20, 20, 20),
    _row(6, "online", "minimally_overt", 50, 9, 10, 10, 10, 11),
    _row(6, "offline", "overt", 100, 20, 20, 20, 20, 20),
    _row(6, "online", "overt", 50, 9, 9, 9, 10, 13),
    _row(7, "offline", "covert", 100, 20, 20, 20, 20, 20),
    _row(7, "online", "covert", 50, 10, 10, 10, 10, 10),
    _row(7, "offline", "minimally_overt", 100, 20, 20, 20, 20, 20),
    _row(7, "online", "minimally_overt", 50, 10, 10, 10, 10, 10),
    _row(7, "offline", "overt", 100, 20, 20, 20, 20, 20),
    _row(7, "online", "overt", 50, 11, 9, 10, 10, 10),
    _row(8, "offline", "covert", 100, 20, 20, 20, 20, 20),
    _row(8, "online", "covert", 50, 11, 10, 9, 9, 11),
    _row(8, "offline", "minimally_overt", 100, 20, 20, 20, 20, 20),
    _row(8, "online", "minimally_overt", 50, 11, 9, 12, 9, 9),
    _row(8, "offline", "overt", 100, 20, 20, 20, 20, 20),
    _row(8, "online", "overt", 50, 10, 9, 10, 10, 11),
    _row(9, "offline", "covert", 100, 20, 20, 20, 20, 20),
    _row(9, "online", "covert", 50, 11, 8, 12, 10, 9),
    _row(9, "offline", "minimally_overt", 100, 20, 20, 20, 20, 20),
    _row(9, "online", "minimally_overt", 50, 14, 8, 9, 12, 7),
    _row(9, "offline", "overt", 100, 20, 20, 20, 20, 20),
    _row(9, "online", "overt", 50, 13, 9, 8, 10, 10),
)


@dataclass(frozen=True)
class TrialCountTotal:
    recording_type: str
    condition: str
    n_total: int
    green: int
    magenta: int
    orange: int
    violet: int
    yellow: int


# Supplementary Table S4 (totals across subjects).
SUPPLEMENTARY_TABLE_S4: tuple[TrialCountTotal, ...] = (
    TrialCountTotal("offline", "covert", 900, 210, 197, 175, 159, 159),
    TrialCountTotal("online", "covert", 450, 98, 103, 92, 79, 78),
    TrialCountTotal("offline", "minimally_overt", 900, 201, 195, 176, 167, 161),
    TrialCountTotal("online", "minimally_overt", 450, 114, 95, 88, 77, 76),
    TrialCountTotal("offline", "overt", 900, 199, 194, 175, 168, 164),
    TrialCountTotal("online", "overt", 450, 99, 100, 84, 80, 87),
)
