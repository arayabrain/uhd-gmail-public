"""CPU tests for manuscript Fig. 1b voice-volume stats from shipped CSV."""

from __future__ import annotations

from pathlib import Path

import pytest

from uhd_eeg.analysis.voice_volume import (
    MANUSCRIPT_CONDITION_MEANS,
    MANUSCRIPT_FRIEDMAN_CHI2,
    MANUSCRIPT_FRIEDMAN_P,
    MANUSCRIPT_PAIRWISE_P_ADJ,
    assert_matches_manuscript,
    condition_means,
    load_subject_voice_volume,
    voice_volume_stats,
)

SUBJECT_CSV = Path("data/voice_volume/voice_volume_subject.csv")


@pytest.fixture(scope="module")
def subject_df():
    if not SUBJECT_CSV.is_file():
        pytest.skip("shipped voice_volume_subject.csv missing")
    return load_subject_voice_volume(SUBJECT_CSV)


def test_voice_volume_condition_means(subject_df):
    means = condition_means(subject_df)
    for condition, expected in MANUSCRIPT_CONDITION_MEANS.items():
        assert means[condition] == pytest.approx(expected, abs=5e-3)


def test_voice_volume_friedman_and_wilcoxon(subject_df):
    stats = voice_volume_stats(subject_df)
    assert stats["n_subjects"] == 9
    assert round(stats["friedman"]["statistic"], 2) == MANUSCRIPT_FRIEDMAN_CHI2
    assert stats["friedman"]["p_value"] == pytest.approx(MANUSCRIPT_FRIEDMAN_P, abs=5e-7)

    by_pair = {(r["condition1"], r["condition2"]): r for r in stats["pairwise"]}
    assert by_pair[("overt", "minimally_overt")]["p_value_corrected"] == pytest.approx(
        MANUSCRIPT_PAIRWISE_P_ADJ[("overt", "minimally_overt")], abs=5e-4
    )
    assert by_pair[("overt", "covert")]["p_value_corrected"] == pytest.approx(
        MANUSCRIPT_PAIRWISE_P_ADJ[("overt", "covert")], abs=5e-4
    )
    assert by_pair[("minimally_overt", "covert")]["p_value_corrected"] == pytest.approx(
        MANUSCRIPT_PAIRWISE_P_ADJ[("minimally_overt", "covert")], abs=5e-4
    )
    assert by_pair[("overt", "minimally_overt")]["star"] == "*"
    assert by_pair[("overt", "covert")]["star"] == "*"
    assert by_pair[("minimally_overt", "covert")]["star"] == ""


def test_voice_volume_assert_matches_manuscript(subject_df):
    assert_matches_manuscript(voice_volume_stats(subject_df))
