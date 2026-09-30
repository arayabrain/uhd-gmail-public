"""SHA-256 check for the authors' public CBraMod backbone file."""

from __future__ import annotations

from pathlib import Path

import pytest

from uhd_eeg.models.Transformer.cBraMod.cBraMod import (
    EXPECTED_BACKBONE_SHA256,
    _sha256_file,
    warn_if_backbone_hash_mismatch,
)


def test_sha256_helper_matches_known_bytes(tmp_path: Path):
    path = tmp_path / "dummy.pth"
    path.write_bytes(b"cbramod-test")
    assert len(_sha256_file(path)) == 64


def test_warn_if_backbone_hash_mismatch_warns_on_wrong_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    path = tmp_path / "pretrained_weights.pth"
    path.write_bytes(b"not-the-official-release")
    assert _sha256_file(path) != EXPECTED_BACKBONE_SHA256
    warn_if_backbone_hash_mismatch(path)
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert EXPECTED_BACKBONE_SHA256 in captured.out


def test_config_backbone_path_uses_upstream_filename():
    text = Path("configs/trainer/config_color_within_offline_split.yaml").read_text(
        encoding="utf-8"
    )
    assert "data/weights/cBraMod/pretrained_weights.pth" in text
    assert "backbone_weights.pth" not in text
