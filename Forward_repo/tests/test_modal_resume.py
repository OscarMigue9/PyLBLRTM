"""The Modal resume flag must select an existing run checkpoint."""

import pytest

from pipelines.modal_final import _resume_sew_path


def test_modal_resume_selects_only_training_checkpoint(tmp_path):
    checkpoint = tmp_path / "sew.pt"
    assert _resume_sew_path("train-unrolled", False, tmp_path) is None
    with pytest.raises(FileNotFoundError, match="No persisted sew checkpoint"):
        _resume_sew_path("train-unrolled", True, tmp_path)
    checkpoint.write_bytes(b"persisted")
    assert _resume_sew_path("train-unrolled", True, tmp_path) == checkpoint
    with pytest.raises(ValueError, match="only to train-unrolled"):
        _resume_sew_path("evaluate-simulated", True, tmp_path)
    with pytest.raises(ValueError, match="requires --resume-sew"):
        _resume_sew_path("train-unrolled", False, tmp_path, "a" * 64)
