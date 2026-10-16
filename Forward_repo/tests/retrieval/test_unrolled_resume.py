"""An interrupted sew run must continue from the next optimizer update."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from iasi_retrieval.state import Standardizer
from iasi_retrieval.unrolled import SEW_INPUT_IDENTIFIERS, _save_checkpoint_atomic, train_unrolled


INPUT_SHA256 = {
    identifier: hashlib.sha256(identifier.encode("ascii")).hexdigest()
    for identifier in SEW_INPUT_IDENTIFIERS
}


class PlannedStop(Exception):
    """Simulate a container ending after a complete periodic checkpoint."""


class TinyRetrieval:
    """Exercise stochastic sampling, AdamW and torch RNG without a LUT."""

    def __init__(self, initial: dict[str, torch.Tensor], stop_after: int | None = None):
        self.step = nn.Linear(1, 1)
        self.step.load_state_dict(initial)
        self.observed = torch.zeros(12, 1)
        self.residual_sd = torch.ones(2145)
        self.standardizer = Standardizer(np.zeros(818), np.ones(818))
        self.stop_after = stop_after
        self.calls = 0

    def evaluate(self, rows, truth, iterations, batch_size):
        return {"weight": float(self.step.weight.detach().item())}

    def run(self, rows, iterations, checkpoint_forward):
        self.calls += 1
        if self.stop_after is not None and self.calls > self.stop_after:
            raise PlannedStop
        feature = torch.as_tensor(rows, dtype=torch.float32).reshape(-1, 1) / 12
        feature = feature + 0.01 * torch.randn_like(feature)
        update = self.step(feature)
        normalized = update.expand(-1, 818)
        return normalized, normalized, update, [update] * iterations


@pytest.mark.parametrize("legacy_checkpoint", [False, True])
def test_resume_reproduces_uninterrupted_trajectory(tmp_path, legacy_checkpoint):
    torch.manual_seed(61)
    initial = nn.Linear(1, 1).state_dict()
    splits = {"train": np.arange(10), "val": np.arange(10, 12)}
    truth = np.zeros((12, 818), dtype=np.float32)
    kwargs = dict(
        steps=6,
        batch_size=4,
        supervised_rows=10,
        iterations=3,
        seed=0,
        trace_every=2,
        trace_rows=2,
        checkpoint_every=3,
        input_sha256=INPUT_SHA256,
    )
    reference_model = TinyRetrieval(initial)
    reference = train_unrolled(reference_model, truth, splits, **kwargs)

    checkpoint = tmp_path / "sew.pt"
    interrupted_model = TinyRetrieval(initial, stop_after=3)
    with pytest.raises(PlannedStop):
        train_unrolled(interrupted_model, truth, splits, checkpoint_path=checkpoint, **kwargs)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["paso"] == 3
    assert saved["input_sha256"] == INPUT_SHA256
    legacy_sha256 = None
    if legacy_checkpoint:
        saved.pop("training")
        saved.pop("input_sha256")
        torch.save(saved, checkpoint)
        legacy_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="independently verified SHA256"):
            train_unrolled(
                TinyRetrieval(initial),
                truth,
                splits,
                resume_path=checkpoint,
                **kwargs,
            )
        with pytest.raises(ValueError, match="does not match the approved digest"):
            train_unrolled(
                TinyRetrieval(initial),
                truth,
                splits,
                resume_path=checkpoint,
                legacy_checkpoint_sha256="0" * 64,
                **kwargs,
            )
    else:
        changed_inputs = {**INPUT_SHA256, "prior": "0" * 64}
        with pytest.raises(ValueError, match="input SHA256 manifest differs"):
            train_unrolled(
                TinyRetrieval(initial),
                truth,
                splits,
                resume_path=checkpoint,
                **{**kwargs, "input_sha256": changed_inputs},
            )

    resumed_model = TinyRetrieval(initial)
    resumed = train_unrolled(
        resumed_model,
        truth,
        splits,
        checkpoint_path=checkpoint,
        resume_path=checkpoint,
        legacy_checkpoint_sha256=legacy_sha256,
        **kwargs,
    )
    assert resumed["resumed_from_step"] == 3
    assert resumed["legacy_resume_without_embedded_input_hashes"] == legacy_checkpoint
    assert resumed["trace"] == reference["trace"]
    np.testing.assert_array_equal(resumed["curve"], reference["curve"])
    np.testing.assert_array_equal(resumed["update_norms"], reference["update_norms"])
    for key, value in reference_model.step.state_dict().items():
        torch.testing.assert_close(resumed_model.step.state_dict()[key], value, rtol=0, atol=0)
    assert torch.load(checkpoint, map_location="cpu", weights_only=False)["paso"] == 6


def test_checkpoint_requires_complete_input_manifest(tmp_path):
    torch.manual_seed(61)
    initial = nn.Linear(1, 1).state_dict()
    with pytest.raises(ValueError, match="complete 16-file input SHA256 manifest"):
        train_unrolled(
            TinyRetrieval(initial),
            np.zeros((12, 818), dtype=np.float32),
            {"train": np.arange(10), "val": np.arange(10, 12)},
            steps=1,
            checkpoint_path=tmp_path / "sew.pt",
            input_sha256={"configuration": "0" * 64},
        )


def test_atomic_save_preserves_previous_checkpoint_on_failure(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sew.pt"
    checkpoint.write_bytes(b"previous complete checkpoint")

    def broken_save(payload, handle):
        handle.write(b"partial")
        raise OSError("simulated serialization failure")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="serialization failure"):
        _save_checkpoint_atomic({"paso": 2}, checkpoint)
    assert checkpoint.read_bytes() == b"previous complete checkpoint"
    assert list(tmp_path.iterdir()) == [checkpoint]
