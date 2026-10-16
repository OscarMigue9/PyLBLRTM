"""Predeclared numerical gate for the isolated Modal LUT pilot."""

from pipelines.modal_lut_canary import (
    MAX_ABS_LOG10_SIGMA,
    MAX_RMS_LOG10_SIGMA,
    assess_comparison,
)


def test_pilot_gate_requires_both_finite_metrics_within_limits() -> None:
    assert assess_comparison(
        {
            "rms_log10_sigma": MAX_RMS_LOG10_SIGMA,
            "max_abs_log10_sigma": MAX_ABS_LOG10_SIGMA,
        }
    )["passed"]
    assert not assess_comparison(
        {
            "rms_log10_sigma": MAX_RMS_LOG10_SIGMA * 1.01,
            "max_abs_log10_sigma": 0,
        }
    )["passed"]
    assert not assess_comparison(
        {
            "rms_log10_sigma": 0,
            "max_abs_log10_sigma": float("nan"),
        }
    )["passed"]
