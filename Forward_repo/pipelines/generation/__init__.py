"""Candidate, source-tracked regeneration of the historical IASI inputs.

The archived August 2026 factors predate their first Git snapshot. Functions in
this package produce candidate artifacts until numerical canaries and the
scientific baseline gate establish parity with the frozen inputs.
"""

from .lut import build_self_broadening, expand_lut, inspect_blocks

__all__ = ["build_self_broadening", "expand_lut", "inspect_blocks"]
