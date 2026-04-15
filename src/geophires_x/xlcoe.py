from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geophires_x.Economics import Economics
    from geophires_x.Model import Model


def calculate_xlcoe_outputs(econ: Economics, model: Model) -> tuple[float, float]:
    """
    Phase 1 scaffolding for XLCOE.

    The calculation path stays inert unless explicitly enabled. Once enabled, the
    implementation must be provided before the model can complete.
    """
    if not econ.DoXLCOECalculations.value:
        return 0.0, 0.0

    raise NotImplementedError(
        "XLCOE calculations are not implemented yet. "
        "See docs/XLCOE-design.md for the phased implementation plan."
    )
