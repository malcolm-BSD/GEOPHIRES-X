"""Units manager wrapper using Pint (optional).

Provides a thin wrapper around pint.UnitRegistry to allow the rest of the
absorption chiller code to accept and return pint.Quantity objects when
enabled, and plain numeric numpy arrays when disabled.

This module is PEP8/257/484 compliant and intentionally lightweight.
"""
from typing import Any, Optional

try:
    import pint

    _HAS_PINT = True
except Exception:  # pragma: no cover - optional dependency
    pint = None  # type: ignore
    _HAS_PINT = False


class UnitsManager:
    """Manage unit handling with Pint.

    If Pint is available and enabled, UnitsManager provides helpers to create
    and convert quantities. If disabled, methods return raw values.
    """

    enabled: bool

    def __init__(self, enabled: bool = True) -> None:
        """Initialize the UnitsManager.

        Parameters
        ----------
        enabled:
            Whether to enable Pint unit handling. Default True. If Pint is not
            installed, this will be silently disabled.
        """
        self.enabled = bool(enabled and _HAS_PINT)
        if self.enabled:
            self._ureg = pint.UnitRegistry()
        else:
            self._ureg = None

    def quantity(self, value: Any, units: str):
        """Return a quantity or raw value.

        If Pint is enabled returns a pint.Quantity otherwise the raw value.
        """
        if self.enabled and self._ureg is not None:
            return value * self._ureg(units)
        return value

    def to(self, q: Any, units: str):
        """Convert quantity to target units; if disabled return numeric.

        If Pint is disabled this returns the input unchanged.
        """
        if self.enabled and hasattr(q, "to"):
            return q.to(units)
        return q

    def strip(self, q: Any) -> Any:
        """Return the raw numeric value without units.

        If Pint is enabled this returns q.magnitude; otherwise returns q.
        """
        if self.enabled and hasattr(q, "magnitude"):
            return q.magnitude
        return q

