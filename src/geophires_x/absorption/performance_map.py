"""PerformanceMap: part-load and temperature-dependent performance curves.

Provides a simple parametric fallback and an interface to build from a
manufacturer lookup table.
"""
from typing import Any, Dict, Optional


class PerformanceMap:
    """Hold PLR→COP and auxiliary power curves and perform interpolation.

    If a lookup_table is not provided a simple parametric curve is used with
    default alpha/beta parameters.
    """

    def __init__(self, rated_cop: float, plr_curve_params: Optional[Dict[str, float]] = None) -> None:
        """Construct a simple parametric performance map.

        Parameters
        ----------
        rated_cop:
            Rated COP at PLR=1.0.
        plr_curve_params:
            Optional params {'alpha': float, 'beta': float} for PLR correction.
        """
        self.rated_cop = float(rated_cop)
        params = plr_curve_params or {}
        self.alpha = float(params.get("alpha", 0.15))
        self.beta = float(params.get("beta", 1.2))

    def evaluate(self, plr: float, t_gen_c: float, t_evap_c: float, t_cond_c: float) -> Dict[str, float]:
        """Evaluate the performance map.

        Returns a dict with keys: 'cop' and 'aux_power_factor'. The simple
        parametric fallback reduces COP gently at low PLR.
        """
        plr_clamped = max(0.0, min(1.0, float(plr)))
        cop = self.rated_cop * (1.0 - self.alpha * (1.0 - plr_clamped) ** self.beta)
        return {"cop": float(cop), "aux_power_factor": 1.0}

    @classmethod
    def from_lookup(cls, lookup_table: Any) -> "PerformanceMap":
        """Create a PerformanceMap from a manufacturer lookup table.

        The lookup_table format is implementation-specific. This constructor is
        a placeholder for later expansion.
        """
        # Placeholder: accept rated COP if provided
        rated = float(getattr(lookup_table, "rated_cop", 0.7))
        return cls(rated_cop=rated)

