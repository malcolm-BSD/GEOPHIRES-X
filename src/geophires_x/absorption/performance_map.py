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
        self.generator_ref_c = float(params.get("generator_ref_c", 95.0))
        self.evaporator_ref_c = float(params.get("evaporator_ref_c", 7.0))
        self.condenser_ref_c = float(params.get("condenser_ref_c", 30.0))
        self.generator_slope_per_c = float(params.get("generator_slope_per_c", 0.006))
        self.evaporator_slope_per_c = float(params.get("evaporator_slope_per_c", 0.010))
        self.condenser_slope_per_c = float(params.get("condenser_slope_per_c", -0.008))

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def evaluate(self, plr: float, t_gen_c: float, t_evap_c: float, t_cond_c: float) -> Dict[str, float]:
        """Evaluate the performance map.

        Returns a dict with keys: 'cop' and 'aux_power_factor'. The simple
        parametric fallback reduces COP gently at low PLR.
        """
        plr_clamped = max(0.0, min(1.0, float(plr)))
        part_load_factor = 1.0 - self.alpha * (1.0 - plr_clamped) ** self.beta
        temperature_factor = (
            1.0
            + self.generator_slope_per_c * (float(t_gen_c) - self.generator_ref_c)
            + self.evaporator_slope_per_c * (float(t_evap_c) - self.evaporator_ref_c)
            + self.condenser_slope_per_c * (float(t_cond_c) - self.condenser_ref_c)
        )
        temperature_factor = self._clamp(temperature_factor, 0.50, 1.25)
        cop = self.rated_cop * part_load_factor * temperature_factor
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
