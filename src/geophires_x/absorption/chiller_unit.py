"""Representation of a commercial absorption chiller unit.

This module defines ChillerUnit which encapsulates nameplate data and a
PerformanceMap. The implementation provides a small, well-documented API that
will be extended as manufacturer data becomes available.
"""
from typing import Dict, Optional

from .performance_map import PerformanceMap


class ChillerUnit:
    """Represent a commercial absorption chiller model.

    Attributes are documented and typed. This class uses a PerformanceMap to
    provide PLR-dependent COP and auxiliary power.
    """

    def __init__(
        self,
        model_id: str,
        manufacturer: str,
        nominal_cooling_kW: float,
        nominal_COP: float = 0.7,
        effect_type: str = "single",
        refrigerant_family: str = "LiBr-water",
        min_PLR: float = 0.2,
        turndown_ratio: float = 3.0,
        nominal_chilled_flow_kg_s: Optional[float] = None,
        nominal_hot_flow_kg_s: Optional[float] = None,
        pump_head_m: Optional[float] = 30.0,
        electrical_aux_kW: float = 5.0,
        installed_cost_USD: Optional[float] = None,
        performance_map: Optional[PerformanceMap] = None,
    ) -> None:
        """Initialize the ChillerUnit.

        Parameters
        ----------
        model_id: Unique model identifier.
        manufacturer: Manufacturer name.
        nominal_cooling_kW: Nominal cooling capacity in kW.
        nominal_COP: Rated COP at nominal conditions.
        effect_type: 'single'|'double'|'triple'
        refrigerant_family: 'LiBr-water'|'NH3-water'
        min_PLR: Minimum practical part-load ratio.
        turndown_ratio: Typical turndown (e.g., 3.0 => 3:1).
        nominal_chilled_flow_kg_s: Optional chilled-water design flow.
        nominal_hot_flow_kg_s: Optional hot-side design flow.
        pump_head_m: Typical pump head for this unit.
        electrical_aux_kW: Electrical auxiliary power at nominal.
        installed_cost_USD: Optional installed cost estimate.
        performance_map: Optional PerformanceMap instance.
        """
        self.model_id = model_id
        self.manufacturer = manufacturer
        self.nominal_cooling_kW = float(nominal_cooling_kW)
        self.nominal_COP = float(nominal_COP)
        self.effect_type = effect_type
        self.refrigerant_family = refrigerant_family
        self.min_PLR = float(min_PLR)
        self.turndown_ratio = float(turndown_ratio)
        self.nominal_chilled_flow_kg_s = nominal_chilled_flow_kg_s
        self.nominal_hot_flow_kg_s = nominal_hot_flow_kg_s
        self.pump_head_m = pump_head_m
        self.electrical_aux_kW = float(electrical_aux_kW)
        self.installed_cost_USD = installed_cost_USD
        self.performance_map = performance_map or PerformanceMap(self.nominal_COP)

    def performance_at_plr(self, plr: float, t_gen_c: float, t_evap_c: float, t_cond_c: float) -> Dict[str, float]:
        """Return performance at provided PLR and temperatures.

        Returns a dictionary with keys: 'cop', 'cooling_kW', 'fuel_input_kW', 'electrical_aux_kW'.
        """
        plr_clamped = max(0.0, min(1.0, float(plr)))
        perf = self.performance_map.evaluate(plr_clamped, t_gen_c, t_evap_c, t_cond_c)
        cop = float(perf.get("cop", self.nominal_COP))
        cooling = self.nominal_cooling_kW * plr_clamped
        fuel_input_kW = cooling / cop if cop > 0 else float("inf")
        aux = self.electrical_aux_kW * plr_clamped
        return {"cop": cop, "cooling_kW": cooling, "fuel_input_kW": fuel_input_kW, "electrical_aux_kW": aux}

    def feasible_for_conditions(self, t_gen_c: float, t_evap_c: float, t_cond_c: float) -> bool:
        """Return True if the unit can operate under the provided temperatures.

        For now this checks only that generator temperature is reasonable for the
        declared effect type; more complex checks may be added later.
        """
        if self.effect_type == "single":
            return t_gen_c >= 80.0
        if self.effect_type == "double":
            return t_gen_c >= 140.0
        if self.effect_type == "triple":
            return t_gen_c >= 170.0
        return False
