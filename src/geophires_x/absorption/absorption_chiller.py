"""High-level AbsorptionChiller controller.

This module exposes :class:`AbsorptionChiller` which ties together
``UnitsManager``, ``FluidPropsAdapter``, :class:`~.catalog.Catalog` and
:class:`~.chiller_bank.ChillerBank`.

Selection of commercial units via :meth:`Catalog.select_min_cost_set` will
attempt to use an integer-programming solver (PuLP) for exact installed-cost
minimization when available; otherwise it falls back to a greedy packer.

The class provides ``evaluate_hourly`` for hourly performance outputs and is
designed to be used by the surface plant integration layer.
"""
from typing import Any, Dict, Optional
import numpy as np

from .units_manager import UnitsManager
from .fluid_props_adapter import FluidPropsAdapter
from .catalog import Catalog
from .chiller_bank import ChillerBank


class AbsorptionChiller:
    """High-level API class that orchestrates chiller calculations.

    This class is intended to be the primary entry point for SurfacePlant
    integration. It supports 'dispatch' and 'baseload' modes.
    """

    def __init__(
        self,
        catalog: Optional[Catalog] = None,
        units_manager: Optional[UnitsManager] = None,
        fluid_adapter: Optional[FluidPropsAdapter] = None,
        refrigerant_family: str = "LiBr-water",
        effect_type: str = "single",
        rated_COP: float = 0.7,
        min_part_load_ratio: float = 0.2,
        turndown_ratio: float = 3.0,
        chilled_deltaT_K: float = 5.0,
        pump_head_m: float = 30.0,
        pump_efficiency: float = 0.70,
        use_pint: bool = True,
        use_coolprop: bool = True,
        effect_multiplier_override: Optional[float] = None,
    ) -> None:
        self.units_manager = units_manager or UnitsManager(enabled=use_pint)
        self.fluid_adapter = fluid_adapter or FluidPropsAdapter(use_coolprop=use_coolprop)
        self.catalog = catalog or Catalog()
        self.refrigerant_family = refrigerant_family
        self.effect_type = effect_type
        self.rated_COP = float(rated_COP)
        self.min_part_load_ratio = float(min_part_load_ratio)
        self.turndown_ratio = float(turndown_ratio)
        self.chilled_deltaT_K = float(chilled_deltaT_K)
        self.pump_head_m = float(pump_head_m)
        self.pump_efficiency = float(pump_efficiency)
        self.effect_multiplier_override = effect_multiplier_override

    def _effect_multiplier(self) -> float:
        if self.effect_multiplier_override is not None:
            return float(self.effect_multiplier_override)
        if self.effect_type == "single":
            return 1.0
        if self.effect_type == "double":
            return 1.65
        if self.effect_type == "triple":
            return 2.0
        return 1.0

    def evaluate_hourly(
        self,
        cooling_demand_hourly: "numpy.ndarray",
        geo_inlet_temp_hourly: "numpy.ndarray",
        chilled_supply_setpoint_c: float = 7.0,
        ambient_temp_hourly: Optional["numpy.ndarray"] = None,
        mode: str = "dispatch",
    ) -> Dict[str, Any]:
        """Evaluate plant performance over hourly timesteps.

        This simple implementation will be expanded; for now it sizes a simple
        bank using the catalog and runs a naive dispatch.
        """
        hours = len(cooling_demand_hourly)
        # select a candidate set (naive full-size selection)
        peak = float(np.max(cooling_demand_hourly))
        selection = self.catalog.select_min_cost_set(peak)
        # build chiller bank from selection (very naive)
        bank = ChillerBank()
        for s in selection.get("selected", []):
            # find matching entry
            entries = [e for e in self.catalog.entries if e.get("model_id") == s.get("model_id")]
            if not entries:
                continue
            entry = entries[0]
            # build ChillerUnit on the fly
            from .chiller_unit import ChillerUnit

            unit = ChillerUnit(
                model_id=entry.get("model_id"),
                manufacturer=entry.get("manufacturer"),
                nominal_cooling_kW=float(entry.get("nominal_cooling_kW", 0)),
                nominal_COP=float(entry.get("nominal_COP", self.rated_COP)),
                effect_type=entry.get("effect_type", self.effect_type),
                refrigerant_family=entry.get("refrigerant_family", self.refrigerant_family),
                min_PLR=float(entry.get("min_PLR", self.min_part_load_ratio)),
                turndown_ratio=float(entry.get("turndown_ratio", self.turndown_ratio)),
                pump_head_m=float(entry.get("pump_head_m", self.pump_head_m)),
                electrical_aux_kW=float(entry.get("electrical_aux_kW", 0)),
            )
            bank.add_unit(unit, int(s.get("count", 1)))

        results = bank.dispatch_hourly(cooling_demand_hourly, generator_heat_available_kW_hourly=None, temps=None, mode=mode)

        # apply effect multiplier to COP arrays
        mult = self._effect_multiplier()
        cop_arr = results.get("COP_hourly", np.zeros(hours))
        cop_arr = cop_arr * mult
        results["COP_hourly"] = cop_arr
        # recompute q_gen if COP changed
        cooling = results.get("cooling_produced_hourly", np.zeros(hours))
        q_gen = np.zeros(hours)
        with np.errstate(divide="ignore", invalid="ignore"):
            q_gen = np.where(cop_arr > 0, cooling / cop_arr, 0.0)
        results["q_gen_hourly"] = q_gen

        # compute chilled and geo mass flows
        cp_water = self.fluid_adapter.cp("Water", chilled_supply_setpoint_c)
        dT = max(1.0, self.chilled_deltaT_K)
        chilled_mdot = (cooling * 1000.0) / (cp_water * dT)
        results["chilled_mdot_hourly"] = chilled_mdot

        # compute geothermal mass flow from q_gen using default dT_geo = 5 K
        dT_geo = 5.0
        cp_geo = self.fluid_adapter.cp("Water", geo_inlet_temp_hourly.mean() if hasattr(geo_inlet_temp_hourly, "mean") else float(geo_inlet_temp_hourly))
        geofluid_mdot = (q_gen * 1000.0) / (cp_geo * dT_geo)
        results["geofluid_mdot_hourly"] = geofluid_mdot

        return results

    def size_for_capacity(self, peak_cooling_kW: float) -> Dict[str, Any]:
        """Return a CatalogSelection for the requested peak capacity.

        Proxy to Catalog.select_min_cost_set.
        """
        return self.catalog.select_min_cost_set(peak_cooling_kW)

