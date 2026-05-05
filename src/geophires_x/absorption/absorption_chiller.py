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
from .chiller_unit import ChillerUnit


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
        n_segments: int = 5,
        use_hourly_temps: bool = False,
        dispatch_strategy: str = "min_cost",
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
        # dispatch/solver tuning passed to ChillerBank
        self.n_segments = int(n_segments)
        self.use_hourly_temps = bool(use_hourly_temps)
        self.dispatch_strategy = dispatch_strategy

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
        temps: Optional[Dict[str, "numpy.ndarray"]] = None,
        mode: str = "dispatch",
        use_milp: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate plant performance over hourly timesteps.

        This simple implementation will be expanded; for now it sizes a simple
        bank using the catalog and runs a naive dispatch.
        """
        hours = len(cooling_demand_hourly)
        if temps is None and self.use_hourly_temps:
            temps = {
                "t_gen": self._coerce_hourly_array(geo_inlet_temp_hourly, hours),
                "t_evap": self._coerce_hourly_array(chilled_supply_setpoint_c, hours),
                "t_cond": self._coerce_hourly_array(
                    30.0 if ambient_temp_hourly is None else ambient_temp_hourly,
                    hours,
                ),
            }
        peak = float(np.max(cooling_demand_hourly))
        bank = self.build_bank(peak)

        results = bank.dispatch_hourly(
            cooling_demand_hourly,
            generator_heat_available_kW_hourly=None,
            temps=temps,
            mode=mode,
            use_milp=use_milp,
        )

        cop_arr = results.get("COP_hourly", np.zeros(hours))
        cooling = results.get("cooling_produced_hourly", np.zeros(hours))
        q_gen = results.get("q_gen_hourly", np.zeros(hours))

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

    @staticmethod
    def _coerce_hourly_array(values: Any, hours: int) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            return np.full(hours, float(array))
        if array.ndim == 2:
            if array.shape[1] < 2:
                array = array.reshape(-1)
            else:
                array = array[:, 1]
        elif array.ndim > 1:
            array = array.reshape(-1)
        if array.size == hours:
            return array
        if array.size == 1:
            return np.full(hours, float(array[0]))
        return np.full(hours, float(np.mean(array)))

    def size_for_capacity(self, peak_cooling_kW: float) -> Dict[str, Any]:
        """Return a CatalogSelection for the requested peak capacity.

        Proxy to Catalog.select_min_cost_set.
        """
        return self.catalog.select_min_cost_set(
            peak_cooling_kW,
            refrigerant_family=self.refrigerant_family,
            effect_type=self.effect_type,
        )

    def build_bank(self, required_capacity_kW: float) -> ChillerBank:
        """Build a dispatchable chiller bank sized for the requested capacity."""
        selection = self.size_for_capacity(required_capacity_kW)
        bank = ChillerBank(
            dispatch_strategy=self.dispatch_strategy,
            n_segments=self.n_segments,
            use_hourly_temps=self.use_hourly_temps,
        )
        for selected in selection.get("selected", []):
            entries = [entry for entry in self.catalog.entries if entry.get("model_id") == selected.get("model_id")]
            if not entries:
                continue
            entry = entries[0]
            unit = ChillerUnit(
                model_id=entry.get("model_id"),
                manufacturer=entry.get("manufacturer"),
                nominal_cooling_kW=float(entry.get("nominal_cooling_kW", 0)),
                nominal_COP=float(entry.get("nominal_COP", self.rated_COP)) * self._effect_multiplier(),
                effect_type=entry.get("effect_type", self.effect_type),
                refrigerant_family=entry.get("refrigerant_family", self.refrigerant_family),
                min_PLR=float(entry.get("min_PLR", self.min_part_load_ratio)),
                turndown_ratio=float(entry.get("turndown_ratio", self.turndown_ratio)),
                pump_head_m=float(entry.get("pump_head_m", self.pump_head_m)),
                electrical_aux_kW=float(entry.get("electrical_aux_kW", 0)),
                installed_cost_USD=float(entry.get("installed_cost_USD", 0) or 0),
            )
            bank.add_unit(unit, int(selected.get("count", 1)))

        return bank

