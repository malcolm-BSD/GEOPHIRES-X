"""ChillerBank: coordinate multiple chiller units and perform dispatch.

This module provides a simple dispatch implementation as a starting point; it
will be extended to support optimization strategies and baseload mode.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .chiller_unit import ChillerUnit


class ChillerBank:
    """Aggregate multiple ChillerUnit instances and perform dispatch.

    The bank contains tuples (unit, count). The dispatch implementation here
    is a simple greedy assignment that meets hourly load by turning on the
    required number of units at nominal PLR until load is satisfied.
    """

    def __init__(self, units: Optional[List[Tuple[ChillerUnit, int]]] = None, dispatch_strategy: str = "min_cost") -> None:
        self.units: List[Tuple[ChillerUnit, int]] = units or []
        self.dispatch_strategy = dispatch_strategy

    def add_unit(self, unit: ChillerUnit, count: int = 1) -> None:
        """Add `count` copies of a ChillerUnit to the bank."""
        self.units.append((unit, int(count)))

    def dispatch_hourly(
        self,
        cooling_load_kW_hourly: "numpy.ndarray",
        generator_heat_available_kW_hourly: Optional["numpy.ndarray"] = None,
        temps: Optional[Dict[str, "numpy.ndarray"]] = None,
        mode: str = "dispatch",
    ) -> Dict[str, Any]:
        """Dispatch units hourly.

        This simple implementation assumes all units are identical for
        simplicity and dispatchs them greedily. Returns a dictionary of
        aggregated hourly arrays.
        """
        hours = len(cooling_load_kW_hourly)
        cooling = np.zeros(hours)
        q_gen = np.zeros(hours)
        cop = np.zeros(hours)
        chilled_mdot = np.zeros(hours)
        pump_power = np.zeros(hours)

        # Flatten unit list to individual units
        unit_list: List[ChillerUnit] = []
        for unit, count in self.units:
            for _ in range(count):
                unit_list.append(unit)

        if not unit_list:
            return {
                "cooling_produced_hourly": cooling,
                "q_gen_hourly": q_gen,
                "COP_hourly": cop,
                "chilled_mdot_hourly": chilled_mdot,
                "pump_power_hourly": pump_power,
                "unit_dispatch": np.zeros((0, hours), dtype=int),
            }

        # For simplicity assume all units have same nominal capacity
        nominal = unit_list[0].nominal_cooling_kW
        for i, load in enumerate(cooling_load_kW_hourly):
            remaining = float(load)
            units_on = 0
            for u in unit_list:
                if remaining <= 0:
                    break
                # turn unit on at PLR to try to consume remaining
                plr = min(1.0, remaining / u.nominal_cooling_kW)
                perf = u.performance_at_plr(plr, 100.0, 7.0, 30.0)
                cooling[i] += perf["cooling_kW"]
                q_gen[i] += perf["fuel_input_kW"]
                cop[i] = perf["cop"] if cop[i] == 0 else cop[i]
                units_on += 1
                remaining = max(0.0, remaining - perf["cooling_kW"])
            # estimate chilled mdot and pump_power (approx)
            if cooling[i] > 0:
                cp = 4186.0
                dT = 5.0
                chilled_mdot[i] = (cooling[i] * 1000.0) / (cp * dT)
                rho = 997.0
                g = 9.81
                H = unit_list[0].pump_head_m or 30.0
                Vdot = chilled_mdot[i] / rho
                P_hydrau = rho * g * H * Vdot
                pump_power[i] = P_hydrau / 0.7 / 1000.0

        unit_dispatch = np.zeros((len(unit_list), hours), dtype=int)
        # naive: first N units considered "on" when used
        # compute counts per hour
        for i in range(hours):
            remaining = float(cooling[i])
            idx = 0
            while remaining > 0 and idx < len(unit_list):
                cap = unit_list[idx].nominal_cooling_kW
                unit_dispatch[idx, i] = 1
                remaining -= cap
                idx += 1

        return {
            "cooling_produced_hourly": cooling,
            "q_gen_hourly": q_gen,
            "COP_hourly": cop,
            "chilled_mdot_hourly": chilled_mdot,
            "pump_power_hourly": pump_power,
            "unit_dispatch": unit_dispatch,
        }

    def _dispatch_hour(self, hour_idx: int, load_kW: float, available_heat_kW: Optional[float]) -> Dict[str, Any]:
        """Internal placeholder for per-hour dispatch logic."""
        raise NotImplementedError

