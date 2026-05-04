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

        # Choose dispatch ordering depending on strategy
        if self.dispatch_strategy == "min_units":
            unit_list.sort(key=lambda u: u.nominal_cooling_kW, reverse=True)
        elif self.dispatch_strategy == "min_cost":
            # prefer lower installed cost per kW
            def cost_key(u):
                try:
                    return float(u.installed_cost_USD) / float(u.nominal_cooling_kW)
                except Exception:
                    return float(u.nominal_cooling_kW) * 1.0

            unit_list.sort(key=cost_key)
        else:
            # default: keep original order (first-fit)
            pass

        # If baseload mode requested, compute steady target output
        target_base = None
        if mode == "baseload":
            target_base = float(np.mean(cooling_load_kW_hourly))

        for i, load in enumerate(cooling_load_kW_hourly):
            requested = float(load if target_base is None else target_base)
            remaining = float(requested)
            units_on = 0
            for u in unit_list:
                if remaining <= 0:
                    break
                # determine feasible plr (respecting min_PLR)
                max_unit_cap = u.nominal_cooling_kW
                desired_plr = min(1.0, remaining / max_unit_cap)
                if desired_plr > 0 and desired_plr < u.min_PLR:
                    # if below min PLR, either run at min_PLR (if helps) or skip
                    if u.min_PLR * max_unit_cap <= remaining:
                        plr = u.min_PLR
                    else:
                        # skip this unit; it cannot operate effectively for remaining
                        continue
                else:
                    plr = desired_plr

                perf = u.performance_at_plr(plr, 100.0, 7.0, 30.0)
                cooling[i] += perf["cooling_kW"]
                q_gen[i] += perf["fuel_input_kW"]
                # set cop as weighted average if multiple units used
                cop[i] = perf["cop"] if cop[i] == 0 else (cop[i] + perf["cop"]) / 2.0
                units_on += 1
                remaining = max(0.0, remaining - perf["cooling_kW"])

            # If generator heat constraint provided and strategy is follow_heat, limit q_gen
            if generator_heat_available_kW_hourly is not None and self.dispatch_strategy == "follow_heat":
                available = float(generator_heat_available_kW_hourly[i])
                if q_gen[i] > available:
                    # scale down cooling and q_gen proportionally
                    scale = available / q_gen[i] if q_gen[i] > 0 else 0.0
                    cooling[i] *= scale
                    q_gen[i] = available
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

