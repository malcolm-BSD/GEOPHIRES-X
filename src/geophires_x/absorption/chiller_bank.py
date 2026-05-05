"""ChillerBank: coordinate multiple chiller units and perform dispatch.

This module provides a simple dispatch implementation as a starting point; it
will be extended to support optimization strategies and baseload mode.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import warnings

from .chiller_unit import ChillerUnit


class ChillerBank:
    """Aggregate multiple ChillerUnit instances and perform dispatch.

    The bank contains tuples (unit, count). The dispatch implementation here
    is a simple greedy assignment that meets hourly load by turning on the
    required number of units at nominal PLR until load is satisfied.
    """

    def __init__(
        self,
        units: Optional[List[Tuple[ChillerUnit, int]]] = None,
        dispatch_strategy: str = "min_cost",
        n_segments: int = 5,
        use_hourly_temps: bool = False,
    ) -> None:
        """Initialize the ChillerBank.

        Parameters:
            units: optional list of (ChillerUnit, count) tuples
            dispatch_strategy: 'min_cost'|'min_units'|'follow_heat'
            n_segments: number of PLR segments used for piecewise-linear COP approximation in the MILP
        """
        self.units: List[Tuple[ChillerUnit, int]] = units or []
        self.dispatch_strategy = dispatch_strategy
        self.n_segments = int(n_segments)
        self.use_hourly_temps = bool(use_hourly_temps)

    def add_unit(self, unit: ChillerUnit, count: int = 1) -> None:
        """Add `count` copies of a ChillerUnit to the bank."""
        self.units.append((unit, int(count)))

    def dispatch_hourly(
        self,
        cooling_load_kW_hourly: "numpy.ndarray",
        generator_heat_available_kW_hourly: Optional["numpy.ndarray"] = None,
        temps: Optional[Dict[str, "numpy.ndarray"]] = None,
        mode: str = "dispatch",
        use_milp: bool = True,
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

        # Validate hourly temps arrays if requested
        use_hourly_temps_enabled = False
        t_gen_arr = t_evap_arr = t_cond_arr = None
        if self.use_hourly_temps:
            if temps is None:
                warnings.warn("ChillerBank: use_hourly_temps=True but no 'temps' dict provided to dispatch_hourly; falling back to default stub temperatures.")
                use_hourly_temps_enabled = False
            else:
                def _find_array(key_candidates):
                    for kk in key_candidates:
                        if kk in temps:
                            try:
                                arr = temps[kk]
                                if len(arr) == hours:
                                    return arr
                            except Exception:
                                continue
                    return None

                t_gen_arr = _find_array(["t_gen", "geo_in", "generator_in", "generator", "geo_inlet"])
                t_evap_arr = _find_array(["t_evap", "chilled_supply", "evaporator", "chilled_supply_setpoint"])
                t_cond_arr = _find_array(["t_cond", "ambient", "condensor", "ambient_temp"])

                missing = []
                if t_gen_arr is None:
                    missing.append("generator inlet (t_gen)")
                if t_evap_arr is None:
                    missing.append("evaporator/chilled supply (t_evap)")
                if t_cond_arr is None:
                    missing.append("condenser/ambient (t_cond)")

                if missing:
                    warnings.warn(
                        f"ChillerBank: use_hourly_temps=True but temps dict is missing or has wrong-length arrays for: {', '.join(missing)}; falling back to default stub temperatures."
                    )
                    use_hourly_temps_enabled = False
                else:
                    use_hourly_temps_enabled = True

        # Keep units as types with counts for integer programming dispatch
        unit_types: List[Tuple[ChillerUnit, int]] = list(self.units)
        # Flatten convenience list for fallbacks
        unit_list_flat: List[ChillerUnit] = []
        for unit, count in unit_types:
            for _ in range(count):
                unit_list_flat.append(unit)

        if not unit_list_flat:
            return {
                "cooling_produced_hourly": cooling,
                "q_gen_hourly": q_gen,
                "COP_hourly": cop,
                "chilled_mdot_hourly": chilled_mdot,
                "pump_power_hourly": pump_power,
                "unit_dispatch": np.zeros((0, hours), dtype=int),
            }

        unit_dispatch = np.zeros((len(unit_list_flat), hours), dtype=int)

        # Note: ordering used only by greedy fallback. For ILP dispatch we will use unit_types and counts.
        # Choose dispatch ordering depending on strategy for fallback greedy
        unit_order = list(enumerate(unit_list_flat))
        if self.dispatch_strategy == "min_units":
            unit_order.sort(key=lambda item: item[1].nominal_cooling_kW, reverse=True)
        elif self.dispatch_strategy == "min_cost":
            # prefer lower installed cost per kW
            def cost_key(u):
                try:
                    return float(u.installed_cost_USD) / float(u.nominal_cooling_kW)
                except Exception:
                    return float(u.nominal_cooling_kW) * 1.0

            unit_order.sort(key=lambda item: cost_key(item[1]))
        else:
            # default: keep original order (first-fit)
            pass

        # Attempt to use integer programming per-hour dispatch when PuLP is available
        try:
            import pulp  # type: ignore
            _HAS_PULP = True
        except Exception:
            _HAS_PULP = False

        # If baseload mode requested, compute steady target output
        target_base = None
        if mode == "baseload":
            target_base = float(np.mean(cooling_load_kW_hourly))

        for i, load in enumerate(cooling_load_kW_hourly):
            requested = float(load if target_base is None else target_base)
            remaining = float(requested)
            units_on = 0
            if use_hourly_temps_enabled:
                try:
                    t_gen_c = float(t_gen_arr[i])
                except Exception:
                    t_gen_c = 100.0
                try:
                    t_evap_c = float(t_evap_arr[i])
                except Exception:
                    t_evap_c = 7.0
                try:
                    t_cond_c = float(t_cond_arr[i])
                except Exception:
                    t_cond_c = 30.0
            else:
                t_gen_c = 100.0
                t_evap_c = 7.0
                t_cond_c = 30.0

            milp_solved = False
            # If PuLP is available, solve a small MILP per-hour that uses binary
            # on/off variables for each unit instance and continuous PLR variables
            # to allow partial loading subject to unit minimum PLR constraints.
            if use_milp and _HAS_PULP and unit_list_flat:
                try:
                    # Build MILP
                    prob = pulp.LpProblem(f"dispatch_hour_{i}", pulp.LpMinimize)
                    y_vars = {}
                    # We'll use a piecewise-linear representation of COP(PLR).
                    # For each unit k create a binary y_k and cooling-by-segment q_k_s
                    # variables. This keeps the MILP linear because fuel = sum(q_k_s / COP_seg).
                    q_vars = {}
                    unmet_demand = pulp.LpVariable(f"unmet_{i}", lowBound=0.0, cat=pulp.LpContinuous)
                    n_segments = int(self.n_segments)
                    for k, u in enumerate(unit_list_flat):
                        y_vars[k] = pulp.LpVariable(f"y_{i}_{k}", cat=pulp.LpBinary)
                        q_vars[k] = {}
                        # create PLR breakpoints [0, 1] divided into n_segments
                        for s in range(n_segments):
                            q_vars[k][s] = pulp.LpVariable(f"q_{i}_{k}_{s}", lowBound=0.0, cat=pulp.LpContinuous)

                    # Objective: minimize sum(installed_cost * y_k) as a proxy for cost
                    def _cost_of(u: ChillerUnit) -> float:
                        try:
                            return float(getattr(u, "installed_cost_USD", 0.0) or 0.0)
                        except Exception:
                            return 0.0

                    unmet_penalty = max(sum(_cost_of(u) for u in unit_list_flat), 1.0e6)
                    prob += (
                        unmet_penalty * unmet_demand
                        + pulp.lpSum([_cost_of(unit_list_flat[k]) * y_vars[k] for k in range(len(unit_list_flat))])
                    )

                    total_cooling = pulp.lpSum(
                        [q_vars[k][s] for k in range(len(unit_list_flat)) for s in range(n_segments)]
                    )
                    prob += total_cooling + unmet_demand == requested

                    # If generator heat constraints are provided, enforce them
                    # regardless of the objective strategy.
                    if generator_heat_available_kW_hourly is not None:
                        try:
                            available_heat = float(generator_heat_available_kW_hourly[i])
                        except Exception:
                            available_heat = None
                        if available_heat is not None:
                            # Build piecewise-linear approximation of COP(plr) using n_segments.
                            # For segment s we evaluate COP at the segment midpoint and use
                            # that as the constant COP for the segment. Fuel (generator heat)
                            # is linear: fuel = sum_s q_k_s / COP_seg.
                            cop_seg = {}
                            caps = {}
                            for k, u in enumerate(unit_list_flat):
                                caps[k] = float(u.nominal_cooling_kW)
                                cop_seg[k] = {}
                                for s in range(n_segments):
                                    plr_low = float(s) / float(n_segments)
                                    plr_high = float(s + 1) / float(n_segments)
                                    plr_mid = 0.5 * (plr_low + plr_high)
                                    try:
                                        perf_mid = u.performance_at_plr(plr_mid if plr_mid > 0 else 1.0, t_gen_c, t_evap_c, t_cond_c)
                                        cop_mid = float(perf_mid.get("cop", getattr(u, "nominal_COP", 1.0)))
                                    except Exception:
                                        cop_mid = float(getattr(u, "nominal_COP", 1.0) or 1.0)
                                    if cop_mid <= 0:
                                        cop_mid = 1e-6
                                    cop_seg[k][s] = cop_mid

                            # linear constraint: sum_k sum_s q_k_s / COP_seg[k][s] <= available_heat
                            prob += pulp.lpSum([
                                q_vars[k][s] / cop_seg[k][s]
                                for k in range(len(unit_list_flat)) for s in range(n_segments)
                            ]) <= available_heat

                    # Logical constraints connecting binary and plr variables
                    for k, u in enumerate(unit_list_flat):
                        min_plr = float(getattr(u, "min_PLR", 0.0) or 0.0)
                        cap_k = float(u.nominal_cooling_kW)
                        # total cooling from this unit
                        prob += pulp.lpSum([q_vars[k][s] for s in range(n_segments)]) <= cap_k * y_vars[k]
                        # enforce minimum PLR when unit is on
                        if min_plr > 0:
                            prob += pulp.lpSum([q_vars[k][s] for s in range(n_segments)]) >= cap_k * min_plr * y_vars[k]
                        # upper bounds per segment: q_k_s <= cap_k * (plr_high - plr_low)
                        for s in range(n_segments):
                            plr_low = float(s) / float(n_segments)
                            plr_high = float(s + 1) / float(n_segments)
                            seg_cap = cap_k * (plr_high - plr_low)
                            prob += q_vars[k][s] <= seg_cap

                    # Solve with default CBC solver (quiet)
                    prob.solve(pulp.PULP_CBC_CMD(msg=False))
                    if pulp.LpStatus.get(prob.status) != "Optimal":
                        raise RuntimeError(f"MILP dispatch did not find an optimal solution: {pulp.LpStatus.get(prob.status)}")

                    # Extract solution and compute performance (using per-unit aggregated q)
                    for k, u in enumerate(unit_list_flat):
                        try:
                            yv = pulp.value(y_vars[k])
                        except Exception:
                            yv = None
                        if yv is None or float(yv) < 0.5:
                            continue
                        # sum cooling across segments
                        q_sum = 0.0
                        for s in range(n_segments):
                            try:
                                q_val = float(pulp.value(q_vars[k][s]) or 0.0)
                            except Exception:
                                q_val = 0.0
                            q_sum += q_val
                        if q_sum <= 0:
                            continue
                        # compute PLR implied and actual performance
                        cap_k = float(u.nominal_cooling_kW)
                        plr = min(1.0, q_sum / cap_k)
                        perf = u.performance_at_plr(plr, t_gen_c, t_evap_c, t_cond_c)
                        # cooling should approximately equal q_sum; use perf cooling for consistency
                        cooling[i] += perf["cooling_kW"]
                        q_gen[i] += perf["fuel_input_kW"]
                        cop[i] = perf["cop"] if cop[i] == 0 else (cop[i] + perf["cop"]) / 2.0
                        units_on += 1
                        unit_dispatch[k, i] = 1
                        remaining = max(0.0, remaining - perf["cooling_kW"])
                    milp_solved = True
                except Exception:
                    # If MILP solver fails for any reason fall through to greedy
                    pass

            if not milp_solved:
                # Greedy per-unit dispatch fallback
                for unit_index, u in unit_order:
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
                            # skip this unit; it cannot operate effectively for remaining without overproducing
                            continue
                    else:
                        plr = desired_plr

                    perf = u.performance_at_plr(plr, t_gen_c, t_evap_c, t_cond_c)
                    cooling[i] += perf["cooling_kW"]
                    q_gen[i] += perf["fuel_input_kW"]
                    # set cop as weighted average if multiple units used
                    cop[i] = perf["cop"] if cop[i] == 0 else (cop[i] + perf["cop"]) / 2.0
                    units_on += 1
                    unit_dispatch[unit_index, i] = 1
                    remaining = max(0.0, remaining - perf["cooling_kW"])

            # If generator heat constraint provided, limit any fallback/rounding excess.
            if generator_heat_available_kW_hourly is not None:
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
                H = unit_list_flat[0].pump_head_m or 30.0
                Vdot = chilled_mdot[i] / rho
                P_hydrau = rho * g * H * Vdot
                pump_power[i] = P_hydrau / 0.7 / 1000.0

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

