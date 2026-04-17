from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from geophires_x.levelized_costs import (
    COOLING_COMMODITY,
    ELECTRICITY_COMMODITY,
    HEAT_COMMODITY,
    LevelizedCostBasis,
    build_levelized_cost_bases,
)

if TYPE_CHECKING:
    from geophires_x.Economics import Economics
    from geophires_x.Model import Model


@dataclass(frozen=True)
class ValueAdjustmentInputs:
    active_base_cost: float
    system_energy_value: float = 0.0
    technology_energy_value: float = 0.0
    system_capacity_value: float = 0.0
    technology_capacity_value: float = 0.0
    system_flexibility_value: float = 0.0
    technology_flexibility_value: float = 0.0


@dataclass(frozen=True)
class ValueAdjustmentResult:
    active_base_cost: float = 0.0
    valco: float = 0.0
    energy_adjustment: float = 0.0
    capacity_adjustment: float = 0.0
    flexibility_adjustment: float = 0.0


def calculate_value_adjusted_cost(inputs: ValueAdjustmentInputs) -> ValueAdjustmentResult:
    energy_adjustment = inputs.system_energy_value - inputs.technology_energy_value
    capacity_adjustment = inputs.system_capacity_value - inputs.technology_capacity_value
    flexibility_adjustment = inputs.system_flexibility_value - inputs.technology_flexibility_value
    return ValueAdjustmentResult(
        active_base_cost=inputs.active_base_cost,
        valco=inputs.active_base_cost + energy_adjustment + capacity_adjustment + flexibility_adjustment,
        energy_adjustment=energy_adjustment,
        capacity_adjustment=capacity_adjustment,
        flexibility_adjustment=flexibility_adjustment,
    )


def calculate_value_adjusted_costs_from_inputs(
    commodity_inputs: dict[str, ValueAdjustmentInputs],
) -> dict[str, ValueAdjustmentResult]:
    return {
        commodity: calculate_value_adjusted_cost(inputs)
        for commodity, inputs in commodity_inputs.items()
    }


def _xlco_market_output_for_commodity(econ: Economics, commodity: str):
    if commodity == ELECTRICITY_COMMODITY:
        return getattr(econ, "XLCOE_Market", None)
    if commodity == HEAT_COMMODITY:
        return getattr(econ, "XLCOH_Market", None)
    if commodity == COOLING_COMMODITY:
        return getattr(econ, "XLCOC_Market", None)
    return None


def select_active_valco_base_costs(econ: Economics, model: Model) -> dict[str, float]:
    bases = build_levelized_cost_bases(econ, model)
    use_xlco = bool(getattr(getattr(econ, "DoXLCOCalculations", None), "value", False))
    active_base_costs: dict[str, float] = {}
    for commodity, basis in bases.items():
        if basis.discounted_output <= 0.0:
            continue

        active_base_cost = basis.public_value
        if use_xlco:
            xlco_output = _xlco_market_output_for_commodity(econ, commodity)
            if xlco_output is not None:
                active_base_cost = float(xlco_output.value)
        active_base_costs[commodity] = active_base_cost

    return active_base_costs


def _empty_value_adjustment_result(base_cost: float = 0.0) -> ValueAdjustmentResult:
    return ValueAdjustmentResult(active_base_cost=base_cost, valco=base_cost)


def assign_value_adjusted_levelized_cost_outputs(
    econ: Economics,
    commodity_results: dict[str, ValueAdjustmentResult],
) -> dict[str, ValueAdjustmentResult]:
    electricity_result = commodity_results.get(ELECTRICITY_COMMODITY, _empty_value_adjustment_result())
    heat_result = commodity_results.get(HEAT_COMMODITY, _empty_value_adjustment_result())
    cooling_result = commodity_results.get(COOLING_COMMODITY, _empty_value_adjustment_result())

    if hasattr(econ, "VALCOE"):
        econ.VALCOE.value = electricity_result.valco
    if hasattr(econ, "VALCOE_EnergyAdjustment"):
        econ.VALCOE_EnergyAdjustment.value = electricity_result.energy_adjustment
    if hasattr(econ, "VALCOE_CapacityAdjustment"):
        econ.VALCOE_CapacityAdjustment.value = electricity_result.capacity_adjustment
    if hasattr(econ, "VALCOE_FlexibilityAdjustment"):
        econ.VALCOE_FlexibilityAdjustment.value = electricity_result.flexibility_adjustment

    if hasattr(econ, "VALCOH"):
        econ.VALCOH.value = heat_result.valco
    if hasattr(econ, "VALCOH_EnergyAdjustment"):
        econ.VALCOH_EnergyAdjustment.value = heat_result.energy_adjustment
    if hasattr(econ, "VALCOH_CapacityAdjustment"):
        econ.VALCOH_CapacityAdjustment.value = heat_result.capacity_adjustment
    if hasattr(econ, "VALCOH_FlexibilityAdjustment"):
        econ.VALCOH_FlexibilityAdjustment.value = heat_result.flexibility_adjustment

    if hasattr(econ, "VALCOC"):
        econ.VALCOC.value = cooling_result.valco
    if hasattr(econ, "VALCOC_EnergyAdjustment"):
        econ.VALCOC_EnergyAdjustment.value = cooling_result.energy_adjustment
    if hasattr(econ, "VALCOC_CapacityAdjustment"):
        econ.VALCOC_CapacityAdjustment.value = cooling_result.capacity_adjustment
    if hasattr(econ, "VALCOC_FlexibilityAdjustment"):
        econ.VALCOC_FlexibilityAdjustment.value = cooling_result.flexibility_adjustment

    return commodity_results


def build_default_value_adjustment_inputs(econ: Economics, model: Model) -> dict[str, ValueAdjustmentInputs]:
    return {
        commodity: ValueAdjustmentInputs(active_base_cost=base_cost)
        for commodity, base_cost in select_active_valco_base_costs(econ, model).items()
    }

