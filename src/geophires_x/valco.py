from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from geophires_x.GeoPHIRESUtils import quantity
from geophires_x.levelized_costs import (
    COOLING_COMMODITY,
    ELECTRICITY_COMMODITY,
    HEAT_COMMODITY,
    build_levelized_cost_bases,
)
from geophires_x.Units import EnergyCostUnit

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


def calculate_annual_mwh_per_kw_year(utilization_factor: float) -> float:
    return float(utilization_factor) * 8.76


def convert_dollars_per_mwh_to_electricity_unit(value: float, target_unit) -> float:
    return float(quantity(value, EnergyCostUnit.DOLLARSPERMWH.value).to(target_unit.value).magnitude)


def derive_valcoe_technology_capacity_value(
    basis_capacity_value_per_kw_year: float,
    capacity_credit: float,
    utilization_factor: float,
    target_unit,
) -> float:
    annual_mwh_per_kw_year = calculate_annual_mwh_per_kw_year(utilization_factor)
    if annual_mwh_per_kw_year <= 0.0:
        return 0.0

    dollars_per_mwh = float(capacity_credit) * float(basis_capacity_value_per_kw_year) / annual_mwh_per_kw_year
    return convert_dollars_per_mwh_to_electricity_unit(dollars_per_mwh, target_unit)


def derive_valcoe_technology_flexibility_value(
    base_flexibility_value_per_kw_year: float,
    flexibility_multiplier: float,
    utilization_factor: float,
    target_unit,
) -> float:
    annual_mwh_per_kw_year = calculate_annual_mwh_per_kw_year(utilization_factor)
    if annual_mwh_per_kw_year <= 0.0:
        return 0.0

    dollars_per_mwh = float(flexibility_multiplier) * float(base_flexibility_value_per_kw_year) / annual_mwh_per_kw_year
    return convert_dollars_per_mwh_to_electricity_unit(dollars_per_mwh, target_unit)


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


def _direct_value_adjustment_inputs_from_parameters(econ: Economics, model: Model) -> dict[str, ValueAdjustmentInputs]:
    active_base_costs = select_active_valco_base_costs(econ, model)
    commodity_inputs: dict[str, ValueAdjustmentInputs] = {}

    if ELECTRICITY_COMMODITY in active_base_costs:
        commodity_inputs[ELECTRICITY_COMMODITY] = ValueAdjustmentInputs(
            active_base_cost=active_base_costs[ELECTRICITY_COMMODITY],
            system_energy_value=float(econ.VALCOESystemAverageEnergyValue.value),
            technology_energy_value=float(econ.VALCOETechnologyEnergyValue.value),
            system_capacity_value=float(econ.VALCOESystemAverageCapacityValue.value),
            technology_capacity_value=float(econ.VALCOETechnologyCapacityValue.value),
            system_flexibility_value=float(econ.VALCOESystemAverageFlexibilityValue.value),
            technology_flexibility_value=float(econ.VALCOETechnologyFlexibilityValue.value),
        )

    if HEAT_COMMODITY in active_base_costs:
        commodity_inputs[HEAT_COMMODITY] = ValueAdjustmentInputs(
            active_base_cost=active_base_costs[HEAT_COMMODITY],
            system_energy_value=float(econ.VALCOHSystemAverageEnergyValue.value),
            technology_energy_value=float(econ.VALCOHTechnologyEnergyValue.value),
            system_capacity_value=float(econ.VALCOHSystemAverageCapacityValue.value),
            technology_capacity_value=float(econ.VALCOHTechnologyCapacityValue.value),
            system_flexibility_value=float(econ.VALCOHSystemAverageFlexibilityValue.value),
            technology_flexibility_value=float(econ.VALCOHTechnologyFlexibilityValue.value),
        )

    if COOLING_COMMODITY in active_base_costs:
        commodity_inputs[COOLING_COMMODITY] = ValueAdjustmentInputs(
            active_base_cost=active_base_costs[COOLING_COMMODITY],
            system_energy_value=float(econ.VALCOCSystemAverageEnergyValue.value),
            technology_energy_value=float(econ.VALCOCTechnologyEnergyValue.value),
            system_capacity_value=float(econ.VALCOCSystemAverageCapacityValue.value),
            technology_capacity_value=float(econ.VALCOCTechnologyCapacityValue.value),
            system_flexibility_value=float(econ.VALCOCSystemAverageFlexibilityValue.value),
            technology_flexibility_value=float(econ.VALCOCTechnologyFlexibilityValue.value),
        )

    return commodity_inputs


def _derived_value_adjustment_inputs_from_parameters(econ: Economics, model: Model) -> dict[str, ValueAdjustmentInputs]:
    commodity_inputs = _direct_value_adjustment_inputs_from_parameters(econ, model)
    active_base_costs = select_active_valco_base_costs(econ, model)

    if ELECTRICITY_COMMODITY not in active_base_costs:
        return commodity_inputs

    direct_inputs = commodity_inputs.get(ELECTRICITY_COMMODITY)
    if direct_inputs is None:
        return commodity_inputs

    electricity_units = getattr(getattr(econ, "LCOE", None), "CurrentUnits", EnergyCostUnit.CENTSSPERKWH)
    utilization_factor = float(model.surfaceplant.utilization_factor.value)
    commodity_inputs[ELECTRICITY_COMMODITY] = ValueAdjustmentInputs(
        active_base_cost=direct_inputs.active_base_cost,
        system_energy_value=direct_inputs.system_energy_value,
        technology_energy_value=direct_inputs.technology_energy_value,
        system_capacity_value=direct_inputs.system_capacity_value,
        technology_capacity_value=derive_valcoe_technology_capacity_value(
            basis_capacity_value_per_kw_year=float(econ.VALCOEBasisCapacityValue.value),
            capacity_credit=float(econ.VALCOECapacityCredit.value),
            utilization_factor=utilization_factor,
            target_unit=electricity_units,
        ),
        system_flexibility_value=direct_inputs.system_flexibility_value,
        technology_flexibility_value=derive_valcoe_technology_flexibility_value(
            base_flexibility_value_per_kw_year=float(econ.VALCOEBaseFlexibilityValue.value),
            flexibility_multiplier=float(econ.VALCOEFlexibilityMultiplier.value),
            utilization_factor=utilization_factor,
            target_unit=electricity_units,
        ),
    )
    return commodity_inputs


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


def calculate_value_adjusted_levelized_costs(econ: Economics, model: Model) -> dict[str, ValueAdjustmentResult]:
    if not bool(getattr(getattr(econ, "DoVALCOCalculations", None), "value", False)):
        return {}

    calculation_mode = str(getattr(getattr(econ, "VALCOCalculationMode", None), "value", "Direct")).strip().lower()
    if calculation_mode == "direct":
        return calculate_value_adjusted_costs_from_inputs(_direct_value_adjustment_inputs_from_parameters(econ, model))
    if calculation_mode == "derived":
        return calculate_value_adjusted_costs_from_inputs(_derived_value_adjustment_inputs_from_parameters(econ, model))
    raise NotImplementedError("VALCO Calculation Mode currently supports only Direct or Derived.")


def assign_value_adjusted_levelized_cost_outputs(
    econ: Economics,
    commodity_results: dict[str, ValueAdjustmentResult],
) -> dict[str, ValueAdjustmentResult]:
    electricity_result = commodity_results.get(ELECTRICITY_COMMODITY, _empty_value_adjustment_result())
    heat_result = commodity_results.get(HEAT_COMMODITY, _empty_value_adjustment_result())
    cooling_result = commodity_results.get(COOLING_COMMODITY, _empty_value_adjustment_result())
    electricity_units = getattr(getattr(econ, "LCOE", None), "CurrentUnits", getattr(getattr(econ, "VALCOE", None), "CurrentUnits", None))
    heat_units = getattr(getattr(econ, "LCOH", None), "CurrentUnits", getattr(getattr(econ, "VALCOH", None), "CurrentUnits", None))
    cooling_units = getattr(getattr(econ, "LCOC", None), "CurrentUnits", getattr(getattr(econ, "VALCOC", None), "CurrentUnits", None))

    if hasattr(econ, "VALCOE"):
        if electricity_units is not None:
            econ.VALCOE.CurrentUnits = electricity_units
        econ.VALCOE.value = electricity_result.valco
    if hasattr(econ, "VALCOE_EnergyAdjustment"):
        if electricity_units is not None:
            econ.VALCOE_EnergyAdjustment.CurrentUnits = electricity_units
        econ.VALCOE_EnergyAdjustment.value = electricity_result.energy_adjustment
    if hasattr(econ, "VALCOE_CapacityAdjustment"):
        if electricity_units is not None:
            econ.VALCOE_CapacityAdjustment.CurrentUnits = electricity_units
        econ.VALCOE_CapacityAdjustment.value = electricity_result.capacity_adjustment
    if hasattr(econ, "VALCOE_FlexibilityAdjustment"):
        if electricity_units is not None:
            econ.VALCOE_FlexibilityAdjustment.CurrentUnits = electricity_units
        econ.VALCOE_FlexibilityAdjustment.value = electricity_result.flexibility_adjustment

    if hasattr(econ, "VALCOH"):
        if heat_units is not None:
            econ.VALCOH.CurrentUnits = heat_units
        econ.VALCOH.value = heat_result.valco
    if hasattr(econ, "VALCOH_EnergyAdjustment"):
        if heat_units is not None:
            econ.VALCOH_EnergyAdjustment.CurrentUnits = heat_units
        econ.VALCOH_EnergyAdjustment.value = heat_result.energy_adjustment
    if hasattr(econ, "VALCOH_CapacityAdjustment"):
        if heat_units is not None:
            econ.VALCOH_CapacityAdjustment.CurrentUnits = heat_units
        econ.VALCOH_CapacityAdjustment.value = heat_result.capacity_adjustment
    if hasattr(econ, "VALCOH_FlexibilityAdjustment"):
        if heat_units is not None:
            econ.VALCOH_FlexibilityAdjustment.CurrentUnits = heat_units
        econ.VALCOH_FlexibilityAdjustment.value = heat_result.flexibility_adjustment

    if hasattr(econ, "VALCOC"):
        if cooling_units is not None:
            econ.VALCOC.CurrentUnits = cooling_units
        econ.VALCOC.value = cooling_result.valco
    if hasattr(econ, "VALCOC_EnergyAdjustment"):
        if cooling_units is not None:
            econ.VALCOC_EnergyAdjustment.CurrentUnits = cooling_units
        econ.VALCOC_EnergyAdjustment.value = cooling_result.energy_adjustment
    if hasattr(econ, "VALCOC_CapacityAdjustment"):
        if cooling_units is not None:
            econ.VALCOC_CapacityAdjustment.CurrentUnits = cooling_units
        econ.VALCOC_CapacityAdjustment.value = cooling_result.capacity_adjustment
    if hasattr(econ, "VALCOC_FlexibilityAdjustment"):
        if cooling_units is not None:
            econ.VALCOC_FlexibilityAdjustment.CurrentUnits = cooling_units
        econ.VALCOC_FlexibilityAdjustment.value = cooling_result.flexibility_adjustment

    return commodity_results


def build_default_value_adjustment_inputs(econ: Economics, model: Model) -> dict[str, ValueAdjustmentInputs]:
    return {
        commodity: ValueAdjustmentInputs(active_base_cost=base_cost)
        for commodity, base_cost in select_active_valco_base_costs(econ, model).items()
    }


def calculate_and_assign_value_adjusted_levelized_cost_outputs(
    econ: Economics,
    model: Model,
) -> dict[str, ValueAdjustmentResult]:
    return assign_value_adjusted_levelized_cost_outputs(econ, calculate_value_adjusted_levelized_costs(econ, model))
