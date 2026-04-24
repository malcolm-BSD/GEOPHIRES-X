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
    """Inputs required to calculate a single VALCO adjustment.

    Attributes:
        active_base_cost: Baseline public cost used as the starting point for VALCO.
        system_energy_value: System-average energy-value term.
        technology_energy_value: Technology-specific energy-value term.
        system_capacity_value: System-average capacity-value term.
        technology_capacity_value: Technology-specific capacity-value term.
        system_flexibility_value: System-average flexibility-value term.
        technology_flexibility_value: Technology-specific flexibility-value term.
    """

    active_base_cost: float
    system_energy_value: float = 0.0
    technology_energy_value: float = 0.0
    system_capacity_value: float = 0.0
    technology_capacity_value: float = 0.0
    system_flexibility_value: float = 0.0
    technology_flexibility_value: float = 0.0


@dataclass(frozen=True)
class ValueAdjustmentResult:
    """Calculated VALCO result for a single commodity.

    Attributes:
        active_base_cost: Starting cost basis used for the calculation.
        valco: Final value-adjusted levelized cost.
        energy_adjustment: Net energy-value adjustment.
        capacity_adjustment: Net capacity-value adjustment.
        flexibility_adjustment: Net flexibility-value adjustment.
    """

    active_base_cost: float = 0.0
    valco: float = 0.0
    energy_adjustment: float = 0.0
    capacity_adjustment: float = 0.0
    flexibility_adjustment: float = 0.0


def calculate_annual_mwh_per_kw_year(utilization_factor: float) -> float:
    """Convert utilization factor to annual MWh per kW-year.

    Args:
        utilization_factor: Plant utilization factor expressed as a decimal fraction.

    Returns:
        Annual MWh produced per installed kW-year.
    """
    return float(utilization_factor) * 8.76


def convert_dollars_per_mwh_to_unit(value: float, target_unit) -> float:
    """Convert a ``USD/MWh`` value into an active public energy-cost unit.

    Args:
        value: Value in ``USD/MWh``.
        target_unit: Target public energy-cost unit enum.

    Returns:
        Converted value in the requested public units.
    """
    return float(quantity(value, EnergyCostUnit.DOLLARSPERMWH.value).to(target_unit.value).magnitude)


def derive_technology_value_from_annualized_inputs(
    basis_value_per_kw_year: float,
    multiplier: float,
    utilization_factor: float,
    target_unit,
) -> float:
    """Derive a technology-specific value term from annualized ``USD/kW-year`` inputs.

    Args:
        basis_value_per_kw_year: Annualized basis value in ``USD/kW-year``.
        multiplier: Commodity-specific credit or multiplier applied to the basis value.
        utilization_factor: Utilization factor used to convert annualized capacity to annual output.
        target_unit: Public output unit to convert the derived value into.

    Returns:
        Technology-specific value term in the requested public output units.
    """
    annual_mwh_per_kw_year = calculate_annual_mwh_per_kw_year(utilization_factor)
    if annual_mwh_per_kw_year <= 0.0:
        return 0.0

    dollars_per_mwh = float(multiplier) * float(basis_value_per_kw_year) / annual_mwh_per_kw_year
    return convert_dollars_per_mwh_to_unit(dollars_per_mwh, target_unit)


def derive_valcoe_technology_capacity_value(
    basis_capacity_value_per_kw_year: float,
    capacity_credit: float,
    utilization_factor: float,
    target_unit,
) -> float:
    """Derive the electricity capacity-value term for VALCOE.

    Args:
        basis_capacity_value_per_kw_year: Basis capacity value in ``USD/kW-year``.
        capacity_credit: Capacity-credit multiplier applied to the basis value.
        utilization_factor: Utilization factor used for annualization.
        target_unit: Public output unit to convert the derived value into.

    Returns:
        Technology capacity-value term in the requested public output units.
    """
    return derive_technology_value_from_annualized_inputs(
        basis_value_per_kw_year=basis_capacity_value_per_kw_year,
        multiplier=capacity_credit,
        utilization_factor=utilization_factor,
        target_unit=target_unit,
    )


def derive_valcoe_technology_flexibility_value(
    base_flexibility_value_per_kw_year: float,
    flexibility_multiplier: float,
    utilization_factor: float,
    target_unit,
) -> float:
    """Derive the electricity flexibility-value term for VALCOE.

    Args:
        base_flexibility_value_per_kw_year: Base flexibility value in ``USD/kW-year``.
        flexibility_multiplier: Multiplier applied to the base flexibility value.
        utilization_factor: Utilization factor used for annualization.
        target_unit: Public output unit to convert the derived value into.

    Returns:
        Technology flexibility-value term in the requested public output units.
    """
    return derive_technology_value_from_annualized_inputs(
        basis_value_per_kw_year=base_flexibility_value_per_kw_year,
        multiplier=flexibility_multiplier,
        utilization_factor=utilization_factor,
        target_unit=target_unit,
    )


def calculate_value_adjusted_cost(inputs: ValueAdjustmentInputs) -> ValueAdjustmentResult:
    """Calculate a value-adjusted levelized cost from already-normalized component inputs.

    Args:
        inputs: Active base cost and all system/technology value terms for a single commodity.

    Returns:
        A :class:`ValueAdjustmentResult` containing the net component adjustments and final VALCO.
    """
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
    """Calculate VALCO results for multiple commodities from normalized inputs.

    Args:
        commodity_inputs: Mapping of commodity name to normalized value-adjustment inputs.

    Returns:
        Mapping of commodity name to :class:`ValueAdjustmentResult`.
    """
    return {
        commodity: calculate_value_adjusted_cost(inputs)
        for commodity, inputs in commodity_inputs.items()
    }


def _direct_value_adjustment_inputs_from_parameters(econ: Economics, model: Model) -> dict[str, ValueAdjustmentInputs]:
    """Build normalized direct-mode VALCO inputs from public economics parameters.

    Args:
        econ: Economics object containing public VALCO parameters.
        model: Full GEOPHIRES model for the current run.

    Returns:
        Mapping of active commodity name to normalized direct-mode inputs.
    """
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
    """Build normalized derived-mode VALCO inputs from public economics parameters.

    Direct energy terms are still provided explicitly in this mode. Capacity and flexibility
    technology terms are derived from annualized ``USD/kW-year`` inputs and utilization factor.

    Args:
        econ: Economics object containing public VALCO parameters.
        model: Full GEOPHIRES model for the current run.

    Returns:
        Mapping of active commodity name to normalized derived-mode inputs.
    """
    commodity_inputs = _direct_value_adjustment_inputs_from_parameters(econ, model)
    active_base_costs = select_active_valco_base_costs(econ, model)
    utilization_factor = float(model.surfaceplant.utilization_factor.value)

    if ELECTRICITY_COMMODITY in active_base_costs:
        # Electricity keeps its source-grounded helper names, but the underlying annualized
        # derivation is shared with heat and cooling to keep the mode behavior consistent.
        direct_inputs = commodity_inputs.get(ELECTRICITY_COMMODITY)
        if direct_inputs is not None:
            electricity_units = getattr(getattr(econ, "LCOE", None), "CurrentUnits", EnergyCostUnit.CENTSSPERKWH)
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

    if HEAT_COMMODITY in active_base_costs:
        # Heat derived mode is an explicit GEOPHIRES extension that reuses the same annualized
        # capacity/flexibility structure without claiming a source-paper thermal market model.
        direct_inputs = commodity_inputs.get(HEAT_COMMODITY)
        if direct_inputs is not None:
            heat_units = getattr(getattr(econ, "LCOH", None), "CurrentUnits", getattr(getattr(econ, "VALCOH", None), "CurrentUnits", None))
            commodity_inputs[HEAT_COMMODITY] = ValueAdjustmentInputs(
                active_base_cost=direct_inputs.active_base_cost,
                system_energy_value=direct_inputs.system_energy_value,
                technology_energy_value=direct_inputs.technology_energy_value,
                system_capacity_value=direct_inputs.system_capacity_value,
                technology_capacity_value=derive_technology_value_from_annualized_inputs(
                    basis_value_per_kw_year=float(econ.VALCOHBasisCapacityValue.value),
                    multiplier=float(econ.VALCOHCapacityCredit.value),
                    utilization_factor=utilization_factor,
                    target_unit=heat_units,
                ),
                system_flexibility_value=direct_inputs.system_flexibility_value,
                technology_flexibility_value=derive_technology_value_from_annualized_inputs(
                    basis_value_per_kw_year=float(econ.VALCOHBaseFlexibilityValue.value),
                    multiplier=float(econ.VALCOHFlexibilityMultiplier.value),
                    utilization_factor=utilization_factor,
                    target_unit=heat_units,
                ),
            )

    if COOLING_COMMODITY in active_base_costs:
        # Cooling follows the same extension pattern as heat: direct energy terms plus derived
        # annualized capacity and flexibility terms in the active public output units.
        direct_inputs = commodity_inputs.get(COOLING_COMMODITY)
        if direct_inputs is not None:
            cooling_units = getattr(getattr(econ, "LCOC", None), "CurrentUnits", getattr(getattr(econ, "VALCOC", None), "CurrentUnits", None))
            commodity_inputs[COOLING_COMMODITY] = ValueAdjustmentInputs(
                active_base_cost=direct_inputs.active_base_cost,
                system_energy_value=direct_inputs.system_energy_value,
                technology_energy_value=direct_inputs.technology_energy_value,
                system_capacity_value=direct_inputs.system_capacity_value,
                technology_capacity_value=derive_technology_value_from_annualized_inputs(
                    basis_value_per_kw_year=float(econ.VALCOCBasisCapacityValue.value),
                    multiplier=float(econ.VALCOCCapacityCredit.value),
                    utilization_factor=utilization_factor,
                    target_unit=cooling_units,
                ),
                system_flexibility_value=direct_inputs.system_flexibility_value,
                technology_flexibility_value=derive_technology_value_from_annualized_inputs(
                    basis_value_per_kw_year=float(econ.VALCOCBaseFlexibilityValue.value),
                    multiplier=float(econ.VALCOCFlexibilityMultiplier.value),
                    utilization_factor=utilization_factor,
                    target_unit=cooling_units,
                ),
            )

    return commodity_inputs


def _xlco_market_output_for_commodity(econ: Economics, commodity: str):
    """Return the XLCO market output parameter for a commodity, if present.

    Args:
        econ: Economics object containing XLCO outputs.
        commodity: Commodity identifier.

    Returns:
        The matching XLCO market output parameter object, or ``None`` if unavailable.
    """
    if commodity == ELECTRICITY_COMMODITY:
        return getattr(econ, "XLCOE_Market", None)
    if commodity == HEAT_COMMODITY:
        return getattr(econ, "XLCOH_Market", None)
    if commodity == COOLING_COMMODITY:
        return getattr(econ, "XLCOC_Market", None)
    return None


def select_active_valco_base_costs(econ: Economics, model: Model) -> dict[str, float]:
    """Select the active baseline cost used by VALCO for each commodity.

    When XLCO is active for a commodity, the base is ``XLCO*_Market``. Otherwise the base is the
    standard public ``LCO*`` output reconstructed from the shared levelized-cost basis helper.

    Args:
        econ: Economics object containing baseline and XLCO outputs.
        model: Full GEOPHIRES model for the current run.

    Returns:
        Mapping of active commodity name to public baseline cost.
    """
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
    """Return a zero-adjustment VALCO result.

    Args:
        base_cost: Base cost to preserve in the result.

    Returns:
        A zero-adjustment :class:`ValueAdjustmentResult`.
    """
    return ValueAdjustmentResult(active_base_cost=base_cost, valco=base_cost)


def calculate_value_adjusted_levelized_costs(econ: Economics, model: Model) -> dict[str, ValueAdjustmentResult]:
    """Calculate VALCO results for all active commodities in the current model.

    Args:
        econ: Economics object containing VALCO parameters and baseline outputs.
        model: Full GEOPHIRES model for the current run.

    Returns:
        Mapping of active commodity name to :class:`ValueAdjustmentResult`.

    Raises:
        NotImplementedError: If ``VALCO Calculation Mode`` is not one of the supported modes.
    """
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
    """Write calculated VALCO values back to the economics output parameters.

    Args:
        econ: Economics object whose ``VALCO*`` output parameters will be updated.
        commodity_results: Mapping of commodity name to calculated VALCO results.

    Returns:
        The same ``commodity_results`` mapping after write-back.
    """
    electricity_result = commodity_results.get(ELECTRICITY_COMMODITY, _empty_value_adjustment_result())
    heat_result = commodity_results.get(HEAT_COMMODITY, _empty_value_adjustment_result())
    cooling_result = commodity_results.get(COOLING_COMMODITY, _empty_value_adjustment_result())
    electricity_units = getattr(getattr(econ, "LCOE", None), "CurrentUnits", getattr(getattr(econ, "VALCOE", None), "CurrentUnits", None))
    heat_units = getattr(getattr(econ, "LCOH", None), "CurrentUnits", getattr(getattr(econ, "VALCOH", None), "CurrentUnits", None))
    cooling_units = getattr(getattr(econ, "LCOC", None), "CurrentUnits", getattr(getattr(econ, "VALCOC", None), "CurrentUnits", None))

    if hasattr(econ, "VALCOE"):
        # Each output is written in the same public units as its paired baseline LCO output so the
        # reported adjustments remain directly comparable in text output, client parsing, and schema use.
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
    """Build zero-adjustment VALCO inputs anchored to the active baseline costs.

    Args:
        econ: Economics object containing baseline and optional XLCO outputs.
        model: Full GEOPHIRES model for the current run.

    Returns:
        Mapping of active commodity name to zero-adjustment :class:`ValueAdjustmentInputs`.
    """
    return {
        commodity: ValueAdjustmentInputs(active_base_cost=base_cost)
        for commodity, base_cost in select_active_valco_base_costs(econ, model).items()
    }


def calculate_and_assign_value_adjusted_levelized_cost_outputs(
    econ: Economics,
    model: Model,
) -> dict[str, ValueAdjustmentResult]:
    """Calculate VALCO results and immediately write them back to economics outputs.

    Args:
        econ: Economics object whose VALCO outputs should be populated.
        model: Full GEOPHIRES model for the current run.

    Returns:
        Mapping of active commodity name to :class:`ValueAdjustmentResult`.
    """
    return assign_value_adjusted_levelized_cost_outputs(econ, calculate_value_adjusted_levelized_costs(econ, model))
