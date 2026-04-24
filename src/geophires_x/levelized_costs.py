from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from geophires_x.GeoPHIRESUtils import quantity
from geophires_x.OptionList import EconomicModel, EndUseOptions, PlantType
from geophires_x.Units import EnergyCostUnit
from geophires_x.Units import convertible_unit

if TYPE_CHECKING:
    from geophires_x.Economics import Economics
    from geophires_x.Model import Model


ELECTRICITY_COMMODITY = 'electricity'
HEAT_COMMODITY = 'heat'
COOLING_COMMODITY = 'cooling'

_ELECTRICITY_PRICE_FACTOR = 1.0e8
_HEAT_AND_COOLING_PRICE_FACTOR = 1.0e8 * 2.931
_DISTRICT_HEATING_PRICE_FACTOR = 1.0e2 * 2.931
_DOLLARS_PER_MWH_PRICE_FACTOR = 1.0e9


@dataclass(frozen=True)
class LevelizedCostBasis:
    """Discounted cost/output basis for a single commodity.

    Attributes:
        commodity: Commodity identifier such as ``electricity``, ``heat``, or ``cooling``.
        public_value: Baseline levelized cost in the commodity's current public output units.
        baseline_discounted_cost_musd: Discounted lifecycle cost numerator in million USD.
        discounted_output: Discounted lifecycle output denominator in the commodity's native
            discounted-energy units.
        public_price_factor: Scaling factor used to convert the internal discounted ratio into
            the public levelized-cost units exposed by GEOPHIRES.
    """

    commodity: str
    public_value: float
    baseline_discounted_cost_musd: float
    discounted_output: float
    public_price_factor: float


def _build_basis(
    commodity: str,
    baseline_discounted_cost_musd: float,
    discounted_output: float,
    public_price_factor: float,
) -> LevelizedCostBasis:
    """Build a normalized levelized-cost basis object.

    Args:
        commodity: Commodity identifier for the basis being created.
        baseline_discounted_cost_musd: Discounted lifecycle cost numerator in million USD.
        discounted_output: Discounted lifecycle commodity output denominator.
        public_price_factor: Unit scaling factor that maps the internal discounted ratio to
            the public output units.

    Returns:
        A populated :class:`LevelizedCostBasis`.
    """
    public_value = 0.0
    if discounted_output > 0.0:
        public_value = baseline_discounted_cost_musd / discounted_output * public_price_factor

    return LevelizedCostBasis(
        commodity=commodity,
        public_value=float(public_value),
        baseline_discounted_cost_musd=float(baseline_discounted_cost_musd),
        discounted_output=float(discounted_output),
        public_price_factor=float(public_price_factor),
    )


def _output_price_factor(unit, default_factor: float) -> float:
    """Return the public price scaling factor for the requested output unit.

    Args:
        unit: The current public output unit for the levelized-cost result.
        default_factor: The default scaling factor for the commodity family.

    Returns:
        The scaling factor needed to express the result in the requested output units.
    """
    if unit == EnergyCostUnit.DOLLARSPERMWH:
        return _DOLLARS_PER_MWH_PRICE_FACTOR
    return default_factor


def build_levelized_cost_bases(econ: Economics, model: Model) -> dict[str, LevelizedCostBasis]:
    """Build discounted cost/output bases for all active commodities in the current model.

    The resulting dictionary is the shared source of truth for baseline ``LCOE``, ``LCOH``,
    ``LCOC``, ``XLCO*``, and ``VALCO*`` calculations. It mirrors the economic-model and
    end-use branching already present in GEOPHIRES and normalizes every path to the same
    discounted numerator/denominator representation.

    Args:
        econ: Economics object containing the current financial-model state and cost values.
        model: Full GEOPHIRES model containing surface-plant outputs and option selections.

    Returns:
        A dictionary keyed by commodity name with one :class:`LevelizedCostBasis` per active
        commodity.
    """
    bases: dict[str, LevelizedCostBasis] = {}

    ccap_elec = econ.CCap.value * econ.CAPEX_heat_electricity_plant_ratio.value
    coam_elec = econ.Coam.value * econ.CAPEX_heat_electricity_plant_ratio.value
    ccap_heat = econ.CCap.value * (1.0 - econ.CAPEX_heat_electricity_plant_ratio.value)
    coam_heat = econ.Coam.value * (1.0 - econ.CAPEX_heat_electricity_plant_ratio.value)

    def _capex_total_plus_construction_inflation() -> float:
        """Return total CAPEX including construction-period inflation."""
        econ.inflation_cost_during_construction.value = quantity(
            econ.CCap.value * econ.inflrateconstruction.value,
            econ.CCap.CurrentUnits,
        ).to(econ.inflation_cost_during_construction.CurrentUnits).magnitude
        return econ.CCap.value + econ.inflation_cost_during_construction.value

    def _construction_inflation_cost_elec_heat() -> tuple[float, float]:
        """Split construction inflation across electricity and heat cost allocations."""
        construction_inflation_cost_elec = ccap_elec * econ.inflrateconstruction.value
        construction_inflation_cost_heat = ccap_heat * econ.inflrateconstruction.value
        econ.inflation_cost_during_construction.value = quantity(
            construction_inflation_cost_elec + construction_inflation_cost_heat,
            econ.CCap.CurrentUnits,
        ).to(econ.inflation_cost_during_construction.CurrentUnits).magnitude
        return ccap_elec + construction_inflation_cost_elec, ccap_heat + construction_inflation_cost_heat

    enduse_option = model.surfaceplant.enduse_option.value
    plant_type = model.surfaceplant.plant_type.value

    if econ.econmodel.value == EconomicModel.FCR:
        # The fixed-charge-rate model keeps the numerator annualized, so each branch only needs
        # the average annual commodity output that corresponds to its levelized denominator.
        capex_total_plus_infl = _capex_total_plus_construction_inflation()

        if enduse_option == EndUseOptions.ELECTRICITY:
            bases[ELECTRICITY_COMMODITY] = _build_basis(
                ELECTRICITY_COMMODITY,
                econ.FCR.value * capex_total_plus_infl + econ.Coam.value,
                float(np.average(model.surfaceplant.NetkWhProduced.value)),
                _ELECTRICITY_PRICE_FACTOR,
            )
        elif enduse_option == EndUseOptions.HEAT and plant_type not in [
            PlantType.ABSORPTION_CHILLER,
            PlantType.HEAT_PUMP,
            PlantType.DISTRICT_HEATING,
        ]:
            bases[HEAT_COMMODITY] = _build_basis(
                HEAT_COMMODITY,
                econ.FCR.value * capex_total_plus_infl + econ.Coam.value + econ.averageannualpumpingcosts.value,
                float(np.average(model.surfaceplant.HeatkWhProduced.value)),
                _HEAT_AND_COOLING_PRICE_FACTOR,
            )
        elif enduse_option in [
            EndUseOptions.COGENERATION_TOPPING_EXTRA_HEAT,
            EndUseOptions.COGENERATION_TOPPING_EXTRA_ELECTRICITY,
            EndUseOptions.COGENERATION_BOTTOMING_EXTRA_ELECTRICITY,
            EndUseOptions.COGENERATION_BOTTOMING_EXTRA_HEAT,
            EndUseOptions.COGENERATION_PARALLEL_EXTRA_HEAT,
            EndUseOptions.COGENERATION_PARALLEL_EXTRA_ELECTRICITY,
        ]:
            capex_elec_plus_infl, capex_heat_plus_infl = _construction_inflation_cost_elec_heat()
            bases[ELECTRICITY_COMMODITY] = _build_basis(
                ELECTRICITY_COMMODITY,
                econ.FCR.value * capex_elec_plus_infl + coam_elec,
                float(np.average(model.surfaceplant.NetkWhProduced.value)),
                _ELECTRICITY_PRICE_FACTOR,
            )
            bases[HEAT_COMMODITY] = _build_basis(
                HEAT_COMMODITY,
                econ.FCR.value * capex_heat_plus_infl + coam_heat + econ.averageannualpumpingcosts.value,
                float(np.average(model.surfaceplant.HeatkWhProduced.value)),
                _HEAT_AND_COOLING_PRICE_FACTOR,
            )
        elif enduse_option == EndUseOptions.HEAT and plant_type == PlantType.ABSORPTION_CHILLER:
            bases[COOLING_COMMODITY] = _build_basis(
                COOLING_COMMODITY,
                econ.FCR.value * capex_total_plus_infl + econ.Coam.value + econ.averageannualpumpingcosts.value,
                float(np.average(model.surfaceplant.cooling_kWh_Produced.value)),
                _HEAT_AND_COOLING_PRICE_FACTOR,
            )
        elif enduse_option == EndUseOptions.HEAT and plant_type == PlantType.HEAT_PUMP:
            bases[HEAT_COMMODITY] = _build_basis(
                HEAT_COMMODITY,
                econ.FCR.value * capex_total_plus_infl
                + econ.Coam.value
                + econ.averageannualpumpingcosts.value
                + econ.averageannualheatpumpelectricitycost.value,
                float(np.average(model.surfaceplant.HeatkWhProduced.value)),
                _HEAT_AND_COOLING_PRICE_FACTOR,
            )
        elif enduse_option == EndUseOptions.HEAT and plant_type == PlantType.DISTRICT_HEATING:
            bases[HEAT_COMMODITY] = _build_basis(
                HEAT_COMMODITY,
                econ.FCR.value * capex_total_plus_infl
                + econ.Coam.value
                + econ.averageannualpumpingcosts.value
                + econ.averageannualngcost.value,
                float(model.surfaceplant.annual_heating_demand.value),
                _DISTRICT_HEATING_PRICE_FACTOR,
            )

    elif econ.econmodel.value == EconomicModel.STANDARDIZED_LEVELIZED_COST:
        # The standardized levelized-cost model discounts annual cost and output streams explicitly.
        discount_vector = 1.0 / np.power(
            1 + econ.discountrate.value,
            np.linspace(0, model.surfaceplant.plant_lifetime.value - 1, model.surfaceplant.plant_lifetime.value),
        )
        capex_total_plus_infl = _capex_total_plus_construction_inflation()

        if enduse_option == EndUseOptions.ELECTRICITY:
            bases[ELECTRICITY_COMMODITY] = _build_basis(
                ELECTRICITY_COMMODITY,
                capex_total_plus_infl + np.sum(econ.Coam.value * discount_vector),
                float(np.sum(model.surfaceplant.NetkWhProduced.value * discount_vector)),
                _ELECTRICITY_PRICE_FACTOR,
            )
        elif enduse_option == EndUseOptions.HEAT and plant_type not in [
            PlantType.ABSORPTION_CHILLER,
            PlantType.HEAT_PUMP,
            PlantType.DISTRICT_HEATING,
        ]:
            econ.averageannualpumpingcosts.value = (
                np.average(model.surfaceplant.PumpingkWh.value) * model.surfaceplant.electricity_cost_to_buy.value / 1e6
            )
            bases[HEAT_COMMODITY] = _build_basis(
                HEAT_COMMODITY,
                capex_total_plus_infl
                + np.sum(
                    (
                        econ.Coam.value
                        + model.surfaceplant.PumpingkWh.value * model.surfaceplant.electricity_cost_to_buy.value / 1e6
                    )
                    * discount_vector
                ),
                float(np.sum(model.surfaceplant.HeatkWhProduced.value * discount_vector)),
                _HEAT_AND_COOLING_PRICE_FACTOR,
            )
        elif enduse_option in [
            EndUseOptions.COGENERATION_TOPPING_EXTRA_HEAT,
            EndUseOptions.COGENERATION_TOPPING_EXTRA_ELECTRICITY,
            EndUseOptions.COGENERATION_BOTTOMING_EXTRA_ELECTRICITY,
            EndUseOptions.COGENERATION_BOTTOMING_EXTRA_HEAT,
            EndUseOptions.COGENERATION_PARALLEL_EXTRA_HEAT,
            EndUseOptions.COGENERATION_PARALLEL_EXTRA_ELECTRICITY,
        ]:
            capex_elec_plus_infl, capex_heat_plus_infl = _construction_inflation_cost_elec_heat()
            bases[ELECTRICITY_COMMODITY] = _build_basis(
                ELECTRICITY_COMMODITY,
                capex_elec_plus_infl + np.sum(coam_elec * discount_vector),
                float(np.sum(model.surfaceplant.NetkWhProduced.value * discount_vector)),
                _ELECTRICITY_PRICE_FACTOR,
            )
            bases[HEAT_COMMODITY] = _build_basis(
                HEAT_COMMODITY,
                capex_heat_plus_infl
                + np.sum(
                    (
                        coam_heat
                        + model.surfaceplant.PumpingkWh.value * model.surfaceplant.electricity_cost_to_buy.value / 1e6
                    )
                    * discount_vector
                ),
                float(np.sum(model.surfaceplant.HeatkWhProduced.value * discount_vector)),
                _HEAT_AND_COOLING_PRICE_FACTOR,
            )
        elif enduse_option == EndUseOptions.HEAT and plant_type == PlantType.ABSORPTION_CHILLER:
            bases[COOLING_COMMODITY] = _build_basis(
                COOLING_COMMODITY,
                capex_total_plus_infl
                + np.sum(
                    (
                        econ.Coam.value
                        + model.surfaceplant.PumpingkWh.value * model.surfaceplant.electricity_cost_to_buy.value / 1e6
                    )
                    * discount_vector
                ),
                float(np.sum(model.surfaceplant.cooling_kWh_Produced.value * discount_vector)),
                _HEAT_AND_COOLING_PRICE_FACTOR,
            )
        elif enduse_option == EndUseOptions.HEAT and plant_type == PlantType.HEAT_PUMP:
            bases[HEAT_COMMODITY] = _build_basis(
                HEAT_COMMODITY,
                capex_total_plus_infl
                + np.sum(
                    (
                        econ.Coam.value
                        + model.surfaceplant.PumpingkWh.value * model.surfaceplant.electricity_cost_to_buy.value / 1e6
                        + model.surfaceplant.heat_pump_electricity_kwh_used.value
                        * model.surfaceplant.electricity_cost_to_buy.value
                        / 1e6
                    )
                    * discount_vector
                ),
                float(np.sum(model.surfaceplant.HeatkWhProduced.value * discount_vector)),
                _HEAT_AND_COOLING_PRICE_FACTOR,
            )
        elif enduse_option == EndUseOptions.HEAT and plant_type == PlantType.DISTRICT_HEATING:
            bases[HEAT_COMMODITY] = _build_basis(
                HEAT_COMMODITY,
                capex_total_plus_infl
                + np.sum(
                    (
                        econ.Coam.value
                        + model.surfaceplant.PumpingkWh.value * model.surfaceplant.electricity_cost_to_buy.value / 1e6
                        + econ.annualngcost.value
                    )
                    * discount_vector
                ),
                float(np.sum(model.surfaceplant.annual_heating_demand.value * discount_vector)),
                _DISTRICT_HEATING_PRICE_FACTOR,
            )

    elif econ.econmodel.value == EconomicModel.SAM_SINGLE_OWNER_PPA:
        # SAM already returns a public LCOE. Reconstruct the discounted basis so downstream XLCO
        # and VALCO calculations can operate on the same normalized representation as other models.
        lcoe = econ.sam_economics_calculations.lcoe_nominal.quantity().to(
            convertible_unit(econ.LCOE.CurrentUnits.value)
        ).magnitude
        discounted_output = float(np.sum(model.surfaceplant.NetkWhProduced.value))
        baseline_discounted_cost_musd = lcoe * discounted_output / _ELECTRICITY_PRICE_FACTOR
        bases[ELECTRICITY_COMMODITY] = LevelizedCostBasis(
            commodity=ELECTRICITY_COMMODITY,
            public_value=float(lcoe),
            baseline_discounted_cost_musd=float(baseline_discounted_cost_musd),
            discounted_output=float(discounted_output),
            public_price_factor=_ELECTRICITY_PRICE_FACTOR,
        )

    elif econ.econmodel.value == EconomicModel.CLGS:
        # The CLGS/AGS path stores annual arrays directly instead of using the standard GEOPHIRES
        # discounted-stream fields, so we rebuild the discounted numerator and denominator here.
        discount_rate = getattr(econ, 'Discount_rate', econ.discountrate.value)
        plant_lifetime = getattr(model.surfaceplant, 'Lifetime', model.surfaceplant.plant_lifetime.value)
        discount_vector = 1.0 / np.power(1.0 + discount_rate, np.linspace(0, plant_lifetime - 1, plant_lifetime))

        total_capex = float(getattr(econ, 'TotalCAPEX', econ.CCap.value))
        annual_opex = np.asarray(getattr(econ, 'OPEX_Plant', econ.Coam.value / 1_000.0), dtype=float)
        if annual_opex.ndim == 0:
            annual_opex = np.full(plant_lifetime, float(annual_opex))

        baseline_discounted_cost_musd = total_capex + float(np.sum(annual_opex * discount_vector))
        if enduse_option == EndUseOptions.ELECTRICITY:
            discounted_output = float(
                np.sum(
                    (
                        np.asarray(model.surfaceplant.Annual_electricity_production, dtype=float)
                        - np.asarray(model.surfaceplant.Annual_pumping_power, dtype=float)
                    )
                    * discount_vector
                )
            )
            bases[ELECTRICITY_COMMODITY] = _build_basis(
                ELECTRICITY_COMMODITY,
                baseline_discounted_cost_musd,
                discounted_output,
                _output_price_factor(econ.LCOE.CurrentUnits, _ELECTRICITY_PRICE_FACTOR),
            )
        elif enduse_option == EndUseOptions.HEAT:
            discounted_output = float(
                np.sum(np.asarray(model.surfaceplant.Annual_heat_production, dtype=float) * discount_vector)
            )
            bases[HEAT_COMMODITY] = _build_basis(
                HEAT_COMMODITY,
                baseline_discounted_cost_musd,
                discounted_output,
                _output_price_factor(econ.LCOH.CurrentUnits, _HEAT_AND_COOLING_PRICE_FACTOR),
            )

    else:
        # The remaining economic-model branch uses the tax-aware BICYCLE style formulation. The
        # goal here is not to replicate each public LCO output formula separately, but to capture
        # the discounted project-cost numerator that those formulas imply for each commodity.
        i_ave = econ.FIB.value * econ.BIR.value * (1 - econ.CTR.value) + (1 - econ.FIB.value) * econ.EIR.value
        crf = i_ave / (1 - np.power(1 + i_ave, -model.surfaceplant.plant_lifetime.value))
        inflation_vector = np.power(
            1 + econ.RINFL.value, np.linspace(1, model.surfaceplant.plant_lifetime.value, model.surfaceplant.plant_lifetime.value)
        )
        discount_vector = 1.0 / np.power(
            1 + i_ave, np.linspace(1, model.surfaceplant.plant_lifetime.value, model.surfaceplant.plant_lifetime.value)
        )
        capex_total_plus_infl = _capex_total_plus_construction_inflation()

        npv_cap = np.sum(capex_total_plus_infl * crf * discount_vector)
        npv_fc = np.sum(capex_total_plus_infl * econ.PTR.value * inflation_vector * discount_vector)
        npv_it = np.sum(
            econ.CTR.value
            / (1 - econ.CTR.value)
            * (capex_total_plus_infl * crf - econ.CCap.value / model.surfaceplant.plant_lifetime.value)
            * discount_vector
        )
        npv_itc = capex_total_plus_infl * econ.RITC.value / (1 - econ.CTR.value)

        if enduse_option == EndUseOptions.ELECTRICITY:
            npv_oandm = np.sum(econ.Coam.value * inflation_vector * discount_vector)
            npv_grt = econ.GTR.value / (1 - econ.GTR.value) * (npv_cap + npv_oandm + npv_fc + npv_it - npv_itc)
            bases[ELECTRICITY_COMMODITY] = _build_basis(
                ELECTRICITY_COMMODITY,
                npv_cap + npv_oandm + npv_fc + npv_it + npv_grt - npv_itc,
                float(np.sum(model.surfaceplant.NetkWhProduced.value * inflation_vector * discount_vector)),
                _ELECTRICITY_PRICE_FACTOR,
            )
        elif enduse_option == EndUseOptions.HEAT and plant_type not in [
            PlantType.ABSORPTION_CHILLER,
            PlantType.HEAT_PUMP,
            PlantType.DISTRICT_HEATING,
        ]:
            pumping_costs = model.surfaceplant.PumpingkWh.value * model.surfaceplant.electricity_cost_to_buy.value / 1e6
            npv_oandm = np.sum((econ.Coam.value + pumping_costs) * inflation_vector * discount_vector)
            npv_grt = econ.GTR.value / (1 - econ.GTR.value) * (npv_cap + npv_oandm + npv_fc + npv_it - npv_itc)
            bases[HEAT_COMMODITY] = _build_basis(
                HEAT_COMMODITY,
                npv_cap + npv_oandm + npv_fc + npv_it + npv_grt - npv_itc,
                float(np.sum(model.surfaceplant.HeatkWhProduced.value * inflation_vector * discount_vector)),
                _HEAT_AND_COOLING_PRICE_FACTOR,
            )
        elif enduse_option in [
            EndUseOptions.COGENERATION_TOPPING_EXTRA_HEAT,
            EndUseOptions.COGENERATION_TOPPING_EXTRA_ELECTRICITY,
            EndUseOptions.COGENERATION_BOTTOMING_EXTRA_ELECTRICITY,
            EndUseOptions.COGENERATION_BOTTOMING_EXTRA_HEAT,
            EndUseOptions.COGENERATION_PARALLEL_EXTRA_HEAT,
            EndUseOptions.COGENERATION_PARALLEL_EXTRA_ELECTRICITY,
        ]:
            capex_elec_plus_infl, capex_heat_plus_infl = _construction_inflation_cost_elec_heat()

            npvcap_elec = np.sum(capex_elec_plus_infl * crf * discount_vector)
            npvfc_elec = np.sum(capex_elec_plus_infl * econ.PTR.value * inflation_vector * discount_vector)
            npvit_elec = np.sum(
                econ.CTR.value
                / (1 - econ.CTR.value)
                * (capex_elec_plus_infl * crf - ccap_elec / model.surfaceplant.plant_lifetime.value)
                * discount_vector
            )
            npvitc_elec = capex_elec_plus_infl * econ.RITC.value / (1 - econ.CTR.value)
            npvoandm_elec = np.sum(coam_elec * inflation_vector * discount_vector)
            npvgrt_elec = econ.GTR.value / (1 - econ.GTR.value) * (
                npvcap_elec + npvoandm_elec + npvfc_elec + npvit_elec - npvitc_elec
            )
            bases[ELECTRICITY_COMMODITY] = _build_basis(
                ELECTRICITY_COMMODITY,
                npvcap_elec + npvoandm_elec + npvfc_elec + npvit_elec + npvgrt_elec - npvitc_elec,
                float(np.sum(model.surfaceplant.NetkWhProduced.value * inflation_vector * discount_vector)),
                _ELECTRICITY_PRICE_FACTOR,
            )

            npv_cap_heat = np.sum(capex_heat_plus_infl * crf * discount_vector)
            npv_fc_heat = np.sum(
                (1 + econ.inflrateconstruction.value)
                * (econ.CCap.value * (1.0 - econ.CAPEX_heat_electricity_plant_ratio.value))
                * econ.PTR.value
                * inflation_vector
                * discount_vector
            )
            npv_it_heat = np.sum(
                econ.CTR.value
                / (1 - econ.CTR.value)
                * (capex_heat_plus_infl * crf - ccap_heat / model.surfaceplant.plant_lifetime.value)
                * discount_vector
            )
            npv_itc_heat = capex_heat_plus_infl * econ.RITC.value / (1 - econ.CTR.value)
            npv_oandm_heat = np.sum(
                (econ.Coam.value * (1.0 - econ.CAPEX_heat_electricity_plant_ratio.value))
                * inflation_vector
                * discount_vector
            )
            npv_grt_heat = econ.GTR.value / (1 - econ.GTR.value) * (
                npv_cap_heat + npv_oandm_heat + npv_fc_heat + npv_it_heat - npv_itc_heat
            )
            bases[HEAT_COMMODITY] = _build_basis(
                HEAT_COMMODITY,
                npv_cap_heat + npv_oandm_heat + npv_fc_heat + npv_it_heat + npv_grt_heat - npv_itc_heat,
                float(np.sum(model.surfaceplant.HeatkWhProduced.value * inflation_vector * discount_vector)),
                _HEAT_AND_COOLING_PRICE_FACTOR,
            )

        elif enduse_option == EndUseOptions.HEAT and plant_type == PlantType.ABSORPTION_CHILLER:
            pumping_costs = model.surfaceplant.PumpingkWh.value * model.surfaceplant.electricity_cost_to_buy.value / 1e6
            npv_oandm = np.sum((econ.Coam.value + pumping_costs) * inflation_vector * discount_vector)
            npv_grt = econ.GTR.value / (1 - econ.GTR.value) * (npv_cap + npv_oandm + npv_fc + npv_it - npv_itc)
            bases[COOLING_COMMODITY] = _build_basis(
                COOLING_COMMODITY,
                npv_cap + npv_oandm + npv_fc + npv_it + npv_grt - npv_itc,
                float(np.sum(model.surfaceplant.cooling_kWh_Produced.value * inflation_vector * discount_vector)),
                _HEAT_AND_COOLING_PRICE_FACTOR,
            )

        elif enduse_option == EndUseOptions.HEAT and plant_type == PlantType.HEAT_PUMP:
            pumping_costs = model.surfaceplant.PumpingkWh.value * model.surfaceplant.electricity_cost_to_buy.value / 1e6
            heat_pump_elec_costs = (
                model.surfaceplant.heat_pump_electricity_kwh_used.value
                * model.surfaceplant.electricity_cost_to_buy.value
                / 1e6
            )
            npv_oandm = np.sum((econ.Coam.value + pumping_costs + heat_pump_elec_costs) * inflation_vector * discount_vector)
            npv_grt = econ.GTR.value / (1 - econ.GTR.value) * (npv_cap + npv_oandm + npv_fc + npv_it - npv_itc)
            bases[HEAT_COMMODITY] = _build_basis(
                HEAT_COMMODITY,
                npv_cap + npv_oandm + npv_fc + npv_it + npv_grt - npv_itc,
                float(np.sum(model.surfaceplant.HeatkWhProduced.value * inflation_vector * discount_vector)),
                _HEAT_AND_COOLING_PRICE_FACTOR,
            )

        elif enduse_option == EndUseOptions.HEAT and plant_type == PlantType.DISTRICT_HEATING:
            pumping_costs = model.surfaceplant.PumpingkWh.value * model.surfaceplant.electricity_cost_to_buy.value / 1e6
            npv_oandm = np.sum((econ.Coam.value + pumping_costs + econ.annualngcost.value) * inflation_vector * discount_vector)
            npv_grt = econ.GTR.value / (1 - econ.GTR.value) * (npv_cap + npv_oandm + npv_fc + npv_it - npv_itc)
            bases[HEAT_COMMODITY] = _build_basis(
                HEAT_COMMODITY,
                npv_cap + npv_oandm + npv_fc + npv_it + npv_grt - npv_itc,
                float(np.sum(model.surfaceplant.annual_heating_demand.value * inflation_vector * discount_vector)),
                _DISTRICT_HEATING_PRICE_FACTOR,
            )

    return bases


def calculate_levelized_cost_outputs(econ: Economics, model: Model) -> tuple[float, float, float]:
    """Calculate public ``LCOE``, ``LCOH``, and ``LCOC`` values from the shared basis helper.

    Args:
        econ: Economics object containing the active model state.
        model: Full GEOPHIRES model for the current run.

    Returns:
        A 3-tuple of ``(lcoe, lcoh, lcoc)`` in the same public units used by the current output
        parameters. Missing commodities are returned as ``0.0``.
    """
    bases = build_levelized_cost_bases(econ, model)
    return (
        bases.get(ELECTRICITY_COMMODITY, LevelizedCostBasis(ELECTRICITY_COMMODITY, 0.0, 0.0, 0.0, _ELECTRICITY_PRICE_FACTOR)).public_value,
        bases.get(HEAT_COMMODITY, LevelizedCostBasis(HEAT_COMMODITY, 0.0, 0.0, 0.0, _HEAT_AND_COOLING_PRICE_FACTOR)).public_value,
        bases.get(COOLING_COMMODITY, LevelizedCostBasis(COOLING_COMMODITY, 0.0, 0.0, 0.0, _HEAT_AND_COOLING_PRICE_FACTOR)).public_value,
    )
