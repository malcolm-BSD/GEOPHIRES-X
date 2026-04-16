from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from geophires_x.levelized_costs import (
    COOLING_COMMODITY,
    ELECTRICITY_COMMODITY,
    HEAT_COMMODITY,
    LevelizedCostBasis,
    build_levelized_cost_bases,
)
from geophires_x.OptionList import EconomicModel

if TYPE_CHECKING:
    from geophires_x.Economics import Economics
    from geophires_x.Model import Model

@dataclass(frozen=True)
class ExtendedCostResult:
    market: float = 0.0
    market_social: float = 0.0


@dataclass(frozen=True)
class CommodityBenefitStreams:
    annual_output: np.ndarray
    annual_baseline_costs_musd: np.ndarray
    annual_market_benefits_musd: np.ndarray
    annual_social_benefits_musd: np.ndarray
    market_discount_rate: float
    social_discount_rate: float
    public_price_factor: float


@dataclass(frozen=True)
class CommodityBenefitInputs:
    basis: LevelizedCostBasis
    discounted_market_benefits_musd: float
    discounted_social_benefits_musd: float


def _discount_vector(rate: float, count: int, start: int = 0) -> np.ndarray:
    return 1.0 / np.power(1.0 + rate, np.arange(start, start + count))


def _to_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def calculate_extended_cost_from_explicit_streams(streams: CommodityBenefitStreams) -> ExtendedCostResult:
    annual_output = _to_float_array(streams.annual_output)
    annual_baseline_costs_musd = _to_float_array(streams.annual_baseline_costs_musd)
    annual_market_benefits_musd = _to_float_array(streams.annual_market_benefits_musd)
    annual_social_benefits_musd = _to_float_array(streams.annual_social_benefits_musd)

    year_count = len(annual_output)
    if not (
        len(annual_baseline_costs_musd)
        == len(annual_market_benefits_musd)
        == len(annual_social_benefits_musd)
        == year_count
    ):
        raise ValueError('Extended cost streams must have the same length.')

    discounted_output = float(np.sum(annual_output * _discount_vector(streams.market_discount_rate, year_count)))
    if discounted_output <= 0.0:
        return ExtendedCostResult()

    discounted_baseline_cost_musd = float(
        np.sum(annual_baseline_costs_musd * _discount_vector(streams.market_discount_rate, year_count))
    )
    discounted_market_benefits_musd = float(
        np.sum(annual_market_benefits_musd * _discount_vector(streams.market_discount_rate, year_count))
    )
    discounted_social_benefits_musd = float(
        np.sum(annual_social_benefits_musd * _discount_vector(streams.social_discount_rate, year_count))
    )

    market = (
        (discounted_baseline_cost_musd - discounted_market_benefits_musd)
        / discounted_output
        * streams.public_price_factor
    )
    market_social = (
        (discounted_baseline_cost_musd - discounted_market_benefits_musd - discounted_social_benefits_musd)
        / discounted_output
        * streams.public_price_factor
    )
    return ExtendedCostResult(market=market, market_social=market_social)


def calculate_extended_costs_from_explicit_streams(
    commodity_streams: dict[str, CommodityBenefitStreams]
) -> dict[str, ExtendedCostResult]:
    return {
        commodity: calculate_extended_cost_from_explicit_streams(streams)
        for commodity, streams in commodity_streams.items()
    }


def calculate_extended_cost_from_basis(inputs: CommodityBenefitInputs) -> ExtendedCostResult:
    if inputs.basis.discounted_output <= 0.0:
        return ExtendedCostResult()

    market = (
        (inputs.basis.baseline_discounted_cost_musd - inputs.discounted_market_benefits_musd)
        / inputs.basis.discounted_output
        * inputs.basis.public_price_factor
    )
    market_social = (
        (
            inputs.basis.baseline_discounted_cost_musd
            - inputs.discounted_market_benefits_musd
            - inputs.discounted_social_benefits_musd
        )
        / inputs.basis.discounted_output
        * inputs.basis.public_price_factor
    )
    return ExtendedCostResult(market=market, market_social=market_social)


def calculate_xlcoe_from_explicit_streams(
    annual_baseline_costs_musd: np.ndarray,
    annual_net_generation_kwh: np.ndarray,
    annual_market_benefits_musd: np.ndarray,
    annual_social_benefits_musd: np.ndarray,
    market_discount_rate: float,
    social_discount_rate: float,
) -> tuple[float, float, float]:
    streams = CommodityBenefitStreams(
        annual_output=_to_float_array(annual_net_generation_kwh),
        annual_baseline_costs_musd=_to_float_array(annual_baseline_costs_musd),
        annual_market_benefits_musd=_to_float_array(annual_market_benefits_musd),
        annual_social_benefits_musd=_to_float_array(annual_social_benefits_musd),
        market_discount_rate=market_discount_rate,
        social_discount_rate=social_discount_rate,
        public_price_factor=1.0e8,
    )
    result = calculate_extended_cost_from_explicit_streams(streams)
    discounted_output = float(
        np.sum(streams.annual_output * _discount_vector(streams.market_discount_rate, len(streams.annual_output)))
    )
    if discounted_output <= 0.0:
        return 0.0, 0.0, 0.0

    discounted_baseline_cost_musd = float(
        np.sum(
            streams.annual_baseline_costs_musd
            * _discount_vector(streams.market_discount_rate, len(streams.annual_output))
        )
    )
    lcoe = discounted_baseline_cost_musd / discounted_output * streams.public_price_factor
    return lcoe, result.market, result.market_social


def _active_xlco_commodities(econ: Economics, model: Model) -> dict[str, LevelizedCostBasis]:
    if not econ.DoXLCOECalculations.value:
        return {}

    if econ.econmodel.value == EconomicModel.CLGS:
        return {}

    bases = build_levelized_cost_bases(econ, model)
    return {
        commodity: basis
        for commodity, basis in bases.items()
        if commodity in {ELECTRICITY_COMMODITY, HEAT_COMMODITY, COOLING_COMMODITY} and basis.discounted_output > 0.0
    }


def _discounted_operational_benefit_musd(
    annual_output_kwh,
    annual_benefit_usd_per_mwh: float,
    discount_rate: float,
    start: int = 0,
) -> float:
    annual_output_mwh = _to_float_array(annual_output_kwh) / 1_000.0
    annual_benefits_musd = annual_output_mwh * annual_benefit_usd_per_mwh / 1_000_000.0
    return float(np.sum(annual_benefits_musd * _discount_vector(discount_rate, len(annual_benefits_musd), start=start)))


def _discounted_constant_annual_benefit_musd(
    annual_benefit_musd: float,
    periods: int,
    discount_rate: float,
    start: int = 0,
) -> float:
    if periods <= 0 or annual_benefit_musd == 0.0:
        return 0.0
    return float(np.sum(np.full(periods, annual_benefit_musd) * _discount_vector(discount_rate, periods, start=start)))


def _shared_market_benefit_musd(econ: Economics) -> float:
    return float(econ.Cwell.value) * float(econ.IdleRigDiscountRate.value)


def _shared_social_benefit_musd(econ: Economics, model: Model) -> float:
    construction_years = model.surfaceplant.construction_years.value
    total_wells = model.wellbores.nprod.value + model.wellbores.ninj.value
    if construction_years <= 0:
        return 0.0

    construction_jobs_total = (
        total_wells
        * float(econ.XLCOEConstructionJobsPerRig.value)
        * float(econ.XLCOEIndirectJobsMultiplier.value)
    )
    construction_jobs_annual_usd = (
        construction_jobs_total * float(econ.XLCOEAverageMonthlyWage.value) * 12.0 / construction_years
    )
    return _discounted_constant_annual_benefit_musd(
        construction_jobs_annual_usd / 1_000_000.0,
        construction_years,
        econ.social_discountrate.value,
    )


def _discounted_social_jobs_benefit_musd(
    average_output_mw: float,
    jobs_per_mw: float,
    econ: Economics,
    model: Model,
) -> float:
    annual_jobs_usd = (
        average_output_mw
        * jobs_per_mw
        * float(econ.XLCOEIndirectJobsMultiplier.value)
        * float(econ.XLCOEAverageMonthlyWage.value)
        * 12.0
    )
    return _discounted_constant_annual_benefit_musd(
        annual_jobs_usd / 1_000_000.0,
        model.surfaceplant.plant_lifetime.value,
        econ.social_discountrate.value,
        start=model.surfaceplant.construction_years.value,
    )


def _electricity_direct_market_benefit_musd(econ: Economics, model: Model) -> float:
    annual_benefit_usd_per_mwh = (
        float(econ.AvoidedEmissionsIntensity.value) * float(econ.XLCOECarbonPrice.value)
        + float(econ.XLCOERECPrice.value)
    )
    return _discounted_operational_benefit_musd(
        model.surfaceplant.NetkWhProduced.value,
        annual_benefit_usd_per_mwh,
        econ.discountrate.value,
    )


def _heat_direct_market_benefit_musd(econ: Economics, model: Model) -> float:
    annual_benefit_usd_per_mwh = (
        float(econ.XLCOHAvoidedEmissionsIntensity.value) * float(econ.XLCOHCarbonPrice.value)
        + float(econ.XLCOHThermalCreditPrice.value)
    )
    return _discounted_operational_benefit_musd(
        model.surfaceplant.HeatkWhProduced.value,
        annual_benefit_usd_per_mwh,
        econ.discountrate.value,
    )


def _cooling_direct_market_benefit_musd(econ: Economics, model: Model) -> float:
    annual_benefit_usd_per_mwh = (
        float(econ.XLCOCAvoidedEmissionsIntensity.value) * float(econ.XLCOCCarbonPrice.value)
        + float(econ.XLCOCCoolingCreditPrice.value)
    )
    return _discounted_operational_benefit_musd(
        model.surfaceplant.cooling_kWh_Produced.value,
        annual_benefit_usd_per_mwh,
        econ.discountrate.value,
    )


def _electricity_direct_social_benefit_musd(econ: Economics, model: Model) -> float:
    plant_lifetime = model.surfaceplant.plant_lifetime.value
    water_benefit = _discounted_operational_benefit_musd(
        model.surfaceplant.NetkWhProduced.value,
        float(econ.XLCOEDisplacedWaterUseIntensity.value) * float(econ.XLCOEWaterShadowPrice.value),
        econ.social_discountrate.value,
        start=model.surfaceplant.construction_years.value,
    )
    average_generation_mw = float(np.average(_to_float_array(model.surfaceplant.ElectricityProduced.value)))
    jobs_benefit = _discounted_social_jobs_benefit_musd(
        average_generation_mw,
        float(econ.XLCOEOperationsJobsPerMW.value),
        econ,
        model,
    )
    return water_benefit + jobs_benefit


def _heat_direct_social_benefit_musd(econ: Economics, model: Model) -> float:
    water_benefit = _discounted_operational_benefit_musd(
        model.surfaceplant.HeatkWhProduced.value,
        float(econ.XLCOHDisplacedWaterUseIntensity.value) * float(econ.XLCOHWaterShadowPrice.value),
        econ.social_discountrate.value,
        start=model.surfaceplant.construction_years.value,
    )
    average_heat_mw = float(np.average(_to_float_array(model.surfaceplant.HeatProduced.value)))
    jobs_benefit = _discounted_social_jobs_benefit_musd(
        average_heat_mw,
        float(econ.XLCOHOperationsJobsPerMW.value),
        econ,
        model,
    )
    return water_benefit + jobs_benefit


def _cooling_direct_social_benefit_musd(econ: Economics, model: Model) -> float:
    water_benefit = _discounted_operational_benefit_musd(
        model.surfaceplant.cooling_kWh_Produced.value,
        float(econ.XLCOCDisplacedWaterUseIntensity.value) * float(econ.XLCOCWaterShadowPrice.value),
        econ.social_discountrate.value,
        start=model.surfaceplant.construction_years.value,
    )
    average_cooling_mw = float(np.average(_to_float_array(model.surfaceplant.cooling_produced.value)))
    jobs_benefit = _discounted_social_jobs_benefit_musd(
        average_cooling_mw,
        float(econ.XLCOCOperationsJobsPerMW.value),
        econ,
        model,
    )
    return water_benefit + jobs_benefit


def _build_commodity_benefit_inputs(econ: Economics, model: Model) -> dict[str, CommodityBenefitInputs]:
    active_bases = _active_xlco_commodities(econ, model)
    total_baseline_discounted_cost = float(
        sum(basis.baseline_discounted_cost_musd for basis in active_bases.values())
    )
    shared_market_benefit_musd = _shared_market_benefit_musd(econ)
    shared_social_benefit_musd = _shared_social_benefit_musd(econ, model)
    commodity_inputs: dict[str, CommodityBenefitInputs] = {}

    for commodity, basis in active_bases.items():
        cost_share = 0.0
        if total_baseline_discounted_cost > 0.0:
            cost_share = basis.baseline_discounted_cost_musd / total_baseline_discounted_cost

        if commodity == ELECTRICITY_COMMODITY:
            commodity_inputs[commodity] = CommodityBenefitInputs(
                basis=basis,
                discounted_market_benefits_musd=_electricity_direct_market_benefit_musd(econ, model)
                + cost_share * shared_market_benefit_musd,
                discounted_social_benefits_musd=_electricity_direct_social_benefit_musd(econ, model)
                + cost_share * shared_social_benefit_musd,
            )
        elif commodity == HEAT_COMMODITY:
            commodity_inputs[commodity] = CommodityBenefitInputs(
                basis=basis,
                discounted_market_benefits_musd=_heat_direct_market_benefit_musd(econ, model)
                + cost_share * shared_market_benefit_musd,
                discounted_social_benefits_musd=_heat_direct_social_benefit_musd(econ, model)
                + cost_share * shared_social_benefit_musd,
            )
        elif commodity == COOLING_COMMODITY:
            commodity_inputs[commodity] = CommodityBenefitInputs(
                basis=basis,
                discounted_market_benefits_musd=_cooling_direct_market_benefit_musd(econ, model)
                + cost_share * shared_market_benefit_musd,
                discounted_social_benefits_musd=_cooling_direct_social_benefit_musd(econ, model)
                + cost_share * shared_social_benefit_musd,
            )

    return commodity_inputs


def calculate_extended_levelized_costs(econ: Economics, model: Model) -> dict[str, ExtendedCostResult]:
    return {
        commodity: calculate_extended_cost_from_basis(inputs)
        for commodity, inputs in _build_commodity_benefit_inputs(econ, model).items()
    }


def calculate_xlcoe_outputs(econ: Economics, model: Model) -> tuple[float, float]:
    """
    Calculate current public XLCOE outputs via the generalized commodity-aware engine.
    """
    electricity_result = calculate_extended_levelized_costs(econ, model).get(ELECTRICITY_COMMODITY)
    if electricity_result is None:
        return 0.0, 0.0
    return electricity_result.market, electricity_result.market_social


def calculate_xlcoh_outputs(econ: Economics, model: Model) -> tuple[float, float]:
    heat_result = calculate_extended_levelized_costs(econ, model).get(HEAT_COMMODITY)
    if heat_result is None:
        return 0.0, 0.0
    return heat_result.market, heat_result.market_social


def calculate_xlcoc_outputs(econ: Economics, model: Model) -> tuple[float, float]:
    cooling_result = calculate_extended_levelized_costs(econ, model).get(COOLING_COMMODITY)
    if cooling_result is None:
        return 0.0, 0.0
    return cooling_result.market, cooling_result.market_social
