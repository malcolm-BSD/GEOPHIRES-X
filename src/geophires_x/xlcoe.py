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
from geophires_x.OptionList import EconomicModel, EndUseOptions

if TYPE_CHECKING:
    from geophires_x.Economics import Economics
    from geophires_x.Model import Model


ACTIVE_XLCOE_COMMODITIES = frozenset({ELECTRICITY_COMMODITY})


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
    annual_market_benefits_musd: np.ndarray
    annual_social_benefits_musd: np.ndarray
    market_discount_rate: float
    social_discount_rate: float


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
    annual_market_benefits_musd = _to_float_array(inputs.annual_market_benefits_musd)
    annual_social_benefits_musd = _to_float_array(inputs.annual_social_benefits_musd)

    if len(annual_market_benefits_musd) != len(annual_social_benefits_musd):
        raise ValueError('Extended cost streams must have the same length.')

    if inputs.basis.discounted_output <= 0.0:
        return ExtendedCostResult()

    discounted_market_benefits_musd = float(
        np.sum(
            annual_market_benefits_musd
            * _discount_vector(inputs.market_discount_rate, len(annual_market_benefits_musd))
        )
    )
    discounted_social_benefits_musd = float(
        np.sum(
            annual_social_benefits_musd
            * _discount_vector(inputs.social_discount_rate, len(annual_social_benefits_musd))
        )
    )

    market = (
        (inputs.basis.baseline_discounted_cost_musd - discounted_market_benefits_musd)
        / inputs.basis.discounted_output
        * inputs.basis.public_price_factor
    )
    market_social = (
        (inputs.basis.baseline_discounted_cost_musd - discounted_market_benefits_musd - discounted_social_benefits_musd)
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
        if commodity in ACTIVE_XLCOE_COMMODITIES and basis.discounted_output > 0.0
    }


def _electricity_market_benefit_stream_musd(econ: Economics, model: Model) -> np.ndarray:
    annual_net_generation_mwh = _to_float_array(model.surfaceplant.NetkWhProduced.value) / 1_000.0
    carbon_benefit_usd_per_mwh = float(econ.AvoidedEmissionsIntensity.value) * float(econ.XLCOECarbonPrice.value)
    rec_benefit_usd_per_mwh = float(econ.XLCOERECPrice.value)
    total_market_benefit_usd_per_mwh = carbon_benefit_usd_per_mwh + rec_benefit_usd_per_mwh
    annual_market_benefits_musd = annual_net_generation_mwh * total_market_benefit_usd_per_mwh / 1_000_000.0
    annual_market_benefits_musd[0] += float(econ.Cwell.value) * float(econ.IdleRigDiscountRate.value)
    return annual_market_benefits_musd


def _electricity_social_benefit_stream_musd(econ: Economics, model: Model) -> np.ndarray:
    plant_lifetime = model.surfaceplant.plant_lifetime.value
    construction_years = model.surfaceplant.construction_years.value
    annual_net_generation_mwh = _to_float_array(model.surfaceplant.NetkWhProduced.value) / 1_000.0

    annual_water_benefit_musd = (
        annual_net_generation_mwh
        * float(econ.XLCOEDisplacedWaterUseIntensity.value)
        * float(econ.XLCOEWaterShadowPrice.value)
        / 1_000_000.0
    )

    total_wells = model.wellbores.nprod.value + model.wellbores.ninj.value
    annual_construction_jobs_benefit_musd = np.zeros(plant_lifetime)
    if construction_years > 0:
        construction_jobs_total = (
            total_wells
            * float(econ.XLCOEConstructionJobsPerRig.value)
            * float(econ.XLCOEIndirectJobsMultiplier.value)
        )
        construction_jobs_annual_usd = (
            construction_jobs_total * float(econ.XLCOEAverageMonthlyWage.value) * 12.0 / construction_years
        )
        annual_construction_jobs_benefit_musd[:construction_years] = construction_jobs_annual_usd / 1_000_000.0

    average_generation_mw = float(np.average(_to_float_array(model.surfaceplant.ElectricityProduced.value)))
    operations_jobs_annual_usd = (
        average_generation_mw
        * float(econ.XLCOEOperationsJobsPerMW.value)
        * float(econ.XLCOEIndirectJobsMultiplier.value)
        * float(econ.XLCOEAverageMonthlyWage.value)
        * 12.0
    )
    annual_operations_jobs_benefit_musd = np.full(plant_lifetime, operations_jobs_annual_usd / 1_000_000.0)

    return annual_water_benefit_musd + annual_construction_jobs_benefit_musd + annual_operations_jobs_benefit_musd


def _build_commodity_benefit_inputs(econ: Economics, model: Model) -> dict[str, CommodityBenefitInputs]:
    active_bases = _active_xlco_commodities(econ, model)
    commodity_inputs: dict[str, CommodityBenefitInputs] = {}

    for commodity, basis in active_bases.items():
        if commodity == ELECTRICITY_COMMODITY:
            commodity_inputs[commodity] = CommodityBenefitInputs(
                basis=basis,
                annual_market_benefits_musd=_electricity_market_benefit_stream_musd(econ, model),
                annual_social_benefits_musd=_electricity_social_benefit_stream_musd(econ, model),
                market_discount_rate=econ.discountrate.value,
                social_discount_rate=econ.social_discountrate.value,
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
