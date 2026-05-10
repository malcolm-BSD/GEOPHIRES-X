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
    """Public XLCO outputs for a single commodity.

    Attributes:
        market: Extended levelized cost after applying market-only benefits.
        market_social: Extended levelized cost after applying both market and social benefits.
    """

    market: float = 0.0
    market_social: float = 0.0


@dataclass(frozen=True)
class CommodityBenefitStreams:
    """Explicit annual streams used to calculate an extended cost directly.

    Attributes:
        annual_output: Annual commodity output stream.
        annual_baseline_costs_musd: Annual baseline cost stream in million USD.
        annual_market_benefits_musd: Annual market-benefit stream in million USD.
        annual_social_benefits_musd: Annual social-benefit stream in million USD.
        market_discount_rate: Discount rate applied to the market numerator and denominator.
        social_discount_rate: Discount rate applied to social-benefit streams.
        public_price_factor: Unit scaling factor used to convert the internal discounted ratio to
            public XLCO output units.
    """

    annual_output: np.ndarray
    annual_baseline_costs_musd: np.ndarray
    annual_market_benefits_musd: np.ndarray
    annual_social_benefits_musd: np.ndarray
    market_discount_rate: float
    social_discount_rate: float
    public_price_factor: float


@dataclass(frozen=True)
class CommodityBenefitInputs:
    """Discounted XLCO inputs derived from a model-backed levelized-cost basis.

    Attributes:
        basis: Shared baseline levelized-cost basis for the commodity.
        discounted_market_benefits_musd: Discounted market benefits in million USD.
        discounted_social_benefits_musd: Discounted social benefits in million USD.
    """

    basis: LevelizedCostBasis
    discounted_market_benefits_musd: float
    discounted_social_benefits_musd: float


def _discount_vector(rate: float, count: int, start: int = 0) -> np.ndarray:
    """Return a discount vector for ``count`` periods starting at ``start``.

    Args:
        rate: Discount rate expressed as a decimal fraction.
        count: Number of periods to generate.
        start: Index offset applied to the first discounted period.

    Returns:
        A NumPy array of discount factors.
    """
    return 1.0 / np.power(1.0 + rate, np.arange(start, start + count))


def _to_float_array(values) -> np.ndarray:
    """Convert an arbitrary sequence-like input to a float NumPy array.

    Args:
        values: Sequence or array-like values to normalize.

    Returns:
        A NumPy array with ``dtype=float``.
    """
    return np.asarray(values, dtype=float)


def calculate_extended_cost_from_explicit_streams(streams: CommodityBenefitStreams) -> ExtendedCostResult:
    """Calculate XLCO results from explicit annual streams.

    Args:
        streams: Fully specified annual output, baseline-cost, and benefit streams.

    Returns:
        An :class:`ExtendedCostResult` containing market-only and market-plus-social XLCO values.

    Raises:
        ValueError: If the provided annual streams do not have matching lengths.
    """
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
    """Calculate explicit-stream XLCO results for multiple commodities.

    Args:
        commodity_streams: Mapping of commodity name to explicit benefit streams.

    Returns:
        A mapping of commodity name to :class:`ExtendedCostResult`.
    """
    return {
        commodity: calculate_extended_cost_from_explicit_streams(streams)
        for commodity, streams in commodity_streams.items()
    }


def calculate_extended_cost_from_basis(inputs: CommodityBenefitInputs) -> ExtendedCostResult:
    """Calculate XLCO results from a discounted baseline basis plus discounted benefits.

    Args:
        inputs: Discounted baseline basis and benefit totals for a single commodity.

    Returns:
        An :class:`ExtendedCostResult`. If the discounted output denominator is zero or negative,
        both results are returned as ``0.0``.
    """
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
    """Calculate ``LCOE``, ``XLCOE_Market``, and ``XLCOE_MarketSocial`` from explicit streams.

    Args:
        annual_baseline_costs_musd: Annual baseline cost stream in million USD.
        annual_net_generation_kwh: Annual net electricity generation stream in kWh.
        annual_market_benefits_musd: Annual market-benefit stream in million USD.
        annual_social_benefits_musd: Annual social-benefit stream in million USD.
        market_discount_rate: Market discount rate.
        social_discount_rate: Social discount rate.

    Returns:
        A tuple of ``(lcoe, xlcoe_market, xlcoe_market_social)`` in ``cents/kWh``.
    """
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
    """Return the active commodity bases eligible for XLCO calculations.

    Args:
        econ: Economics object containing the current XLCO switch state.
        model: Full GEOPHIRES model for the current run.

    Returns:
        A dictionary of active commodity bases with positive discounted output denominators.
    """
    if not econ.DoXLCOCalculations.value:
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
    """Discount an output-linked operational benefit stream to a single present-value total.

    Args:
        annual_output_kwh: Annual commodity output stream in kWh.
        annual_benefit_usd_per_mwh: Benefit value per unit of annual output in USD/MWh.
        discount_rate: Discount rate applied to the stream.
        start: Period offset for the first discounted year.

    Returns:
        Discounted benefit total in million USD.
    """
    annual_output_mwh = _to_float_array(annual_output_kwh) / 1_000.0
    annual_benefits_musd = annual_output_mwh * annual_benefit_usd_per_mwh / 1_000_000.0
    return float(np.sum(annual_benefits_musd * _discount_vector(discount_rate, len(annual_benefits_musd), start=start)))


def _discounted_constant_annual_benefit_musd(
    annual_benefit_musd: float,
    periods: int,
    discount_rate: float,
    start: int = 0,
) -> float:
    """Discount a constant annual benefit to present value.

    Args:
        annual_benefit_musd: Constant annual benefit in million USD.
        periods: Number of years to apply the benefit.
        discount_rate: Discount rate applied to the stream.
        start: Period offset for the first discounted year.

    Returns:
        Discounted benefit total in million USD.
    """
    if periods <= 0 or annual_benefit_musd == 0.0:
        return 0.0
    return float(np.sum(np.full(periods, annual_benefit_musd) * _discount_vector(discount_rate, periods, start=start)))


def _shared_market_benefit_musd(econ: Economics) -> float:
    """Return the shared idle-rig market benefit in million USD.

    Args:
        econ: Economics object containing drilling CAPEX and idle-rig discount settings.

    Returns:
        Shared market benefit in million USD before commodity allocation.
    """
    return float(econ.Cwell.value) * float(econ.IdleRigDiscountRate.value)


def _shared_social_benefit_musd(econ: Economics, model: Model) -> float:
    """Return the shared construction-jobs social benefit in million USD.

    Args:
        econ: Economics object containing jobs and wage assumptions.
        model: Full GEOPHIRES model providing well count and construction duration.

    Returns:
        Discounted shared construction-jobs benefit in million USD.
    """
    construction_years = model.surfaceplant.construction_years.value
    total_wells = model.wellbores.nprod.value + model.wellbores.ninj.value
    if construction_years <= 0:
        return 0.0

    construction_jobs_total = (
        total_wells
        * float(econ.XLCOConstructionJobsPerRig.value)
        * float(econ.XLCOIndirectJobsMultiplier.value)
    )
    construction_jobs_annual_usd = (
        construction_jobs_total * float(econ.XLCOAverageMonthlyWage.value) * 12.0 / construction_years
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
    """Discount operations-jobs benefits tied to average commodity output.

    Args:
        average_output_mw: Average commodity output in MW-equivalent.
        jobs_per_mw: Jobs-per-MW coefficient for the active commodity.
        econ: Economics object containing multiplier and wage assumptions.
        model: Full GEOPHIRES model containing plant lifetime and construction period.

    Returns:
        Discounted operations-jobs benefit in million USD.
    """
    annual_jobs_usd = (
        average_output_mw
        * jobs_per_mw
        * float(econ.XLCOIndirectJobsMultiplier.value)
        * float(econ.XLCOAverageMonthlyWage.value)
        * 12.0
    )
    return _discounted_constant_annual_benefit_musd(
        annual_jobs_usd / 1_000_000.0,
        model.surfaceplant.plant_lifetime.value,
        econ.social_discountrate.value,
        start=model.surfaceplant.construction_years.value,
    )


def _electricity_direct_market_benefit_musd(econ: Economics, model: Model) -> float:
    """Calculate discounted electricity-market benefits in million USD."""
    annual_benefit_usd_per_mwh = (
        float(econ.XLCOEAvoidedEmissionsIntensity.value) * float(econ.XLCOCarbonPrice.value)
        + float(econ.XLCOERECPrice.value)
    )
    return _discounted_operational_benefit_musd(
        model.surfaceplant.NetkWhProduced.value,
        annual_benefit_usd_per_mwh,
        econ.discountrate.value,
    )


def _heat_direct_market_benefit_musd(econ: Economics, model: Model) -> float:
    """Calculate discounted heat-market benefits in million USD."""
    annual_benefit_usd_per_mwh = (
        float(econ.XLCOHAvoidedEmissionsIntensity.value) * float(econ.XLCOCarbonPrice.value)
        + float(econ.XLCOHREC.value)
    )
    return _discounted_operational_benefit_musd(
        model.surfaceplant.HeatkWhProduced.value,
        annual_benefit_usd_per_mwh,
        econ.discountrate.value,
    )


def _cooling_direct_market_benefit_musd(econ: Economics, model: Model) -> float:
    """Calculate discounted cooling-market benefits in million USD."""
    annual_benefit_usd_per_mwh = (
        float(econ.XLCOCAvoidedEmissionsIntensity.value) * float(econ.XLCOCarbonPrice.value)
        + float(econ.XLCOCREC.value)
    )
    return _discounted_operational_benefit_musd(
        model.surfaceplant.cooling_kWh_Produced.value,
        annual_benefit_usd_per_mwh,
        econ.discountrate.value,
    )


def _electricity_direct_social_benefit_musd(econ: Economics, model: Model) -> float:
    """Calculate discounted electricity-social benefits in million USD."""
    water_benefit = _discounted_operational_benefit_musd(
        model.surfaceplant.NetkWhProduced.value,
        float(econ.XLCOEDisplacedWaterUseIntensity.value) * float(econ.XLCOWaterShadowPrice.value),
        econ.social_discountrate.value,
        start=model.surfaceplant.construction_years.value,
    )
    average_generation_mw = float(np.average(_to_float_array(model.surfaceplant.ElectricityProduced.value)))
    jobs_benefit = _discounted_social_jobs_benefit_musd(
        average_generation_mw,
        float(econ.XLCOOperationsJobsPerMW.value),
        econ,
        model,
    )
    return water_benefit + jobs_benefit


def _heat_direct_social_benefit_musd(econ: Economics, model: Model) -> float:
    """Calculate discounted heat-social benefits in million USD."""
    water_benefit = _discounted_operational_benefit_musd(
        model.surfaceplant.HeatkWhProduced.value,
        float(econ.XLCOHDisplacedWaterUseIntensity.value) * float(econ.XLCOWaterShadowPrice.value),
        econ.social_discountrate.value,
        start=model.surfaceplant.construction_years.value,
    )
    average_heat_mw = float(np.average(_to_float_array(model.surfaceplant.HeatProduced.value)))
    jobs_benefit = _discounted_social_jobs_benefit_musd(
        average_heat_mw,
        float(econ.XLCOOperationsJobsPerMW.value),
        econ,
        model,
    )
    return water_benefit + jobs_benefit


def _cooling_direct_social_benefit_musd(econ: Economics, model: Model) -> float:
    """Calculate discounted cooling-social benefits in million USD."""
    water_benefit = _discounted_operational_benefit_musd(
        model.surfaceplant.cooling_kWh_Produced.value,
        float(econ.XLCOCDisplacedWaterUseIntensity.value) * float(econ.XLCOWaterShadowPrice.value),
        econ.social_discountrate.value,
        start=model.surfaceplant.construction_years.value,
    )
    average_cooling_mw = float(np.average(_to_float_array(model.surfaceplant.cooling_produced.value)))
    jobs_benefit = _discounted_social_jobs_benefit_musd(
        average_cooling_mw,
        float(econ.XLCOOperationsJobsPerMW.value),
        econ,
        model,
    )
    return water_benefit + jobs_benefit


def _build_commodity_benefit_inputs(econ: Economics, model: Model) -> dict[str, CommodityBenefitInputs]:
    """Build discounted XLCO inputs for every active commodity.

    Args:
        econ: Economics object containing XLCO settings and project costs.
        model: Full GEOPHIRES model with active commodity outputs.

    Returns:
        Per-commodity discounted inputs ready for XLCO calculation.
    """
    active_bases = _active_xlco_commodities(econ, model)
    total_baseline_discounted_cost = float(
        sum(basis.baseline_discounted_cost_musd for basis in active_bases.values())
    )
    shared_market_benefit_musd = _shared_market_benefit_musd(econ)
    shared_social_benefit_musd = _shared_social_benefit_musd(econ, model)
    commodity_inputs: dict[str, CommodityBenefitInputs] = {}

    for commodity, basis in active_bases.items():
        # Shared market and social benefits are allocated by baseline discounted-cost share so that
        # mixed-output projects receive a stable, economically consistent split across commodities.
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
    """Calculate XLCO results for all active commodities.

    Args:
        econ: Economics object containing XLCO settings and baseline economics.
        model: Full GEOPHIRES model for the current run.

    Returns:
        Mapping of commodity name to :class:`ExtendedCostResult`.
    """
    return {
        commodity: calculate_extended_cost_from_basis(inputs)
        for commodity, inputs in _build_commodity_benefit_inputs(econ, model).items()
    }


def assign_extended_levelized_cost_outputs(econ: Economics, model: Model) -> dict[str, ExtendedCostResult]:
    """Calculate XLCO values and write them back to the economics output parameters.

    Args:
        econ: Economics object whose ``XLCO*`` output parameters will be updated.
        model: Full GEOPHIRES model for the current run.

    Returns:
        The same per-commodity XLCO mapping that was written back to ``econ``.
    """
    extended_costs = calculate_extended_levelized_costs(econ, model)
    electricity_costs = extended_costs.get(ELECTRICITY_COMMODITY)
    heat_costs = extended_costs.get(HEAT_COMMODITY)
    cooling_costs = extended_costs.get(COOLING_COMMODITY)

    econ.XLCOE_Market.value = electricity_costs.market if electricity_costs is not None else 0.0
    econ.XLCOE_MarketSocial.value = electricity_costs.market_social if electricity_costs is not None else 0.0
    econ.XLCOH_Market.value = heat_costs.market if heat_costs is not None else 0.0
    econ.XLCOH_MarketSocial.value = heat_costs.market_social if heat_costs is not None else 0.0
    econ.XLCOC_Market.value = cooling_costs.market if cooling_costs is not None else 0.0
    econ.XLCOC_MarketSocial.value = cooling_costs.market_social if cooling_costs is not None else 0.0
    return extended_costs


def calculate_xlcoe_outputs(econ: Economics, model: Model) -> tuple[float, float]:
    """Calculate electricity-only XLCO outputs from the generalized engine.

    Args:
        econ: Economics object containing XLCO settings.
        model: Full GEOPHIRES model for the current run.

    Returns:
        Tuple of ``(xlcoe_market, xlcoe_market_social)``. If electricity is inactive, both values
        are returned as ``0.0``.
    """
    electricity_result = calculate_extended_levelized_costs(econ, model).get(ELECTRICITY_COMMODITY)
    if electricity_result is None:
        return 0.0, 0.0
    return electricity_result.market, electricity_result.market_social


def calculate_xlcoh_outputs(econ: Economics, model: Model) -> tuple[float, float]:
    """Calculate heat-only XLCO outputs from the generalized engine.

    Args:
        econ: Economics object containing XLCO settings.
        model: Full GEOPHIRES model for the current run.

    Returns:
        Tuple of ``(xlcoh_market, xlcoh_market_social)``. If heat is inactive, both values are
        returned as ``0.0``.
    """
    heat_result = calculate_extended_levelized_costs(econ, model).get(HEAT_COMMODITY)
    if heat_result is None:
        return 0.0, 0.0
    return heat_result.market, heat_result.market_social


def calculate_xlcoc_outputs(econ: Economics, model: Model) -> tuple[float, float]:
    """Calculate cooling-only XLCO outputs from the generalized engine.

    Args:
        econ: Economics object containing XLCO settings.
        model: Full GEOPHIRES model for the current run.

    Returns:
        Tuple of ``(xlcoc_market, xlcoc_market_social)``. If cooling is inactive, both values are
        returned as ``0.0``.
    """
    cooling_result = calculate_extended_levelized_costs(econ, model).get(COOLING_COMMODITY)
    if cooling_result is None:
        return 0.0, 0.0
    return cooling_result.market, cooling_result.market_social
