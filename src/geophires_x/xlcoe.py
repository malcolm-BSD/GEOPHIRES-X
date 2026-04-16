from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from geophires_x.OptionList import EndUseOptions, EconomicModel

if TYPE_CHECKING:
    from geophires_x.Economics import Economics
    from geophires_x.Model import Model


def _discounted_energy_kwh(econ: Economics, model: Model) -> float:
    plant_lifetime = model.surfaceplant.plant_lifetime.value
    discount_vector = 1.0 / np.power(1.0 + econ.discountrate.value, np.arange(plant_lifetime))
    return float(np.sum(np.asarray(model.surfaceplant.NetkWhProduced.value) * discount_vector))


def _discount_vector(rate: float, count: int, start: int = 0) -> np.ndarray:
    return 1.0 / np.power(1.0 + rate, np.arange(start, start + count))


def _discounted_market_benefits_musd(econ: Economics, model: Model) -> float:
    plant_lifetime = model.surfaceplant.plant_lifetime.value
    discount_vector = _discount_vector(econ.discountrate.value, plant_lifetime)
    annual_net_generation_mwh = np.asarray(model.surfaceplant.NetkWhProduced.value) / 1_000.0

    carbon_benefit_usd_per_mwh = float(econ.AvoidedEmissionsIntensity.value) * float(econ.XLCOECarbonPrice.value)
    rec_benefit_usd_per_mwh = float(econ.XLCOERECPrice.value)
    total_market_benefit_usd_per_mwh = carbon_benefit_usd_per_mwh + rec_benefit_usd_per_mwh

    annual_market_benefits_musd = annual_net_generation_mwh * total_market_benefit_usd_per_mwh / 1_000_000.0
    discounted_operational_market_benefits = float(np.sum(annual_market_benefits_musd * discount_vector))
    idle_rig_discount_benefit_musd = float(econ.Cwell.value) * float(econ.IdleRigDiscountRate.value)
    return discounted_operational_market_benefits + idle_rig_discount_benefit_musd


def _discounted_social_benefits_musd(econ: Economics, model: Model) -> float:
    plant_lifetime = model.surfaceplant.plant_lifetime.value
    construction_years = model.surfaceplant.construction_years.value
    annual_net_generation_mwh = np.asarray(model.surfaceplant.NetkWhProduced.value) / 1_000.0

    annual_water_benefit_musd = (
        annual_net_generation_mwh
        * float(econ.XLCOEDisplacedWaterUseIntensity.value)
        * float(econ.XLCOEWaterShadowPrice.value)
        / 1_000_000.0
    )

    total_wells = model.wellbores.nprod.value + model.wellbores.ninj.value
    annual_construction_jobs_benefit_musd = np.zeros(construction_years)
    if construction_years > 0:
        construction_jobs_total = (
            total_wells
            * float(econ.XLCOEConstructionJobsPerRig.value)
            * float(econ.XLCOEIndirectJobsMultiplier.value)
        )
        construction_jobs_annual_usd = (
            construction_jobs_total * float(econ.XLCOEAverageMonthlyWage.value) * 12.0 / construction_years
        )
        annual_construction_jobs_benefit_musd.fill(construction_jobs_annual_usd / 1_000_000.0)

    average_generation_mw = float(np.average(np.asarray(model.surfaceplant.ElectricityProduced.value)))
    operations_jobs_annual_usd = (
        average_generation_mw
        * float(econ.XLCOEOperationsJobsPerMW.value)
        * float(econ.XLCOEIndirectJobsMultiplier.value)
        * float(econ.XLCOEAverageMonthlyWage.value)
        * 12.0
    )
    annual_operations_jobs_benefit_musd = np.full(plant_lifetime, operations_jobs_annual_usd / 1_000_000.0)

    social_rate = econ.social_discountrate.value
    discounted_construction_jobs = float(
        np.sum(annual_construction_jobs_benefit_musd * _discount_vector(social_rate, construction_years))
    )
    discounted_operational_social = float(
        np.sum(
            (annual_water_benefit_musd + annual_operations_jobs_benefit_musd)
            * _discount_vector(social_rate, plant_lifetime, start=construction_years)
        )
    )
    return discounted_construction_jobs + discounted_operational_social


def calculate_xlcoe_outputs(econ: Economics, model: Model) -> tuple[float, float]:
    """
    Calculate Phase 3 XLCOE market and social outputs.

    Market-priced benefits are discounted using the standard discount rate, while
    social-value benefits are discounted separately using the social discount rate.
    """
    if not econ.DoXLCOECalculations.value:
        return 0.0, 0.0

    if model.surfaceplant.enduse_option.value != EndUseOptions.ELECTRICITY:
        return 0.0, 0.0

    if econ.econmodel.value == EconomicModel.CLGS:
        return 0.0, 0.0

    discounted_energy_kwh = _discounted_energy_kwh(econ, model)
    if discounted_energy_kwh <= 0.0:
        return 0.0, 0.0

    baseline_discounted_cost_musd = econ.LCOE.value * discounted_energy_kwh / 1.0e8
    discounted_market_benefits_musd = _discounted_market_benefits_musd(econ, model)
    discounted_social_benefits_musd = _discounted_social_benefits_musd(econ, model)

    xlcoe_market = (baseline_discounted_cost_musd - discounted_market_benefits_musd) / discounted_energy_kwh * 1.0e8
    xlcoe_market_social = (
        baseline_discounted_cost_musd - discounted_market_benefits_musd - discounted_social_benefits_musd
    ) / discounted_energy_kwh * 1.0e8
    return xlcoe_market, xlcoe_market_social
