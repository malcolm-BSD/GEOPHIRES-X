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


def _discounted_market_benefits_musd(econ: Economics, model: Model) -> float:
    plant_lifetime = model.surfaceplant.plant_lifetime.value
    discount_vector = 1.0 / np.power(1.0 + econ.discountrate.value, np.arange(plant_lifetime))
    annual_net_generation_mwh = np.asarray(model.surfaceplant.NetkWhProduced.value) / 1_000.0

    carbon_benefit_usd_per_mwh = float(econ.AvoidedEmissionsIntensity.value) * float(econ.XLCOECarbonPrice.value)
    rec_benefit_usd_per_mwh = float(econ.XLCOERECPrice.value)
    total_market_benefit_usd_per_mwh = carbon_benefit_usd_per_mwh + rec_benefit_usd_per_mwh

    annual_market_benefits_musd = annual_net_generation_mwh * total_market_benefit_usd_per_mwh / 1_000_000.0
    return float(np.sum(annual_market_benefits_musd * discount_vector))


def calculate_xlcoe_outputs(econ: Economics, model: Model) -> tuple[float, float]:
    """
    Calculate Phase 2 XLCOE market outputs.

    This phase implements the market-priced electricity modifiers first. Until
    social-value benefit streams are added, `XLCOE_MarketSocial` remains equal to
    `XLCOE_Market`.
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

    xlcoe_market = (baseline_discounted_cost_musd - discounted_market_benefits_musd) / discounted_energy_kwh * 1.0e8
    xlcoe_market_social = xlcoe_market
    return xlcoe_market, xlcoe_market_social
