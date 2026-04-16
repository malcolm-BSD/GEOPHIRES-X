from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from geophires_x.Model import Model
from geophires_x.levelized_costs import ELECTRICITY_COMMODITY, HEAT_COMMODITY, build_levelized_cost_bases
from geophires_x.Units import CO2ProductionUnit, CostPerMassUnit, CurrencyUnit, EnergyCostUnit
from geophires_x.xlcoe import (
    CommodityBenefitStreams,
    calculate_extended_cost_from_explicit_streams,
    calculate_extended_costs_from_explicit_streams,
    calculate_extended_levelized_costs,
    calculate_xlcoe_from_explicit_streams,
)
from geophires_x_client import GeophiresInputParameters
from tests.base_test_case import BaseTestCase


class XLCOETestCase(BaseTestCase):

    def test_generalized_extended_cost_helper_rejects_mismatched_stream_lengths(self):
        with self.assertRaises(ValueError):
            calculate_extended_cost_from_explicit_streams(
                CommodityBenefitStreams(
                    annual_output=[1.0, 1.0],
                    annual_baseline_costs_musd=[1.0],
                    annual_market_benefits_musd=[0.0, 0.0],
                    annual_social_benefits_musd=[0.0, 0.0],
                    market_discount_rate=0.07,
                    social_discount_rate=0.07,
                    public_price_factor=1.0e8,
                )
            )

    def test_generalized_extended_cost_helper_supports_multiple_commodities(self):
        results = calculate_extended_costs_from_explicit_streams(
            {
                'electricity': CommodityBenefitStreams(
                    annual_output=[100.0, 100.0],
                    annual_baseline_costs_musd=[1.0, 1.0],
                    annual_market_benefits_musd=[0.1, 0.1],
                    annual_social_benefits_musd=[0.05, 0.05],
                    market_discount_rate=0.0,
                    social_discount_rate=0.0,
                    public_price_factor=100.0,
                ),
                'heat': CommodityBenefitStreams(
                    annual_output=[50.0, 50.0],
                    annual_baseline_costs_musd=[0.5, 0.5],
                    annual_market_benefits_musd=[0.0, 0.0],
                    annual_social_benefits_musd=[0.1, 0.1],
                    market_discount_rate=0.0,
                    social_discount_rate=0.0,
                    public_price_factor=100.0,
                ),
            }
        )

        self.assertAlmostEqual(0.9, results['electricity'].market)
        self.assertAlmostEqual(0.85, results['electricity'].market_social)
        self.assertAlmostEqual(1.0, results['heat'].market)
        self.assertAlmostEqual(0.8, results['heat'].market_social)

    def test_xlcoe_parameters_and_outputs_are_defined(self):
        model = self._new_model()

        self.assertIn('Do XLCOE Calculations', model.economics.ParameterDict)
        self.assertIn('Social Discount Rate', model.economics.ParameterDict)
        self.assertIn('Avoided Emissions Intensity', model.economics.ParameterDict)
        self.assertIn('XLCOE Carbon Price', model.economics.ParameterDict)
        self.assertIn('XLCOE REC Price', model.economics.ParameterDict)
        self.assertIn('XLCOE Displaced Water Use Intensity', model.economics.ParameterDict)
        self.assertIn('XLCOE Water Shadow Price', model.economics.ParameterDict)
        self.assertIn('Idle Rig Discount Rate', model.economics.ParameterDict)
        self.assertIn('XLCOE Construction Jobs Per Rig', model.economics.ParameterDict)
        self.assertIn('XLCOE Operations Jobs Per MW', model.economics.ParameterDict)
        self.assertIn('XLCOE Indirect Jobs Multiplier', model.economics.ParameterDict)
        self.assertIn('XLCOE Average Monthly Wage', model.economics.ParameterDict)
        self.assertIn('XLCOH Avoided Emissions Intensity', model.economics.ParameterDict)
        self.assertIn('XLCOH Carbon Price', model.economics.ParameterDict)
        self.assertIn('XLCOH Thermal Credit Price', model.economics.ParameterDict)
        self.assertIn('XLCOH Displaced Water Use Intensity', model.economics.ParameterDict)
        self.assertIn('XLCOH Water Shadow Price', model.economics.ParameterDict)
        self.assertIn('XLCOH Operations Jobs Per MW', model.economics.ParameterDict)
        self.assertIn('XLCOC Avoided Emissions Intensity', model.economics.ParameterDict)
        self.assertIn('XLCOC Carbon Price', model.economics.ParameterDict)
        self.assertIn('XLCOC Cooling Credit Price', model.economics.ParameterDict)
        self.assertIn('XLCOC Displaced Water Use Intensity', model.economics.ParameterDict)
        self.assertIn('XLCOC Water Shadow Price', model.economics.ParameterDict)
        self.assertIn('XLCOC Operations Jobs Per MW', model.economics.ParameterDict)
        self.assertIn('XLCOE_Market', model.economics.OutputParameterDict)
        self.assertIn('XLCOE_MarketSocial', model.economics.OutputParameterDict)
        self.assertIn('XLCOH_Market', model.economics.OutputParameterDict)
        self.assertIn('XLCOH_MarketSocial', model.economics.OutputParameterDict)
        self.assertIn('XLCOC_Market', model.economics.OutputParameterDict)
        self.assertIn('XLCOC_MarketSocial', model.economics.OutputParameterDict)

        self.assertFalse(model.economics.DoXLCOECalculations.value)
        self.assertEqual(0.07, model.economics.social_discountrate.value)
        self.assertEqual(0.0, model.economics.AvoidedEmissionsIntensity.value)
        self.assertEqual(0.0, model.economics.XLCOECarbonPrice.value)
        self.assertEqual(0.0, model.economics.XLCOERECPrice.value)
        self.assertEqual(0.0, model.economics.XLCOEDisplacedWaterUseIntensity.value)
        self.assertEqual(0.0, model.economics.XLCOEWaterShadowPrice.value)
        self.assertEqual(0.0, model.economics.IdleRigDiscountRate.value)
        self.assertEqual(0.0, model.economics.XLCOEConstructionJobsPerRig.value)
        self.assertEqual(0.0, model.economics.XLCOEOperationsJobsPerMW.value)
        self.assertEqual(1.0, model.economics.XLCOEIndirectJobsMultiplier.value)
        self.assertEqual(0.0, model.economics.XLCOEAverageMonthlyWage.value)
        self.assertEqual(0.0, model.economics.XLCOHAvoidedEmissionsIntensity.value)
        self.assertEqual(0.0, model.economics.XLCOHCarbonPrice.value)
        self.assertEqual(0.0, model.economics.XLCOHThermalCreditPrice.value)
        self.assertEqual(0.0, model.economics.XLCOHDisplacedWaterUseIntensity.value)
        self.assertEqual(0.0, model.economics.XLCOHWaterShadowPrice.value)
        self.assertEqual(0.0, model.economics.XLCOHOperationsJobsPerMW.value)
        self.assertEqual(0.0, model.economics.XLCOCAvoidedEmissionsIntensity.value)
        self.assertEqual(0.0, model.economics.XLCOCCarbonPrice.value)
        self.assertEqual(0.0, model.economics.XLCOCCoolingCreditPrice.value)
        self.assertEqual(0.0, model.economics.XLCOCDisplacedWaterUseIntensity.value)
        self.assertEqual(0.0, model.economics.XLCOCWaterShadowPrice.value)
        self.assertEqual(0.0, model.economics.XLCOCOperationsJobsPerMW.value)
        self.assertEqual(EnergyCostUnit.CENTSSPERKWH, model.economics.XLCOE_Market.CurrentUnits)
        self.assertEqual(EnergyCostUnit.CENTSSPERKWH, model.economics.XLCOE_MarketSocial.CurrentUnits)
        self.assertEqual(CO2ProductionUnit.TONNEPERMWH, model.economics.XLCOHAvoidedEmissionsIntensity.CurrentUnits)
        self.assertEqual(CostPerMassUnit.DOLLARSPERTONNE, model.economics.XLCOHCarbonPrice.CurrentUnits)
        self.assertEqual(EnergyCostUnit.DOLLARSPERMWH, model.economics.XLCOHThermalCreditPrice.CurrentUnits)
        self.assertEqual(CurrencyUnit.DOLLARS, model.economics.XLCOHWaterShadowPrice.CurrentUnits)
        self.assertEqual(CO2ProductionUnit.TONNEPERMWH, model.economics.XLCOCAvoidedEmissionsIntensity.CurrentUnits)
        self.assertEqual(CostPerMassUnit.DOLLARSPERTONNE, model.economics.XLCOCCarbonPrice.CurrentUnits)
        self.assertEqual(EnergyCostUnit.DOLLARSPERMWH, model.economics.XLCOCCoolingCreditPrice.CurrentUnits)
        self.assertEqual(CurrencyUnit.DOLLARS, model.economics.XLCOCWaterShadowPrice.CurrentUnits)
        self.assertEqual(EnergyCostUnit.DOLLARSPERMMBTU, model.economics.XLCOH_Market.CurrentUnits)
        self.assertEqual(EnergyCostUnit.DOLLARSPERMMBTU, model.economics.XLCOH_MarketSocial.CurrentUnits)
        self.assertEqual(EnergyCostUnit.DOLLARSPERMMBTU, model.economics.XLCOC_Market.CurrentUnits)
        self.assertEqual(EnergyCostUnit.DOLLARSPERMMBTU, model.economics.XLCOC_MarketSocial.CurrentUnits)

    def test_xlcoe_disabled_leaves_lcoe_unchanged_and_outputs_zero(self):
        baseline_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/Fervo_Project_Cape-5.txt')),
            read_and_calculate=True,
        )
        disabled_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/Fervo_Project_Cape-5.txt')),
            additional_params={'Do XLCOE Calculations': False},
            read_and_calculate=True,
        )

        self.assertGreater(baseline_model.economics.LCOE.value, 0.0)
        self.assertAlmostEqual(baseline_model.economics.LCOE.value, disabled_model.economics.LCOE.value, places=7)
        self.assertEqual(0.0, disabled_model.economics.XLCOE_Market.value)
        self.assertEqual(0.0, disabled_model.economics.XLCOE_MarketSocial.value)

    def test_xlcoe_enabled_with_zero_market_inputs_matches_lcoe(self):
        model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example1.txt')),
            additional_params={'Do XLCOE Calculations': True},
            read_and_calculate=True,
        )

        self.assertGreater(model.economics.LCOE.value, 0.0)
        self.assertAlmostEqual(model.economics.LCOE.value, model.economics.XLCOE_Market.value, places=7)
        self.assertAlmostEqual(model.economics.XLCOE_Market.value, model.economics.XLCOE_MarketSocial.value, places=7)

    def test_xlcoh_enabled_with_zero_inputs_matches_lcoh(self):
        model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example2.txt')),
            additional_params={'Do XLCOE Calculations': True},
            read_and_calculate=True,
        )

        self.assertGreater(model.economics.LCOH.value, 0.0)
        self.assertAlmostEqual(model.economics.LCOH.value, model.economics.XLCOH_Market.value, places=7)
        self.assertAlmostEqual(model.economics.XLCOH_Market.value, model.economics.XLCOH_MarketSocial.value, places=7)
        self.assertEqual(0.0, model.economics.XLCOE_Market.value)
        self.assertEqual(0.0, model.economics.XLCOC_Market.value)

    def test_xlcoc_enabled_with_zero_inputs_matches_lcoc(self):
        model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example11_AC.txt')),
            additional_params={'Do XLCOE Calculations': True},
            read_and_calculate=True,
        )

        self.assertGreater(model.economics.LCOC.value, 0.0)
        self.assertAlmostEqual(model.economics.LCOC.value, model.economics.XLCOC_Market.value, places=7)
        self.assertAlmostEqual(model.economics.XLCOC_Market.value, model.economics.XLCOC_MarketSocial.value, places=7)
        self.assertEqual(0.0, model.economics.XLCOE_Market.value)

    def test_xlcoe_market_benefits_reduce_breakeven_price(self):
        baseline_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example1.txt')),
            additional_params={'Do XLCOE Calculations': True},
            read_and_calculate=True,
        )
        benefit_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example1.txt')),
            additional_params={
                'Do XLCOE Calculations': True,
                'Avoided Emissions Intensity': 0.44,
                'XLCOE Carbon Price': 35.0,
                'XLCOE REC Price': 7.0,
            },
            read_and_calculate=True,
        )

        self.assertLess(benefit_model.economics.XLCOE_Market.value, baseline_model.economics.XLCOE_Market.value)
        self.assertLessEqual(benefit_model.economics.XLCOE_Market.value, benefit_model.economics.LCOE.value)
        self.assertAlmostEqual(
            benefit_model.economics.XLCOE_Market.value,
            benefit_model.economics.XLCOE_MarketSocial.value,
            places=7,
        )

    def test_xlcoh_market_benefits_reduce_heat_breakeven_price(self):
        baseline_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example2.txt')),
            additional_params={'Do XLCOE Calculations': True},
            read_and_calculate=True,
        )
        benefit_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example2.txt')),
            additional_params={
                'Do XLCOE Calculations': True,
                'XLCOH Avoided Emissions Intensity': 0.25,
                'XLCOH Carbon Price': 30.0,
                'XLCOH Thermal Credit Price': 4.0,
            },
            read_and_calculate=True,
        )

        self.assertLess(benefit_model.economics.XLCOH_Market.value, baseline_model.economics.XLCOH_Market.value)
        self.assertLessEqual(benefit_model.economics.XLCOH_Market.value, benefit_model.economics.LCOH.value)
        self.assertAlmostEqual(
            benefit_model.economics.XLCOH_Market.value,
            benefit_model.economics.XLCOH_MarketSocial.value,
            places=7,
        )

    def test_xlcoe_idle_rig_discount_reduces_market_output(self):
        baseline_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example1.txt')),
            additional_params={'Do XLCOE Calculations': True},
            read_and_calculate=True,
        )
        idle_rig_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example1.txt')),
            additional_params={
                'Do XLCOE Calculations': True,
                'Idle Rig Discount Rate': 0.05,
            },
            read_and_calculate=True,
        )

        self.assertGreater(idle_rig_model.economics.Cwell.value, 0.0)
        self.assertLess(idle_rig_model.economics.XLCOE_Market.value, baseline_model.economics.XLCOE_Market.value)
        self.assertLessEqual(
            idle_rig_model.economics.XLCOE_MarketSocial.value,
            idle_rig_model.economics.XLCOE_Market.value,
        )

    def test_xlcoe_social_benefits_reduce_market_social_breakeven_price(self):
        market_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example1.txt')),
            additional_params={
                'Do XLCOE Calculations': True,
                'Avoided Emissions Intensity': 0.44,
                'XLCOE Carbon Price': 35.0,
                'XLCOE REC Price': 7.0,
            },
            read_and_calculate=True,
        )
        social_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example1.txt')),
            additional_params={
                'Do XLCOE Calculations': True,
                'Avoided Emissions Intensity': 0.44,
                'XLCOE Carbon Price': 35.0,
                'XLCOE REC Price': 7.0,
                'XLCOE Displaced Water Use Intensity': 1.0,
                'XLCOE Water Shadow Price': 0.25,
                'XLCOE Construction Jobs Per Rig': 37.0,
                'XLCOE Operations Jobs Per MW': 1.7,
                'XLCOE Indirect Jobs Multiplier': 1.0,
                'XLCOE Average Monthly Wage': 4000.0,
            },
            read_and_calculate=True,
        )

        self.assertAlmostEqual(social_model.economics.XLCOE_Market.value, market_model.economics.XLCOE_Market.value, places=7)
        self.assertLess(
            social_model.economics.XLCOE_MarketSocial.value,
            social_model.economics.XLCOE_Market.value,
        )

    def test_xlcoc_social_benefits_reduce_market_social_breakeven_price(self):
        market_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example11_AC.txt')),
            additional_params={'Do XLCOE Calculations': True},
            read_and_calculate=True,
        )
        social_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example11_AC.txt')),
            additional_params={
                'Do XLCOE Calculations': True,
                'XLCOC Displaced Water Use Intensity': 1.2,
                'XLCOC Water Shadow Price': 0.4,
                'XLCOC Operations Jobs Per MW': 1.1,
                'XLCOE Indirect Jobs Multiplier': 1.0,
                'XLCOE Average Monthly Wage': 4000.0,
            },
            read_and_calculate=True,
        )

        self.assertAlmostEqual(social_model.economics.XLCOC_Market.value, market_model.economics.XLCOC_Market.value, places=7)
        self.assertLess(
            social_model.economics.XLCOC_MarketSocial.value,
            social_model.economics.XLCOC_Market.value,
        )

    def test_cogeneration_project_computes_both_electricity_and_heat_extended_outputs(self):
        model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example13.txt')),
            additional_params={'Do XLCOE Calculations': True},
            read_and_calculate=True,
        )

        self.assertGreater(model.economics.LCOE.value, 0.0)
        self.assertGreater(model.economics.LCOH.value, 0.0)
        self.assertAlmostEqual(model.economics.LCOE.value, model.economics.XLCOE_Market.value, places=7)
        self.assertAlmostEqual(model.economics.LCOH.value, model.economics.XLCOH_Market.value, places=7)

    def test_cogeneration_idle_rig_benefit_is_allocated_by_baseline_cost_share(self):
        baseline_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example13.txt')),
            additional_params={'Do XLCOE Calculations': True},
            read_and_calculate=True,
        )
        benefit_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example13.txt')),
            additional_params={
                'Do XLCOE Calculations': True,
                'Idle Rig Discount Rate': 0.05,
            },
            read_and_calculate=True,
        )

        bases = build_levelized_cost_bases(benefit_model.economics, benefit_model)
        total_discounted_cost = sum(basis.baseline_discounted_cost_musd for basis in bases.values())
        total_shared_benefit = benefit_model.economics.Cwell.value * benefit_model.economics.IdleRigDiscountRate.value

        electricity_observed = self._observed_discounted_benefit_musd(
            bases[ELECTRICITY_COMMODITY],
            baseline_model.economics.XLCOE_Market.value,
            benefit_model.economics.XLCOE_Market.value,
        )
        heat_observed = self._observed_discounted_benefit_musd(
            bases[HEAT_COMMODITY],
            baseline_model.economics.XLCOH_Market.value,
            benefit_model.economics.XLCOH_Market.value,
        )

        self.assertAlmostEqual(total_shared_benefit, electricity_observed + heat_observed, places=7)
        self.assertAlmostEqual(
            total_shared_benefit * bases[ELECTRICITY_COMMODITY].baseline_discounted_cost_musd / total_discounted_cost,
            electricity_observed,
            places=7,
        )
        self.assertAlmostEqual(
            total_shared_benefit * bases[HEAT_COMMODITY].baseline_discounted_cost_musd / total_discounted_cost,
            heat_observed,
            places=7,
        )

    def test_cogeneration_construction_jobs_benefit_is_allocated_by_baseline_cost_share(self):
        baseline_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example13.txt')),
            additional_params={'Do XLCOE Calculations': True},
            read_and_calculate=True,
        )
        benefit_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example13.txt')),
            additional_params={
                'Do XLCOE Calculations': True,
                'XLCOE Construction Jobs Per Rig': 37.0,
                'XLCOE Indirect Jobs Multiplier': 1.0,
                'XLCOE Average Monthly Wage': 4000.0,
            },
            read_and_calculate=True,
        )

        bases = build_levelized_cost_bases(benefit_model.economics, benefit_model)
        total_discounted_cost = sum(basis.baseline_discounted_cost_musd for basis in bases.values())
        construction_years = benefit_model.surfaceplant.construction_years.value
        total_wells = benefit_model.wellbores.nprod.value + benefit_model.wellbores.ninj.value
        construction_jobs_total = (
            total_wells
            * benefit_model.economics.XLCOEConstructionJobsPerRig.value
            * benefit_model.economics.XLCOEIndirectJobsMultiplier.value
        )
        annual_construction_jobs_musd = (
            construction_jobs_total
            * benefit_model.economics.XLCOEAverageMonthlyWage.value
            * 12.0
            / construction_years
            / 1_000_000.0
        )
        total_shared_social_benefit = sum(
            annual_construction_jobs_musd / (1.0 + benefit_model.economics.social_discountrate.value) ** year
            for year in range(construction_years)
        )

        electricity_observed = self._observed_discounted_benefit_musd(
            bases[ELECTRICITY_COMMODITY],
            baseline_model.economics.XLCOE_MarketSocial.value,
            benefit_model.economics.XLCOE_MarketSocial.value,
        )
        heat_observed = self._observed_discounted_benefit_musd(
            bases[HEAT_COMMODITY],
            baseline_model.economics.XLCOH_MarketSocial.value,
            benefit_model.economics.XLCOH_MarketSocial.value,
        )

        self.assertAlmostEqual(total_shared_social_benefit, electricity_observed + heat_observed, places=7)
        self.assertAlmostEqual(
            total_shared_social_benefit * bases[ELECTRICITY_COMMODITY].baseline_discounted_cost_musd / total_discounted_cost,
            electricity_observed,
            places=7,
        )
        self.assertAlmostEqual(
            total_shared_social_benefit * bases[HEAT_COMMODITY].baseline_discounted_cost_musd / total_discounted_cost,
            heat_observed,
            places=7,
        )

    def test_cogeneration_direct_benefits_do_not_leak_between_commodities(self):
        baseline_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example13.txt')),
            additional_params={'Do XLCOE Calculations': True},
            read_and_calculate=True,
        )
        electricity_benefit_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example13.txt')),
            additional_params={
                'Do XLCOE Calculations': True,
                'Avoided Emissions Intensity': 0.44,
                'XLCOE Carbon Price': 35.0,
                'XLCOE REC Price': 7.0,
            },
            read_and_calculate=True,
        )
        heat_benefit_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example13.txt')),
            additional_params={
                'Do XLCOE Calculations': True,
                'XLCOH Avoided Emissions Intensity': 0.25,
                'XLCOH Carbon Price': 30.0,
                'XLCOH Thermal Credit Price': 4.0,
            },
            read_and_calculate=True,
        )

        self.assertLess(
            electricity_benefit_model.economics.XLCOE_Market.value,
            baseline_model.economics.XLCOE_Market.value,
        )
        self.assertAlmostEqual(
            electricity_benefit_model.economics.XLCOH_Market.value,
            baseline_model.economics.XLCOH_Market.value,
            places=7,
        )
        self.assertLess(
            heat_benefit_model.economics.XLCOH_Market.value,
            baseline_model.economics.XLCOH_Market.value,
        )
        self.assertAlmostEqual(
            heat_benefit_model.economics.XLCOE_Market.value,
            baseline_model.economics.XLCOE_Market.value,
            places=7,
        )

    def test_xlcoe_social_discount_rate_only_changes_social_output(self):
        low_discount_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example1.txt')),
            additional_params={
                'Do XLCOE Calculations': True,
                'XLCOE Displaced Water Use Intensity': 1.0,
                'XLCOE Water Shadow Price': 0.25,
                'XLCOE Construction Jobs Per Rig': 37.0,
                'XLCOE Operations Jobs Per MW': 1.7,
                'XLCOE Indirect Jobs Multiplier': 1.0,
                'XLCOE Average Monthly Wage': 4000.0,
                'Social Discount Rate': 0.04,
            },
            read_and_calculate=True,
        )
        high_discount_model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example1.txt')),
            additional_params={
                'Do XLCOE Calculations': True,
                'XLCOE Displaced Water Use Intensity': 1.0,
                'XLCOE Water Shadow Price': 0.25,
                'XLCOE Construction Jobs Per Rig': 37.0,
                'XLCOE Operations Jobs Per MW': 1.7,
                'XLCOE Indirect Jobs Multiplier': 1.0,
                'XLCOE Average Monthly Wage': 4000.0,
                'Social Discount Rate': 0.10,
            },
            read_and_calculate=True,
        )

        self.assertAlmostEqual(
            low_discount_model.economics.XLCOE_Market.value,
            high_discount_model.economics.XLCOE_Market.value,
            places=7,
        )
        self.assertLess(
            low_discount_model.economics.XLCOE_MarketSocial.value,
            high_discount_model.economics.XLCOE_MarketSocial.value,
        )

    def test_generalized_engine_returns_no_active_results_when_xlcoe_disabled(self):
        model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example1.txt')),
            additional_params={'Do XLCOE Calculations': False},
            read_and_calculate=True,
        )

        self.assertEqual({}, calculate_extended_levelized_costs(model.economics, model))

    def test_xlcoe_paper_low_fixture_matches_published_values(self):
        fixture = self._load_paper_fixture('../examples/example_XLCOE_paper_low.txt')

        lcoe, xlcoe_market, xlcoe_market_social = calculate_xlcoe_from_explicit_streams(
            fixture['annual_baseline_costs_musd'],
            fixture['annual_net_generation_kwh'],
            fixture['annual_market_benefits_musd'],
            fixture['annual_social_benefits_musd'],
            fixture['Market Discount Rate'],
            fixture['Social Discount Rate'],
        )

        self.assertAlmostEqual(fixture['Expected LCOE cents/kWh'], lcoe, places=7)
        self.assertAlmostEqual(fixture['Expected XLCOE Market cents/kWh'], xlcoe_market, places=7)
        self.assertAlmostEqual(fixture['Expected XLCOE Market Social cents/kWh'], xlcoe_market_social, places=7)

    def test_xlcoe_paper_high_fixture_matches_published_values(self):
        fixture = self._load_paper_fixture('../examples/example_XLCOE_paper_high.txt')

        lcoe, xlcoe_market, xlcoe_market_social = calculate_xlcoe_from_explicit_streams(
            fixture['annual_baseline_costs_musd'],
            fixture['annual_net_generation_kwh'],
            fixture['annual_market_benefits_musd'],
            fixture['annual_social_benefits_musd'],
            fixture['Market Discount Rate'],
            fixture['Social Discount Rate'],
        )

        self.assertAlmostEqual(fixture['Expected LCOE cents/kWh'], lcoe, places=7)
        self.assertAlmostEqual(fixture['Expected XLCOE Market cents/kWh'], xlcoe_market, places=7)
        self.assertAlmostEqual(fixture['Expected XLCOE Market Social cents/kWh'], xlcoe_market_social, places=7)

    def test_xlcoe_paper_fixture_component_sensitivities_have_expected_sign(self):
        fixture = self._load_paper_fixture('../examples/example_XLCOE_paper_low.txt')
        baseline = self._calculate_paper_fixture_outputs(fixture)

        for component_name in [
            'Annual Carbon Benefit MUSD',
            'Annual REC Benefit MUSD',
            'Idle Rig Discount Benefit MUSD',
        ]:
            adjusted_fixture = dict(fixture)
            adjusted_fixture[component_name] *= 1.10
            adjusted = self._calculate_paper_fixture_outputs(adjusted_fixture)
            self.assertLess(adjusted[1], baseline[1], msg=component_name)
            self.assertLess(adjusted[2], baseline[2], msg=component_name)

        for component_name in [
            'Annual Water Benefit MUSD',
            'Annual Construction Jobs Benefit MUSD',
            'Annual Operations Jobs Benefit MUSD',
        ]:
            adjusted_fixture = dict(fixture)
            adjusted_fixture[component_name] *= 1.10
            adjusted = self._calculate_paper_fixture_outputs(adjusted_fixture)
            self.assertAlmostEqual(adjusted[1], baseline[1], places=7, msg=component_name)
            self.assertLess(adjusted[2], baseline[2], msg=component_name)

    # noinspection PyMethodMayBeStatic
    def _new_model(
        self, input_file: Path | None = None, additional_params: dict[str, Any] | None = None, read_and_calculate=False
    ) -> Model:
        model_args = {'enable_geophires_logging_config': False}

        if input_file is not None:
            if additional_params is not None:
                params = GeophiresInputParameters(from_file_path=input_file, params=additional_params)
                input_file = params.as_file_path()

            model_args['input_file'] = input_file

        stash_cwd = Path.cwd()
        stash_sys_argv = sys.argv

        sys.argv = ['']

        model = Model(**model_args)

        sys.argv = stash_sys_argv
        os.chdir(stash_cwd)

        if read_and_calculate:
            model.read_parameters()
            model.Calculate()

        return model

    def _load_paper_fixture(self, relative_path: str) -> dict[str, Any]:
        fixture: dict[str, Any] = {}
        with open(self._get_test_file_path(relative_path), encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if line == '' or line.startswith('#'):
                    continue
                key, value = [part.strip() for part in line.split(',', maxsplit=1)]
                fixture[key] = float(value)

        return self._expand_paper_fixture(fixture)

    def _calculate_paper_fixture_outputs(self, fixture: dict[str, Any]) -> tuple[float, float, float]:
        fixture = self._expand_paper_fixture(dict(fixture))
        return calculate_xlcoe_from_explicit_streams(
            fixture['annual_baseline_costs_musd'],
            fixture['annual_net_generation_kwh'],
            fixture['annual_market_benefits_musd'],
            fixture['annual_social_benefits_musd'],
            fixture['Market Discount Rate'],
            fixture['Social Discount Rate'],
        )

    def _expand_paper_fixture(self, fixture: dict[str, Any]) -> dict[str, Any]:
        construction_years = int(fixture['Construction Years'])
        operating_years = int(fixture['Operating Years'])
        total_years = construction_years + operating_years

        fixture['annual_net_generation_kwh'] = [0.0] * construction_years + [
            fixture['Annual Net Generation kWh']
        ] * operating_years
        fixture['annual_baseline_costs_musd'] = [0.0] * construction_years + [
            fixture['Annual Baseline Cost MUSD']
        ] * operating_years

        annual_market_benefits_musd = [0.0] * total_years
        annual_market_benefits_musd[0] = fixture['Idle Rig Discount Benefit MUSD']
        for year in range(construction_years, total_years):
            annual_market_benefits_musd[year] = (
                fixture['Annual Carbon Benefit MUSD'] + fixture['Annual REC Benefit MUSD']
            )
        fixture['annual_market_benefits_musd'] = annual_market_benefits_musd

        fixture['annual_social_benefits_musd'] = [fixture['Annual Construction Jobs Benefit MUSD']] * construction_years + [
            fixture['Annual Water Benefit MUSD'] + fixture['Annual Operations Jobs Benefit MUSD']
        ] * operating_years
        return fixture

    @staticmethod
    def _observed_discounted_benefit_musd(basis, baseline_public_value: float, benefit_public_value: float) -> float:
        return (
            (baseline_public_value - benefit_public_value)
            * basis.discounted_output
            / basis.public_price_factor
        )
