from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from geophires_x.Model import Model
from geophires_x.Units import EnergyCostUnit
from geophires_x_client import GeophiresInputParameters
from tests.base_test_case import BaseTestCase


class XLCOETestCase(BaseTestCase):

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
        self.assertIn('XLCOE_Market', model.economics.OutputParameterDict)
        self.assertIn('XLCOE_MarketSocial', model.economics.OutputParameterDict)

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
        self.assertEqual(EnergyCostUnit.CENTSSPERKWH, model.economics.XLCOE_Market.CurrentUnits)
        self.assertEqual(EnergyCostUnit.CENTSSPERKWH, model.economics.XLCOE_MarketSocial.CurrentUnits)

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
