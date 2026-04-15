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
        self.assertIn('XLCOE_Market', model.economics.OutputParameterDict)
        self.assertIn('XLCOE_MarketSocial', model.economics.OutputParameterDict)

        self.assertFalse(model.economics.DoXLCOECalculations.value)
        self.assertEqual(0.07, model.economics.social_discountrate.value)
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

    def test_xlcoe_enabled_raises_not_implemented_error(self):
        with self.assertRaisesRegex(NotImplementedError, 'XLCOE calculations are not implemented yet'):
            self._new_model(
                input_file=Path(self._get_test_file_path('../examples/Fervo_Project_Cape-5.txt')),
                additional_params={'Do XLCOE Calculations': True},
                read_and_calculate=True,
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
