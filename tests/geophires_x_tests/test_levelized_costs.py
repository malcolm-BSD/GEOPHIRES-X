from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from geophires_x.Model import Model
from geophires_x.levelized_costs import (
    COOLING_COMMODITY,
    ELECTRICITY_COMMODITY,
    HEAT_COMMODITY,
    build_levelized_cost_bases,
)
from geophires_x_client import GeophiresInputParameters
from tests.base_test_case import BaseTestCase


class LevelizedCostBasisTestCase(BaseTestCase):

    def test_electricity_basis_matches_lcoe_output(self):
        model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example1.txt')),
            read_and_calculate=True,
        )

        basis = build_levelized_cost_bases(model.economics, model)[ELECTRICITY_COMMODITY]

        self.assertGreater(basis.discounted_output, 0.0)
        self.assertGreater(basis.baseline_discounted_cost_musd, 0.0)
        self.assertAlmostEqual(model.economics.LCOE.value, basis.public_value, places=7)

    def test_heat_pump_basis_matches_lcoh_output(self):
        model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example10_HP.txt')),
            additional_params={'Economic Model': 3},
            read_and_calculate=True,
        )

        basis = build_levelized_cost_bases(model.economics, model)[HEAT_COMMODITY]

        self.assertGreater(basis.discounted_output, 0.0)
        self.assertGreater(basis.baseline_discounted_cost_musd, 0.0)
        self.assertAlmostEqual(model.economics.LCOH.value, basis.public_value, places=7)

    def test_absorption_chiller_basis_matches_lcoc_output(self):
        model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example11_AC.txt')),
            read_and_calculate=True,
        )

        basis = build_levelized_cost_bases(model.economics, model)[COOLING_COMMODITY]

        self.assertGreater(basis.discounted_output, 0.0)
        self.assertGreater(basis.baseline_discounted_cost_musd, 0.0)
        self.assertAlmostEqual(model.economics.LCOC.value, basis.public_value, places=7)

    def test_district_heating_basis_matches_lcoh_output(self):
        model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example12_DH.txt')),
            additional_params={
                'District Heating Demand File Name': self._get_test_file_path('../examples/cornell_heat_demand.csv')
            },
            read_and_calculate=True,
        )

        basis = build_levelized_cost_bases(model.economics, model)[HEAT_COMMODITY]

        self.assertGreater(basis.discounted_output, 0.0)
        self.assertGreater(basis.baseline_discounted_cost_musd, 0.0)
        self.assertAlmostEqual(model.economics.LCOH.value, basis.public_value, places=7)

    def test_cogeneration_basis_matches_both_outputs(self):
        model = self._new_model(
            input_file=Path(self._get_test_file_path('../examples/example13.txt')),
            read_and_calculate=True,
        )

        bases = build_levelized_cost_bases(model.economics, model)

        self.assertIn(ELECTRICITY_COMMODITY, bases)
        self.assertIn(HEAT_COMMODITY, bases)
        self.assertAlmostEqual(model.economics.LCOE.value, bases[ELECTRICITY_COMMODITY].public_value, places=7)
        self.assertAlmostEqual(model.economics.LCOH.value, bases[HEAT_COMMODITY].public_value, places=7)

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
