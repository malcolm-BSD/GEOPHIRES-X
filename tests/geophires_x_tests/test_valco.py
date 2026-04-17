from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from geophires_x.levelized_costs import COOLING_COMMODITY
from geophires_x.levelized_costs import ELECTRICITY_COMMODITY
from geophires_x.levelized_costs import HEAT_COMMODITY
from geophires_x.Model import Model
from geophires_x.valco import ValueAdjustmentInputs
from geophires_x.valco import assign_value_adjusted_levelized_cost_outputs
from geophires_x.valco import build_default_value_adjustment_inputs
from geophires_x.valco import calculate_value_adjusted_cost
from geophires_x.valco import calculate_value_adjusted_costs_from_inputs
from geophires_x.valco import select_active_valco_base_costs
from geophires_x_client import GeophiresInputParameters
from tests.base_test_case import BaseTestCase


class VALCOTestCase(BaseTestCase):
    def test_value_adjusted_cost_applies_component_deltas_to_active_base_cost(self):
        result = calculate_value_adjusted_cost(
            ValueAdjustmentInputs(
                active_base_cost=8.0,
                system_energy_value=5.0,
                technology_energy_value=7.0,
                system_capacity_value=2.0,
                technology_capacity_value=1.0,
                system_flexibility_value=0.5,
                technology_flexibility_value=0.25,
            )
        )

        self.assertAlmostEqual(8.0, result.active_base_cost)
        self.assertAlmostEqual(-2.0, result.energy_adjustment)
        self.assertAlmostEqual(1.0, result.capacity_adjustment)
        self.assertAlmostEqual(0.25, result.flexibility_adjustment)
        self.assertAlmostEqual(7.25, result.valco)

    def test_calculate_value_adjusted_costs_supports_multiple_commodities(self):
        results = calculate_value_adjusted_costs_from_inputs(
            {
                ELECTRICITY_COMMODITY: ValueAdjustmentInputs(active_base_cost=8.0),
                HEAT_COMMODITY: ValueAdjustmentInputs(
                    active_base_cost=12.0,
                    system_capacity_value=2.0,
                    technology_capacity_value=4.0,
                ),
            }
        )

        self.assertAlmostEqual(8.0, results[ELECTRICITY_COMMODITY].valco)
        self.assertAlmostEqual(-2.0, results[HEAT_COMMODITY].capacity_adjustment)
        self.assertAlmostEqual(10.0, results[HEAT_COMMODITY].valco)

    def test_select_active_valco_base_costs_uses_lco_basis_without_xlco(self):
        model = self._new_model(
            input_file=self._get_test_file_path("../examples/example1.txt"),
            read_and_calculate=True,
        )

        base_costs = select_active_valco_base_costs(model.economics, model)

        self.assertIn(ELECTRICITY_COMMODITY, base_costs)
        self.assertAlmostEqual(model.economics.LCOE.value, base_costs[ELECTRICITY_COMMODITY], places=7)

    def test_select_active_valco_base_costs_uses_xlco_market_when_available(self):
        model = self._new_model(
            input_file=self._get_test_file_path("../examples/example1.txt"),
            additional_params={
                "Do XLCO(E|H|C) Calculations": True,
                "XLCOE Avoided Emissions Intensity": 0.44,
                "XLCO(E|H|C) Carbon Price": 35.0,
                "XLCOE REC Price": 7.0,
            },
            read_and_calculate=True,
        )

        base_costs = select_active_valco_base_costs(model.economics, model)

        self.assertIn(ELECTRICITY_COMMODITY, base_costs)
        self.assertAlmostEqual(model.economics.XLCOE_Market.value, base_costs[ELECTRICITY_COMMODITY], places=7)
        self.assertNotAlmostEqual(model.economics.LCOE.value, base_costs[ELECTRICITY_COMMODITY], places=5)

    def test_build_default_value_adjustment_inputs_uses_active_base_costs(self):
        model = self._new_model(
            input_file=self._get_test_file_path("../examples/example2.txt"),
            read_and_calculate=True,
        )

        commodity_inputs = build_default_value_adjustment_inputs(model.economics, model)

        self.assertIn(HEAT_COMMODITY, commodity_inputs)
        self.assertAlmostEqual(model.economics.LCOH.value, commodity_inputs[HEAT_COMMODITY].active_base_cost, places=7)

    def test_assign_value_adjusted_levelized_cost_outputs_sets_available_outputs(self):
        econ = SimpleNamespace(
            VALCOE=SimpleNamespace(value=0.0),
            VALCOE_EnergyAdjustment=SimpleNamespace(value=0.0),
            VALCOE_CapacityAdjustment=SimpleNamespace(value=0.0),
            VALCOE_FlexibilityAdjustment=SimpleNamespace(value=0.0),
            VALCOH=SimpleNamespace(value=0.0),
            VALCOC=SimpleNamespace(value=0.0),
        )
        commodity_results = calculate_value_adjusted_costs_from_inputs(
            {
                ELECTRICITY_COMMODITY: ValueAdjustmentInputs(
                    active_base_cost=8.0,
                    system_energy_value=1.0,
                    technology_energy_value=2.0,
                ),
                HEAT_COMMODITY: ValueAdjustmentInputs(active_base_cost=12.0),
                COOLING_COMMODITY: ValueAdjustmentInputs(active_base_cost=9.5),
            }
        )

        assign_value_adjusted_levelized_cost_outputs(econ, commodity_results)

        self.assertAlmostEqual(7.0, econ.VALCOE.value)
        self.assertAlmostEqual(-1.0, econ.VALCOE_EnergyAdjustment.value)
        self.assertAlmostEqual(0.0, econ.VALCOE_CapacityAdjustment.value)
        self.assertAlmostEqual(0.0, econ.VALCOE_FlexibilityAdjustment.value)
        self.assertAlmostEqual(12.0, econ.VALCOH.value)
        self.assertAlmostEqual(9.5, econ.VALCOC.value)

    def _new_model(
        self, input_file: Path | None = None, additional_params: dict[str, Any] | None = None, read_and_calculate=False
    ) -> Model:
        model_args = {"enable_geophires_logging_config": False}

        if input_file is not None:
            if additional_params is not None:
                params = GeophiresInputParameters(from_file_path=input_file, params=additional_params)
                input_file = params.as_file_path()

            model_args["input_file"] = input_file

        stash_cwd = Path.cwd()
        stash_sys_argv = sys.argv

        sys.argv = [""]

        model = Model(**model_args)

        sys.argv = stash_sys_argv
        os.chdir(stash_cwd)

        if read_and_calculate:
            model.read_parameters()
            model.Calculate()

        return model
