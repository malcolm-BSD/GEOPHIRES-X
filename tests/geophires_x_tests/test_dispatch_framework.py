import os
import sys
from pathlib import Path

from geophires_x.Dispatch import DemandProfileFactory
from geophires_x.Model import Model
from geophires_x.OptionList import DispatchDemandSource
from geophires_x.OptionList import DispatchFlowStrategy
from geophires_x.OptionList import OperatingMode
from geophires_x.Parameter import ParameterEntry
from tests.base_test_case import BaseTestCase


class DispatchFrameworkTestCase(BaseTestCase):
    def _new_model(self) -> Model:
        stash_cwd = Path.cwd()
        stash_sys_argv = sys.argv
        sys.argv = [""]
        model = Model(enable_geophires_logging_config=False)
        sys.argv = stash_sys_argv
        os.chdir(stash_cwd)
        return model

    def test_operating_mode_from_input_string(self):
        self.assertEqual(OperatingMode.from_input_string("Baseload"), OperatingMode.BASELOAD)
        self.assertEqual(OperatingMode.from_input_string("dispatchable"), OperatingMode.DISPATCHABLE)

    def test_dispatch_parameter_parsing(self):
        model = self._new_model()
        model.InputParameters = {
            "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
            "Dispatch Demand Source": ParameterEntry(Name="Dispatch Demand Source", sValue="Annual Heat Demand"),
            "Dispatch Flow Strategy": ParameterEntry(Name="Dispatch Flow Strategy", sValue="Demand Following"),
        }

        model.surfaceplant.read_parameters(model)

        self.assertEqual(OperatingMode.DISPATCHABLE, model.surfaceplant.operating_mode.value)
        self.assertEqual(DispatchDemandSource.ANNUAL_HEAT_DEMAND, model.surfaceplant.dispatch_demand_source.value)
        self.assertEqual(DispatchFlowStrategy.DEMAND_FOLLOWING, model.surfaceplant.dispatch_flow_strategy.value)

    def test_demand_profile_factory_uses_hourly_historical_array(self):
        model = self._new_model()
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        model.InputParameters = {
            "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=csv_file),
        }

        model.surfaceplant.read_parameters(model)
        profile = DemandProfileFactory.from_model(model)

        self.assertEqual(8760, profile.num_timesteps)
        self.assertAlmostEqual(13188.2, profile.series[0], places=1)

    def test_dispatchable_calculate_raises_clear_placeholder_error(self):
        model = self._new_model()
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        model.reserv = type(
            "CylindricalReservoir",
            (),
            {
                "read_parameters": lambda self, model: None,
                "Calculate": lambda self, model: None,
            },
        )()
        model.InputParameters = {
            "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
            "End-Use Option": ParameterEntry(Name="End-Use Option", sValue="2"),
            "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=csv_file),
        }

        model.read_parameters()

        with self.assertRaisesRegex(NotImplementedError, "hourly dispatch simulation loop is not implemented yet"):
            model.Calculate()
