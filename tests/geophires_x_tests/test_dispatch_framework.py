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

    def test_dispatchable_cylindrical_run_populates_dispatch_results_and_economics(self):
        from geophires_x.CylindricalReservoir import CylindricalReservoir

        model = self._new_model()
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        model.reserv = CylindricalReservoir(model)
        model.InputParameters = {
            "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
            "End-Use Option": ParameterEntry(Name="End-Use Option", sValue="2"),
            "Plant Lifetime": ParameterEntry(Name="Plant Lifetime", sValue="1"),
            "Reservoir Model": ParameterEntry(Name="Reservoir Model", sValue="0"),
            "Power Plant Type": ParameterEntry(Name="Power Plant Type", sValue="9"),
            "Number of Multilateral Sections": ParameterEntry(Name="Number of Multilateral Sections", sValue="1"),
            "Maximum Dispatch Flow Fraction": ParameterEntry(Name="Maximum Dispatch Flow Fraction", sValue="1.2"),
            "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=csv_file),
        }

        model.read_parameters()
        model.Calculate()

        self.assertEqual(8760, len(model.dispatch_results.hourly_produced_temperature))
        self.assertEqual(8760, len(model.surfaceplant.HeatProduced.value))
        self.assertGreater(model.dispatch_results.summary_metrics["design_heat_extracted_mw"], 0.0)
        self.assertGreater(model.dispatch_results.summary_metrics["annual_served_heat_kwh"], 0.0)
        self.assertGreaterEqual(model.economics.LCOH.value, 0.0)
        self.assertEqual(8760, model.economics.timestepsperyear.value)

    def test_dispatchable_reduced_order_reservoirs_run(self):
        from geophires_x.LHSReservoir import LHSReservoir
        from geophires_x.MPFReservoir import MPFReservoir
        from geophires_x.SFReservoir import SFReservoir

        reservoir_models = {
            "MPFReservoir": (MPFReservoir, "1"),
            "LHSReservoir": (LHSReservoir, "2"),
            "SFReservoir": (SFReservoir, "3"),
        }
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")

        for reservoir_name, reservoir_def in reservoir_models.items():
            with self.subTest(reservoir=reservoir_name):
                reservoir_class, reservoir_model_value = reservoir_def
                model = self._new_model()
                model.reserv = reservoir_class(model)
                model.InputParameters = {
                    "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
                    "End-Use Option": ParameterEntry(Name="End-Use Option", sValue="2"),
                    "Plant Lifetime": ParameterEntry(Name="Plant Lifetime", sValue="1"),
                    "Reservoir Model": ParameterEntry(Name="Reservoir Model", sValue=reservoir_model_value),
                    "Reservoir Depth": ParameterEntry(Name="Reservoir Depth", sValue="3"),
                    "Gradient 1": ParameterEntry(Name="Gradient 1", sValue="70"),
                    "Power Plant Type": ParameterEntry(Name="Power Plant Type", sValue="9"),
                    "Maximum Dispatch Flow Fraction": ParameterEntry(
                        Name="Maximum Dispatch Flow Fraction", sValue="1.1"
                    ),
                    "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=csv_file),
                }

                model.read_parameters()
                model.Calculate()

                self.assertEqual(8760, len(model.dispatch_results.hourly_produced_temperature))
                self.assertGreater(model.dispatch_results.summary_metrics["annual_served_heat_kwh"], 0.0)
                self.assertGreaterEqual(model.economics.LCOH.value, 0.0)

    def test_dispatchable_upp_run(self):
        from geophires_x.UPPReservoir import UPPReservoir

        model = self._new_model()
        model.reserv = UPPReservoir(model)
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        model.InputParameters = {
            "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
            "End-Use Option": ParameterEntry(Name="End-Use Option", sValue="2"),
            "Plant Lifetime": ParameterEntry(Name="Plant Lifetime", sValue="1"),
            "Reservoir Model": ParameterEntry(Name="Reservoir Model", sValue="5"),
            "Power Plant Type": ParameterEntry(Name="Power Plant Type", sValue="9"),
            "Maximum Dispatch Flow Fraction": ParameterEntry(Name="Maximum Dispatch Flow Fraction", sValue="1.1"),
            "Reservoir Output Profile": ParameterEntry(
                Name="Reservoir Output Profile",
                sValue="160,159,158,157,156,155,154,153,152,151,150",
            ),
            "Reservoir Output Profile Time Step": ParameterEntry(
                Name="Reservoir Output Profile Time Step",
                sValue="0.1",
            ),
            "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=csv_file),
        }

        model.read_parameters()
        model.Calculate()

        self.assertEqual(8760, len(model.dispatch_results.hourly_produced_temperature))
        self.assertGreater(model.dispatch_results.summary_metrics["annual_served_heat_kwh"], 0.0)
        self.assertGreaterEqual(model.economics.LCOH.value, 0.0)

    def test_dispatchable_sbt_is_explicitly_unsupported_for_now(self):
        from geophires_x.SBTEconomics import SBTEconomics
        from geophires_x.SBTReservoir import SBTReservoir
        from geophires_x.SBTWellbores import SBTWellbores

        model = self._new_model()
        model.reserv = SBTReservoir(model)
        model.wellbores = SBTWellbores(model)
        model.economics = SBTEconomics(model)
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        model.InputParameters = {
            "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
            "End-Use Option": ParameterEntry(Name="End-Use Option", sValue="2"),
            "Plant Lifetime": ParameterEntry(Name="Plant Lifetime", sValue="1"),
            "Reservoir Model": ParameterEntry(Name="Reservoir Model", sValue="8"),
            "Power Plant Type": ParameterEntry(Name="Power Plant Type", sValue="9"),
            "Maximum Dispatch Flow Fraction": ParameterEntry(Name="Maximum Dispatch Flow Fraction", sValue="1.05"),
            "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=csv_file),
        }

        model.read_parameters()
        with self.assertRaisesRegex(
            NotImplementedError,
            "Dispatchable timestep simulation has not been implemented yet for SBTReservoir",
        ):
            model.Calculate()
