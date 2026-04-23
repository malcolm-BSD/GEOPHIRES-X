import os
import sys
from pathlib import Path
from typing import Optional

from geophires_x.Dispatch import CylindricalDispatchPlantAdapter
from geophires_x.Dispatch import DemandFollowingDispatchStrategy
from geophires_x.Dispatch import DemandProfileFactory
from geophires_x.Dispatch import NoRecoveryModel
from geophires_x.Dispatch import ReducedOrderRecoveryModel
from geophires_x.Model import Model
from geophires_x.OptionList import DispatchDemandSource
from geophires_x.OptionList import DispatchFlowStrategy
from geophires_x.OptionList import OperatingMode
from geophires_x.Parameter import ParameterEntry
from tests.base_test_case import BaseTestCase


class DispatchFrameworkTestCase(BaseTestCase):
    def _new_model(self, input_file: Optional[str] = None) -> Model:
        stash_cwd = Path.cwd()
        stash_sys_argv = sys.argv
        sys.argv = [""]
        model = Model(enable_geophires_logging_config=False, input_file=input_file)
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
        self.assertEqual(1, model.surfaceplant.dispatch_analysis_start_year.value)
        self.assertEqual(2, model.surfaceplant.dispatch_analysis_end_year.value)

    def test_dispatch_parameter_parsing_for_electricity(self):
        model = self._new_model(input_file=str(Path(__file__).resolve().parents[1] / "examples" / "example1.txt"))
        model.InputParameters.update(
            {
                "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
                "Dispatch Demand Source": ParameterEntry(
                    Name="Dispatch Demand Source", sValue="Annual Electricity Demand"
                ),
                "Dispatch Flow Strategy": ParameterEntry(Name="Dispatch Flow Strategy", sValue="Demand Following"),
            }
        )

        model.surfaceplant.read_parameters(model)

        self.assertEqual(OperatingMode.DISPATCHABLE, model.surfaceplant.operating_mode.value)
        self.assertEqual(
            DispatchDemandSource.ANNUAL_ELECTRICITY_DEMAND, model.surfaceplant.dispatch_demand_source.value
        )
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
        self.assertEqual("MW", profile.units)
        self.assertAlmostEqual(13.1882, profile.series[0], places=4)

    def test_demand_profile_factory_uses_hourly_electricity_array(self):
        model = self._new_model(input_file=str(Path(__file__).resolve().parents[1] / "examples" / "example1.txt"))
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        model.InputParameters.update(
            {
                "Dispatch Demand Source": ParameterEntry(
                    Name="Dispatch Demand Source", sValue="Annual Electricity Demand"
                ),
                "Annual Electricity Demand": ParameterEntry(Name="Annual Electricity Demand", sValue=csv_file),
            }
        )

        model.surfaceplant.read_parameters(model)
        profile = DemandProfileFactory.from_model(model)

        self.assertEqual(8760, profile.num_timesteps)
        self.assertEqual("MW", profile.units)
        self.assertEqual("electric", profile.demand_type)
        self.assertAlmostEqual(13.1882, profile.series[0], places=4)

    def test_recovery_models_only_recover_during_shut_in(self):
        model = ReducedOrderRecoveryModel(
            equilibrium_state=100.0, recovery_time_constant_hours=24.0, state_bounds=(0.0, 100.0)
        )
        self.assertEqual(40.0, NoRecoveryModel().update(40.0, dt_hours=12.0, is_shut_in=True))
        self.assertEqual(40.0, model.update(40.0, dt_hours=12.0, is_shut_in=False))
        self.assertGreater(model.update(40.0, dt_hours=12.0, is_shut_in=True), 40.0)

    def _new_cylindrical_dispatch_adapter(self):
        from geophires_x.CylindricalReservoir import CylindricalReservoir

        model = self._new_model()
        model.reserv = CylindricalReservoir(model)
        model.InputParameters = {
            "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
            "End-Use Option": ParameterEntry(Name="End-Use Option", sValue="2"),
            "Plant Lifetime": ParameterEntry(Name="Plant Lifetime", sValue="1"),
            "Reservoir Model": ParameterEntry(Name="Reservoir Model", sValue="0"),
            "Power Plant Type": ParameterEntry(Name="Power Plant Type", sValue="9"),
            "Number of Multilateral Sections": ParameterEntry(Name="Number of Multilateral Sections", sValue="1"),
            "Maximum Dispatch Flow Fraction": ParameterEntry(Name="Maximum Dispatch Flow Fraction", sValue="1.2"),
        }
        model.read_parameters()
        adapter = CylindricalDispatchPlantAdapter()
        adapter.initialize(model, design_state={})
        return model, adapter

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
        self.assertEqual(1.0, model.dispatch_results.summary_metrics["dispatch_analysis_start_year"])
        self.assertEqual(2.0, model.dispatch_results.summary_metrics["dispatch_analysis_end_year"])
        self.assertEqual(1.0, model.dispatch_results.summary_metrics["dispatch_analysis_year_count"])
        self.assertGreater(model.dispatch_results.summary_metrics["design_heat_extracted_mw"], 0.0)
        self.assertGreater(model.dispatch_results.summary_metrics["annual_served_heat_kwh"], 0.0)
        self.assertGreater(model.dispatch_results.summary_metrics["peak_hourly_demand_mw"], 0.0)
        self.assertGreaterEqual(model.economics.LCOH.value, 0.0)
        self.assertEqual(8760, model.economics.timestepsperyear.value)

    def test_dispatchable_analysis_window_can_target_later_operating_years(self):
        from geophires_x.CylindricalReservoir import CylindricalReservoir

        model = self._new_model()
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        model.reserv = CylindricalReservoir(model)
        model.InputParameters = {
            "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
            "End-Use Option": ParameterEntry(Name="End-Use Option", sValue="2"),
            "Plant Lifetime": ParameterEntry(Name="Plant Lifetime", sValue="5"),
            "Reservoir Model": ParameterEntry(Name="Reservoir Model", sValue="0"),
            "Power Plant Type": ParameterEntry(Name="Power Plant Type", sValue="9"),
            "Number of Multilateral Sections": ParameterEntry(Name="Number of Multilateral Sections", sValue="1"),
            "Maximum Dispatch Flow Fraction": ParameterEntry(Name="Maximum Dispatch Flow Fraction", sValue="1.2"),
            "Dispatch Analysis Start Year": ParameterEntry(Name="Dispatch Analysis Start Year", sValue="3"),
            "Dispatch Analysis End Year": ParameterEntry(Name="Dispatch Analysis End Year", sValue="5"),
            "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=csv_file),
        }

        model.read_parameters()
        model.Calculate()

        self.assertEqual(8760 * 2, len(model.dispatch_results.hourly_produced_temperature))
        self.assertEqual(3, model.dispatch_results.analysis_start_year)
        self.assertEqual(5, model.dispatch_results.analysis_end_year)
        self.assertEqual((3 - 1) * 8760 + 1, model.dispatch_results.simulation_start_hour)
        self.assertEqual([3, 4], model.dispatch_results.annual_aggregates["analysis_years"])
        self.assertEqual(3.0, model.dispatch_results.summary_metrics["dispatch_analysis_start_year"])
        self.assertEqual(5.0, model.dispatch_results.summary_metrics["dispatch_analysis_end_year"])
        self.assertEqual(2.0, model.dispatch_results.summary_metrics["dispatch_analysis_year_count"])

    def test_cylindrical_recovery_restores_state_during_shut_in_period(self):
        model, recovering_adapter = self._new_cylindrical_dispatch_adapter()

        dispatch_strategy = DemandFollowingDispatchStrategy()
        _, nonrecovering_adapter = self._new_cylindrical_dispatch_adapter()
        nonrecovering_adapter._recovery_model = NoRecoveryModel()

        def run_cycle(adapter):
            hour_index = 0

            def step(demand_mw: float) -> None:
                nonlocal hour_index
                nominal_state = adapter.thermal_state_for_flow_fraction(1.0)
                dispatch_command = dispatch_strategy.dispatch(
                    {
                        "nominal_output_mw": nominal_state["dispatch_output_mw"],
                        "maximum_dispatch_flow_fraction": model.surfaceplant.maximum_dispatch_flow_fraction.value,
                        "minimum_dispatch_flow_fraction": model.surfaceplant.minimum_dispatch_flow_fraction.value,
                        "minimum_dispatch_runtime_fraction": model.surfaceplant.minimum_dispatch_runtime_fraction.value,
                    },
                    demand_mw,
                )
                adapter.evaluate_timestep(dispatch_command, hour_index)
                hour_index += 1

            for _ in range(720):
                step(20.0)
            depleted_heat_content = adapter._remaining_heat_content_pj
            for _ in range(720):
                step(0.0)

            return depleted_heat_content, adapter._remaining_heat_content_pj, adapter._current_reservoir_temperature()

        recovering_depleted, recovering_recovered, recovering_temperature = run_cycle(recovering_adapter)
        nonrecovering_depleted, nonrecovering_recovered, nonrecovering_temperature = run_cycle(nonrecovering_adapter)

        self.assertAlmostEqual(recovering_depleted, nonrecovering_depleted, places=6)
        self.assertGreater(recovering_recovered, nonrecovering_recovered)
        self.assertGreater(recovering_temperature, nonrecovering_temperature)

    def test_cylindrical_recovery_improves_served_energy_when_post_shut_in_demand_is_supply_limited(self):
        model, recovering_adapter = self._new_cylindrical_dispatch_adapter()
        _, nonrecovering_adapter = self._new_cylindrical_dispatch_adapter()
        nonrecovering_adapter._recovery_model = NoRecoveryModel()
        dispatch_strategy = DemandFollowingDispatchStrategy()

        def run_cycle(adapter):
            served_heat = 0.0
            hour_index = 0

            def step(demand_mw: float) -> None:
                nonlocal served_heat, hour_index
                nominal_state = adapter.thermal_state_for_flow_fraction(1.0)
                dispatch_command = dispatch_strategy.dispatch(
                    {
                        "nominal_output_mw": nominal_state["dispatch_output_mw"],
                        "maximum_dispatch_flow_fraction": model.surfaceplant.maximum_dispatch_flow_fraction.value,
                        "minimum_dispatch_flow_fraction": model.surfaceplant.minimum_dispatch_flow_fraction.value,
                        "minimum_dispatch_runtime_fraction": model.surfaceplant.minimum_dispatch_runtime_fraction.value,
                    },
                    demand_mw,
                )
                result = adapter.evaluate_timestep(dispatch_command, hour_index)
                served_heat += result.served_demand
                hour_index += 1

            for _ in range(720):
                step(40.0)
            for _ in range(1440):
                step(0.0)
            for _ in range(720):
                step(40.0)

            return served_heat

        self.assertGreater(run_cycle(recovering_adapter), run_cycle(nonrecovering_adapter))

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

    def test_dispatchable_electricity_run_populates_dispatch_results_and_economics(self):
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        model = self._new_model(input_file=str(Path(__file__).resolve().parents[1] / "examples" / "example1.txt"))
        model.InputParameters.update(
            {
                "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
                "Dispatch Demand Source": ParameterEntry(
                    Name="Dispatch Demand Source", sValue="Annual Electricity Demand"
                ),
                "Dispatch Flow Strategy": ParameterEntry(Name="Dispatch Flow Strategy", sValue="Demand Following"),
                "Plant Lifetime": ParameterEntry(Name="Plant Lifetime", sValue="1"),
                "Annual Electricity Demand": ParameterEntry(Name="Annual Electricity Demand", sValue=csv_file),
            }
        )

        model.read_parameters()
        model.Calculate()

        self.assertEqual("electric", model.dispatch_results.demand_type)
        self.assertEqual(8760, len(model.surfaceplant.NetElectricityProduced.value))
        self.assertGreater(model.dispatch_results.summary_metrics["design_net_electricity_produced_mw"], 0.0)
        self.assertGreater(model.dispatch_results.summary_metrics["annual_served_electricity_kwh"], 0.0)
        self.assertGreaterEqual(model.economics.LCOE.value, 0.0)

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

    def test_dispatchable_sbt_uloop_run(self):
        example_file = str(Path(__file__).resolve().parents[1] / "examples" / "example_SBT_ULoop.txt")
        model = self._new_model(example_file)
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        model.InputParameters["Operating Mode"] = ParameterEntry(Name="Operating Mode", sValue="Dispatchable")
        model.InputParameters["End-Use Option"] = ParameterEntry(Name="End-Use Option", sValue="2")
        model.InputParameters["Plant Lifetime"] = ParameterEntry(Name="Plant Lifetime", sValue="1")
        model.InputParameters["Economic Model"] = ParameterEntry(Name="Economic Model", sValue="2")
        model.InputParameters["Power Plant Type"] = ParameterEntry(Name="Power Plant Type", sValue="9")
        model.InputParameters["Maximum Dispatch Flow Fraction"] = ParameterEntry(
            Name="Maximum Dispatch Flow Fraction", sValue="1.05"
        )
        model.InputParameters["Annual Heat Demand"] = ParameterEntry(Name="Annual Heat Demand", sValue=csv_file)
        model.InputParameters["SBT Generate Wireframe Graphics"] = ParameterEntry(
            Name="SBT Generate Wireframe Graphics", sValue="False"
        )

        model.read_parameters()
        model.Calculate()

        self.assertEqual("SBTDispatchPlantAdapter", type(model.dispatch_adapter).__name__)
        self.assertEqual(8760, len(model.dispatch_results.hourly_produced_temperature))
        self.assertEqual(8760, len(model.surfaceplant.HeatProduced.value))
        self.assertGreater(model.dispatch_results.summary_metrics["design_heat_extracted_mw"], 0.0)
        self.assertGreater(model.dispatch_results.summary_metrics["annual_served_heat_kwh"], 0.0)
        self.assertGreater(model.dispatch_results.summary_metrics["peak_hourly_demand_mw"], 0.0)
        self.assertGreaterEqual(model.economics.LCOH.value, 0.0)
