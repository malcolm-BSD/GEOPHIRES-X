import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from geophires_x.Dispatch import CylindricalDispatchPlantAdapter
from geophires_x.Dispatch import DemandFollowingDispatchStrategy
from geophires_x.Dispatch import DemandProfileFactory
from geophires_x.Dispatch import NoRecoveryModel
from geophires_x.Dispatch import ReducedOrderRecoveryModel
from geophires_x.Model import Model
from geophires_x.OptionList import DispatchDemandSource
from geophires_x.OptionList import DispatchFlowStrategy
from geophires_x.OptionList import OperatingMode
from geophires_x.OptionList import TESSChargeControlStrategy
from geophires_x.OptionList import TESSPressureMode
from geophires_x.OptionList import TESSWorkingFluid
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

    def test_tess_defaults_to_disabled(self):
        model = self._new_model()

        model.surfaceplant.read_parameters(model)

        self.assertFalse(model.surfaceplant.tess_enabled.value)
        self.assertEqual(TESSWorkingFluid.WATER, model.surfaceplant.tess_working_fluid.value)
        self.assertEqual(TESSPressureMode.AUTO, model.surfaceplant.tess_pressure_mode.value)
        self.assertEqual(
            TESSChargeControlStrategy.TEMPERATURE_BAND,
            model.surfaceplant.tess_charge_control_strategy.value,
        )

    def test_tess_parameter_parsing(self):
        model = self._new_model()
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        model.InputParameters = {
            "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
            "End-Use Option": ParameterEntry(Name="End-Use Option", sValue="2"),
            "TESS Enabled": ParameterEntry(Name="TESS Enabled", sValue="True"),
            "TESS Volume": ParameterEntry(Name="TESS Volume", sValue="5000"),
            "TESS Cost per Cubic Meter": ParameterEntry(Name="TESS Cost per Cubic Meter", sValue="750"),
            "TESS Deadband Range": ParameterEntry(Name="TESS Deadband Range", sValue="12"),
            "TESS Charge Control Strategy": ParameterEntry(
                Name="TESS Charge Control Strategy", sValue="Moving Average"
            ),
            "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=csv_file),
        }

        model.surfaceplant.read_parameters(model)

        self.assertTrue(model.surfaceplant.tess_enabled.value)
        self.assertEqual(5000.0, model.surfaceplant.tess_volume.value)
        self.assertEqual(750.0, model.surfaceplant.tess_cost_per_cubic_meter.value)
        self.assertEqual(12.0, model.surfaceplant.tess_deadband_range.value)
        self.assertEqual(
            TESSChargeControlStrategy.MOVING_AVERAGE,
            model.surfaceplant.tess_charge_control_strategy.value,
        )

    def test_tess_default_maximum_discharge_power_autosizes_from_peak_heat_demand(self):
        model = self._new_model()
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        model.InputParameters = {
            "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
            "End-Use Option": ParameterEntry(Name="End-Use Option", sValue="2"),
            "TESS Enabled": ParameterEntry(Name="TESS Enabled", sValue="True"),
            "TESS Subhourly Demand Peak Multiplier": ParameterEntry(
                Name="TESS Subhourly Demand Peak Multiplier", sValue="1.5"
            ),
            "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=csv_file),
        }

        model.surfaceplant.read_parameters(model)

        demand_profile = np.asarray(model.surfaceplant.HeatingDemand.value, dtype=float)
        heat_demand_kwh = demand_profile[:, 1] if demand_profile.ndim == 2 else demand_profile
        expected_maximum_discharge_power_mw = float(np.max(heat_demand_kwh) / 1000.0 * 1.5)
        self.assertAlmostEqual(
            expected_maximum_discharge_power_mw,
            model.surfaceplant.tess_maximum_discharge_power.value,
            places=6,
        )

    def test_tess_rejects_invalid_temperature_bounds(self):
        model = self._new_model()
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        model.InputParameters = {
            "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
            "End-Use Option": ParameterEntry(Name="End-Use Option", sValue="2"),
            "TESS Enabled": ParameterEntry(Name="TESS Enabled", sValue="True"),
            "TESS Minimum Useful Temperature": ParameterEntry(Name="TESS Minimum Useful Temperature", sValue="150"),
            "TESS Maximum Temperature": ParameterEntry(Name="TESS Maximum Temperature", sValue="140"),
            "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=csv_file),
        }

        with self.assertRaisesRegex(ValueError, "TESS Maximum Temperature"):
            model.surfaceplant.read_parameters(model)

    def test_tess_rejects_user_pressure_below_liquid_requirement(self):
        model = self._new_model()
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        model.InputParameters = {
            "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
            "End-Use Option": ParameterEntry(Name="End-Use Option", sValue="2"),
            "TESS Enabled": ParameterEntry(Name="TESS Enabled", sValue="True"),
            "TESS Pressure Mode": ParameterEntry(Name="TESS Pressure Mode", sValue="User Specified"),
            "TESS Pressure": ParameterEntry(Name="TESS Pressure", sValue="0.1"),
            "TESS Maximum Temperature": ParameterEntry(Name="TESS Maximum Temperature", sValue="180"),
            "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=csv_file),
        }

        with self.assertRaisesRegex(ValueError, "TESS Pressure"):
            model.surfaceplant.read_parameters(model)

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

    def _run_direct_use_cylindrical_dispatch(self, overrides: Optional[dict[str, str]] = None) -> Model:
        from geophires_x.CylindricalReservoir import CylindricalReservoir

        model = self._new_model()
        model.reserv = CylindricalReservoir(model)
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        input_values = {
            "Operating Mode": "Dispatchable",
            "End-Use Option": "2",
            "Plant Lifetime": "1",
            "Reservoir Model": "0",
            "Power Plant Type": "9",
            "Number of Multilateral Sections": "1",
            "Maximum Dispatch Flow Fraction": "1.2",
            "Annual Heat Demand": csv_file,
        }
        if overrides is not None:
            input_values.update(overrides)

        model.InputParameters = {name: ParameterEntry(Name=name, sValue=value) for name, value in input_values.items()}
        model.read_parameters()
        model.Calculate()
        return model

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

    def test_dispatchable_tess_disabled_matches_legacy_dispatch(self):
        legacy_model = self._run_direct_use_cylindrical_dispatch()
        disabled_tess_model = self._run_direct_use_cylindrical_dispatch({"TESS Enabled": "False"})

        np.testing.assert_allclose(
            legacy_model.dispatch_results.hourly_demand_served,
            disabled_tess_model.dispatch_results.hourly_demand_served,
        )
        np.testing.assert_allclose(
            legacy_model.dispatch_results.hourly_geothermal_thermal_output,
            disabled_tess_model.dispatch_results.hourly_geothermal_thermal_output,
        )
        np.testing.assert_allclose(
            legacy_model.dispatch_results.hourly_unmet_demand,
            disabled_tess_model.dispatch_results.hourly_unmet_demand,
        )
        np.testing.assert_allclose(
            legacy_model.dispatch_results.hourly_flow,
            disabled_tess_model.dispatch_results.hourly_flow,
        )
        np.testing.assert_allclose(
            legacy_model.dispatch_results.hourly_runtime_fraction,
            disabled_tess_model.dispatch_results.hourly_runtime_fraction,
        )
        np.testing.assert_allclose(
            legacy_model.dispatch_results.hourly_pumping_power,
            disabled_tess_model.dispatch_results.hourly_pumping_power,
        )
        np.testing.assert_allclose(
            legacy_model.surfaceplant.HeatkWhProduced.value,
            disabled_tess_model.surfaceplant.HeatkWhProduced.value,
        )
        np.testing.assert_allclose(
            legacy_model.surfaceplant.PumpingkWh.value,
            disabled_tess_model.surfaceplant.PumpingkWh.value,
        )
        self.assertEqual(legacy_model.economics.CCap.value, disabled_tess_model.economics.CCap.value)
        self.assertEqual(legacy_model.economics.Coam.value, disabled_tess_model.economics.Coam.value)
        self.assertEqual(legacy_model.economics.LCOH.value, disabled_tess_model.economics.LCOH.value)
        self.assertEqual(0.0, disabled_tess_model.economics.tess_capital_cost.value)
        self.assertEqual(0.0, disabled_tess_model.economics.tess_o_and_m_cost.value)
        self.assertEqual(0.0, float(np.sum(disabled_tess_model.dispatch_results.hourly_tess_discharge_to_load)))
        self.assertNotIn("tess_enabled", disabled_tess_model.dispatch_results.summary_metrics)

    def test_dispatchable_tess_costs_feed_economics_when_enabled(self):
        volume_m3 = 2000.0
        cost_per_m3 = 750.0
        fixed_om_fraction = 0.02
        no_cost_model = self._run_direct_use_cylindrical_dispatch(
            {
                "TESS Enabled": "True",
                "TESS Volume": f"{volume_m3}",
                "TESS Cost per Cubic Meter": "0",
                "TESS Fixed O&M Fraction": "0",
            }
        )
        costed_model = self._run_direct_use_cylindrical_dispatch(
            {
                "TESS Enabled": "True",
                "TESS Volume": f"{volume_m3}",
                "TESS Cost per Cubic Meter": f"{cost_per_m3}",
                "TESS Fixed O&M Fraction": f"{fixed_om_fraction}",
            }
        )

        expected_capex_musd = volume_m3 * cost_per_m3 / 1.0e6
        expected_om_musd_per_year = expected_capex_musd * fixed_om_fraction

        self.assertAlmostEqual(expected_capex_musd, costed_model.economics.tess_capital_cost.value)
        self.assertAlmostEqual(expected_om_musd_per_year, costed_model.economics.tess_o_and_m_cost.value)
        self.assertAlmostEqual(
            expected_capex_musd,
            costed_model.economics.CCap.value - no_cost_model.economics.CCap.value,
        )
        self.assertAlmostEqual(
            expected_om_musd_per_year,
            costed_model.economics.Coam.value - no_cost_model.economics.Coam.value,
        )
        self.assertGreater(costed_model.economics.LCOH.value, no_cost_model.economics.LCOH.value)

    def test_dispatchable_tess_costs_respect_total_cost_overrides(self):
        model = self._run_direct_use_cylindrical_dispatch(
            {
                "TESS Enabled": "True",
                "TESS Volume": "2000",
                "TESS Cost per Cubic Meter": "750",
                "TESS Fixed O&M Fraction": "0.02",
                "Total Capital Cost": "123",
                "Total O&M Cost": "7",
            }
        )

        self.assertAlmostEqual(1.5, model.economics.tess_capital_cost.value)
        self.assertAlmostEqual(0.03, model.economics.tess_o_and_m_cost.value)
        self.assertAlmostEqual(123.0, model.economics.CCap.value)
        self.assertAlmostEqual(7.0, model.economics.Coam.value)

    def test_dispatchable_tess_serves_demand_from_initial_storage(self):
        model = self._run_direct_use_cylindrical_dispatch(
            {
                "TESS Enabled": "True",
                "TESS Volume": "3000000",
                "TESS Initial Temperature": "160",
                "TESS Daily Heat Loss Fraction": "0",
                "TESS Charge Flow Fraction": "0",
            }
        )

        self.assertEqual(0.0, float(np.sum(model.dispatch_results.hourly_unmet_demand)))
        self.assertEqual(0.0, float(np.max(model.dispatch_results.hourly_flow)))
        self.assertEqual(0.0, float(np.sum(model.dispatch_results.hourly_tess_charge_from_geothermal)))
        np.testing.assert_allclose(
            model.dispatch_results.hourly_demand_served / 1000.0,
            model.dispatch_results.hourly_tess_discharge_to_load,
        )
        self.assertGreater(model.dispatch_results.summary_metrics["annual_tess_served_heat_kwh"], 0.0)
        self.assertAlmostEqual(
            model.dispatch_results.summary_metrics["annual_served_heat_kwh"],
            model.dispatch_results.summary_metrics["annual_tess_served_heat_kwh"],
        )

    def test_dispatchable_tess_temperature_band_charges_and_shuts_off(self):
        model = self._run_direct_use_cylindrical_dispatch(
            {
                "TESS Enabled": "True",
                "TESS Volume": "10000",
                "TESS Daily Heat Loss Fraction": "0",
            }
        )

        charge_command = model.dispatch_results.hourly_geothermal_charge_command
        tess_temperature = model.dispatch_results.hourly_tess_temperature
        geothermal_output_kwh = model.dispatch_results.hourly_geothermal_thermal_output * 1000.0

        self.assertGreater(int(np.count_nonzero(charge_command)), 0)
        self.assertGreater(int(np.count_nonzero(charge_command == 0.0)), 0)
        self.assertTrue(np.any((tess_temperature >= 155.0) & (charge_command == 0.0)))
        self.assertFalse(np.allclose(geothermal_output_kwh, model.dispatch_results.hourly_demand_served))
        self.assertGreater(model.dispatch_results.summary_metrics["tess_annual_charge_kwh"], 0.0)
        self.assertGreater(model.dispatch_results.summary_metrics["peak_geothermal_charge_mw"], 0.0)

    def test_dispatchable_tess_reports_unmet_demand_when_discharge_limited(self):
        model = self._run_direct_use_cylindrical_dispatch(
            {
                "TESS Enabled": "True",
                "TESS Volume": "3000000",
                "TESS Initial Temperature": "160",
                "TESS Daily Heat Loss Fraction": "0",
                "TESS Charge Flow Fraction": "0",
                "TESS Maximum Discharge Power": "1",
            }
        )

        self.assertGreater(float(np.sum(model.dispatch_results.hourly_unmet_demand)), 0.0)
        self.assertAlmostEqual(1000.0, float(np.max(model.dispatch_results.hourly_demand_served)))
        self.assertGreater(model.dispatch_results.summary_metrics["peak_unmet_heat_kwh"], 0.0)

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

    def test_dispatchable_chp_heat_following_run_populates_both_heat_and_electric_outputs(self):
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        model = self._new_model(input_file=str(Path(__file__).resolve().parents[1] / "examples" / "example1.txt"))
        model.InputParameters.update(
            {
                "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
                "End-Use Option": ParameterEntry(Name="End-Use Option", sValue="31"),
                "Dispatch Demand Source": ParameterEntry(Name="Dispatch Demand Source", sValue="Annual Heat Demand"),
                "Dispatch Flow Strategy": ParameterEntry(Name="Dispatch Flow Strategy", sValue="Demand Following"),
                "Plant Lifetime": ParameterEntry(Name="Plant Lifetime", sValue="1"),
                "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=csv_file),
            }
        )

        model.read_parameters()
        model.Calculate()

        self.assertEqual("thermal", model.dispatch_results.demand_type)
        self.assertGreater(model.dispatch_results.summary_metrics["annual_served_heat_kwh"], 0.0)
        self.assertGreater(model.dispatch_results.summary_metrics["annual_served_electricity_kwh"], 0.0)
        self.assertGreater(model.dispatch_results.summary_metrics["design_net_electricity_produced_mw"], 0.0)
        self.assertGreater(model.surfaceplant.HeatkWhProduced.value[0], 0.0)
        self.assertGreater(model.surfaceplant.NetkWhProduced.value[0], 0.0)
        self.assertGreaterEqual(model.economics.LCOH.value, 0.0)
        self.assertGreaterEqual(model.economics.LCOE.value, 0.0)

    def test_dispatchable_chp_electricity_following_run_populates_both_heat_and_electric_outputs(self):
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        model = self._new_model(input_file=str(Path(__file__).resolve().parents[1] / "examples" / "example1.txt"))
        model.InputParameters.update(
            {
                "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
                "End-Use Option": ParameterEntry(Name="End-Use Option", sValue="52"),
                "Dispatch Demand Source": ParameterEntry(
                    Name="Dispatch Demand Source", sValue="Annual Electricity Demand"
                ),
                "Dispatch Flow Strategy": ParameterEntry(Name="Dispatch Flow Strategy", sValue="Demand Following"),
                "Plant Lifetime": ParameterEntry(Name="Plant Lifetime", sValue="1"),
                "Annual Electricity Demand": ParameterEntry(Name="Annual Electricity Demand", sValue=csv_file),
                "CHP Fraction": ParameterEntry(Name="CHP Fraction", sValue="0.4"),
            }
        )

        model.read_parameters()
        model.Calculate()

        self.assertEqual("electric", model.dispatch_results.demand_type)
        self.assertGreater(model.dispatch_results.summary_metrics["annual_served_electricity_kwh"], 0.0)
        self.assertGreater(model.dispatch_results.summary_metrics["annual_served_heat_kwh"], 0.0)
        self.assertGreater(model.dispatch_results.summary_metrics["design_heat_produced_mw"], 0.0)
        self.assertGreater(model.surfaceplant.NetkWhProduced.value[0], 0.0)
        self.assertGreater(model.surfaceplant.HeatkWhProduced.value[0], 0.0)
        self.assertGreaterEqual(model.economics.LCOH.value, 0.0)
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
