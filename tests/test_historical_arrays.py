import os
import sys
import unittest
from pathlib import Path

import numpy as np

from geophires_x.Model import Model
from geophires_x.Parameter import ParameterEntry
from geophires_x.Parameter import ReadParameter
from geophires_x.Parameter import TimeSeriesParameter
from geophires_x.Parameter import floatParameter
from geophires_x.Units import EnergyUnit
from geophires_x.Units import TemperatureUnit
from geophires_x.Units import Units


class HistoricalArrayTestCase(unittest.TestCase):
    def _new_model(self):
        stash_cwd = Path.cwd()
        stash_sys_argv = sys.argv
        sys.argv = [""]
        m = Model(enable_geophires_logging_config=False)
        sys.argv = stash_sys_argv
        os.chdir(stash_cwd)
        return m

    def test_load_time_temperature_from_file(self):
        model = self._new_model()
        param = TimeSeriesParameter(
            Name="Ambient Temperature",
            DefaultValue=[12, 15, 30],
            Min=-50,
            Max=50,
            UnitType=Units.TEMPERATURE,
            PreferredUnits=TemperatureUnit.CELSIUS,
            CurrentUnits=TemperatureUnit.CELSIUS,
            AllowExtendedInput=True,
            PreferredXUnits="hour",
            PreferredYUnits="celsius",
            ResampleToHourlyYear=True,
        )
        root = str(Path(__file__).resolve().parents[0])  # adjust depth as needed
        csv_file = root + "/assets/params/ambient_temperature.csv"

        ReadParameter(
            ParameterEntry(Name="Ambient Temperature", sValue=csv_file, raw_entry=f"Ambient Temperature, {csv_file}"),
            param,
            model,
        )

        self.assertEqual(8760, len(param.value))
        self.assertEqual(8760, len(param.value))
        self.assertEqual("hr", param.CurrentXUnits)
        self.assertEqual("degC", param.CurrentYUnits)
        self.assertEqual(param.value[0][0], 0.0)
        self.assertEqual(param.value[0][1], 15.600000000000023)

    def test_load_time_cooling_demand_from_file(self):
        model = self._new_model()
        param = TimeSeriesParameter(
            Name="Cooling Demand",
            DefaultValue=[],
            YMin=-50,
            YMax=50,
            UnitType=Units.ENERGY,
            PreferredUnits=EnergyUnit.KWH,
            AllowExtendedInput=True,
            PreferredXUnits="hr",
            PreferredYUnits="kWh",
            ResampleToHourlyYear=True,
        )
        root = str(Path(__file__).resolve().parents[0])  # adjust depth as needed
        csv_file = root + "/assets/params/annual_cooling_demand.csv"

        ReadParameter(
            ParameterEntry(
                Name="Cooling Demand",
                sValue=csv_file,
                raw_entry=f"Cooling Demand, {csv_file}",
            ),
            param,
            model,
        )

        self.assertEqual(8760, len(param.value))
        self.assertEqual(8760, len(param.value))
        self.assertEqual(param.PreferredXUnits, param.CurrentXUnits)
        self.assertEqual(param.PreferredYUnits, param.CurrentYUnits)
        self.assertEqual(param.value[0][0], 0.0)
        self.assertEqual(param.value[0][1], 15239.69777777778)

    def test_load_time_temperature_from_downsampled_file(self):
        model = self._new_model()
        param = TimeSeriesParameter(
            Name="Annual Cooling Demand Downsampled",
            DefaultValue=[],
            YMin=-50,
            YMax=50,
            UnitType=Units.ENERGY,
            PreferredUnits=EnergyUnit.KWH,
            AllowExtendedInput=True,
            PreferredXUnits="hr",
            PreferredYUnits="kWh",
            ResampleToHourlyYear=True,
        )
        root = str(Path(__file__).resolve().parents[0])  # adjust depth as needed
        csv_file = root + "/assets/params/annual_cooling_demand_downsampled.csv"

        ReadParameter(
            ParameterEntry(
                Name="Cooling Demand Downsampled",
                sValue=csv_file,
                raw_entry=f"Cooling Demand Downsampled, {csv_file}",
            ),
            param,
            model,
        )

        self.assertEqual(8760, len(param.value))
        self.assertEqual(8760, len(param.value))
        self.assertEqual("hr", param.CurrentXUnits.value)
        self.assertEqual("kWh", param.CurrentYUnits.value)
        self.assertEqual(param.value[1][0], 1.0)
        self.assertEqual(param.value[1][1], 14653.555555555557)

    def test_load_time_temperature_from_daily_file(self):
        model = self._new_model()
        param = TimeSeriesParameter(
            Name="Annual Cooling Demand Downsampled to Daily",
            DefaultValue=[],
            YMin=-50,
            YMax=50,
            UnitType=Units.ENERGY,
            PreferredUnits=EnergyUnit.KWH,
            AllowExtendedInput=True,
            PreferredXUnits="hr",
            PreferredYUnits="kWh",
            ResampleToHourlyYear=True,
        )
        root = str(Path(__file__).resolve().parents[0])  # adjust depth as needed
        csv_file = root + "/assets/params/annual_cooling_demand_daily.csv"

        ReadParameter(
            ParameterEntry(
                Name="Cooling Demand Daily",
                sValue=csv_file,
                raw_entry=f"Cooling Demand Daily, {csv_file}",
            ),
            param,
            model,
        )

        self.assertEqual(8760, len(param.value))
        self.assertEqual(8760, len(param.value))
        self.assertEqual("hr", param.CurrentXUnits.value)
        self.assertEqual("kWh", param.CurrentYUnits.value)
        self.assertEqual(param.value[1][0], 1.0)
        self.assertEqual(param.value[1][1], 485875.2683333333)

    def test_load_time_temperature_from_weekly_file(self):
        model = self._new_model()
        param = TimeSeriesParameter(
            Name="Annual Cooling Demand Downsampled to Weekly",
            DefaultValue=[],
            YMin=-50,
            YMax=50,
            UnitType=Units.ENERGY,
            PreferredUnits=EnergyUnit.KWH,
            AllowExtendedInput=True,
            PreferredXUnits="hr",
            PreferredYUnits="kWh",
            ResampleToHourlyYear=True,
        )
        root = str(Path(__file__).resolve().parents[0])  # adjust depth as needed
        csv_file = root + "/assets/params/annual_cooling_demand_weekly.csv"

        ReadParameter(
            ParameterEntry(
                Name="Cooling Demand Weekly",
                sValue=csv_file,
                raw_entry=f"Cooling Demand Weekly, {csv_file}",
            ),
            param,
            model,
        )

        self.assertEqual(8760, len(param.value))
        self.assertEqual(8760, len(param.value))
        self.assertEqual("hr", param.CurrentXUnits.value)
        self.assertEqual("kWh", param.CurrentYUnits.value)
        self.assertEqual(param.value[1][0], 1.0)
        self.assertEqual(param.value[1][1], 2972133.572619048)

    def test_load_time_temperature_from_monthly_file(self):
        model = self._new_model()
        param = TimeSeriesParameter(
            Name="Annual Cooling Demand Downsampled to Monthly",
            DefaultValue=[],
            YMin=-50,
            YMax=50,
            UnitType=Units.ENERGY,
            PreferredUnits=EnergyUnit.KWH,
            AllowExtendedInput=True,
            PreferredXUnits="hr",
            PreferredYUnits="kWh",
            ResampleToHourlyYear=True,
        )
        root = str(Path(__file__).resolve().parents[0])  # adjust depth as needed
        csv_file = root + "/assets/params/annual_cooling_demand_monthly.csv"

        ReadParameter(
            ParameterEntry(
                Name="Cooling Demand Monthly",
                sValue=csv_file,
                raw_entry=f"Cooling Demand Monthly, {csv_file}",
            ),
            param,
            model,
        )

        self.assertEqual(8760, len(param.value))
        self.assertEqual(8760, len(param.value))
        self.assertEqual("hr", param.CurrentXUnits.value)
        self.assertEqual("kWh", param.CurrentYUnits.value)
        self.assertEqual(param.value[1][0], 1.0)
        self.assertEqual(param.value[1][1], 9954261.032083334)

    def test_read_parameter_historical_array_sets_series_and_scalar(self):
        model = self._new_model()
        param = TimeSeriesParameter(
            Name="Ambient Temperature",
            DefaultValue=[12, 15, 30],
            Min=-50,
            Max=50,
            UnitType=Units.TEMPERATURE,
            PreferredUnits=TemperatureUnit.CELSIUS,
            CurrentUnits=TemperatureUnit.CELSIUS,
            AllowExtendedInput=True,
            PreferredXUnits="hour",
            PreferredYUnits="degC",
            ResampleToHourlyYear=True,
        )

        entry = ParameterEntry(
            Name="Ambient Temperature",
            sValue="Time (hours), Temperature (Celsius)\n0,20\n2,22\n",
            raw_entry="Ambient Temperature, Time (hours), Temperature (Celsius)\n0,20\n2,22\n",
        )
        ReadParameter(entry, param, model)

        self.assertTrue(param.Provided)
        self.assertTrue(param.Valid)
        self.assertEqual(21.0, param.value[1][1])

    def test_read_parameter_historical_array_allows_scalar_with_units(self):
        model = self._new_model()
        param = floatParameter(
            Name="Ambient Temperature",
            DefaultValue=15.0,
            Min=-50,
            Max=50,
            UnitType=Units.TEMPERATURE,
            PreferredUnits=TemperatureUnit.CELSIUS,
            CurrentUnits=TemperatureUnit.CELSIUS,
        )
        param.AllowHistoricalArrayInput = True
        param.HistoricalXDimension = "time"
        param.HistoricalYDimension = "temperature"
        param.HistoricalDefaultXUnits = "hour"
        param.HistoricalDefaultYUnits = "celsius"
        param.HistoricalResampleToHourlyYear = True

        entry = ParameterEntry(
            Name="Ambient Temperature",
            sValue="10 degC",
            raw_entry="Ambient Temperature, 10 degC",
        )
        ReadParameter(entry, param, model)

        self.assertTrue(param.Provided)
        self.assertTrue(param.Valid)
        self.assertAlmostEqual(10.0, param.value, places=6)

    def test_read_parameter_allows_pair_vector_csv_file_for_float_parameter(self):
        model = self._new_model()
        param = floatParameter(
            Name="Ambient Temperature",
            DefaultValue=150.0,
            Min=0.0,
            Max=500.0,
            PreferredUnits=TemperatureUnit.CELSIUS,
            AllowExtendedInput=True,
        )
        param.AllowPairVectorInput = True

        root = str(Path(__file__).resolve().parents[0])  # adjust depth as needed
        csv_file = root + "/assets/params/ambient_temperature.csv"

        ReadParameter(
            ParameterEntry(Name="Ambient Temperature", sValue=csv_file, raw_entry=f"Ambient Temperature, {csv_file}"),
            param,
            model,
        )

        self.assertIsInstance(param.value, list)
        self.assertTrue(
            np.array_equal(
                param.value[0:2],
                np.array(
                    [
                        [
                            "Time (hour)",
                            "Temperature (degF)",
                            "datetime_local",
                            "Tdb_F",
                            "Tdp_F",
                            "Twb_F",
                            "RH_pct",
                            "W_kgkg",
                            "W_grains_lbda",
                            "h_kJ_kgda",
                            "h_Btu_lbda",
                            "rho_kgm3",
                            "rho_lbft3",
                            "P_pa",
                            "CDH65_Fhr",
                            "HDH65_Fhr",
                        ],
                        [
                            0,
                            60.08,
                            "2023-01-01 00:00:00-06:00",
                            60.08,
                            57.92,
                            58.73808439,
                            92.56272759,
                            0.010226531,
                            71.58571707,
                            41.56688708,
                            17.87054475,
                            1.203797811,
                            0.075150642,
                            101420,
                            0,
                            4.92,
                        ],
                    ]
                ),
            )
        )

    def test_read_parameter_allows_pair_vector_csv_url_for_float_parameter(self):
        model = self._new_model()
        param = floatParameter(
            Name="Reservoir Temperature",
            DefaultValue=150.0,
            Min=0.0,
            Max=500.0,
            PreferredUnits=TemperatureUnit.CELSIUS,
            AllowExtendedInput=True,
        )
        ReadParameter(
            ParameterEntry(
                Name="Ambient Temperature",
                sValue="https://raw.githubusercontent.com/malcolm-BSD/GEOPHIRES-X/test-assets-v1.1/tests/assets/params/ambient_temperature.csv",
                raw_entry="Ambient Temperature, https://raw.githubusercontent.com/malcolm-BSD/GEOPHIRES-X/test-assets-v1.1/tests/assets/params/ambient_temperature.csv",
            ),
            param,
            model,
        )

        self.assertIsInstance(param.value, list)
        self.assertTrue(
            np.array_equal(
                param.value[0:2],
                np.array(
                    [
                        [
                            "Time (hour)",
                            "Temperature (degF)",
                            "datetime_local",
                            "Tdb_F",
                            "Tdp_F",
                            "Twb_F",
                            "RH_pct",
                            "W_kgkg",
                            "W_grains_lbda",
                            "h_kJ_kgda",
                            "h_Btu_lbda",
                            "rho_kgm3",
                            "rho_lbft3",
                            "P_pa",
                            "CDH65_Fhr",
                            "HDH65_Fhr",
                        ],
                        [
                            0,
                            60.08,
                            "2023-01-01 00:00:00-06:00",
                            60.08,
                            57.92,
                            58.73808439,
                            92.56272759,
                            0.010226531,
                            71.58571707,
                            41.56688708,
                            17.87054475,
                            1.203797811,
                            0.075150642,
                            101420,
                            0,
                            4.92,
                        ],
                    ]
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
