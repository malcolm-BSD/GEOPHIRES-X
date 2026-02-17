import os
import sys
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from geophires_x.Model import Model
from geophires_x.Parameter import ParameterEntry
from geophires_x.Parameter import ReadParameter
from geophires_x.Parameter import floatParameter
from geophires_x.Parameter import listParameter
from geophires_x.Reservoir import _derive_numseg_from_gradient_thickness
from geophires_x.Units import TemperatureUnit
from geophires_x.Units import Units
from geophires_x.historical_arrays import detect_header_units
from geophires_x.historical_arrays import load_xy_series_from_source


class HistoricalArrayTestCase(unittest.TestCase):
    def _new_model(self):
        stash_cwd = Path.cwd()
        stash_sys_argv = sys.argv
        sys.argv = ['']
        m = Model(enable_geophires_logging_config=False)
        sys.argv = stash_sys_argv
        os.chdir(stash_cwd)
        return m

    def test_detect_header_units(self):
        x_u, y_u = detect_header_units('Time (minutes), Temperature (Fahrenheit)')
        self.assertEqual('minutes', x_u)
        self.assertEqual('Fahrenheit', y_u)

    def test_load_time_temperature_resamples_to_8760(self):
        content = 'Time (hours), Temperature (Celsius)\n0,10\n24,12\n48,11\n'
        with NamedTemporaryFile(mode='w+', suffix='.csv', delete=True) as f:
            f.write(content)
            f.flush()
            series = load_xy_series_from_source(
                f.name,
                x_dimension='time',
                y_dimension='temperature',
                default_x_units='hour',
                default_y_units='celsius',
                resample_to_hourly=True,
            )

        self.assertEqual(8760, len(series.x_canonical))
        self.assertEqual(8760, len(series.y_canonical))
        self.assertEqual('hour', series.x_units_canonical)
        self.assertEqual('degC', series.y_units_canonical)

    def test_read_parameter_historical_array_sets_series_and_scalar(self):
        model = self._new_model()
        param = floatParameter(
            Name='Ambient Temperature',
            DefaultValue=15.0,
            Min=-50,
            Max=50,
            UnitType=Units.TEMPERATURE,
            PreferredUnits=TemperatureUnit.CELSIUS,
            CurrentUnits=TemperatureUnit.CELSIUS,
        )
        param.AllowHistoricalArrayInput = True
        param.HistoricalXDimension = 'time'
        param.HistoricalYDimension = 'temperature'
        param.HistoricalDefaultXUnits = 'hour'
        param.HistoricalDefaultYUnits = 'celsius'
        param.HistoricalResampleToHourlyYear = True

        entry = ParameterEntry(
            Name='Ambient Temperature',
            sValue='Time (hours), Temperature (Celsius)\n0,20\n2,22\n',
            raw_entry='Ambient Temperature, Time (hours), Temperature (Celsius)\n0,20\n2,22\n',
        )
        ReadParameter(entry, param, model)

        self.assertTrue(param.Provided)
        self.assertTrue(param.Valid)
        self.assertIsNotNone(param.HistoricalData)
        self.assertEqual(8760, len(param.HistoricalData.y_canonical))
        self.assertAlmostEqual(20.0, param.value, places=6)


    def test_read_parameter_historical_array_allows_scalar_with_units(self):
        model = self._new_model()
        param = floatParameter(
            Name='Ambient Temperature',
            DefaultValue=15.0,
            Min=-50,
            Max=50,
            UnitType=Units.TEMPERATURE,
            PreferredUnits=TemperatureUnit.CELSIUS,
            CurrentUnits=TemperatureUnit.CELSIUS,
        )
        param.AllowHistoricalArrayInput = True
        param.HistoricalXDimension = 'time'
        param.HistoricalYDimension = 'temperature'
        param.HistoricalDefaultXUnits = 'hour'
        param.HistoricalDefaultYUnits = 'celsius'
        param.HistoricalResampleToHourlyYear = True

        entry = ParameterEntry(
            Name='Ambient Temperature',
            sValue='10 degC',
            raw_entry='Ambient Temperature, 10 degC',
        )
        ReadParameter(entry, param, model)

        self.assertTrue(param.Provided)
        self.assertTrue(param.Valid)
        self.assertIsNone(param.HistoricalData)
        self.assertAlmostEqual(10.0, param.value, places=6)

    def test_read_parameter_historical_gradient_keeps_array_length(self):
        model = self._new_model()
        param = listParameter(Name='Gradients', DefaultValue=[])
        param.AllowHistoricalArrayInput = True
        param.HistoricalXDimension = 'distance'
        param.HistoricalYDimension = 'temperature'
        param.HistoricalDefaultXUnits = 'meter'
        param.HistoricalDefaultYUnits = 'celsius'
        param.HistoricalResampleToHourlyYear = False

        entry = ParameterEntry(
            Name='Gradients',
            sValue='Distance (feet), Temperature (Fahrenheit)\n0,32\n10,50\n',
            raw_entry='Gradients, Distance (feet), Temperature (Fahrenheit)\n0,32\n10,50\n',
        )
        ReadParameter(entry, param, model)

        self.assertEqual(2, len(param.value))
        self.assertTrue(np.allclose(param.value[0], 0.0))

    def test_derive_numseg_from_gradient_thickness(self):
        self.assertEqual(3, _derive_numseg_from_gradient_thickness([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]))


if __name__ == '__main__':
    unittest.main()
