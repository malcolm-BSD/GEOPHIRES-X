import os
import sys
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import numpy as np

from geophires_x.Model import Model
from geophires_x.Parameter import ConvertUnitsBack
from geophires_x.Parameter import OutputParameter
from geophires_x.Parameter import Parameter
from geophires_x.Parameter import ParameterEntry
from geophires_x.Parameter import ReadParameter
from geophires_x.Parameter import floatParameter
from geophires_x.Parameter import listParameter
from geophires_x.Units import CostPerMassUnit
from geophires_x.Units import CurrencyUnit
from geophires_x.Units import EnergyCostUnit
from geophires_x.Units import LengthUnit
from geophires_x.Units import PressureUnit
from geophires_x.Units import TemperatureUnit
from geophires_x.Units import Units
from tests.base_test_case import BaseTestCase


class ParameterTestCase(BaseTestCase):

    def test_read_parameter_allows_pair_vector_inline_for_float_parameter(self):
        model = self._new_model()
        param = floatParameter(Name='Reservoir Temperature', DefaultValue=150.0, Min=0.0, Max=500.0)
        param.AllowPairVectorInput = True

        ReadParameter(
            ParameterEntry(
                Name='Reservoir Temperature',
                sValue='[1000',
                raw_entry='Reservoir Temperature, [1000, 200]'
            ),
            param,
            model,
        )

        self.assertIsInstance(param.value, np.ndarray)
        self.assertTrue(np.array_equal(param.value, np.array([1000.0, 200.0])))
        self.assertTrue(param.Provided)
        self.assertTrue(param.Valid)

    def test_read_parameter_allows_pair_vector_csv_file_for_float_parameter(self):
        model = self._new_model()
        param = floatParameter(Name='Reservoir Temperature', DefaultValue=150.0, Min=0.0, Max=500.0)
        param.AllowPairVectorInput = True

        with NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as f:
            f.write('1000,200\n')
            csv_file = f.name

        try:
            ReadParameter(
                ParameterEntry(
                    Name='Reservoir Temperature',
                    sValue=csv_file,
                    raw_entry=f'Reservoir Temperature, {csv_file}'
                ),
                param,
                model,
            )
        finally:
            os.remove(csv_file)

        self.assertIsInstance(param.value, np.ndarray)
        self.assertTrue(np.array_equal(param.value, np.array([1000.0, 200.0])))

    def test_read_parameter_allows_pair_vector_csv_url_for_float_parameter(self):
        model = self._new_model()
        param = floatParameter(Name='Reservoir Temperature', DefaultValue=150.0, Min=0.0, Max=500.0)
        param.AllowPairVectorInput = True

        class _MockResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, n=-1):
                return b'1000,200\n'

        with patch('geophires_x.Parameter.urlopen', return_value=_MockResponse()):
            ReadParameter(
                ParameterEntry(
                    Name='Reservoir Temperature',
                    sValue='https://example.com/pair.csv',
                    raw_entry='Reservoir Temperature, https://example.com/pair.csv'
                ),
                param,
                model,
            )

        self.assertIsInstance(param.value, np.ndarray)
        self.assertTrue(np.array_equal(param.value, np.array([1000.0, 200.0])))

    def test_read_parameter_pair_vector_applies_convert_units_in_loop(self):
        model = self._new_model()
        param = floatParameter(
            Name='Reservoir Temperature',
            DefaultValue=150.0,
            Min=0.0,
            Max=500.0,
            UnitType=Units.TEMPERATURE,
            PreferredUnits=TemperatureUnit.CELSIUS,
            CurrentUnits=TemperatureUnit.CELSIUS,
        )
        param.AllowPairVectorInput = True

        ReadParameter(
            ParameterEntry(
                Name='Reservoir Temperature',
                sValue='[10, 212 degF]'
            ),
            param,
            model,
        )

        self.assertIsInstance(param.value, np.ndarray)
        self.assertAlmostEqual(param.value[0], 10.0, places=6)
        self.assertAlmostEqual(param.value[1], 100.0, places=6)

    def test_read_list_parameter_allows_csv_file_input_with_units(self):
        model = self._new_model()
        param = listParameter(
            Name='Thicknesses',
            DefaultValue=[1000.0],
            Min=0.0,
            Max=100000.0,
            UnitType=Units.LENGTH,
            PreferredUnits=LengthUnit.METERS,
            CurrentUnits=LengthUnit.METERS,
        )

        with NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as f:
            f.write('1 km\n2 km\n3 km\n')
            csv_file = f.name

        try:
            ReadParameter(
                ParameterEntry(
                    Name='Thicknesses',
                    sValue=csv_file,
                    raw_entry=f'Thicknesses, {csv_file}'
                ),
                param,
                model,
            )
        finally:
            os.remove(csv_file)

        self.assertEqual([1000.0, 2000.0, 3000.0], param.value)

    def test_read_list_parameter_allows_csv_url_input_with_units(self):
        model = self._new_model()
        param = listParameter(
            Name='Thicknesses',
            DefaultValue=[1000.0],
            Min=0.0,
            Max=100000.0,
            UnitType=Units.LENGTH,
            PreferredUnits=LengthUnit.METERS,
            CurrentUnits=LengthUnit.METERS,
        )

        class _MockResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, n=-1):
                return b'1 km\n2 km\n'

        with patch('geophires_x.Parameter.urlopen', return_value=_MockResponse()):
            ReadParameter(
                ParameterEntry(
                    Name='Thicknesses',
                    sValue='https://example.com/thicknesses.csv',
                    raw_entry='Thicknesses, https://example.com/thicknesses.csv'
                ),
                param,
                model,
            )

        self.assertEqual([1000.0, 2000.0], param.value)

    def test_convert_units_back(self):
        model = self._new_model()  # TODO mock instead

        param_to_modify: Parameter = floatParameter(
            Name='Production Well Diameter',
            Required=True,
            Provided=True,
            Valid=True,
            ErrMessage='assume default production well diameter (8 inch)',
            InputComment='',
            ToolTipText='Inner diameter of production wellbore (assumed constant along the wellbore) to calculate             frictional pressure drop and wellbore heat transmission with Rameys model',
            UnitType=Units.LENGTH,
            PreferredUnits=LengthUnit.INCHES,
            CurrentUnits=LengthUnit.METERS,
            value=0.17779999999999999,
            DefaultValue=8.0,
            Min=1.0,
            Max=30.0,
        )
        self.assertFalse(param_to_modify.UnitsMatch)

        ConvertUnitsBack(param_to_modify, model)

        self.assertEqual(param_to_modify.value, 7.0)
        self.assertEqual(param_to_modify.CurrentUnits, LengthUnit.INCHES)

    def test_set_default_value(self):
        without_val = floatParameter(
            'Average Reservoir Pressure',
            DefaultValue=29430,  # Calculated from example1
            Min=1e2,
            Max=1e5,
            UnitType=Units.PRESSURE,
            PreferredUnits=PressureUnit.KPASCAL,
            CurrentUnits=PressureUnit.KPASCAL,
            ErrMessage='calculate reservoir pressure using built-in correlation',
            ToolTipText='Reservoir hydrostatic far-field pressure.  Default value is calculated with built-in modified \
                    Xie-Bloomfield-Shook equation (DOE, 2016).',
        )
        self.assertEqual(29430, without_val.value)

        with_val = floatParameter(
            'Average Reservoir Pressure',
            value=1e2,
            DefaultValue=29430,
            Min=1e2,
            Max=1e5,
            UnitType=Units.PRESSURE,
            PreferredUnits=PressureUnit.KPASCAL,
            CurrentUnits=PressureUnit.KPASCAL,
            ErrMessage='calculate reservoir pressure using built-in correlation',
            ToolTipText='Reservoir hydrostatic far-field pressure.  Default value is calculated with built-in modified \
                    Xie-Bloomfield-Shook equation (DOE, 2016).',
        )
        self.assertEqual(1e2, with_val.value)

    def test_set_default_value_list(self):
        without_val = listParameter(
            'Thicknesses',
            DefaultValue=[100_000.0, 0.01, 0.01, 0.01, 0.01],
            Min=0.01,
            Max=100.0,
            UnitType=Units.LENGTH,
            PreferredUnits=LengthUnit.KILOMETERS,
            CurrentUnits=LengthUnit.KILOMETERS,
            ErrMessage='assume default layer thicknesses (100,000, 0, 0, 0 km)',
            ToolTipText='Thicknesses of rock segments',
        )

        self.assertEqual([100_000.0, 0.01, 0.01, 0.01, 0.01], without_val.value)

        with_val = listParameter(
            'Thicknesses',
            value=[1, 2, 3],
            DefaultValue=[100_000.0, 0.01, 0.01, 0.01, 0.01],
            Min=0.01,
            Max=100.0,
            UnitType=Units.LENGTH,
            PreferredUnits=LengthUnit.KILOMETERS,
            CurrentUnits=LengthUnit.KILOMETERS,
            ErrMessage='assume default layer thicknesses (100,000, 0, 0, 0 km)',
            ToolTipText='Thicknesses of rock segments',
        )

        self.assertEqual([1, 2, 3], with_val.value)

    def test_output_parameter_with_preferred_units(self):
        op: OutputParameter = OutputParameter(
            Name='Electricity Sale Price Model',
            value=[
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
                0.055,
            ],
            ToolTipText='This is ToolTip Text',
            UnitType=Units.ENERGYCOST,
            PreferredUnits=EnergyCostUnit.CENTSSPERKWH,
            CurrentUnits=EnergyCostUnit.DOLLARSPERKWH,
        )

        result = op.with_preferred_units()
        self.assertIsNotNone(result)
        self.assertEqual(5.5, result.value[0])
        self.assertEqual(5.5, result.value[-1])

    def test_output_parameter_json_types(self):
        cases = [
            ('foo', 'string'),
            (1, 'number'),
            (44.4, 'number'),
            (True, 'boolean'),
            ([1, 2, 3], 'array'),
            ({4, 5, 6}, 'array'),
            (None, 'object'),
            ({'foo': 'bar'}, 'object'),
        ]

        for case in cases:
            with self.subTest(case=case):
                jpt = OutputParameter(value=case[0]).json_parameter_type
                self.assertEqual(case[1], jpt)

    def test_convert_units_back_currency(self):
        model = self._new_model()

        param = floatParameter(
            'CAPEX',
            DefaultValue=1379.0,
            UnitType=Units.COSTPERMASS,
            PreferredUnits=CostPerMassUnit.DOLLARSPERMT,
            CurrentUnits=CostPerMassUnit.CENTSSPERMT,
        )

        ConvertUnitsBack(param, model)
        self.assertEqual(param.CurrentUnits, CostPerMassUnit.DOLLARSPERMT)
        self.assertAlmostEqual(param.value, 13.79, places=2)

        with self.assertRaises(RuntimeError) as re:
            # TODO update once https://github.com/NREL/GEOPHIRES-X/issues/236?title=Currency+conversions+disabled is
            #   addressed
            param2 = floatParameter(
                'OPEX',
                DefaultValue=240,
                UnitType=Units.CURRENCY,
                PreferredUnits=CurrencyUnit.DOLLARS,
                CurrentUnits=CurrencyUnit.EUR,
            )
            ConvertUnitsBack(param2, model)

            self.assertIn('GEOPHIRES failed to convert your units for OPEX', str(re))

    def _new_model(self) -> Model:
        stash_cwd = Path.cwd()
        stash_sys_argv = sys.argv

        sys.argv = ['']

        m = Model(enable_geophires_logging_config=False)

        sys.argv = stash_sys_argv
        os.chdir(stash_cwd)

        return m


if __name__ == '__main__':
    unittest.main()
