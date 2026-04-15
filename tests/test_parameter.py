import os
import sys
import tempfile
import unittest
from pathlib import Path

from geophires_x.formula_evaluator import evaluate_formula_expression
from geophires_x.formula_evaluator import resolve_parameter_formulas
from geophires_x.Model import Model
from geophires_x.Parameter import ConvertUnitsBack
from geophires_x.Parameter import OutputParameter
from geophires_x.Parameter import Parameter
from geophires_x.Parameter import ParameterEntry
from geophires_x.Parameter import ReadParameter
from geophires_x.Parameter import floatParameter
from geophires_x.Parameter import intParameter
from geophires_x.Parameter import listParameter
from geophires_x.Units import CostPerMassUnit
from geophires_x.Units import CurrencyUnit
from geophires_x.Units import EnergyCostUnit
from geophires_x.Units import LengthUnit
from geophires_x.Units import PressureUnit
from geophires_x.Units import Units
from geophires_x_client import GeophiresXClient
from geophires_x_client import GeophiresXResult
from geophires_x_client import ImmutableGeophiresInputParameters
from tests.base_test_case import BaseTestCase


class ParameterTestCase(BaseTestCase):
    def test_evaluate_formula_expression_supports_numeric_literals(self):
        result = evaluate_formula_expression('42.5', lambda _: None)

        self.assertEqual(42.5, result)

    def test_evaluate_formula_expression_supports_arithmetic_operators_and_parentheses(self):
        result = evaluate_formula_expression('(2 + 3) * 4 - 5 / (1 + 1)', lambda _: None)

        self.assertAlmostEqual(17.5, result)

    def test_evaluate_formula_expression_supports_small_set_of_math_functions(self):
        result = evaluate_formula_expression(
            'sqrt(81) + abs(-4) + max(1, 5, 3) - min(2, 7) + round(2.49)',
            lambda _: None,
        )

        self.assertEqual(18.0, result)

    def test_evaluate_formula_expression_supports_named_variables_from_symbol_table(self):
        symbols = {
            'number_of_injection_wells': 2,
            'well_spacing': 1000,
        }

        result = evaluate_formula_expression(
            'number_of_injection_wells * 1.5 + well_spacing / 1000',
            lambda name: symbols[name],
        )

        self.assertEqual(4.0, result)

    def test_read_parameter_rejects_formula_when_parameter_does_not_allow_formula_input(self):
        model = self._new_model()
        param = intParameter(
            'Number of Injection Wells',
            DefaultValue=2,
            AllowableRange=list(range(201)),
            UnitType=Units.NONE,
        )

        with self.assertRaises(ValueError) as exc:
            ReadParameter(
                ParameterEntry(Name='Number of Injection Wells', sValue='= number_of_production_wells'),
                param,
                model,
            )

        self.assertIn('does not allow formula input', str(exc.exception))
        self.assertIn('Number of Injection Wells', str(exc.exception))

    def test_evaluate_formula_expression_rejects_non_finite_values(self):
        with self.assertRaises(ValueError) as exc:
            evaluate_formula_expression('1e309', lambda _: None)

        self.assertIn('did not evaluate to a finite number', str(exc.exception))

    def test_number_of_production_wells_allows_formula_from_number_of_injection_wells(self):
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as tmp:
            tmp.write('Number of Injection Wells, 2\n')
            tmp.write('Number of Production Wells, = number_of_injection_wells * 1.5\n')
            input_file = tmp.name

        try:
            model = Model(enable_geophires_logging_config=False, input_file=input_file)
            model.read_parameters()

            self.assertEqual(2, model.wellbores.ninj.value)
            self.assertEqual(3, model.wellbores.nprod.value)
            self.assertEqual('number_of_injection_wells * 1.5', model.wellbores.nprod.FormulaExpression)
            self.assertTrue(model.wellbores.nprod.EvaluatedFromFormula)
            self.assertTrue(model.wellbores.nprod.Valid)
        finally:
            Path(input_file).unlink()

    def test_number_of_production_wells_formula_resolution_is_order_independent(self):
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as tmp:
            tmp.write('Number of Production Wells, = number_of_injection_wells * 1.5\n')
            tmp.write('Number of Injection Wells, 2\n')
            input_file = tmp.name

        try:
            model = Model(enable_geophires_logging_config=False, input_file=input_file)
            model.read_parameters()

            self.assertEqual(2, model.wellbores.ninj.value)
            self.assertEqual(3, model.wellbores.nprod.value)
            self.assertTrue(model.wellbores.nprod.EvaluatedFromFormula)
        finally:
            Path(input_file).unlink()

    def test_number_of_injection_wells_allows_formula_from_number_of_production_wells(self):
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as tmp:
            tmp.write('Number of Production Wells, 4\n')
            tmp.write('Number of Injection Wells, = number_of_production_wells - 1\n')
            input_file = tmp.name

        try:
            model = Model(enable_geophires_logging_config=False, input_file=input_file)
            model.read_parameters()

            self.assertEqual(4, model.wellbores.nprod.value)
            self.assertEqual(3, model.wellbores.ninj.value)
            self.assertEqual('number_of_production_wells - 1', model.wellbores.ninj.FormulaExpression)
            self.assertTrue(model.wellbores.ninj.EvaluatedFromFormula)
        finally:
            Path(input_file).unlink()

    def test_number_of_doublets_formula_sets_production_and_injection_wells(self):
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as tmp:
            tmp.write('Number of Segments, 3\n')
            tmp.write('Number of Doublets, = number_of_segments + 1\n')
            input_file = tmp.name

        try:
            model = Model(enable_geophires_logging_config=False, input_file=input_file)
            model.read_parameters()

            self.assertEqual(4, model.wellbores.doublets_count.value)
            self.assertEqual(4, model.wellbores.nprod.value)
            self.assertEqual(4, model.wellbores.ninj.value)
            self.assertTrue(model.wellbores.doublets_count.EvaluatedFromFormula)
        finally:
            Path(input_file).unlink()

    def test_number_of_injection_wells_per_production_well_formula_sets_injection_wells(self):
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as tmp:
            tmp.write('Number of Production Wells, 4\n')
            tmp.write('Number of Injection Wells per Production Well, = number_of_production_wells / 8\n')
            input_file = tmp.name

        try:
            model = Model(enable_geophires_logging_config=False, input_file=input_file)
            model.read_parameters()

            self.assertEqual(4, model.wellbores.nprod.value)
            self.assertEqual(2, model.wellbores.ninj.value)
            self.assertEqual(0.5, model.wellbores.ninj_per_production_well.value)
            self.assertTrue(model.wellbores.ninj_per_production_well.EvaluatedFromFormula)
        finally:
            Path(input_file).unlink()

    def test_number_of_production_wells_formula_unknown_symbol_raises_clear_error(self):
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as tmp:
            tmp.write('Number of Production Wells, = unknown_parameter * 1.5\n')
            input_file = tmp.name

        try:
            model = Model(enable_geophires_logging_config=False, input_file=input_file)

            with self.assertRaises(ValueError) as exc:
                model.read_parameters()

            self.assertIn('Number of Production Wells', str(exc.exception))
            self.assertIn('unknown_parameter', str(exc.exception))
        finally:
            Path(input_file).unlink()

    def test_parameter_formula_circular_dependency_raises_clear_error(self):
        nprod = intParameter(
            'Number of Production Wells',
            DefaultValue=2,
            AllowableRange=list(range(1, 201)),
            UnitType=Units.NONE,
            AllowFormulaInput=True,
            FormulaExpression='number_of_injection_wells + 1',
        )
        ninj = intParameter(
            'Number of Injection Wells',
            DefaultValue=2,
            AllowableRange=list(range(201)),
            UnitType=Units.NONE,
            AllowFormulaInput=True,
            FormulaExpression='number_of_production_wells - 1',
        )

        model = self._new_model()

        with self.assertRaises(ValueError) as exc:
            resolve_parameter_formulas([nprod, ninj], model.logger)

        self.assertIn('Number of Production Wells', str(exc.exception))
        self.assertIn('Circular formula dependency detected', str(exc.exception))

    def test_parameter_formula_resolved_value_fails_min_max_validation(self):
        nprod = intParameter(
            'Number of Production Wells',
            DefaultValue=2,
            AllowableRange=list(range(1, 5)),
            UnitType=Units.NONE,
            AllowFormulaInput=True,
            FormulaExpression='number_of_injection_wells * 3',
        )
        ninj = intParameter(
            'Number of Injection Wells',
            value=2,
            DefaultValue=2,
            AllowableRange=list(range(201)),
            UnitType=Units.NONE,
        )

        model = self._new_model()

        with self.assertRaises(ValueError) as exc:
            resolve_parameter_formulas([nprod, ninj], model.logger)

        self.assertIn('Number of Production Wells', str(exc.exception))
        self.assertIn('outside of valid range', str(exc.exception))

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

    def test_convert_cost_per_mass(self):
        result: GeophiresXResult = GeophiresXClient().get_geophires_result(
            ImmutableGeophiresInputParameters(
                from_file_path=self._get_test_file_path('examples/example_SAM-single-owner-PPA-6_carbon-revenue.txt'),
                params={
                    'Starting Carbon Credit Value': '1 USD/kilogram',
                    'Ending Carbon Credit Value': 100,  # arbitrary high number
                    'Carbon Escalation Rate Per Year': 0,
                    'Units:Total Saved Carbon Production': 'kilogram',
                },
            )
        )

        def _cash_flow_row(r: GeophiresXResult, row_name: str) -> str:
            from geophires_x.EconomicsSam import _cash_flow_profile_row
            from geophires_x.GeoPHIRESUtils import is_float

            return [it for it in _cash_flow_profile_row(r.result['SAM CASH FLOW PROFILE'], row_name) if is_float(it)]

        capacity_payment_revenue_usd_row = _cash_flow_row(result, 'Capacity payment revenue ($)')
        total_capacity_payment_revenue_usd = sum(capacity_payment_revenue_usd_row)

        total_avoided_carbon_emissions_vu: dict[str, float] = result.result['SUMMARY OF RESULTS'][
            'Total Avoided Carbon Emissions'
        ]
        self.assertEqual('kilogram', total_avoided_carbon_emissions_vu['unit'])
        self.assertEqual(int(total_avoided_carbon_emissions_vu['value']), total_capacity_payment_revenue_usd)

    # noinspection PyMethodMayBeStatic
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
