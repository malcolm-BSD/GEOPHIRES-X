from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from geophires_x.EconomicsUtils import total_capex_parameter_output_parameter
from geophires_x.formula_evaluator import evaluate_formula_expression
from geophires_x.formula_evaluator import resolve_parameter_formulas
from geophires_x.GeoPHIRESUtils import read_input_file
from geophires_x.Model import Model
from geophires_x.Parameter import ConvertUnitsBack
from geophires_x.Parameter import OutputParameter
from geophires_x.Parameter import Parameter
from geophires_x.Parameter import ParameterEntry
from geophires_x.Parameter import ReadParameter
from geophires_x.Parameter import TemperatureGradientUnit
from geophires_x.Parameter import boolParameter
from geophires_x.Parameter import filenameParameter
from geophires_x.Parameter import floatParameter
from geophires_x.Parameter import intParameter
from geophires_x.Parameter import listParameter
from geophires_x.Parameter import strParameter
from geophires_x.Reservoir import derive_numseg_from_gradient_thickness
from geophires_x.Units import CostPerDistanceUnit
from geophires_x.Units import CostPerMassUnit
from geophires_x.Units import CurrencyUnit
from geophires_x.Units import EnergyCostUnit
from geophires_x.Units import LengthUnit
from geophires_x.Units import PressureUnit
from geophires_x.Units import TemperatureUnit
from geophires_x.Units import Units
from geophires_x_client import GeophiresXClient
from geophires_x_client import GeophiresXResult
from geophires_x_client import ImmutableGeophiresInputParameters
from tests.base_test_case import BaseTestCase


class ParameterTestCase(BaseTestCase):
    def test_parameter_ui_defaults_are_owned_by_each_parameter(self):
        parameter = Parameter()

        self.assertEqual(
            {
                "tab": "Project",
                "group": "",
                "is_controller": False,
                "order": 1,
            },
            parameter.UI,
        )
        self.assertIsNot(parameter.UI, Parameter().UI)

    def test_evaluate_formula_expression_supports_numeric_literals(self):
        result = evaluate_formula_expression("42.5", lambda _: None)

        self.assertEqual(42.5, result)

    def test_evaluate_formula_expression_supports_arithmetic_operators_and_parentheses(self):
        result = evaluate_formula_expression("(2 + 3) * 4 - 5 / (1 + 1)", lambda _: None)

        self.assertAlmostEqual(17.5, result)

    def test_evaluate_formula_expression_supports_small_set_of_math_functions(self):
        result = evaluate_formula_expression(
            "sqrt(81) + abs(-4) + max(1, 5, 3) - min(2, 7) + round(2.49)",
            lambda _: None,
        )

        self.assertEqual(18.0, result)

    def test_evaluate_formula_expression_supports_named_variables_from_symbol_table(self):
        symbols = {
            "number_of_injection_wells": 2,
            "well_spacing": 1000,
        }

        result = evaluate_formula_expression(
            "number_of_injection_wells * 1.5 + well_spacing / 1000",
            lambda name: symbols[name],
        )

        self.assertEqual(4.0, result)

    def test_read_parameter_rejects_formula_when_parameter_does_not_allow_formula_input(self):
        model = self._new_model()
        param = intParameter(
            "Number of Injection Wells",
            DefaultValue=2,
            AllowableRange=list(range(201)),
            UnitType=Units.NONE,
        )

        with self.assertRaises(ValueError) as exc:
            ReadParameter(
                ParameterEntry(Name="Number of Injection Wells", sValue="= number_of_production_wells"),
                param,
                model,
            )

        self.assertIn("does not allow formula input", str(exc.exception))
        self.assertIn("Number of Injection Wells", str(exc.exception))

    def test_evaluate_formula_expression_rejects_non_finite_values(self):
        with self.assertRaises(ValueError) as exc:
            evaluate_formula_expression("1e309", lambda _: None)

        self.assertIn("did not evaluate to a finite number", str(exc.exception))

    def test_number_of_production_wells_allows_formula_from_number_of_injection_wells(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write("Number of Injection Wells, 2\n")
            tmp.write("Number of Production Wells, = number_of_injection_wells * 1.5\n")
            input_file = tmp.name

        try:
            model = Model(enable_geophires_logging_config=False, input_file=input_file)
            model.read_parameters()

            self.assertEqual(2, model.wellbores.ninj.value)
            self.assertEqual(3, model.wellbores.nprod.value)
            self.assertEqual("number_of_injection_wells * 1.5", model.wellbores.nprod.FormulaExpression)
            self.assertTrue(model.wellbores.nprod.EvaluatedFromFormula)
            self.assertTrue(model.wellbores.nprod.Valid)
        finally:
            Path(input_file).unlink()

    def test_number_of_production_wells_formula_resolution_is_order_independent(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write("Number of Production Wells, = number_of_injection_wells * 1.5\n")
            tmp.write("Number of Injection Wells, 2\n")
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
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write("Number of Production Wells, 4\n")
            tmp.write("Number of Injection Wells, = number_of_production_wells - 1\n")
            input_file = tmp.name

        try:
            model = Model(enable_geophires_logging_config=False, input_file=input_file)
            model.read_parameters()

            self.assertEqual(4, model.wellbores.nprod.value)
            self.assertEqual(3, model.wellbores.ninj.value)
            self.assertEqual("number_of_production_wells - 1", model.wellbores.ninj.FormulaExpression)
            self.assertTrue(model.wellbores.ninj.EvaluatedFromFormula)
        finally:
            Path(input_file).unlink()

    def test_number_of_doublets_formula_sets_production_and_injection_wells(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write("Number of Segments, 1\n")
            tmp.write("Number of Doublets, = number_of_segments + 1\n")
            input_file = tmp.name

        try:
            model = Model(enable_geophires_logging_config=False, input_file=input_file)
            model.read_parameters()

            self.assertEqual(2, model.wellbores.doublets_count.value)
            self.assertEqual(2, model.wellbores.nprod.value)
            self.assertEqual(2, model.wellbores.ninj.value)
            self.assertTrue(model.wellbores.doublets_count.EvaluatedFromFormula)
        finally:
            Path(input_file).unlink()

    def test_number_of_injection_wells_per_production_well_formula_sets_injection_wells(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write("Number of Production Wells, 4\n")
            tmp.write("Number of Injection Wells per Production Well, = number_of_production_wells / 8\n")
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

    def test_sbt_geometry_parameters_allow_chained_formulas(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write("Reservoir Model, 8\n")
            tmp.write("Vertical Section Length, = junction_depth - 1000\n")
            tmp.write("Vertical Wellbore Spacing, = vertical_section_length / 2\n")
            tmp.write("Lateral Spacing, = vertical_wellbore_spacing / 10\n")
            tmp.write('Discretization Length, "= min(250, lateral_spacing)"\n')
            tmp.write("Junction Depth, = lateral_endpoint_depth - 2000\n")
            tmp.write("Lateral Endpoint Depth, = 5000\n")
            input_file = tmp.name

        expected_values = {
            "Vertical Section Length": 2000.0,
            "Vertical Wellbore Spacing": 1000.0,
            "Lateral Spacing": 100.0,
            "Discretization Length": 100.0,
            "Junction Depth": 3000.0,
            "Lateral Endpoint Depth": 5000.0,
        }

        try:
            model = Model(enable_geophires_logging_config=False, input_file=input_file)
            model.read_parameters()

            for parameter_name, expected_value in expected_values.items():
                parameter = model.wellbores.ParameterDict[parameter_name]
                self.assertEqual(expected_value, parameter.value)
                self.assertEqual(LengthUnit.METERS, parameter.CurrentUnits)
                self.assertTrue(parameter.EvaluatedFromFormula)
                self.assertTrue(parameter.Valid)
        finally:
            Path(input_file).unlink()

    def test_leaf_parameter_groups_allow_formulas(self):
        formulas = {
            # Dimensionless performance parameters
            "Water Loss Fraction": "= 0.02",
            "Reservoir Porosity": "= water_loss_fraction * 2",
            "Drawdown Parameter": "= reservoir_porosity / 8",
            "Maximum Drawdown": "= 0.2",
            "Overpressure Percentage": "= 120",
            "Overpressure Depletion Rate": "= 2",
            "Circulation Pump Efficiency": "= 0.8",
            "Utilization Factor": "= circulation_pump_efficiency + 0.1",
            "End-Use Efficiency Factor": "= 0.85",
            "CHP Fraction": "= 0.5",
            # Thermophysical properties
            "Reservoir Heat Capacity": "= working_fluid_heat_capacity / 4",
            "Reservoir Density": "= 2600",
            "Reservoir Thermal Conductivity": "= 2.5",
            "Reservoir Permeability": "= 1e-14",
            "Water Thermal Conductivity": "= working_fluid_thermal_conductivity + 0.05",
            "Working Fluid Heat Capacity": "= 4100",
            "Working Fluid Density": "= 990",
            "Working Fluid Thermal Conductivity": "= 0.6",
            "Working Fluid Dynamic Viscosity": "= 0.001",
            # Passive wellbore dimensions and construction properties
            "Production Well Diameter": "= 7.5",
            "Injection Well Diameter": "= production_well_diameter + 0.5",
            "Production Flow Rate per Well": "= 60",
            "Well Separation": "= 40000",
            "Nonvertical Length per Multilateral Section": "= 1200",
            "Nonvertical Wellbore Diameter": "= 0.2",
            "Lateral Casing Thermal Conductivity": "= 40",
            "Lateral Cement Thermal Conductivity": "= 1.5",
            "Lateral Casing Thickness": "= 0.02",
            "Lateral Cement Thickness": "= lateral_casing_thickness * 2",
            # Independent economic percentages
            "Contingency Percentage": "= indirect_capital_cost_percentage - 5",
            "Indirect Capital Cost Percentage": "= 15",
            "Well Drilling and Completion Indirect Capital Cost Percentage": "= indirect_capital_cost_percentage / 3",
            "Reservoir Stimulation Indirect Capital Cost Percentage": "= 6",
        }
        expected_values = {
            "Water Loss Fraction": 0.02,
            "Reservoir Porosity": 0.04,
            "Drawdown Parameter": 0.005,
            "Maximum Drawdown": 0.2,
            "Overpressure Percentage": 120.0,
            "Overpressure Depletion Rate": 2.0,
            "Circulation Pump Efficiency": 0.8,
            "Utilization Factor": 0.9,
            "End-Use Efficiency Factor": 0.85,
            "CHP Fraction": 0.5,
            "Reservoir Heat Capacity": 1025.0,
            "Reservoir Density": 2600.0,
            "Reservoir Thermal Conductivity": 2.5,
            "Reservoir Permeability": 1e-14,
            "Water Thermal Conductivity": 0.65,
            "Working Fluid Heat Capacity": 4100.0,
            "Working Fluid Density": 990.0,
            "Working Fluid Thermal Conductivity": 0.6,
            "Working Fluid Dynamic Viscosity": 0.001,
            "Production Well Diameter": 7.5,
            "Injection Well Diameter": 8.0,
            "Production Flow Rate per Well": 60.0,
            "Well Separation": 40000.0,
            "Nonvertical Length per Multilateral Section": 1200.0,
            "Nonvertical Wellbore Diameter": 0.2,
            "Lateral Casing Thermal Conductivity": 40.0,
            "Lateral Cement Thermal Conductivity": 1.5,
            "Lateral Casing Thickness": 0.02,
            "Lateral Cement Thickness": 0.04,
            "Contingency Percentage": 10.0,
            "Indirect Capital Cost Percentage": 15.0,
            "Well Drilling and Completion Indirect Capital Cost Percentage": 5.0,
            "Reservoir Stimulation Indirect Capital Cost Percentage": 6.0,
        }

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write("Reservoir Model, 4\n")
            for parameter_name, formula in formulas.items():
                tmp.write(f"{parameter_name}, {formula}\n")
            input_file = tmp.name

        try:
            model = Model(enable_geophires_logging_config=False, input_file=input_file)
            model.read_parameters()

            parameters_by_name = {
                parameter.Name: parameter
                for component in (model.reserv, model.wellbores, model.surfaceplant, model.economics)
                for parameter in component.ParameterDict.values()
            }
            self.assertEqual(set(expected_values), set(formulas))
            for parameter_name, expected_value in expected_values.items():
                parameter = parameters_by_name[parameter_name]
                self.assertTrue(
                    np.isclose(expected_value, parameter.value, rtol=1e-12, atol=0.0),
                    msg=f"Unexpected resolved value for {parameter_name}: {parameter.value}",
                )
                self.assertTrue(parameter.AllowFormulaInput)
                self.assertTrue(parameter.EvaluatedFromFormula)
                self.assertTrue(parameter.Valid)
        finally:
            Path(input_file).unlink()

    def test_drawdown_parameter_formula_is_supported_by_single_fracture_reservoir(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write("Reservoir Model, 3\n")
            tmp.write("Drawdown Parameter, = 0.01\n")
            input_file = tmp.name

        try:
            model = Model(enable_geophires_logging_config=False, input_file=input_file)
            model.read_parameters()

            self.assertEqual(0.01, model.reserv.drawdp.value)
            self.assertTrue(model.reserv.drawdp.EvaluatedFromFormula)
            self.assertTrue(model.reserv.drawdp.Valid)
        finally:
            Path(input_file).unlink()

    def test_number_of_production_wells_formula_unknown_symbol_raises_clear_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write("Number of Production Wells, = unknown_parameter * 1.5\n")
            input_file = tmp.name

        try:
            model = Model(enable_geophires_logging_config=False, input_file=input_file)

            with self.assertRaises(ValueError) as exc:
                model.read_parameters()

            self.assertIn("Number of Production Wells", str(exc.exception))
            self.assertIn("unknown_parameter", str(exc.exception))
        finally:
            Path(input_file).unlink()

    def test_model_formula_resolution_ignores_duplicate_symbols_from_add_on_component_inheritance(self):
        model = Model(
            enable_geophires_logging_config=False,
            input_file=self._get_test_file_path("geophires_x_tests/egs-sam-em-add-ons.txt"),
        )

        model.read_parameters()

        self.assertIsNotNone(model.addeconomics)
        self.assertEqual(5, model.economics.econmodel.value.int_value)

    def test_parameter_formula_circular_dependency_raises_clear_error(self):
        nprod = intParameter(
            "Number of Production Wells",
            DefaultValue=2,
            AllowableRange=list(range(1, 201)),
            UnitType=Units.NONE,
            AllowFormulaInput=True,
            FormulaExpression="number_of_injection_wells + 1",
        )
        ninj = intParameter(
            "Number of Injection Wells",
            DefaultValue=2,
            AllowableRange=list(range(201)),
            UnitType=Units.NONE,
            AllowFormulaInput=True,
            FormulaExpression="number_of_production_wells - 1",
        )

        model = self._new_model()

        with self.assertRaises(ValueError) as exc:
            resolve_parameter_formulas([nprod, ninj], model.logger)

        self.assertIn("Number of Production Wells", str(exc.exception))
        self.assertIn("Circular formula dependency detected", str(exc.exception))

    def test_parameter_formula_resolved_value_fails_min_max_validation(self):
        nprod = intParameter(
            "Number of Production Wells",
            DefaultValue=2,
            AllowableRange=list(range(1, 5)),
            UnitType=Units.NONE,
            AllowFormulaInput=True,
            FormulaExpression="number_of_injection_wells * 3",
        )
        ninj = intParameter(
            "Number of Injection Wells",
            value=2,
            DefaultValue=2,
            AllowableRange=list(range(201)),
            UnitType=Units.NONE,
        )

        model = self._new_model()

        with self.assertRaises(ValueError) as exc:
            resolve_parameter_formulas([nprod, ninj], model.logger)

        self.assertIn("Number of Production Wells", str(exc.exception))
        self.assertIn("outside of valid range", str(exc.exception))

    def test_parameter_formula_ignores_duplicate_unreferenced_symbols(self):
        dependent_param = intParameter(
            "Number of Production Wells",
            DefaultValue=2,
            AllowableRange=list(range(1, 201)),
            UnitType=Units.NONE,
            AllowFormulaInput=True,
            FormulaExpression="number_of_injection_wells + 1",
        )
        source_param = intParameter(
            "Number of Injection Wells",
            value=2,
            DefaultValue=2,
            AllowableRange=list(range(201)),
            UnitType=Units.NONE,
        )
        economic_model_1 = intParameter(
            "Economic Model",
            value=2,
            DefaultValue=2,
            AllowableRange=list(range(10)),
            UnitType=Units.NONE,
        )
        economic_model_2 = intParameter(
            "Economic Model",
            value=4,
            DefaultValue=4,
            AllowableRange=list(range(10)),
            UnitType=Units.NONE,
        )

        model = self._new_model()
        resolve_parameter_formulas(
            [dependent_param, source_param, economic_model_1, economic_model_2],
            model.logger,
        )

        self.assertEqual(3, dependent_param.value)
        self.assertTrue(dependent_param.EvaluatedFromFormula)

    def test_read_parameter_allows_pair_vector_inline_for_float_parameter(self):
        model = self._new_model()
        param = floatParameter(
            Name="Reservoir Temperature",
            DefaultValue=150.0,
            Min=0.0,
            Max=500.0,
            UnitType=Units.TEMPERATURE,
            PreferredUnits=TemperatureUnit.CELSIUS,
            CurrentUnits=TemperatureUnit.CELSIUS,
            AllowExtendedInput=True,
        )
        param.AllowPairVectorInput = True

        ReadParameter(
            ParameterEntry(
                Name="Reservoir Temperature", sValue="[1000, 200]", raw_entry="Reservoir Temperature, [1000, 200]"
            ),
            param,
            model,
        )

        self.assertIsInstance(param.value, list)
        self.assertEqual(param.value, [1000.0, 200.0])
        self.assertTrue(param.Provided)
        self.assertTrue(param.Valid)

    def test_read_parameter_pair_vector_applies_convert_units_in_loop(self):
        model = self._new_model()
        param = floatParameter(
            Name="Reservoir Temperature",
            DefaultValue=150.0,
            Min=0.0,
            Max=500.0,
            UnitType=Units.TEMPERATURE,
            PreferredUnits=TemperatureUnit.CELSIUS,
            CurrentUnits=TemperatureUnit.CELSIUS,
            AllowExtendedInput=True,
        )
        param.AllowPairVectorInput = True

        ReadParameter(
            ParameterEntry(Name="Reservoir Temperature", sValue="[212, 220]"),
            param,
            model,
        )

        self.assertIsInstance(param.value, list)
        self.assertAlmostEqual(param.value[0], 212.0, places=6)
        self.assertAlmostEqual(param.value[1], 220.0, places=6)

    def test_convert_units_back(self):
        model = self._new_model()  # TODO mock instead

        param_to_modify: Parameter = floatParameter(
            Name="Production Well Diameter",
            Required=True,
            Provided=True,
            Valid=True,
            ErrMessage="assume default production well diameter (8 inch)",
            InputComment="",
            ToolTipText="Inner diameter of production wellbore (assumed constant along the wellbore) to calculate frictional pressure drop and wellbore heat transmission with Rameys model",
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
            "Average Reservoir Pressure",
            DefaultValue=29430,  # Calculated from example1
            Min=1e2,
            Max=1e5,
            UnitType=Units.PRESSURE,
            PreferredUnits=PressureUnit.KPASCAL,
            CurrentUnits=PressureUnit.KPASCAL,
            ErrMessage="calculate reservoir pressure using built-in correlation",
            ToolTipText="Reservoir hydrostatic far-field pressure.  Default value is calculated with built-in modified \
                    Xie-Bloomfield-Shook equation (DOE, 2016).",
        )
        self.assertEqual(29430, without_val.value)

        with_val = floatParameter(
            "Average Reservoir Pressure",
            value=1e2,
            DefaultValue=29430,
            Min=1e2,
            Max=1e5,
            UnitType=Units.PRESSURE,
            PreferredUnits=PressureUnit.KPASCAL,
            CurrentUnits=PressureUnit.KPASCAL,
            ErrMessage="calculate reservoir pressure using built-in correlation",
            ToolTipText="Reservoir hydrostatic far-field pressure.  Default value is calculated with built-in modified \
                    Xie-Bloomfield-Shook equation (DOE, 2016).",
        )
        self.assertEqual(1e2, with_val.value)

    def test_set_default_value_list(self):
        without_val = listParameter(
            "Thicknesses",
            DefaultValue=[100_000.0, 0.01, 0.01, 0.01, 0.01],
            Min=0.01,
            Max=100.0,
            UnitType=Units.LENGTH,
            PreferredUnits=LengthUnit.KILOMETERS,
            CurrentUnits=LengthUnit.KILOMETERS,
            ErrMessage="assume default layer thicknesses (100,000, 0, 0, 0 km)",
            ToolTipText="Thicknesses of rock segments",
        )

        self.assertEqual([100_000.0, 0.01, 0.01, 0.01, 0.01], without_val.value)

        with_val = listParameter(
            "Thicknesses",
            value=[1, 2, 3],
            DefaultValue=[100_000.0, 0.01, 0.01, 0.01, 0.01],
            Min=0.01,
            Max=100.0,
            UnitType=Units.LENGTH,
            PreferredUnits=LengthUnit.KILOMETERS,
            CurrentUnits=LengthUnit.KILOMETERS,
            ErrMessage="assume default layer thicknesses (100,000, 0, 0, 0 km)",
            ToolTipText="Thicknesses of rock segments",
        )

        self.assertEqual([1, 2, 3], with_val.value)

    def test_output_parameter_with_preferred_units(self):
        op: OutputParameter = OutputParameter(
            Name="Electricity Sale Price Model",
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
            ToolTipText="This is ToolTip Text",
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
            ("foo", "string"),
            (1, "number"),
            (44.4, "number"),
            (True, "boolean"),
            ([1, 2, 3], "array"),
            ({4, 5, 6}, "array"),
            (None, "object"),
            ({"foo": "bar"}, "object"),
        ]

        for case in cases:
            with self.subTest(case=case):
                jpt = OutputParameter(value=case[0]).json_parameter_type
                self.assertEqual(case[1], jpt)

    def test_convert_units_back_currency(self):
        model = self._new_model()

        param = floatParameter(
            "CAPEX",
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
                "OPEX",
                DefaultValue=240,
                UnitType=Units.CURRENCY,
                PreferredUnits=CurrencyUnit.DOLLARS,
                CurrentUnits=CurrencyUnit.EUR,
            )
            ConvertUnitsBack(param2, model)

            self.assertIn("GEOPHIRES failed to convert your units for OPEX", str(re))

    def test_read_parameter_cost_per_distance_kusd_per_km(self):
        model = self._new_model()
        param = floatParameter(
            Name="Transmission/pipeline Cost",
            DefaultValue=750,
            Min=0,
            Max=10000,
            UnitType=Units.COSTPERDISTANCE,
            PreferredUnits=CostPerDistanceUnit.KDOLLARSPERKM,
            CurrentUnits=CostPerDistanceUnit.KDOLLARSPERKM,
        )

        ReadParameter(
            ParameterEntry(
                Name="Transmission/pipeline Cost", sValue="1 USD/m", raw_entry="Transmission/pipeline Cost, 1 USD/m"
            ),
            param,
            model,
        )

        self.assertAlmostEqual(param.value, 1.0)
        self.assertEqual(param.CurrentUnits, CostPerDistanceUnit.KDOLLARSPERKM)

    def test_read_bool_from_file(self):
        model = self._new_model()
        param = boolParameter("Count", DefaultValue=True, AllowExtendedInput=True)
        root = str(Path(__file__).resolve().parents[0])  # adjust depth as needed
        csv_file = root + "/assets/params/bool.txt"
        ReadParameter(
            ParameterEntry(Name="Geophires Enabled", sValue=csv_file, raw_entry=f"Geophires Enabled, {csv_file}"),
            param,
            model,
        )

        self.assertIsInstance(param.value, bool)
        self.assertEqual(param.value, False)

    def test_read_int_from_file(self):
        model = self._new_model()
        param = intParameter("Count", DefaultValue=0, AllowableRange=[0, 113, 1000000], AllowExtendedInput=True)
        root = str(Path(__file__).resolve().parents[0])  # adjust depth as needed
        csv_file = root + "/assets/params/int.txt"

        ReadParameter(
            ParameterEntry(Name="Count", sValue=csv_file, raw_entry=f"Count, {csv_file}"),
            param,
            model,
        )

        self.assertIsInstance(param.value, int)
        self.assertEqual(param.value, 113)

    def test_read_float_from_file(self):
        model = self._new_model()
        param = floatParameter(
            "CAPEX",
            DefaultValue=1379.0,
            UnitType=Units.COSTPERMASS,
            PreferredUnits=CostPerMassUnit.DOLLARSPERMT,
            CurrentUnits=CostPerMassUnit.CENTSSPERMT,
            AllowExtendedInput=True,
        )
        root = str(Path(__file__).resolve().parents[0])  # adjust depth as needed
        csv_file = root + "/assets/params/float.txt"

        ReadParameter(
            ParameterEntry(Name="CAPEX", sValue=csv_file, raw_entry=f"Ambient Temperature, {csv_file}"),
            param,
            model,
        )

        self.assertIsInstance(param.value, float)
        self.assertEqual(param.value, 3.14156)

    def test_read_str_from_file(self):
        model = self._new_model()
        param = strParameter("Gettysburg", DefaultValue="", AllowExtendedInput=True)
        root = str(Path(__file__).resolve().parents[0])  # adjust depth as needed
        csv_file = root + "/assets/params/string.txt"
        ReadParameter(
            ParameterEntry(Name="Gettysburg", sValue=csv_file, raw_entry=f"Geophires Enabled, {csv_file}"), param, model
        )

        self.assertIsInstance(param.value, str)
        self.assertIn("Four score and seven years ago", param.value)

    def test_read_list_from_file(self):
        model = self._new_model()
        param = listParameter("Random", DefaultValue=[], AllowExtendedInput=True)
        root = str(Path(__file__).resolve().parents[0])  # adjust depth as needed
        csv_file = root + "/assets/params/list.txt"

        ReadParameter(ParameterEntry(Name="Random", sValue=csv_file, raw_entry=f"Random, {csv_file}"), param, model)

        self.assertIsInstance(param.value, list)
        self.assertEqual(345, param.value[0])

    def test_read_parameter_values_from_json_file_by_pointer(self):
        model = self._new_model()
        profile = {
            "number": 3.14156,
            "count": 113,
            "enabled": False,
            "label": "Edwards profile",
            "values": [1, 2, 3],
            "distance": "5000 m",
            "nested": {"a/b": {"~value": 7}},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            cases = [
                (floatParameter("Number", DefaultValue=0.0, AllowExtendedInput=True), "/number", 3.14156),
                (
                    intParameter("Count", DefaultValue=0, AllowableRange=[0, 7, 113], AllowExtendedInput=True),
                    "/count",
                    113,
                ),
                (boolParameter("Enabled", DefaultValue=True, AllowExtendedInput=True), "/enabled", False),
                (strParameter("Label", DefaultValue="", AllowExtendedInput=True), "/label", "Edwards profile"),
                (listParameter("Values", DefaultValue=[], AllowExtendedInput=True), "/values", [1, 2, 3]),
                (
                    floatParameter(
                        "Distance",
                        DefaultValue=0.0,
                        UnitType=Units.LENGTH,
                        PreferredUnits=LengthUnit.KILOMETERS,
                        CurrentUnits=LengthUnit.KILOMETERS,
                        AllowExtendedInput=True,
                    ),
                    "/distance",
                    5.0,
                ),
                (
                    intParameter("Escaped", DefaultValue=0, AllowableRange=[0, 7], AllowExtendedInput=True),
                    "/nested/a~1b/~0value",
                    7,
                ),
            ]

            for param, pointer, expected in cases:
                with self.subTest(parameter=param.Name):
                    ReadParameter(
                        ParameterEntry(Name=param.Name, sValue=f"{profile_path}#{pointer}"),
                        param,
                        model,
                    )
                    self.assertEqual(expected, param.value)

    def test_extended_input_is_enabled_by_default_except_for_output_destinations(self):
        model = self._new_model()
        parameters = {}
        for component in vars(model).values():
            parameter_dict = getattr(component, "ParameterDict", None)
            if parameter_dict is not None:
                parameters.update(parameter_dict)

        disabled_parameters = {name for name, parameter in parameters.items() if not parameter.AllowExtendedInput}

        self.assertEqual(
            {
                "Improved Text Output File",
                "HTML Output File",
                "Dispatch Profile Output File",
            },
            disabled_parameters,
        )

    def test_json_array_is_rejected_for_an_ordinary_scalar_parameter(self):
        model = self._new_model()
        param = floatParameter("Number", DefaultValue=0.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "profile.json"
            profile_path.write_text('{"values": [1, 2, 3]}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "requires a scalar value"):
                ReadParameter(
                    ParameterEntry(Name=param.Name, sValue=f"{profile_path}#/values"),
                    param,
                    model,
                )

    def test_json_scalar_is_rejected_for_a_list_parameter(self):
        model = self._new_model()
        param = listParameter("Values", DefaultValue=[])

        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "profile.json"
            profile_path.write_text('{"value": 1}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "requires an array"):
                ReadParameter(
                    ParameterEntry(Name=param.Name, sValue=f"{profile_path}#/value"),
                    param,
                    model,
                )

    def test_invalid_boolean_value_is_rejected(self):
        model = self._new_model()
        param = boolParameter("Enabled", DefaultValue=True)

        with self.assertRaisesRegex(ValueError, "not a recognized boolean value"):
            ReadParameter(ParameterEntry(Name=param.Name, sValue="maybe"), param, model)

        self.assertFalse(param.Valid)

    def test_filename_parameter_retains_a_bare_file_reference(self):
        model = self._new_model()
        param = filenameParameter("Input File", DefaultValue="")

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.csv"
            data_path.write_text("1,2,3", encoding="utf-8")

            ReadParameter(ParameterEntry(Name=param.Name, sValue=str(data_path)), param, model)

        self.assertEqual(str(data_path), param.value)

    def test_filename_parameter_can_be_selected_from_json(self):
        model = self._new_model()
        param = filenameParameter("Input File", DefaultValue="")

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.csv"
            data_path.write_text("1,2,3", encoding="utf-8")
            profile_path = Path(temp_dir) / "profile.json"
            profile_path.write_text(json.dumps({"filename": str(data_path)}), encoding="utf-8")

            ReadParameter(
                ParameterEntry(Name=param.Name, sValue=f"{profile_path}#/filename"),
                param,
                model,
            )

        self.assertEqual(str(data_path), param.value)

    def test_read_parameter_value_from_relative_json_file_by_pointer(self):
        model = self._new_model()
        param = floatParameter("Number", DefaultValue=0.0, AllowExtendedInput=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "profile.json"
            profile_path.write_text('{"nested": {"number": 2.5}}', encoding="utf-8")
            model.input_file_path = str(Path(temp_dir) / "model-input.txt")

            ReadParameter(
                ParameterEntry(Name="Number", sValue="profile.json#/nested/number"),
                param,
                model,
            )

        self.assertEqual(2.5, param.value)

    def test_read_parameter_value_from_json_url_by_pointer(self):
        model = self._new_model()
        param = floatParameter("Number", DefaultValue=0.0, AllowExtendedInput=True)
        url = "https://example.com/profile.json#/number"

        with patch(
            "geophires_x.Parameter.get_data_from_file_or_url_as_string",
            return_value='{"number": 2.5}',
        ) as get_content:
            ReadParameter(ParameterEntry(Name="Number", sValue=url), param, model)

        get_content.assert_called_once_with("https://example.com/profile.json")
        self.assertEqual(2.5, param.value)

    def test_json_pointer_parameter_input_reports_invalid_sources_and_values(self):
        model = self._new_model()

        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_json_path = Path(temp_dir) / "invalid.json"
            invalid_json_path.write_text('{"number":', encoding="utf-8")
            object_path = Path(temp_dir) / "object.json"
            object_path.write_text('{"object": {"number": 2.5}, "null_value": null}', encoding="utf-8")

            cases = [
                (f"{invalid_json_path}#/number", "not valid JSON"),
                (f"{object_path}#/missing", "key was not found"),
                (f"{object_path}#/object", "select one of its values"),
                (f"{object_path}#/null_value", "selected JSON value is null"),
            ]
            for reference, message in cases:
                with self.subTest(reference=reference), self.assertRaisesRegex(ValueError, message):
                    ReadParameter(
                        ParameterEntry(Name="Number", sValue=reference),
                        floatParameter("Number", DefaultValue=0.0, AllowExtendedInput=True),
                        model,
                    )

    def test_profile_parameters_allow_json_pointer_extended_input(self):
        model = self._new_model()
        profile_parameters = [
            model.reserv.depth,
            model.reserv.Tsurf,
            model.surfaceplant.ambient_temperature,
            model.surfaceplant.project_latitude,
            model.surfaceplant.project_longitude,
        ]

        self.assertTrue(all(parameter.AllowExtendedInput for parameter in profile_parameters))

        profile = {
            "target_depth_km": 5,
            "latitude": 34.90808838,
            "longitude": -117.88552852,
            "surface_temperature_c": 14.8,
            "ambient_temperature_c": 14.8,
        }
        parameter_pointers = [
            (model.reserv.depth, "/target_depth_km"),
            (model.reserv.Tsurf, "/surface_temperature_c"),
            (model.surfaceplant.ambient_temperature, "/ambient_temperature_c"),
            (model.surfaceplant.project_latitude, "/latitude"),
            (model.surfaceplant.project_longitude, "/longitude"),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "edwards.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            input_path = Path(temp_dir) / "model-input.txt"
            input_path.write_text(
                "\n".join(f"{parameter.Name}, {profile_path}#{pointer}" for parameter, pointer in parameter_pointers),
                encoding="utf-8",
            )
            parsed_inputs = read_input_file(model.logger, input_path)
            for parameter, _ in parameter_pointers:
                ReadParameter(
                    parsed_inputs[parameter.Name],
                    parameter,
                    model,
                )

        self.assertEqual(5.0, model.reserv.depth.value)
        self.assertEqual(14.8, model.reserv.Tsurf.value)
        self.assertEqual(14.8, model.surfaceplant.ambient_temperature.value)
        self.assertEqual(34.90808838, model.surfaceplant.project_latitude.value)
        self.assertEqual(-117.88552852, model.surfaceplant.project_longitude.value)

    def test_read_bool_from_URL(self):
        model = self._new_model()
        param = boolParameter("Count", DefaultValue=True, AllowExtendedInput=True)

        ReadParameter(
            ParameterEntry(
                Name="Geophires Enabled",
                sValue="https://raw.githubusercontent.com/malcolm-BSD/GEOPHIRES-X/test-assets-v1.2/tests/assets/params/bool.txt",
            ),
            param,
            model,
        )

        self.assertIsInstance(param.value, bool)
        self.assertEqual(param.value, False)

    def test_read_int_from_URL(self):
        model = self._new_model()
        param = intParameter("Count", DefaultValue=0, AllowableRange=[0, 113, 1000000], AllowExtendedInput=True)

        ReadParameter(
            ParameterEntry(
                Name="Count",
                sValue="https://raw.githubusercontent.com/malcolm-BSD/GEOPHIRES-X/test-assets-v1.2/tests/assets/params/int.txt",
            ),
            param,
            model,
        )

        self.assertIsInstance(param.value, int)
        self.assertEqual(param.value, 113)

    def test_read_float_from_URL(self):
        model = self._new_model()
        param = floatParameter(
            "CAPEX",
            DefaultValue=1379.0,
            UnitType=Units.COSTPERMASS,
            PreferredUnits=CostPerMassUnit.DOLLARSPERMT,
            CurrentUnits=CostPerMassUnit.CENTSSPERMT,
            AllowExtendedInput=True,
        )

        ReadParameter(
            ParameterEntry(
                Name="CAPEX",
                sValue="https://raw.githubusercontent.com/malcolm-BSD/GEOPHIRES-X/test-assets-v1.2/tests/assets/params/float.txt",
            ),
            param,
            model,
        )

        self.assertIsInstance(param.value, float)
        self.assertEqual(param.value, 3.14156)

    def test_read_str_from_URL(self):
        model = self._new_model()
        param = strParameter("Gettysburg", DefaultValue="", AllowExtendedInput=True)

        ReadParameter(
            ParameterEntry(
                Name="Gettysburg",
                sValue="https://raw.githubusercontent.com/malcolm-BSD/GEOPHIRES-X/test-assets-v1.2/tests/assets/params/string.txt",
            ),
            param,
            model,
        )

        self.assertIsInstance(param.value, str)
        self.assertIn("Four score and seven years ago", param.value)

    def test_read_list_from_URL(self):
        model = self._new_model()
        param = listParameter("Random", DefaultValue=[], AllowExtendedInput=True)

        ReadParameter(
            ParameterEntry(
                Name="Random",
                sValue="https://raw.githubusercontent.com/malcolm-BSD/GEOPHIRES-X/test-assets-v1.2/tests/assets/params/list.txt",
            ),
            param,
            model,
        )

        self.assertIsInstance(param.value, list)
        self.assertEqual(345, param.value[0])

    def test_read_gradient_from_file(self):
        model = self._new_model()
        param = listParameter(
            Name="Gradients",
            DefaultValue=[1000.0],
            Min=0.0,
            Max=100000.0,
            UnitType=Units.TEMP_GRADIENT,
            PreferredUnits=TemperatureGradientUnit.DEGREESCPERM,
            CurrentUnits=TemperatureGradientUnit.DEGREESFPERFT,
            AllowExtendedInput=True,
        )
        root = str(Path(__file__).resolve().parents[0])  # adjust depth as needed
        csv_file = root + "/assets/params/gradients.csv"

        ReadParameter(
            ParameterEntry(Name="Gradients", sValue=csv_file, raw_entry=f"Gradients, {csv_file}"),
            param,
            model,
        )

        self.assertEqual([0.2916302128900554, 0.32808398950131235, 0.27340332458442695], param.value[0:3])

    def test_read_test_read_thicknesses_from_URL(self):
        model = self._new_model()
        param = listParameter(
            Name="Thicknesses",
            DefaultValue=[1000.0],
            Min=0.0,
            Max=100000.0,
            UnitType=Units.LENGTH,
            PreferredUnits=LengthUnit.METERS,
            CurrentUnits=LengthUnit.METERS,
            AllowExtendedInput=True,
        )
        ReadParameter(
            ParameterEntry(
                Name="Thicknesses",
                sValue="https://raw.githubusercontent.com/malcolm-BSD/GEOPHIRES-X/test-assets-v1.3/tests/assets/params/thicknesses.csv",
                raw_entry="Thicknesses, https://raw.githubusercontent.com/malcolm-BSD/GEOPHIRES-X/test-assets-v1.3/tests/assets/params/thicknesses.csv",
            ),
            param,
            model,
        )

        self.assertEqual([431.5967999999999, 193.85279999999997, 293.82719999999995], param.value[0:3])

    def test_read_parameter_gradient_keeps_array_length(self):
        model = self._new_model()
        param = listParameter(
            Name="Gradients",
            DefaultValue=[],
            PreferredUnits=TemperatureUnit.CELSIUS,
        )
        param.AllowHistoricalArrayInput = True
        param.HistoricalXDimension = "distance"
        param.HistoricalYDimension = "temperature"
        param.HistoricalDefaultXUnits = "meter"
        param.HistoricalDefaultYUnits = "celsius"
        param.HistoricalResampleToHourlyYear = False

        entry = ParameterEntry(
            Name="Gradients",
            sValue="Distance (feet),Temperature (Fahrenheit)\n0,32\n10,50\n",
            raw_entry="Gradients, Distance (feet),Temperature (Fahrenheit)\n0,32\n10,50\n",
        )
        ReadParameter(entry, param, model)

        self.assertEqual(2, len(param.value))
        self.assertTrue(np.allclose(param.value[0], 0.0))

    def test_derive_numseg_from_gradient_thickness(self):
        self.assertEqual(3, derive_numseg_from_gradient_thickness([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0]))

    def test_convert_cost_per_mass(self):
        result: GeophiresXResult = GeophiresXClient().get_geophires_result(
            ImmutableGeophiresInputParameters(
                from_file_path=self._get_test_file_path("examples/example_SAM-single-owner-PPA-6_carbon-revenue.txt"),
                params={
                    "Starting Carbon Credit Value": "1 USD/kilogram",
                    "Ending Carbon Credit Value": 100,  # arbitrary high number
                    "Carbon Escalation Rate Per Year": 0,
                    "Units:Total Saved Carbon Production": "kilogram",
                },
            )
        )

        def _cash_flow_row(r: GeophiresXResult, row_name: str) -> str:
            from geophires_x.EconomicsSam import _cash_flow_profile_row
            from geophires_x.GeoPHIRESUtils import is_float

            return [it for it in _cash_flow_profile_row(r.result["SAM CASH FLOW PROFILE"], row_name) if is_float(it)]

        capacity_payment_revenue_usd_row = _cash_flow_row(result, "Capacity payment revenue ($)")
        total_capacity_payment_revenue_usd = sum(capacity_payment_revenue_usd_row)

        total_avoided_carbon_emissions_vu: dict[str, float] = result.result["SUMMARY OF RESULTS"][
            "Total Avoided Carbon Emissions"
        ]
        self.assertEqual("kilogram", total_avoided_carbon_emissions_vu["unit"])
        self.assertEqual(int(total_avoided_carbon_emissions_vu["value"]), total_capacity_payment_revenue_usd)

    def test_has_quantity_as_units(self):
        param: OutputParameter = total_capex_parameter_output_parameter()
        param.CurrentUnits = CurrencyUnit.MDOLLARS
        param.value = 1
        self.assertEqual(1e6, param.quantity("USD").magnitude)
        self.assertEqual(1e6, param.quantity(CurrencyUnit.DOLLARS).magnitude)

    # noinspection PyMethodMayBeStatic
    def _new_model(self) -> Model:
        stash_cwd = Path.cwd()
        stash_sys_argv = sys.argv

        sys.argv = [""]

        m = Model(enable_geophires_logging_config=False)

        sys.argv = stash_sys_argv
        os.chdir(stash_cwd)

        return m


if __name__ == "__main__":
    unittest.main()
