import json
import unittest

from geophires_x_schema_generator import GeophiresXSchemaGenerator
from geophires_x_schema_generator import HipRaXSchemaGenerator
from tests.base_test_case import BaseTestCase


class GeophiresXSchemaGeneratorTestCase(BaseTestCase):
    def test_geophires_x_enablement_dependencies_are_explicit(self):
        request_schema, _ = GeophiresXSchemaGenerator().generate_json_schema()

        self.assertEqual(
            [{"parameter": "Reservoir Model", "operator": "in", "values": [6]}],
            request_schema["properties"]["TOUGH2 Executable Path"]["x-dependencies"]["conditions"],
        )
        self.assertEqual(
            [
                {"parameter": "Reservoir Model", "operator": "in", "values": [8]},
                {"parameter": "Well Geometry Configuration", "operator": "in", "values": [1, 5]},
                {"parameter": "Flowrate Model", "operator": "in", "values": [2]},
            ],
            request_schema["properties"]["Flowrate File"]["x-dependencies"]["conditions"],
        )
        self.assertEqual(
            "enable",
            request_schema["properties"]["Drawdown Parameter"]["x-dependencies"]["effect"],
        )
        self.assertEqual(
            [{"parameter": "Economic Model", "operator": "in", "values": [1]}],
            request_schema["properties"]["Fixed Charge Rate"]["x-dependencies"]["conditions"],
        )
        self.assertNotIn("x-dependencies", request_schema["properties"]["Discount Rate"])

        input_parameters_json, _ = GeophiresXSchemaGenerator().get_parameters_json()
        input_parameters = json.loads(input_parameters_json)
        source_dependencies = {
            name: parameter["UI"]["dependency"]
            for name, parameter in input_parameters.items()
            if parameter["UI"]["dependency"] is not None
        }
        schema_dependencies = {
            name: definition["x-dependencies"]
            for name, definition in request_schema["properties"].items()
            if "x-dependencies" in definition
        }
        self.assertEqual(318, len(source_dependencies))
        self.assertEqual(set(source_dependencies), set(schema_dependencies))
        for name, source_dependency in source_dependencies.items():
            with self.subTest(parameter=name):
                emitted_dependency = {
                    key: value for key, value in schema_dependencies[name].items() if key != "depends_on"
                }
                self.assertEqual(source_dependency, emitted_dependency)

    def test_economic_model_dependencies_match_runtime_consumers(self):
        input_parameters_json, _ = GeophiresXSchemaGenerator().get_parameters_json()
        input_parameters = json.loads(input_parameters_json)

        for parameter_name in ["Discount Rate", "Investment Tax Credit Rate"]:
            with self.subTest(unrestricted=parameter_name):
                self.assertIsNone(input_parameters[parameter_name]["UI"]["dependency"])

        for parameter_name in [
            "Fraction of Investment in Bonds",
            "Inflated Bond Interest Rate",
            "Combined Income Tax Rate",
            "Property Tax Rate",
        ]:
            with self.subTest(bicycle_and_sam=parameter_name):
                dependency = input_parameters[parameter_name]["UI"]["dependency"]
                self.assertEqual("all", dependency["match"])
                self.assertEqual(
                    [{"parameter": "Economic Model", "operator": "in", "values": [3, 5]}],
                    dependency["conditions"],
                )

        for parameter_name in ["Inflated Equity Interest Rate", "Gross Revenue Tax Rate"]:
            with self.subTest(bicycle_only=parameter_name):
                dependency = input_parameters[parameter_name]["UI"]["dependency"]
                self.assertEqual(
                    [{"parameter": "Economic Model", "operator": "in", "values": [3]}],
                    dependency["conditions"],
                )

        inflation_dependency = input_parameters["Inflation Rate"]["UI"]["dependency"]
        self.assertEqual("any", inflation_dependency["match"])
        self.assertEqual(
            [
                {"parameter": "Economic Model", "operator": "in", "values": [3, 5]},
                {
                    "parameter": "Production Tax Credit Inflation Adjusted",
                    "operator": "in",
                    "values": [True],
                },
            ],
            inflation_dependency["conditions"],
        )

        construction_inflation_dependency = input_parameters["Inflation Rate During Construction"]["UI"]["dependency"]
        self.assertEqual("any", construction_inflation_dependency["match"])
        self.assertEqual(
            [
                {"parameter": "Economic Model", "operator": "in", "values": [1, 2, 3, 5]},
                {"parameter": "Reservoir Model", "operator": "in", "values": [7]},
            ],
            construction_inflation_dependency["conditions"],
        )

        for parameter_name in [
            "Operation & Maintenance Cost of Surface Plant",
            "Capital Cost for Surface Plant for Direct-use System",
        ]:
            with self.subTest(clgs_only=parameter_name):
                dependency = input_parameters[parameter_name]["UI"]["dependency"]
                self.assertEqual("all", dependency["match"])
                self.assertIn(
                    {"parameter": "Economic Model", "operator": "in", "values": [4]},
                    dependency["conditions"],
                )

    def test_xlco_groups_and_dependencies_match_active_commodities(self):
        input_parameters_json, _ = GeophiresXSchemaGenerator().get_parameters_json()
        input_parameters = json.loads(input_parameters_json)

        expected_placement = {
            "Social Discount Rate": ("XLCO Shared Assumptions", 320),
            "XLCO(E|H|C) Carbon Price": ("XLCO Shared Assumptions", 330),
            "XLCO(E|H|C) Water Shadow Price": ("XLCO Shared Assumptions", 340),
            "Idle Rig Discount Rate": ("XLCO Shared Assumptions", 350),
            "XLCO(E|H|C) Construction Jobs Per Rig": ("XLCO Employment Assumptions", 360),
            "XLCO(E|H|C) Operations Jobs Per MW": ("XLCO Employment Assumptions", 370),
            "XLCO(E|H|C) Indirect Jobs Multiplier": ("XLCO Employment Assumptions", 380),
            "XLCO(E|H|C) Average Monthly Wage": ("XLCO Employment Assumptions", 390),
            "XLCOE Avoided Emissions Intensity": ("XLCO Electricity Benefits", 400),
            "XLCOE REC Price": ("XLCO Electricity Benefits", 410),
            "XLCOE Displaced Water Use Intensity": ("XLCO Electricity Benefits", 420),
            "XLCOH Avoided Emissions Intensity": ("XLCO Heat Benefits", 430),
            "XLCOH Thermal REC": ("XLCO Heat Benefits", 440),
            "XLCOH Displaced Water Use Intensity": ("XLCO Heat Benefits", 450),
            "XLCOC Avoided Emissions Intensity": ("XLCO Cooling Benefits", 460),
            "XLCOC Thermal REC": ("XLCO Cooling Benefits", 470),
            "XLCOC Displaced Water Use Intensity": ("XLCO Cooling Benefits", 480),
        }
        for parameter_name, (group, order) in expected_placement.items():
            with self.subTest(placement=parameter_name):
                ui = input_parameters[parameter_name]["UI"]
                self.assertEqual("Economics", ui["tab"])
                self.assertEqual(group, ui["group"])
                self.assertEqual(order, ui["order"])
                self.assertNotEqual("XLCO parameters", ui["group"])

        xlco_enabled = {
            "parameter": "Do XLCO(E|H|C) Calculations",
            "operator": "in",
            "values": [True],
        }
        for parameter_name in list(expected_placement)[:8]:
            with self.subTest(shared_dependency=parameter_name):
                dependency = input_parameters[parameter_name]["UI"]["dependency"]
                self.assertEqual("all", dependency["match"])
                self.assertEqual([xlco_enabled], dependency["conditions"])

        electricity_conditions = [
            xlco_enabled,
            {"parameter": "End-Use Option", "operator": "in", "values": [1, 31, 32, 41, 42, 51, 52]},
        ]
        for parameter_name in list(expected_placement)[8:11]:
            with self.subTest(electricity_dependency=parameter_name):
                self.assertEqual(
                    electricity_conditions,
                    input_parameters[parameter_name]["UI"]["dependency"]["conditions"],
                )

        heat_conditions = [
            xlco_enabled,
            {"parameter": "End-Use Option", "operator": "in", "values": [2, 31, 32, 41, 42, 51, 52]},
            {"parameter": "Power Plant Type", "operator": "in", "values": [1, 2, 3, 4, 6, 7, 8, 9]},
        ]
        for parameter_name in list(expected_placement)[11:14]:
            with self.subTest(heat_dependency=parameter_name):
                self.assertEqual(
                    heat_conditions,
                    input_parameters[parameter_name]["UI"]["dependency"]["conditions"],
                )

        cooling_conditions = [
            xlco_enabled,
            {"parameter": "End-Use Option", "operator": "in", "values": [2]},
            {"parameter": "Power Plant Type", "operator": "in", "values": [5]},
        ]
        for parameter_name in list(expected_placement)[14:]:
            with self.subTest(cooling_dependency=parameter_name):
                self.assertEqual(
                    cooling_conditions,
                    input_parameters[parameter_name]["UI"]["dependency"]["conditions"],
                )

    def test_additional_economic_dependencies_preserve_alternate_runtime_paths(self):
        input_parameters_json, _ = GeophiresXSchemaGenerator().get_parameters_json()
        input_parameters = json.loads(input_parameters_json)

        self.assertEqual(
            {
                "effect": "enable",
                "match": "any",
                "conditions": [
                    {"parameter": "Economic Model", "operator": "in", "values": [4]},
                    {"parameter": "Well Drilling Cost Correlation", "operator": "in", "values": [5]},
                ],
                "reason": "Used by CLGS or by the simple per-meter drilling-cost method.",
            },
            input_parameters["All-in Vertical Drilling Costs"]["UI"]["dependency"],
        )
        self.assertEqual(
            [{"parameter": "Well Geometry Configuration", "operator": "in", "values": [1, 2, 4, 5]}],
            input_parameters["All-in Nonvertical Drilling Costs"]["UI"]["dependency"]["conditions"],
        )
        self.assertEqual(
            [{"parameter": "End-Use Option", "operator": "in", "values": [1, 31, 32, 41, 42, 51, 52]}],
            input_parameters["Capital Cost for Power Plant for Electricity Generation"]["UI"]["dependency"][
                "conditions"
            ],
        )
        transmission_dependency = input_parameters["Transmission/pipeline Cost"]["UI"]["dependency"]
        self.assertEqual("any", transmission_dependency["match"])
        self.assertEqual(
            [
                {"parameter": "Total Capital Cost", "operator": "provided", "values": [False]},
                {"parameter": "Economic Model", "operator": "in", "values": [4]},
            ],
            transmission_dependency["conditions"],
        )

        direct_use_clgs_conditions = input_parameters["Capital Cost for Surface Plant for Direct-use System"]["UI"][
            "dependency"
        ]["conditions"]
        self.assertIn(
            {"parameter": "End-Use Option", "operator": "in", "values": [2]},
            direct_use_clgs_conditions,
        )
        self.assertNotIn(
            {"parameter": "End-Use Option", "operator": "in", "values": [2]},
            input_parameters["Operation & Maintenance Cost of Surface Plant"]["UI"]["dependency"]["conditions"],
        )

        valco_heat_names = [
            "VALCOH System Average Energy Value",
            "VALCOH Technology Energy Value",
            "VALCOH System Average Capacity Value",
            "VALCOH Technology Capacity Value",
            "VALCOH System Average Flexibility Value",
            "VALCOH Technology Flexibility Value",
            "VALCOH Basis Capacity Value",
            "VALCOH Capacity Credit",
            "VALCOH Base Flexibility Value",
            "VALCOH Flexibility Multiplier",
        ]
        heat_plant_condition = {
            "parameter": "Power Plant Type",
            "operator": "in",
            "values": [1, 2, 3, 4, 6, 7, 8, 9],
        }
        for parameter_name in valco_heat_names:
            with self.subTest(valco_heat_dependency=parameter_name):
                self.assertIn(
                    heat_plant_condition,
                    input_parameters[parameter_name]["UI"]["dependency"]["conditions"],
                )

        self.assertEqual(
            [
                {"parameter": "Do Carbon Price Calculations", "operator": "in", "values": [True]},
                {"parameter": "End-Use Option", "operator": "in", "values": [1, 31, 32, 41, 42, 51, 52]},
            ],
            input_parameters["Current Grid CO2 production"]["UI"]["dependency"]["conditions"],
        )
        self.assertEqual(
            [
                {"parameter": "Do Carbon Price Calculations", "operator": "in", "values": [True]},
                {"parameter": "End-Use Option", "operator": "in", "values": [2, 31, 32, 41, 42, 51, 52]},
            ],
            input_parameters["CO2 produced by Natural Gas"]["UI"]["dependency"]["conditions"],
        )
        self.assertEqual(
            [{"parameter": "End-Use Option", "operator": "in", "values": [1, 31, 32, 41, 42, 51, 52]}],
            input_parameters["Estimated Jobs Created per MW of Electricity Produced"]["UI"]["dependency"]["conditions"],
        )

    def test_surface_plant_groups_and_dependencies_match_runtime_routing(self):
        input_parameters_json, _ = GeophiresXSchemaGenerator().get_parameters_json()
        input_parameters = json.loads(input_parameters_json)

        expected_group_members = {
            "Plant configuration": ["Power Plant Type"],
            "Thermal energy storage": ["TESS Enabled"],
            "Absorption chiller": ["Use Advanced Absorption Chiller", "Absorption Chiller COP"],
            "Injection conditions": ["Injection Temperature Model", "Injection Temperature"],
            "Common plant operation": [
                "Circulation Pump Efficiency",
                "Utilization Factor",
                "End-Use Efficiency Factor",
                "Ambient Temperature",
                "Surface Piping Length",
                "Plant Outlet Pressure",
            ],
            "Energy prices": ["Electricity Rate", "Heat Rate"],
            "Weather data": ["Weather Data Year"],
            "Cogeneration configuration": ["CHP Fraction", "CHP Bottoming Entering Temperature"],
            "Project delivery": ["Construction Years"],
            "District heating demand": ["District Heating Demand Option"],
            "Dispatch configuration": [
                "Dispatch Demand Source",
                "Dispatch Flow Strategy",
                "Maximum Dispatch Flow Fraction",
                "Minimum Dispatch Flow Fraction",
                "Minimum Dispatch Runtime Fraction",
                "Dispatch Analysis Start Year",
                "Dispatch Analysis End Year",
            ],
            "Dispatch demand profiles": [
                "Annual Heat Demand",
                "Annual Electricity Demand",
                "Annual Cooling Demand",
            ],
            "TESS design and cost": [
                "TESS Volume",
                "TESS Cost per Cubic Meter",
                "TESS Fixed O&M Fraction",
                "TESS Working Fluid",
                "TESS Pressure Mode",
                "TESS Pressure",
                "TESS Pressure Safety Factor",
            ],
            "TESS thermal limits": [
                "TESS Minimum Useful Temperature",
                "TESS Maximum Temperature",
                "TESS Target Temperature",
                "TESS Initial Temperature",
                "TESS Deadband Range",
            ],
            "TESS performance": [
                "TESS Charge Efficiency",
                "TESS Discharge Efficiency",
                "TESS Daily Heat Loss Fraction",
                "TESS Maximum Charge Power",
                "TESS Maximum Discharge Power",
                "TESS Subhourly Demand Peak Multiplier",
            ],
            "TESS controls": [
                "TESS Charge Control Strategy",
                "TESS Charge Flow Fraction",
                "TESS Moving Average Window",
                "TESS SOC Control Gain",
            ],
            "SBT working-fluid properties": [
                "Working Fluid Heat Capacity",
                "Working Fluid Density",
                "Working Fluid Thermal Conductivity",
                "Working Fluid Dynamic Viscosity",
            ],
            "AGS thermodynamic reference": ["Dead-state Pressure"],
            "AGS CO2 power cycle": [
                "Isentropic Efficiency for CO2 Turbine",
                "Generator Conversion Efficiency",
                "Isentropic Efficiency for CO2 Compressor",
                "CO2 Temperature Decline with Cooling",
                "CO2 Turbine Outlet Pressure",
            ],
            "Advanced chiller model": [
                "Absorption Chiller Use Pint",
                "Absorption Chiller Use CoolProp",
                "Absorption Chiller Refrigerant Family",
                "Absorption Chiller Effect Type",
                "Absorption Chiller Effect Multiplier Override",
                "Absorption Chiller Catalog Path",
            ],
            "Chiller hydraulics": [
                "Absorption Chiller Chilled DeltaT",
                "Absorption Chiller Pump Head",
                "Absorption Chiller Pump Efficiency",
            ],
            "Chiller part-load and dispatch": [
                "Absorption Chiller Min Part Load Ratio",
                "Absorption Chiller Turndown Ratio",
                "Absorption Chiller Dispatch Strategy",
                "Absorption Chiller PLR Segments",
            ],
            "Chiller temperature profiles": [
                "Absorption Chiller Use Hourly Temperatures",
                "Absorption Chiller Generator Temperature Profile",
                "Absorption Chiller Evaporator Temperature Profile",
                "Absorption Chiller Condenser Temperature Profile",
            ],
            "District heating demand file": [
                "District Heating Demand File Name",
                "District Heating Demand Data Time Resolution",
                "District Heating Demand Data Column Number",
            ],
            "District heating weather demand": [
                "Temperature File Name",
                "Temperature Data Column Number",
                "Number of Housing Units",
                "Constant Anchor Demand",
                "US Census Division",
            ],
            "Heat pump": ["Heat Pump COP"],
        }

        surface_parameters = {
            name: parameter["UI"]
            for name, parameter in input_parameters.items()
            if parameter["UI"]["tab"] == "SurfacePlant"
        }
        self.assertEqual(87, len(surface_parameters))
        self.assertEqual(
            set(surface_parameters),
            {name for names in expected_group_members.values() for name in names},
        )
        self.assertEqual(
            len(surface_parameters),
            len({ui["order"] for ui in surface_parameters.values()}),
        )
        for group, parameter_names in expected_group_members.items():
            for parameter_name in parameter_names:
                with self.subTest(group=group, parameter=parameter_name):
                    self.assertEqual(group, surface_parameters[parameter_name]["group"])

        chiller_conditions = [
            {"parameter": "End-Use Option", "operator": "in", "values": [2]},
            {"parameter": "Power Plant Type", "operator": "in", "values": [5]},
        ]
        self.assertEqual(
            chiller_conditions,
            surface_parameters["Use Advanced Absorption Chiller"]["dependency"]["conditions"],
        )
        self.assertEqual(
            [
                *chiller_conditions,
                {"parameter": "Use Advanced Absorption Chiller", "operator": "in", "values": [True]},
            ],
            surface_parameters["Absorption Chiller Effect Type"]["dependency"]["conditions"],
        )

        district_heating_conditions = [
            {"parameter": "End-Use Option", "operator": "in", "values": [2]},
            {"parameter": "Power Plant Type", "operator": "in", "values": [7]},
            {"parameter": "Operating Mode", "operator": "in", "values": [1, "Baseload"]},
        ]
        self.assertEqual(
            district_heating_conditions,
            surface_parameters["District Heating Demand Option"]["dependency"]["conditions"],
        )
        self.assertEqual(
            [
                *district_heating_conditions,
                {"parameter": "District Heating Demand Option", "operator": "in", "values": [1]},
            ],
            surface_parameters["District Heating Demand File Name"]["dependency"]["conditions"],
        )

        injection_temperature_dependency = surface_parameters["Injection Temperature"]["dependency"]
        self.assertEqual("any", injection_temperature_dependency["match"])
        self.assertEqual(
            [
                {"parameter": "Reservoir Model", "operator": "in", "values": [0, 1, 2, 3, 4, 5, 6, 7]},
                {"parameter": "Injection Temperature Model", "operator": "in", "values": [1]},
            ],
            injection_temperature_dependency["conditions"],
        )
        self.assertIsNone(surface_parameters["Ambient Temperature"]["dependency"])

        ags_conditions = [
            {"parameter": "Is AGS", "operator": "in", "values": [True]},
            {"parameter": "Reservoir Model", "operator": "in", "values": [0, 1, 2, 3, 4, 5, 6, 7]},
            {"parameter": "End-Use Option", "operator": "in", "values": [1]},
        ]
        self.assertEqual(
            ags_conditions,
            surface_parameters["Dead-state Pressure"]["dependency"]["conditions"],
        )
        ags_co2_conditions = [
            *ags_conditions,
            {"parameter": "Heat Transfer Fluid", "operator": "in", "values": [2]},
        ]
        for parameter_name in expected_group_members["AGS CO2 power cycle"]:
            with self.subTest(ags_co2_dependency=parameter_name):
                self.assertEqual(
                    ags_co2_conditions,
                    surface_parameters[parameter_name]["dependency"]["conditions"],
                )

    def test_wellbore_groups_and_dependencies_match_runtime_routing(self):
        input_parameters_json, _ = GeophiresXSchemaGenerator().get_parameters_json()
        input_parameters = json.loads(input_parameters_json)

        expected_group_members = {
            "Wellbore model controls": ["Is AGS", "Well Geometry Configuration", "Heat Transfer Fluid"],
            "Wellfield sizing": [
                "Number of Production Wells",
                "Number of Injection Wells",
                "Number of Doublets",
                "Number of Injection Wells per Production Well",
            ],
            "Vertical well geometry": ["Production Well Diameter", "Injection Well Diameter"],
            "Wellbore thermal model": [
                "Ramey Production Wellbore Model",
                "Production Wellbore Temperature Drop",
                "Injection Wellbore Temperature Gain",
            ],
            "Wellbore flow and pressure": [
                "Production Flow Rate per Well",
                "Production Wellhead Pressure",
                "Reservoir Hydrostatic Pressure",
            ],
            "Well operating limits": ["Maximum Drawdown", "Well Integrity Maximum Lifetime"],
            "Injection reservoir overpressure": [
                "Injection Reservoir Temperature",
                "Injection Reservoir Depth",
                "Injection Reservoir Initial Pressure",
                "Injection Reservoir Inflation Rate",
            ],
            "Multilateral geometry": [
                "Nonvertical Length per Multilateral Section",
                "Nonvertical Wellbore Diameter",
                "Number of Multilateral Sections",
                "Number of Multilateral Sections per Vertical Section",
                "Multilaterals Cased",
            ],
            "TOUGH2 doublet geometry": ["Well Separation"],
            "SBT flow control": ["Flowrate Model"],
            "SBT well geometry": [
                "Vertical Section Length",
                "Vertical Wellbore Spacing",
                "Lateral Spacing",
                "Lateral Inclination Angle",
                "Discretization Length",
                "Junction Depth",
                "Lateral Endpoint Depth",
            ],
            "SBT lateral construction": [
                "Lateral Casing Thermal Conductivity",
                "Lateral Cement Thermal Conductivity",
                "Lateral Casing Thickness",
                "Lateral Cement Thickness",
            ],
            "AGS fluid properties": ["Water Thermal Conductivity"],
            "AGS calculation controls": ["Closed Loop Calculation Start Year"],
            "Compatibility aliases": ["Closed-loop Configuration"],
        }

        wellbore_parameters = {
            name: parameter["UI"]
            for name, parameter in input_parameters.items()
            if parameter["UI"]["tab"] == "WellBores"
        }
        self.assertEqual(42, len(wellbore_parameters))
        self.assertEqual(
            set(wellbore_parameters),
            {name for names in expected_group_members.values() for name in names},
        )
        self.assertEqual(
            len(wellbore_parameters),
            len({ui["order"] for ui in wellbore_parameters.values()}),
        )
        for group, parameter_names in expected_group_members.items():
            for parameter_name in parameter_names:
                with self.subTest(group=group, parameter=parameter_name):
                    self.assertEqual(group, wellbore_parameters[parameter_name]["group"])

        ags_conditions = [
            {"parameter": "Is AGS", "operator": "in", "values": [True]},
            {"parameter": "Reservoir Model", "operator": "in", "values": [0, 1, 2, 3, 4, 5, 6, 7]},
        ]
        self.assertEqual(
            ags_conditions,
            wellbore_parameters["Heat Transfer Fluid"]["dependency"]["conditions"],
        )
        self.assertEqual(
            ags_conditions,
            wellbore_parameters["Water Thermal Conductivity"]["dependency"]["conditions"],
        )
        self.assertIsNone(wellbore_parameters["Closed-loop Configuration"]["dependency"])

        self.assertEqual(
            [
                {"parameter": "Reservoir Model", "operator": "in", "values": [0, 1, 2, 3, 4, 5, 6, 8]},
                {"parameter": "Number of Production Wells", "operator": "provided", "values": [False]},
                {"parameter": "Number of Injection Wells", "operator": "provided", "values": [False]},
                {
                    "parameter": "Number of Injection Wells per Production Well",
                    "operator": "provided",
                    "values": [False],
                },
            ],
            wellbore_parameters["Number of Doublets"]["dependency"]["conditions"],
        )
        self.assertEqual(
            [
                {"parameter": "Reservoir Model", "operator": "in", "values": [0, 1, 2, 3, 4, 5, 6, 8]},
                {"parameter": "Number of Doublets", "operator": "provided", "values": [False]},
                {"parameter": "Number of Injection Wells", "operator": "provided", "values": [False]},
            ],
            wellbore_parameters["Number of Injection Wells per Production Well"]["dependency"]["conditions"],
        )

        conventional_reservoir_condition = {
            "parameter": "Reservoir Model",
            "operator": "in",
            "values": [0, 1, 2, 3, 4, 5, 6],
        }
        self.assertEqual(
            [conventional_reservoir_condition],
            wellbore_parameters["Ramey Production Wellbore Model"]["dependency"]["conditions"],
        )
        self.assertEqual(
            [
                conventional_reservoir_condition,
                {"parameter": "Ramey Production Wellbore Model", "operator": "in", "values": [False]},
            ],
            wellbore_parameters["Production Wellbore Temperature Drop"]["dependency"]["conditions"],
        )
        self.assertEqual(
            [{"parameter": "Reservoir Model", "operator": "in", "values": [1, 2, 3, 4, 5, 6, 7]}],
            wellbore_parameters["Injection Wellbore Temperature Gain"]["dependency"]["conditions"],
        )
        flow_dependency = wellbore_parameters["Production Flow Rate per Well"]["dependency"]
        self.assertEqual("any", flow_dependency["match"])
        self.assertEqual(
            [
                {"parameter": "Reservoir Model", "operator": "in", "values": [0, 1, 2, 3, 4, 5, 6, 7]},
                {"parameter": "Flowrate Model", "operator": "in", "values": [1]},
            ],
            flow_dependency["conditions"],
        )
        self.assertEqual(
            [
                {"parameter": "Reservoir Model", "operator": "in", "values": [0, 1, 2, 3, 4, 5, 6, 8]},
                {"parameter": "Reservoir Impedance", "operator": "provided", "values": [False]},
            ],
            wellbore_parameters["Production Wellhead Pressure"]["dependency"]["conditions"],
        )
        conventional_non_ags_conditions = [
            conventional_reservoir_condition,
            {"parameter": "Is AGS", "operator": "in", "values": [False]},
        ]
        for parameter_name in [
            "Reservoir Hydrostatic Pressure",
            "Maximum Drawdown",
            "Well Integrity Maximum Lifetime",
        ]:
            with self.subTest(conventional_wellbore_dependency=parameter_name):
                self.assertEqual(
                    conventional_non_ags_conditions,
                    wellbore_parameters[parameter_name]["dependency"]["conditions"],
                )

        nonvertical_configuration_condition = {
            "parameter": "Well Geometry Configuration",
            "operator": "in",
            "values": [1, 2, 4, 5],
        }
        self.assertEqual(
            [nonvertical_configuration_condition],
            wellbore_parameters["Nonvertical Length per Multilateral Section"]["dependency"]["conditions"],
        )
        self.assertEqual(
            [
                nonvertical_configuration_condition,
                {
                    "parameter": "Number of Multilateral Sections per Vertical Section",
                    "operator": "provided",
                    "values": [False],
                },
            ],
            wellbore_parameters["Number of Multilateral Sections"]["dependency"]["conditions"],
        )
        self.assertEqual(
            [
                nonvertical_configuration_condition,
                {"parameter": "Is AGS", "operator": "in", "values": [False]},
                {"parameter": "Number of Multilateral Sections", "operator": "provided", "values": [False]},
            ],
            wellbore_parameters["Number of Multilateral Sections per Vertical Section"]["dependency"]["conditions"],
        )

        sbt_conditions = [
            {"parameter": "Reservoir Model", "operator": "in", "values": [8]},
            {"parameter": "Well Geometry Configuration", "operator": "in", "values": [1, 5]},
        ]
        self.assertEqual(
            sbt_conditions,
            wellbore_parameters["Flowrate Model"]["dependency"]["conditions"],
        )
        for parameter_name in expected_group_members["SBT well geometry"]:
            with self.subTest(sbt_geometry_dependency=parameter_name):
                self.assertEqual(
                    sbt_conditions,
                    wellbore_parameters[parameter_name]["dependency"]["conditions"],
                )
        cased_sbt_conditions = [
            *sbt_conditions,
            {"parameter": "Multilaterals Cased", "operator": "in", "values": [True]},
        ]
        for parameter_name in expected_group_members["SBT lateral construction"]:
            with self.subTest(sbt_lateral_construction_dependency=parameter_name):
                self.assertEqual(
                    cased_sbt_conditions,
                    wellbore_parameters[parameter_name]["dependency"]["conditions"],
                )

        self.assertEqual(
            [{"parameter": "Overpressure Percentage", "operator": "provided", "values": [True]}],
            wellbore_parameters["Injection Reservoir Initial Pressure"]["dependency"]["conditions"],
        )

    def test_reservoir_groups_and_dependencies_match_runtime_routing(self):
        input_parameters_json, _ = GeophiresXSchemaGenerator().get_parameters_json()
        input_parameters = json.loads(input_parameters_json)

        expected_group_members = {
            "Fracture and reservoir volume": [
                "Reservoir Volume Option",
                "Fracture Shape",
                "Fracture Area",
                "Fracture Height",
                "Fracture Width",
                "Number of Fractures",
                "Number of Fractures per Stimulated Well",
                "Fracture Separation",
                "Reservoir Volume",
            ],
            "Reservoir location and limits": [
                "Reservoir Depth",
                "Surface Temperature",
                "Maximum Temperature",
            ],
            "Geothermal gradient profile": ["Gradients", "Thicknesses", "Number of Segments"],
            "Legacy gradient inputs": [
                "Gradient 1",
                "Gradient 2",
                "Gradient 3",
                "Gradient 4",
                "Thickness 1",
                "Thickness 2",
                "Thickness 3",
                "Thickness 4",
            ],
            "Reservoir rock properties": [
                "Reservoir Heat Capacity",
                "Reservoir Density",
                "Reservoir Thermal Conductivity",
                "Reservoir Permeability",
                "Reservoir Porosity",
            ],
            "Reservoir flow properties": [
                "Reservoir Impedance",
                "Injectivity Index",
                "Productivity Index",
                "Water Loss Fraction",
            ],
            "Reservoir overpressure": ["Overpressure Percentage", "Overpressure Depletion Rate"],
            "Thermal drawdown models": [
                "Drawdown Parameter",
                "Drawdown Parameter Schedule",
                "Gringarten-Stehfest Precision",
            ],
            "User-provided temperature profile": [
                "Reservoir Output File Name",
                "Reservoir Output Profile",
                "Reservoir Output Profile Time Step",
            ],
            "Cylindrical reservoir model": [
                "Cylindrical Reservoir Input Depth",
                "Cylindrical Reservoir Output Depth",
                "Cylindrical Reservoir Length",
                "Cylindrical Reservoir Radius of Effect",
                "Cylindrical Reservoir Radius of Effect Factor",
                "Drilled length",
            ],
            "SBT profile inputs": ["Flowrate File", "Injection Temperature File"],
            "SBT solver controls": [
                "SBT Accuracy Desired",
                "SBT Percent Implicit Euler Scheme",
                "SBT Initial Timestep Count",
                "SBT Final Timestep Count",
                "SBT Initial to Final Timestep Transition",
                "SBT Generate Wireframe Graphics",
            ],
            "SUTRA output files": [
                "SUTRA Annual Heat File Name",
                "SUTRA Heat Budget File Name",
                "SUTRA Balance and Storage Well Output File Name",
            ],
            "TOUGH2 model": [
                "TOUGH2 Executable Path",
                "TOUGH2 Model/File Name",
                "Reservoir Thickness",
                "Reservoir Width",
            ],
        }

        reservoir_parameters = {
            name: parameter["UI"]
            for name, parameter in input_parameters.items()
            if parameter["UI"]["tab"] == "Reservoir"
        }
        self.assertEqual(61, len(reservoir_parameters))
        self.assertEqual(
            set(reservoir_parameters),
            {name for names in expected_group_members.values() for name in names},
        )
        self.assertEqual(
            len(reservoir_parameters),
            len({ui["order"] for ui in reservoir_parameters.values()}),
        )
        for group, parameter_names in expected_group_members.items():
            for parameter_name in parameter_names:
                with self.subTest(group=group, parameter=parameter_name):
                    self.assertEqual(group, reservoir_parameters[parameter_name]["group"])

        self.assertIsNone(reservoir_parameters["Gradients"]["dependency"])
        self.assertIsNone(reservoir_parameters["Gradient 1"]["dependency"])
        self.assertEqual(
            [{"parameter": "Reservoir Model", "operator": "in", "values": [2, 6]}],
            reservoir_parameters["Reservoir Porosity"]["dependency"]["conditions"],
        )
        self.assertEqual(
            [{"parameter": "Reservoir Model", "operator": "in", "values": [3, 4]}],
            reservoir_parameters["Drawdown Parameter"]["dependency"]["conditions"],
        )
        sbt_uloop_conditions = [
            {"parameter": "Reservoir Model", "operator": "in", "values": [8]},
            {"parameter": "Well Geometry Configuration", "operator": "in", "values": [1, 5]},
        ]
        self.assertEqual(
            [
                *sbt_uloop_conditions,
                {"parameter": "Flowrate Model", "operator": "in", "values": [2]},
            ],
            reservoir_parameters["Flowrate File"]["dependency"]["conditions"],
        )
        self.assertEqual(
            sbt_uloop_conditions,
            input_parameters["Injection Temperature Model"]["UI"]["dependency"]["conditions"],
        )
        self.assertEqual(
            [
                *sbt_uloop_conditions,
                {"parameter": "Injection Temperature Model", "operator": "in", "values": [2]},
            ],
            reservoir_parameters["Injection Temperature File"]["dependency"]["conditions"],
        )
        self.assertEqual(
            [
                {"parameter": "Reservoir Model", "operator": "in", "values": [8]},
                {"parameter": "Well Geometry Configuration", "operator": "in", "values": [1, 2, 5]},
            ],
            reservoir_parameters["SBT Accuracy Desired"]["dependency"]["conditions"],
        )
        for parameter_name in ["SBT Percent Implicit Euler Scheme", "SBT Generate Wireframe Graphics"]:
            with self.subTest(sbt_uloop_dependency=parameter_name):
                self.assertEqual(
                    sbt_uloop_conditions,
                    reservoir_parameters[parameter_name]["dependency"]["conditions"],
                )
        for parameter_name in [
            "SBT Initial Timestep Count",
            "SBT Final Timestep Count",
            "SBT Initial to Final Timestep Transition",
        ]:
            with self.subTest(deprecated_sbt_timestep_parameter=parameter_name):
                self.assertIsNone(reservoir_parameters[parameter_name]["dependency"])
                self.assertIn(
                    "Deprecated compatibility input; ignored", input_parameters[parameter_name]["ToolTipText"]
                )
        for parameter_name in expected_group_members["TOUGH2 model"]:
            with self.subTest(tough2_dependency=parameter_name):
                self.assertEqual(
                    [{"parameter": "Reservoir Model", "operator": "in", "values": [6]}],
                    reservoir_parameters[parameter_name]["dependency"]["conditions"],
                )

    def test_geophires_x_ui_metadata_applies_declared_placement_and_priority(self):
        generator = GeophiresXSchemaGenerator()
        input_parameters_json, _ = generator.get_parameters_json()
        input_parameters = json.loads(input_parameters_json)
        request_schema, _ = generator.generate_json_schema()
        properties = request_schema["properties"]

        self.assertEqual(387, len(properties))
        self.assertTrue(all("x-ui" in definition for definition in properties.values()))
        self.assertTrue(all("UI" in definition for definition in input_parameters.values()))
        normalized_tabs = {
            "Project": "project",
            "Reservoir": "reservoir",
            "SurfacePlant": "surface-plant",
            "WellBores": "wellbores",
            "Economics": "economics",
        }
        for parameter_name, parameter in input_parameters.items():
            with self.subTest(parameter=parameter_name):
                declared_ui = dict(parameter["UI"])
                declared_ui["tab"] = normalized_tabs[declared_ui["tab"]]
                self.assertEqual(declared_ui, properties[parameter_name]["x-ui"])
                self.assertNotIn("is_controller", properties[parameter_name]["x-ui"])
        expected_independent_placement = {
            "End-Use Option": ("Project", 1),
            "Operating Mode": ("Project", 2),
            "Economic Model": ("Project", 3),
            "Reservoir Model": ("Project", 4),
            "Plant Lifetime": ("Project", 5),
            "Time steps per year": ("Project", 6),
            "Use Weather Data": ("Project", 7),
            "Do Carbon Price Calculations": ("Project", 8),
            "Do S-DAC-GT Calculations": ("Project", 9),
            "Do AddOn Calculations": ("Project", 10),
            "Do VALCO(E|H|C) Calculations": ("Project", 11),
            "Do XLCO(E|H|C) Calculations": ("Project", 12),
            "Is AGS": ("WellBores", 1),
            "Well Geometry Configuration": ("WellBores", 2),
            "Power Plant Type": ("SurfacePlant", 1),
            "TESS Enabled": ("SurfacePlant", 2),
            "Reservoir Volume Option": ("Reservoir", 10),
            "Reservoir Impedance": ("Reservoir", 300),
            "Total Capital Cost": ("Economics", 2),
            "Total O&M Cost": ("Economics", 3),
            "VALCO Calculation Mode": ("Economics", 5),
        }
        for parameter_name, (tab, order) in expected_independent_placement.items():
            with self.subTest(tab=tab, parameter=parameter_name):
                ui = input_parameters[parameter_name]["UI"]
                self.assertEqual(tab, ui["tab"])
                self.assertEqual(order, ui["order"])
                self.assertIsNone(ui["dependency"])
                self.assertNotIn("x-dependencies", properties[parameter_name])

        self.assertEqual({"tab_labels": {"project": "Project Control"}}, request_schema["x-ui"])
        for order, parameter_name in enumerate(
            [
                "Use Weather Data",
                "Do Carbon Price Calculations",
                "Do S-DAC-GT Calculations",
                "Do AddOn Calculations",
                "Do VALCO(E|H|C) Calculations",
                "Do XLCO(E|H|C) Calculations",
            ],
            start=7,
        ):
            with self.subTest(optional_calculation=parameter_name):
                ui = input_parameters[parameter_name]["UI"]
                self.assertEqual("Optional Calculations", ui["group"])
                self.assertEqual(order, ui["order"])

        weather_dependency = {
            "effect": "enable",
            "match": "all",
            "conditions": [{"parameter": "Use Weather Data", "operator": "in", "values": [True]}],
            "reason": "Used only when weather data are enabled.",
        }
        for order, parameter_name in enumerate(["Project Latitude", "Project Longitude"], start=13):
            with self.subTest(project_location=parameter_name):
                ui = input_parameters[parameter_name]["UI"]
                self.assertEqual("Project", ui["tab"])
                self.assertEqual("Project location", ui["group"])
                self.assertEqual(order, ui["order"])
                self.assertEqual(weather_dependency, ui["dependency"])
                self.assertEqual(
                    {**weather_dependency, "depends_on": ["Use Weather Data"]},
                    properties[parameter_name]["x-dependencies"],
                )

        self.assertEqual(
            [
                {"name": "BASELOAD", "value": "Baseload", "int_value": 1},
                {"name": "DISPATCHABLE", "value": "Dispatchable", "int_value": 2},
            ],
            properties["Operating Mode"]["enum_values"],
        )
        self.assertEqual(
            {"tab", "group", "order", "dependency"},
            set(input_parameters["Reservoir Depth"]["UI"]),
        )
        self.assertIsNone(input_parameters["Reservoir Depth"]["UI"]["dependency"])
        self.assertIsNone(input_parameters["Discount Rate"]["UI"]["dependency"])
        self.assertEqual("project", properties["Economic Model"]["x-ui"]["tab"])
        self.assertEqual("project", properties["Reservoir Model"]["x-ui"]["tab"])
        self.assertEqual("project", properties["Project Latitude"]["x-ui"]["tab"])
        self.assertEqual("reservoir", properties["Reservoir Depth"]["x-ui"]["tab"])
        self.assertEqual(100, properties["Reservoir Depth"]["x-ui"]["order"])
        self.assertEqual("wellbores", properties["Number of Production Wells"]["x-ui"]["tab"])
        self.assertEqual("economics", properties["Fixed Internal Rate"]["x-ui"]["tab"])
        self.assertEqual("surface-plant", properties["Use Advanced Absorption Chiller"]["x-ui"]["tab"])
        self.assertIn("Absorption Chiller Effect Type", properties)
        self.assertIn("Heat Pump COP", properties)
        self.assertIn("District Heating Demand Option", properties)
        self.assertIn("WACC", properties)
        self.assertIn("Improved Text Output File", properties)

        self.assertEqual(
            {"parameter": "Reservoir Impedance", "operator": "provided", "values": [False]},
            properties["Injectivity Index"]["x-dependencies"]["conditions"][0],
        )
        self.assertEqual(
            "Direct",
            properties["VALCOE Technology Capacity Value"]["x-dependencies"]["conditions"][-1]["values"][0],
        )
        self.assertEqual(
            "Derived",
            properties["VALCOE Basis Capacity Value"]["x-dependencies"]["conditions"][-1]["values"][0],
        )

        for parameter_name, definition in properties.items():
            dependency = definition.get("x-dependencies")
            if dependency is None:
                continue
            for condition in dependency["conditions"]:
                controller = condition["parameter"]
                with self.subTest(parameter=parameter_name, controller=controller):
                    self.assertIn(controller, properties)

        expected_tabs = {
            "Is AGS": "wellbores",
            "Fracture Shape": "reservoir",
            "Reservoir Volume Option": "reservoir",
            "Reservoir Impedance": "reservoir",
            "Injectivity Index": "reservoir",
            "Productivity Index": "reservoir",
            "Overpressure Percentage": "reservoir",
            "Overpressure Depletion Rate": "reservoir",
            "Injection Temperature Model": "surface-plant",
            "Power Plant Type": "surface-plant",
            "Injection Temperature": "surface-plant",
            "Closed-loop Configuration": "wellbores",
            "Flowrate Model": "wellbores",
            "Heat Transfer Fluid": "wellbores",
            "Production Wellhead Pressure": "wellbores",
            "Depreciation Schedule": "economics",
            "Well Drilling Cost Correlation": "economics",
        }
        for parameter, expected_tab in expected_tabs.items():
            with self.subTest(parameter=parameter):
                self.assertEqual(expected_tab, properties[parameter]["x-ui"]["tab"])

        for definition in properties.values():
            ui_metadata = definition["x-ui"]
            self.assertGreaterEqual(ui_metadata["order"], 0)
            self.assertIsInstance(ui_metadata["group"], str)

    def test_hip_ra_x_dependency_metadata_is_explicit(self):
        request_schema, _ = HipRaXSchemaGenerator().generate_json_schema()

        self.assertEqual(
            {
                "depends_on": ["Reservoir Temperature"],
                "behavior": "derived_when_omitted_or_below_minimum",
            },
            request_schema["properties"]["Reservoir Depth"]["x-dependencies"],
        )
        self.assertEqual(
            ["Reservoir Temperature", "Reservoir Pressure"],
            request_schema["properties"]["Density Of Reservoir Fluid"]["x-dependencies"]["depends_on"],
        )

    def test_parameters_rst(self):
        g = GeophiresXSchemaGenerator()
        rst = g.generate_parameters_reference_rst()
        self.assertIsNotNone(rst)  # TODO sanity checks on content

    def test_outputs_rst(self):
        g = GeophiresXSchemaGenerator()
        _, output_params_json = g.get_parameters_json()
        rst = g.get_output_params_table_rst(output_params_json)

        self.assertIn(
            """ECONOMIC PARAMETERS
-------------------
    .. list-table:: ECONOMIC PARAMETERS Outputs
       :header-rows: 1

       * - Name
         - Description
         - Preferred Units
         - Default Value Type
       * - Economic Model
""",
            rst,
        )

        self.assertIn(
            """       * - Project IRR
         - Project Internal Rate of Return
         - %
         - number
""",
            rst,
        )

    def test_get_json_schema(self):
        g = GeophiresXSchemaGenerator()
        req_schema, result_schema = g.generate_json_schema()
        self.assertIsNotNone(req_schema)  # TODO sanity checks on content
        self.assertIsNotNone(result_schema)  # TODO sanity checks on content

        print(f"Generated result schema: {json.dumps(result_schema, indent=2)}")

        def get_result_prop(cat: str, name: str) -> dict:
            return result_schema["properties"][cat]["properties"][name]

        self.assertIn(
            "multiple of invested capital",
            get_result_prop("ECONOMIC PARAMETERS", "Project MOIC")["description"].lower(),
        )

        self.assertIn(
            "Wellfield cost. ", get_result_prop("CAPITAL COSTS (M$)", "Drilling and completion costs")["description"]
        )

        self.assertIn("Do XLCO(E|H|C) Calculations", req_schema["properties"])
        self.assertIn("XLCO(E|H|C) Carbon Price", req_schema["properties"])
        self.assertIn("XLCOH Avoided Emissions Intensity", req_schema["properties"])
        self.assertIn("XLCO(E|H|C) Carbon Price", req_schema["properties"])
        self.assertIn("XLCOH Thermal REC", req_schema["properties"])
        self.assertIn("XLCO(E|H|C) Water Shadow Price", req_schema["properties"])
        self.assertIn("XLCOC Avoided Emissions Intensity", req_schema["properties"])
        self.assertIn("XLCO(E|H|C) Carbon Price", req_schema["properties"])
        self.assertIn("XLCOC Thermal REC", req_schema["properties"])
        self.assertIn("XLCO(E|H|C) Water Shadow Price", req_schema["properties"])
        self.assertIn("Idle Rig Discount Rate", req_schema["properties"])
        self.assertIn("Do VALCO(E|H|C) Calculations", req_schema["properties"])
        self.assertIn("VALCO Calculation Mode", req_schema["properties"])
        self.assertIn("Project Latitude", req_schema["properties"])
        self.assertIn("Project Longitude", req_schema["properties"])
        self.assertIn("Weather Data Year", req_schema["properties"])
        self.assertEqual(-90.0, req_schema["properties"]["Project Latitude"]["minimum"])
        self.assertEqual(90.0, req_schema["properties"]["Project Latitude"]["maximum"])
        self.assertEqual(-180.0, req_schema["properties"]["Project Longitude"]["minimum"])
        self.assertEqual(180.0, req_schema["properties"]["Project Longitude"]["maximum"])
        self.assertEqual(2024, req_schema["properties"]["Weather Data Year"]["default"])
        self.assertIn("VALCOE System Average Energy Value", req_schema["properties"])
        self.assertIn("VALCOE Technology Capacity Value", req_schema["properties"])
        self.assertIn("VALCOE Basis Capacity Value", req_schema["properties"])
        self.assertIn("VALCOE Capacity Credit", req_schema["properties"])
        self.assertIn("VALCOE Base Flexibility Value", req_schema["properties"])
        self.assertIn("VALCOE Flexibility Multiplier", req_schema["properties"])
        self.assertIn("VALCOH System Average Flexibility Value", req_schema["properties"])
        self.assertIn("VALCOH Technology Energy Value", req_schema["properties"])
        self.assertIn("VALCOH Basis Capacity Value", req_schema["properties"])
        self.assertIn("VALCOH Capacity Credit", req_schema["properties"])
        self.assertIn("VALCOH Base Flexibility Value", req_schema["properties"])
        self.assertIn("VALCOH Flexibility Multiplier", req_schema["properties"])
        self.assertIn("VALCOC System Average Capacity Value", req_schema["properties"])
        self.assertIn("VALCOC Technology Flexibility Value", req_schema["properties"])
        self.assertIn("VALCOC Basis Capacity Value", req_schema["properties"])
        self.assertIn("VALCOC Capacity Credit", req_schema["properties"])
        self.assertIn("VALCOC Base Flexibility Value", req_schema["properties"])
        self.assertIn("VALCOC Flexibility Multiplier", req_schema["properties"])
        self.assertIn(
            "Extended Electricity Breakeven Price (XLCOE Market)",
            result_schema["properties"]["SUMMARY OF RESULTS"]["properties"],
        )
        self.assertIn(
            "Extended Electricity Breakeven Price (XLCOE Market + Social)",
            result_schema["properties"]["SUMMARY OF RESULTS"]["properties"],
        )
        self.assertIn(
            "Extended Heat Breakeven Price (XLCOH Market)",
            result_schema["properties"]["SUMMARY OF RESULTS"]["properties"],
        )
        self.assertIn(
            "Extended Heat Breakeven Price (XLCOH Market + Social)",
            result_schema["properties"]["SUMMARY OF RESULTS"]["properties"],
        )
        self.assertIn(
            "Extended Cooling Breakeven Price (XLCOC Market)",
            result_schema["properties"]["SUMMARY OF RESULTS"]["properties"],
        )
        self.assertIn(
            "Extended Cooling Breakeven Price (XLCOC Market + Social)",
            result_schema["properties"]["SUMMARY OF RESULTS"]["properties"],
        )
        self.assertIn(
            "Value-Adjusted Electricity Breakeven Price (VALCOE)",
            result_schema["properties"]["SUMMARY OF RESULTS"]["properties"],
        )
        self.assertIn(
            "VALCOE Energy Adjustment",
            result_schema["properties"]["SUMMARY OF RESULTS"]["properties"],
        )
        self.assertIn(
            "Value-Adjusted Heat Breakeven Price (VALCOH)",
            result_schema["properties"]["SUMMARY OF RESULTS"]["properties"],
        )
        self.assertIn(
            "VALCOH Capacity Adjustment",
            result_schema["properties"]["SUMMARY OF RESULTS"]["properties"],
        )
        self.assertIn(
            "Value-Adjusted Cooling Breakeven Price (VALCOC)",
            result_schema["properties"]["SUMMARY OF RESULTS"]["properties"],
        )
        self.assertIn(
            "VALCOC Flexibility Adjustment",
            result_schema["properties"]["SUMMARY OF RESULTS"]["properties"],
        )
        self.assertIn(
            "Maximum Flowrate per production well",
            result_schema["properties"]["SUMMARY OF RESULTS"]["properties"],
        )
        self.assertEqual(
            ["thermal", "cooling", "electric"],
            get_result_prop("Dispatch Summary", "demand_type")["enum"],
        )
        self.assertIn(
            "analysis_window",
            result_schema["properties"]["Dispatch Summary"]["required"],
        )
        self.assertIn("Dispatch Profile", result_schema["properties"])
        self.assertEqual(
            ["schema_version", "columns", "rows"],
            result_schema["properties"]["Dispatch Profile"]["required"],
        )


if __name__ == "__main__":
    unittest.main()
