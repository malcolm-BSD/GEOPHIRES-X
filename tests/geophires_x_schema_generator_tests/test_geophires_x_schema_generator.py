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
        self.assertEqual(307, len(source_dependencies))
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
            "Heat Transfer Fluid": ("WellBores", 3),
            "Power Plant Type": ("SurfacePlant", 1),
            "TESS Enabled": ("SurfacePlant", 2),
            "Use Advanced Absorption Chiller": ("SurfacePlant", 3),
            "Reservoir Volume Option": ("Reservoir", 1),
            "Reservoir Impedance": ("Reservoir", 2),
            "Total Capital Cost": ("Economics", 2),
            "Total O&M Cost": ("Economics", 3),
            "VALCO Calculation Mode": ("Economics", 5),
            "District Heating Demand Option": ("Economics", 6),
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
        self.assertEqual(10, properties["Reservoir Depth"]["x-ui"]["order"])
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
