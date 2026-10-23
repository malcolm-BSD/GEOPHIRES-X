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

    def test_geophires_x_ui_metadata_promotes_controllers_to_project(self):
        generator = GeophiresXSchemaGenerator()
        input_parameters_json, _ = generator.get_parameters_json()
        input_parameters = json.loads(input_parameters_json)
        request_schema, _ = generator.generate_json_schema()
        properties = request_schema["properties"]

        self.assertEqual(334, len(properties))
        self.assertTrue(all("x-ui" in definition for definition in properties.values()))
        self.assertTrue(all("UI" in definition for definition in input_parameters.values()))
        self.assertEqual("Project", input_parameters["Is AGS"]["UI"]["tab"])
        self.assertEqual(
            {"tab", "group", "is_controller", "order"},
            set(input_parameters["Reservoir Depth"]["UI"]),
        )
        self.assertEqual("project", properties["Economic Model"]["x-ui"]["tab"])
        self.assertTrue(properties["Economic Model"]["x-ui"]["is_controller"])
        self.assertEqual("project", properties["Reservoir Model"]["x-ui"]["tab"])
        self.assertTrue(properties["Is AGS"]["x-ui"]["is_controller"])
        self.assertEqual("project", properties["Project Latitude"]["x-ui"]["tab"])
        self.assertEqual("reservoir", properties["Reservoir Depth"]["x-ui"]["tab"])
        self.assertEqual(10, properties["Reservoir Depth"]["x-ui"]["order"])
        self.assertEqual("wellbores", properties["Number of Production Wells"]["x-ui"]["tab"])
        self.assertEqual("economics", properties["Fixed Internal Rate"]["x-ui"]["tab"])

        expected_tabs = {
            "Is AGS": "project",
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

        orders_by_tab: dict[str, list[int]] = {}
        for definition in properties.values():
            ui_metadata = definition["x-ui"]
            self.assertGreater(ui_metadata["order"], 0)
            self.assertIsInstance(ui_metadata["group"], str)
            orders_by_tab.setdefault(ui_metadata["tab"], []).append(ui_metadata["order"])
        self.assertTrue(all(len(orders) == len(set(orders)) for orders in orders_by_tab.values()))

    def test_hip_ra_x_dependency_metadata_is_explicit(self):
        request_schema, _ = HipRaXSchemaGenerator().generate_json_schema()

        self.assertEqual(
            {
                "depends_on": ["Reservoir Temperature"],
                "behavior": "derived_when_omitted",
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
