import logging
import os
import sys
import tempfile
from csv import DictReader
from pathlib import Path

from geophires_x.Model import Model
from geophires_x.OutputsRich import removeDisallowedFilenameChars
from geophires_x.Parameter import ParameterEntry
from geophires_x_client import GeophiresInputParameters
from geophires_x_client import GeophiresXClient
from geophires_x_client.geophires_x_result import GeophiresXResult
from tests.base_test_case import BaseTestCase

_log = logging.getLogger(__name__)


class OutputsTestCase(BaseTestCase):
    def test_html_output_file(self):
        html_path = Path(tempfile.gettempdir(), "example12_DH.html").absolute()
        try:
            GeophiresXClient().get_geophires_result(
                GeophiresInputParameters(
                    from_file_path=self._get_test_file_path("../examples/example12_DH.txt"),
                    params={"HTML Output File": str(html_path)},
                )
            )

            self.assertTrue(html_path.exists())
            with open(html_path, encoding="UTF-8") as f:
                html_content = f.read()
                self.assertIn("***CASE REPORT***", html_content)
                # TODO expand test to assert more about output HTML
        except RuntimeError as e:
            # https://github.com/NREL/GEOPHIRES-X/issues/365
            has_expected_error_msg = (
                "cannot unpack non-iterable NoneType object" in str(e)
                or "Can't find a usable tk.tcl" in str(e)
                or 'invalid command name "tcl_findLibrary"' in str(e)
            )
            if has_expected_error_msg and os.name == "nt" and "TOXPYTHON" in os.environ:
                _log.warning(
                    f"Ignoring error while testing HTML output file "
                    f"since we appear to be running on Windows in GitHub Actions ({e!s})"
                )
            else:
                raise e

    def test_relative_output_file_path(self):
        input_file = GeophiresInputParameters({"HTML Output File": "foo.html"}).as_file_path()
        m = self._new_model(input_file=input_file, original_cwd=Path("/tmp/"))  # noqa: S108
        html_filepath = Path(m.outputs.html_output_file.value)
        self.assertTrue(html_filepath.is_absolute())

        expected_path = str(Path("/tmp/foo.html"))  # noqa: S108
        self._assert_file_paths_equal(self._strip_drive(str(html_filepath)), expected_path)

    def test_absolute_output_file_path(self):
        input_file = GeophiresInputParameters(
            {"HTML Output File": "/home/user/my-geophires-project/foo.html"}
        ).as_file_path()
        m = self._new_model(input_file=input_file, original_cwd=Path("/tmp/"))  # noqa: S108
        html_filepath = Path(m.outputs.html_output_file.value)
        self.assertTrue(html_filepath.is_absolute())
        self._assert_file_paths_equal(
            self._strip_drive(str(html_filepath)), str(Path("/home/user/my-geophires-project/foo.html"))
        )

    def test_dispatch_results_are_written_and_parseable(self):
        from geophires_x.CylindricalReservoir import CylindricalReservoir

        output_path = Path(tempfile.gettempdir(), "dispatch_results_test.out").absolute()
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")

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
            "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=csv_file),
        }

        model.read_parameters()
        model.Calculate()
        model.outputs.output_file = str(output_path)
        model.outputs.PrintOutputs(model)

        with open(output_path, encoding="UTF-8") as f:
            output_text = f.read()
        self.assertIn("***DISPATCH RESULTS***", output_text)

        result = GeophiresXResult(str(output_path))
        dispatch_results = result.result["DISPATCH RESULTS"]
        self.assertEqual(1.0, dispatch_results["Dispatch analysis start year"]["value"])
        self.assertEqual(2.0, dispatch_results["Dispatch analysis end year"]["value"])
        self.assertIsNotNone(dispatch_results["Annual geothermal heat delivered"])
        self.assertGreater(dispatch_results["Annual geothermal heat delivered"]["value"], 0.0)
        self.assertGreater(dispatch_results["Peak hourly demand"]["value"], 0.0)
        self.assertEqual("MW", dispatch_results["Peak hourly demand"]["unit"])

    def test_electric_dispatch_results_are_written_and_parseable(self):
        output_path = Path(tempfile.gettempdir(), "dispatch_results_electric_test.out").absolute()
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
        model.outputs.output_file = str(output_path)
        model.outputs.PrintOutputs(model)

        result = GeophiresXResult(str(output_path))
        dispatch_results = result.result["DISPATCH RESULTS"]
        self.assertIsNotNone(dispatch_results["Annual geothermal electricity delivered"])
        self.assertGreater(dispatch_results["Annual geothermal electricity delivered"]["value"], 0.0)
        self.assertGreater(dispatch_results["Design net electricity produced"]["value"], 0.0)
        self.assertEqual("MW", dispatch_results["Peak hourly demand"]["unit"])

    def test_chp_heat_following_dispatch_results_are_written_and_parseable(self):
        output_path = Path(tempfile.gettempdir(), "dispatch_results_chp_heat_test.out").absolute()
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
        model.outputs.output_file = str(output_path)
        model.outputs.PrintOutputs(model)

        with open(output_path, encoding="UTF-8") as f:
            output_text = f.read()
        self.assertIn("***DISPATCH RESULTS***", output_text)
        self.assertIn("CHP: Percent cost allocation for electrical plant", output_text)

        result = GeophiresXResult(str(output_path))
        dispatch_results = result.result["DISPATCH RESULTS"]
        economics = result.result["ECONOMIC PARAMETERS"]
        self.assertIsNotNone(dispatch_results["Annual geothermal heat delivered"])
        self.assertIsNotNone(dispatch_results["Annual geothermal electricity delivered"])
        self.assertGreater(dispatch_results["Annual geothermal heat delivered"]["value"], 0.0)
        self.assertGreater(dispatch_results["Annual geothermal electricity delivered"]["value"], 0.0)
        self.assertGreater(dispatch_results["Design heat produced"]["value"], 0.0)
        self.assertGreater(dispatch_results["Design net electricity produced"]["value"], 0.0)
        self.assertIsNotNone(economics["CHP: Percent cost allocation for electrical plant"])

    def test_dispatch_profile_csv_is_written(self):
        from geophires_x.CylindricalReservoir import CylindricalReservoir

        output_path = Path(tempfile.gettempdir(), "dispatch_profile_results_test.out").absolute()
        csv_output_path = Path(tempfile.gettempdir(), "dispatch_profile_results_test.csv").absolute()
        demand_csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")

        model = self._new_model()
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
            "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=demand_csv_file),
            "Dispatch Profile Output File": ParameterEntry(
                Name="Dispatch Profile Output File", sValue=str(csv_output_path)
            ),
        }

        model.read_parameters()
        model.Calculate()
        model.outputs.output_file = str(output_path)
        model.outputs.PrintOutputs(model)

        self.assertTrue(csv_output_path.exists())
        with open(csv_output_path, encoding="UTF-8", newline="") as f:
            rows = list(DictReader(f))

        self.assertEqual(8760 * 2, len(rows))
        self.assertEqual("3", rows[0]["Year"])
        self.assertEqual("1", rows[0]["Hour of Year"])
        self.assertEqual(str((3 - 1) * 8760 + 1), rows[0]["Simulation Hour"])
        self.assertAlmostEqual(13.1882, float(rows[0]["Thermal Demand (MW)"]), places=4)
        self.assertGreaterEqual(float(rows[0]["Demand Served (MW)"]), 0.0)
        self.assertGreaterEqual(float(rows[0]["Produced Temperature (degC)"]), 0.0)

    def test_electric_dispatch_profile_csv_is_written(self):
        output_path = Path(tempfile.gettempdir(), "dispatch_profile_electric_results_test.out").absolute()
        csv_output_path = Path(tempfile.gettempdir(), "dispatch_profile_electric_results_test.csv").absolute()
        demand_csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")

        model = self._new_model(input_file=str(Path(__file__).resolve().parents[1] / "examples" / "example1.txt"))
        model.InputParameters.update(
            {
                "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
                "Dispatch Demand Source": ParameterEntry(
                    Name="Dispatch Demand Source", sValue="Annual Electricity Demand"
                ),
                "Dispatch Flow Strategy": ParameterEntry(Name="Dispatch Flow Strategy", sValue="Demand Following"),
                "Plant Lifetime": ParameterEntry(Name="Plant Lifetime", sValue="1"),
                "Annual Electricity Demand": ParameterEntry(Name="Annual Electricity Demand", sValue=demand_csv_file),
                "Dispatch Profile Output File": ParameterEntry(
                    Name="Dispatch Profile Output File", sValue=str(csv_output_path)
                ),
            }
        )

        model.read_parameters()
        model.Calculate()
        model.outputs.output_file = str(output_path)
        model.outputs.PrintOutputs(model)

        self.assertTrue(csv_output_path.exists())
        with open(csv_output_path, encoding="UTF-8", newline="") as f:
            rows = list(DictReader(f))

        self.assertEqual(8760, len(rows))
        self.assertEqual("1", rows[0]["Year"])
        self.assertEqual("1", rows[0]["Hour of Year"])
        self.assertAlmostEqual(13.1882, float(rows[0]["Electricity Demand (MW)"]), places=4)
        self.assertGreaterEqual(float(rows[0]["Geothermal Electric Output (MW)"]), 0.0)
        self.assertGreaterEqual(float(rows[0]["Demand Served (MW)"]), 0.0)

    def test_chp_electric_dispatch_profile_csv_is_written(self):
        output_path = Path(tempfile.gettempdir(), "dispatch_profile_chp_electric_results_test.out").absolute()
        csv_output_path = Path(tempfile.gettempdir(), "dispatch_profile_chp_electric_results_test.csv").absolute()
        demand_csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")

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
                "Annual Electricity Demand": ParameterEntry(Name="Annual Electricity Demand", sValue=demand_csv_file),
                "CHP Fraction": ParameterEntry(Name="CHP Fraction", sValue="0.4"),
                "Dispatch Profile Output File": ParameterEntry(
                    Name="Dispatch Profile Output File", sValue=str(csv_output_path)
                ),
            }
        )

        model.read_parameters()
        model.Calculate()
        model.outputs.output_file = str(output_path)
        model.outputs.PrintOutputs(model)

        self.assertTrue(csv_output_path.exists())
        with open(csv_output_path, encoding="UTF-8", newline="") as f:
            rows = list(DictReader(f))

        self.assertEqual(8760, len(rows))
        self.assertEqual("1", rows[0]["Year"])
        self.assertEqual("1", rows[0]["Hour of Year"])
        self.assertAlmostEqual(13.1882, float(rows[0]["Electricity Demand (MW)"]), places=4)
        self.assertGreaterEqual(float(rows[0]["Geothermal Electric Output (MW)"]), 0.0)
        self.assertGreaterEqual(float(rows[0]["Demand Served (MW)"]), 0.0)

    def test_chp_electric_dispatch_results_are_written_and_parseable(self):
        output_path = Path(tempfile.gettempdir(), "dispatch_results_chp_electric_test.out").absolute()
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
        model.outputs.output_file = str(output_path)
        model.outputs.PrintOutputs(model)

        result = GeophiresXResult(str(output_path))
        dispatch_results = result.result["DISPATCH RESULTS"]
        self.assertIsNotNone(dispatch_results["Annual geothermal electricity delivered"])
        self.assertIsNotNone(dispatch_results["Annual geothermal heat delivered"])
        self.assertGreater(dispatch_results["Annual geothermal electricity delivered"]["value"], 0.0)
        self.assertGreater(dispatch_results["Annual geothermal heat delivered"]["value"], 0.0)
        self.assertGreater(dispatch_results["Design net electricity produced"]["value"], 0.0)
        self.assertGreater(dispatch_results["Design heat produced"]["value"], 0.0)

    def test_chp_dispatch_summary_is_written_to_json_output(self):
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        input_file = GeophiresInputParameters(
            from_file_path=str(Path(__file__).resolve().parents[1] / "examples" / "example1.txt"),
            params={
                "Operating Mode": "Dispatchable",
                "End-Use Option": "31",
                "Dispatch Demand Source": "Annual Heat Demand",
                "Dispatch Flow Strategy": "Demand Following",
                "Plant Lifetime": "1",
                "Annual Heat Demand": csv_file,
            },
        )
        result = GeophiresXClient().get_geophires_result(input_file)
        dispatch_summary = result.dispatch_summary_json
        self.assertIsNotNone(dispatch_summary)
        self.assertEqual("thermal", dispatch_summary["demand_type"])
        self.assertGreater(dispatch_summary["summary_metrics"]["annual_served_heat_kwh"], 0.0)
        self.assertGreater(dispatch_summary["summary_metrics"]["annual_served_electricity_kwh"], 0.0)
        self.assertGreater(dispatch_summary["summary_metrics"]["design_net_electricity_produced_mw"], 0.0)
        self.assertTrue(result.json_output_file_path.exists())

    def test_dispatch_html_graphs_are_generated_when_enabled(self):
        from geophires_x.CylindricalReservoir import CylindricalReservoir

        output_path = Path(tempfile.gettempdir(), "dispatch_graphs_results_test.out").absolute()
        html_output_path = Path(tempfile.gettempdir(), "dispatch_graphs_results_test.html").absolute()
        demand_csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")

        graph_titles = [
            "DISPATCH PROFILE: Demand, Served, and Unmet Heat",
            "DISPATCH PROFILE: Produced Temperature and Flow Rate",
            "DISPATCH PROFILE: Runtime Fraction and Pumping Power",
        ]
        graph_paths = [
            Path(html_output_path.parent, f"{removeDisallowedFilenameChars(title.replace(' ', '_'))}.png")
            for title in graph_titles
        ]
        for graph_path in graph_paths:
            if graph_path.exists():
                graph_path.unlink()

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
            "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=demand_csv_file),
            "HTML Output File": ParameterEntry(Name="HTML Output File", sValue=str(html_output_path)),
            "Generate Dispatch HTML Graphs": ParameterEntry(Name="Generate Dispatch HTML Graphs", sValue="1"),
        }

        model.read_parameters()
        model.Calculate()
        model.outputs.output_file = str(output_path)
        model.outputs.PrintOutputs(model)

        self.assertTrue(html_output_path.exists())
        for graph_path in graph_paths:
            self.assertTrue(graph_path.exists())

    def test_electric_dispatch_html_graphs_are_generated_when_enabled(self):
        output_path = Path(tempfile.gettempdir(), "dispatch_graphs_electric_results_test.out").absolute()
        html_output_path = Path(tempfile.gettempdir(), "dispatch_graphs_electric_results_test.html").absolute()
        demand_csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")

        graph_titles = [
            "DISPATCH PROFILE: Demand, Served, and Unmet Electricity",
            "DISPATCH PROFILE: Produced Temperature and Flow Rate",
            "DISPATCH PROFILE: Runtime Fraction and Electric Output",
        ]
        graph_paths = [
            Path(html_output_path.parent, f"{removeDisallowedFilenameChars(title.replace(' ', '_'))}.png")
            for title in graph_titles
        ]
        for graph_path in graph_paths:
            if graph_path.exists():
                graph_path.unlink()

        model = self._new_model(input_file=str(Path(__file__).resolve().parents[1] / "examples" / "example1.txt"))
        model.InputParameters.update(
            {
                "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
                "Dispatch Demand Source": ParameterEntry(
                    Name="Dispatch Demand Source", sValue="Annual Electricity Demand"
                ),
                "Dispatch Flow Strategy": ParameterEntry(Name="Dispatch Flow Strategy", sValue="Demand Following"),
                "Plant Lifetime": ParameterEntry(Name="Plant Lifetime", sValue="1"),
                "Annual Electricity Demand": ParameterEntry(Name="Annual Electricity Demand", sValue=demand_csv_file),
                "HTML Output File": ParameterEntry(Name="HTML Output File", sValue=str(html_output_path)),
                "Generate Dispatch HTML Graphs": ParameterEntry(Name="Generate Dispatch HTML Graphs", sValue="1"),
            }
        )

        model.read_parameters()
        model.Calculate()
        model.outputs.output_file = str(output_path)
        model.outputs.PrintOutputs(model)

        self.assertTrue(html_output_path.exists())
        for graph_path in graph_paths:
            self.assertTrue(graph_path.exists())

    def test_chp_heat_dispatch_html_graphs_are_generated_when_enabled(self):
        output_path = Path(tempfile.gettempdir(), "dispatch_graphs_chp_heat_results_test.out").absolute()
        html_output_path = Path(tempfile.gettempdir(), "dispatch_graphs_chp_heat_results_test.html").absolute()
        demand_csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")

        graph_titles = [
            "DISPATCH PROFILE: Demand, Served, and Unmet Heat",
            "DISPATCH PROFILE: Produced Temperature and Flow Rate",
            "DISPATCH PROFILE: Runtime Fraction and Pumping Power",
        ]
        graph_paths = [
            Path(html_output_path.parent, f"{removeDisallowedFilenameChars(title.replace(' ', '_'))}.png")
            for title in graph_titles
        ]
        for graph_path in graph_paths:
            if graph_path.exists():
                graph_path.unlink()

        model = self._new_model(input_file=str(Path(__file__).resolve().parents[1] / "examples" / "example1.txt"))
        model.InputParameters.update(
            {
                "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
                "End-Use Option": ParameterEntry(Name="End-Use Option", sValue="31"),
                "Dispatch Demand Source": ParameterEntry(Name="Dispatch Demand Source", sValue="Annual Heat Demand"),
                "Dispatch Flow Strategy": ParameterEntry(Name="Dispatch Flow Strategy", sValue="Demand Following"),
                "Plant Lifetime": ParameterEntry(Name="Plant Lifetime", sValue="1"),
                "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=demand_csv_file),
                "HTML Output File": ParameterEntry(Name="HTML Output File", sValue=str(html_output_path)),
                "Generate Dispatch HTML Graphs": ParameterEntry(Name="Generate Dispatch HTML Graphs", sValue="1"),
            }
        )

        model.read_parameters()
        model.Calculate()
        model.outputs.output_file = str(output_path)
        model.outputs.PrintOutputs(model)

        self.assertTrue(html_output_path.exists())
        for graph_path in graph_paths:
            self.assertTrue(graph_path.exists())

    def test_chp_electric_dispatch_html_graphs_are_generated_when_enabled(self):
        output_path = Path(tempfile.gettempdir(), "dispatch_graphs_chp_electric_results_test.out").absolute()
        html_output_path = Path(tempfile.gettempdir(), "dispatch_graphs_chp_electric_results_test.html").absolute()
        demand_csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")

        graph_titles = [
            "DISPATCH PROFILE: Demand, Served, and Unmet Electricity",
            "DISPATCH PROFILE: Produced Temperature and Flow Rate",
            "DISPATCH PROFILE: Runtime Fraction and Electric Output",
        ]
        graph_paths = [
            Path(html_output_path.parent, f"{removeDisallowedFilenameChars(title.replace(' ', '_'))}.png")
            for title in graph_titles
        ]
        for graph_path in graph_paths:
            if graph_path.exists():
                graph_path.unlink()

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
                "Annual Electricity Demand": ParameterEntry(Name="Annual Electricity Demand", sValue=demand_csv_file),
                "CHP Fraction": ParameterEntry(Name="CHP Fraction", sValue="0.4"),
                "HTML Output File": ParameterEntry(Name="HTML Output File", sValue=str(html_output_path)),
                "Generate Dispatch HTML Graphs": ParameterEntry(Name="Generate Dispatch HTML Graphs", sValue="1"),
            }
        )

        model.read_parameters()
        model.Calculate()
        model.outputs.output_file = str(output_path)
        model.outputs.PrintOutputs(model)

        self.assertTrue(html_output_path.exists())
        for graph_path in graph_paths:
            self.assertTrue(graph_path.exists())

    def test_full_scale_dispatch_example_input_runs(self):
        input_path = Path(__file__).resolve().parent / "example1_dispatchable_full_scale.txt"
        text_output_path = input_path.parent / "example1_dispatchable_full_scale_text.out"
        html_output_path = input_path.parent / "example1_dispatchable_full_scale.html"
        dispatch_profile_path = input_path.parent / "example1_dispatchable_full_scale_dispatch_profile.csv"
        graph_titles = [
            "DISPATCH PROFILE: Demand, Served, and Unmet Heat",
            "DISPATCH PROFILE: Produced Temperature and Flow Rate",
            "DISPATCH PROFILE: Runtime Fraction and Pumping Power",
        ]
        graph_paths = [
            Path(html_output_path.parent, f"{removeDisallowedFilenameChars(title.replace(' ', '_'))}.png")
            for title in graph_titles
        ]

        for artifact_path in [text_output_path, html_output_path, dispatch_profile_path, *graph_paths]:
            if artifact_path.exists():
                artifact_path.unlink()

        try:
            result = GeophiresXClient().get_geophires_result(GeophiresInputParameters(from_file_path=str(input_path)))
        except RuntimeError as e:
            has_expected_error_msg = (
                "cannot unpack non-iterable NoneType object" in str(e)
                or "Can't find a usable tk.tcl" in str(e)
                or 'invalid command name "tcl_findLibrary"' in str(e)
            )
            if has_expected_error_msg and os.name == "nt" and "TOXPYTHON" in os.environ:
                _log.warning(
                    f"Ignoring error while testing full-scale dispatch example HTML output "
                    f"since we appear to be running on Windows in GitHub Actions ({e!s})"
                )
                return
            raise e

        self.assertTrue(Path(result.output_file_path).exists())
        self.assertTrue(text_output_path.exists())
        self.assertTrue(html_output_path.exists())
        self.assertTrue(dispatch_profile_path.exists())
        for graph_path in graph_paths:
            self.assertTrue(graph_path.exists())

        with open(text_output_path, encoding="UTF-8") as f:
            output_text = f.read()
        self.assertIn("***DISPATCH RESULTS***", output_text)

        with open(dispatch_profile_path, encoding="UTF-8", newline="") as f:
            rows = list(DictReader(f))
        self.assertEqual(8760, len(rows))
        self.assertEqual("1", rows[0]["Year"])
        self.assertAlmostEqual(13.1882, float(rows[0]["Thermal Demand (MW)"]), places=4)

    # noinspection PyMethodMayBeStatic
    def _strip_drive(self, p: str) -> str:
        return p.replace("D:", "").replace("C:", "")

    def _assert_file_paths_equal(self, file_path_1, file_path_2):
        try:
            self.assertEqual(file_path_1, file_path_2)
        except AssertionError as e:
            if os.name == "nt" and "TOXPYTHON" in os.environ:
                # FIXME - Python 3.9/10 on Windows seem to have had a backwards-incompatible change introduced on or
                #  around 2025-06-06 which cause failures; examples:
                #  - https://github.com/NREL/GEOPHIRES-X/actions/runs/15499833486/job/43649021692
                #  - https://github.com/NREL/GEOPHIRES-X/actions/runs/15499833486/job/43649021692
                #  - https://github.com/NREL/GEOPHIRES-X/actions/runs/15501867732/job/43650830019?pr=389
                _log.warning(
                    f"Ignoring file path equality assertion error since we appear to be running on Windows "
                    f"in GitHub Actions ({e!s})"
                )
            else:
                raise e

    # noinspection PyMethodMayBeStatic
    def _new_model(self, input_file=None, original_cwd=None) -> Model:
        stash_cwd = Path.cwd()
        stash_sys_argv = sys.argv

        sys.argv = [""]

        if input_file is not None:
            sys.argv.append(input_file)

        m = Model(enable_geophires_logging_config=False)

        if input_file is not None:
            m.read_parameters(default_output_path=original_cwd)

        sys.argv = stash_sys_argv
        os.chdir(stash_cwd)

        return m
