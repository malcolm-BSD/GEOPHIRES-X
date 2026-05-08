import logging
import os
import sys
from csv import DictReader
from pathlib import Path

import numpy as np
import pandas as pd

from geophires_x.Model import Model
from geophires_x.OutputsRich import removeDisallowedFilenameChars
from geophires_x.Parameter import ParameterEntry
from geophires_x.WeatherData import WeatherData
from geophires_x_client import GeophiresInputParameters
from geophires_x_client import GeophiresXClient
from geophires_x_client.geophires_x_result import GeophiresXResult
from tests.base_test_case import BaseTestCase

_log = logging.getLogger(__name__)


class OutputsTestCase(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._output_artifacts: set[Path] = set()

    def tearDown(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        for artifact_path in sorted(self._output_artifacts, key=lambda path: len(path.parts), reverse=True):
            artifact_path = artifact_path.resolve()
            if not artifact_path.is_relative_to(repo_root):
                continue
            if artifact_path.exists() and artifact_path.is_file():
                artifact_path.unlink()
        super().tearDown()

    def _register_output_artifact(self, path: Path) -> Path:
        artifact_path = path.resolve()
        self._output_artifacts.add(artifact_path)
        if artifact_path.suffix.lower() == ".out":
            self._output_artifacts.add(artifact_path.with_suffix(".json"))
        return artifact_path

    def _output_artifact_path(self, filename: str) -> Path:
        return self._register_output_artifact(Path(__file__).resolve().parent / filename)

    def _geophires_input_parameters(
        self,
        output_filename: str,
        params=None,
        from_file_path=None,
    ) -> GeophiresInputParameters:
        return GeophiresInputParameters(
            params=params,
            from_file_path=from_file_path,
            output_file_path=self._output_artifact_path(output_filename),
        )

    def _dispatch_graph_path(self, html_output_path: Path, title: str) -> Path:
        file_stem = (
            f"{removeDisallowedFilenameChars(html_output_path.stem)}_"
            f"{removeDisallowedFilenameChars(title.replace(' ', '_'))}"
        )
        return self._register_output_artifact(Path(html_output_path.parent, f"{file_stem}.png"))

    @staticmethod
    def _normalized_text_output_lines(path: Path) -> list[str]:
        normalized_lines = []
        volatile_metadata_prefixes = (
            " GEOPHIRES Version:",
            " Simulation Date:",
            " Simulation Time:",
            " Calculation Time:",
        )
        for line in path.read_text(encoding="utf-8").splitlines():
            normalized_line = line.rstrip()
            for prefix in volatile_metadata_prefixes:
                if line.startswith(prefix):
                    normalized_line = f"{prefix} <volatile>"
                    break
            normalized_lines.append(normalized_line)

        while normalized_lines and normalized_lines[-1] == "":
            normalized_lines.pop()

        return normalized_lines

    def _legacy_example_graph_path(self, file_stem: str, title: str) -> Path:
        graph_stem = (
            f"{removeDisallowedFilenameChars(file_stem)}_{removeDisallowedFilenameChars(title.replace(' ', '_'))}"
        )
        return self._register_output_artifact(Path.cwd() / f"{graph_stem}.png")

    @staticmethod
    def _weather_data() -> WeatherData:
        hourly_temperature = np.linspace(5.0, 25.0, 8760)
        return WeatherData(
            latitude=39.7392,
            longitude=-104.9903,
            year=2024,
            hourly_data=pd.DataFrame(
                {
                    "time": pd.date_range("2024-01-01", periods=8760, freq="h"),
                    "temperature_2m": hourly_temperature,
                }
            ),
            hourly_units={"temperature_2m": "degC"},
        )

    def test_baseload_text_output_format_matches_example_fixture(self):
        result = GeophiresXClient().get_geophires_result(
            self._geophires_input_parameters(
                "example1_baseload_format.out",
                from_file_path=self._get_test_file_path("../examples/example1.txt"),
            )
        )

        actual_lines = self._normalized_text_output_lines(Path(result.output_file_path))
        expected_lines = self._normalized_text_output_lines(
            Path(__file__).resolve().parent / "baseline_example1_current_format.txt"
        )

        self.assertEqual(expected_lines, actual_lines)

    def test_html_output_file(self):
        html_path = self._output_artifact_path("example12_DH.html")
        try:
            GeophiresXClient().get_geophires_result(
                self._geophires_input_parameters(
                    "example12_DH.out",
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

    def test_text_output_file_contains_xlcoe_summary_lines(self):
        result = GeophiresXClient().get_geophires_result(
            self._geophires_input_parameters(
                "example1_xlcoe.out",
                from_file_path=self._get_test_file_path("../examples/example1.txt"),
                params={
                    "Do XLCO(E|H|C) Calculations": True,
                    "XLCO(E|H|C) Carbon Price": 25.0,
                    "XLCOE REC Price": 15.0,
                },
            )
        )

        with open(result.output_file_path, encoding="utf-8") as f:
            output_content = f.read()

        self.assertIn("Extended Electricity Breakeven Price (XLCOE Market)", output_content)
        self.assertIn("Extended Electricity Breakeven Price (XLCOE Market + Social)", output_content)

    def test_non_dispatch_cashflow_profile_includes_construction_year(self):
        result = GeophiresXClient().get_geophires_result(
            self._geophires_input_parameters(
                "example1_cashflow_profile.out",
                from_file_path=self._get_test_file_path("../examples/example1.txt"),
            )
        )

        cashflow_rows = result.result["REVENUE & CASHFLOW PROFILE"]

        self.assertEqual(0, cashflow_rows[1][0])
        self.assertLess(cashflow_rows[1][-1], 0.0)
        self.assertEqual(1, cashflow_rows[2][0])

    def test_text_output_file_contains_xlcoh_summary_lines(self):
        result = GeophiresXClient().get_geophires_result(
            self._geophires_input_parameters(
                "example2_xlcoh.out",
                from_file_path=self._get_test_file_path("../examples/example2.txt"),
                params={
                    "Do XLCO(E|H|C) Calculations": True,
                    "XLCO(E|H|C) Carbon Price": 25.0,
                    "XLCOH Thermal REC": 15.0,
                },
            )
        )

        with open(result.output_file_path, encoding="utf-8") as f:
            output_content = f.read()

        self.assertIn("Extended Heat Breakeven Price (XLCOH Market)", output_content)
        self.assertIn("Extended Heat Breakeven Price (XLCOH Market + Social)", output_content)

    def test_text_output_file_contains_xlcoc_summary_lines(self):
        result = GeophiresXClient().get_geophires_result(
            self._geophires_input_parameters(
                "example11_xlcoc.out",
                from_file_path=self._get_test_file_path("../examples/example11_AC.txt"),
                params={
                    "Do XLCO(E|H|C) Calculations": True,
                    "XLCOC Thermal REC": 15.0,
                    "XLCO(E|H|C) Water Shadow Price": 0.5,
                },
            )
        )

        with open(result.output_file_path, encoding="utf-8") as f:
            output_content = f.read()

        self.assertIn("Extended Cooling Breakeven Price (XLCOC Market)", output_content)
        self.assertIn("Extended Cooling Breakeven Price (XLCOC Market + Social)", output_content)

    def test_text_output_file_contains_valcoe_summary_lines(self):
        result = GeophiresXClient().get_geophires_result(
            self._geophires_input_parameters(
                "example1_valcoe.out",
                from_file_path=self._get_test_file_path("../examples/example1.txt"),
                params={"Do VALCO(E|H|C) Calculations": True},
            )
        )

        with open(result.output_file_path, encoding="utf-8") as f:
            output_content = f.read()

        self.assertIn("Value-Adjusted Electricity Breakeven Price (VALCOE)", output_content)
        self.assertIn("VALCOE Energy Adjustment", output_content)
        self.assertIn("VALCOE Capacity Adjustment", output_content)
        self.assertIn("VALCOE Flexibility Adjustment", output_content)

    def test_text_output_file_contains_valcoh_summary_lines(self):
        result = GeophiresXClient().get_geophires_result(
            self._geophires_input_parameters(
                "example2_valcoh.out",
                from_file_path=self._get_test_file_path("../examples/example2.txt"),
                params={"Do VALCO(E|H|C) Calculations": True},
            )
        )

        with open(result.output_file_path, encoding="utf-8") as f:
            output_content = f.read()

        self.assertIn("Value-Adjusted Heat Breakeven Price (VALCOH)", output_content)
        self.assertIn("VALCOH Energy Adjustment", output_content)
        self.assertIn("VALCOH Capacity Adjustment", output_content)
        self.assertIn("VALCOH Flexibility Adjustment", output_content)

    def test_text_output_file_contains_valcoc_summary_lines(self):
        result = GeophiresXClient().get_geophires_result(
            self._geophires_input_parameters(
                "example11_valcoc.out",
                from_file_path=self._get_test_file_path("../examples/example11_AC.txt"),
                params={"Do VALCO(E|H|C) Calculations": True},
            )
        )

        with open(result.output_file_path, encoding="utf-8") as f:
            output_content = f.read()

        self.assertIn("Value-Adjusted Cooling Breakeven Price (VALCOC)", output_content)
        self.assertIn("VALCOC Energy Adjustment", output_content)
        self.assertIn("VALCOC Capacity Adjustment", output_content)
        self.assertIn("VALCOC Flexibility Adjustment", output_content)

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

        output_path = self._output_artifact_path("dispatch_results_test.out")
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

    def test_weather_data_results_are_written_and_parseable(self) -> None:
        output_path = self._output_artifact_path("weather_data_results_test.out")

        model = self._new_model(input_file=str(Path(__file__).resolve().parents[1] / "examples" / "example1.txt"))
        model.InputParameters.update(
            {
                "Plant Lifetime": ParameterEntry(Name="Plant Lifetime", sValue="1"),
            }
        )
        model.read_parameters()
        model.weather_data = self._weather_data()
        model.Calculate()
        model.outputs.output_file = str(output_path)
        model.outputs.PrintOutputs(model)

        with open(output_path, encoding="UTF-8") as f:
            output_text = f.read()
        self.assertIn("***WEATHER DATA RESULTS***", output_text)

        result = GeophiresXResult(str(output_path))
        weather_results = result.result["WEATHER DATA RESULTS"]
        self.assertNotIn("Weather data source", weather_results)
        self.assertNotIn("Weather data year", weather_results)
        self.assertNotIn("Project latitude", weather_results)
        self.assertNotIn("Project longitude", weather_results)
        self.assertAlmostEqual(15.0, weather_results["Annual average temperature (from Open-Meteo)"]["value"], places=2)
        self.assertAlmostEqual(5.0, weather_results["Minimum hourly temperature (from Open-Meteo)"]["value"], places=2)
        self.assertAlmostEqual(25.0, weather_results["Maximum hourly temperature (from Open-Meteo)"]["value"], places=2)

    def test_tess_dispatch_outputs_are_written_and_parseable(self) -> None:
        """Verify enabled TESS dispatch text rows and CSV columns are emitted."""
        from geophires_x.CylindricalReservoir import CylindricalReservoir

        output_path = self._output_artifact_path("dispatch_results_tess_test.out")
        text_output_path = self._output_artifact_path("dispatch_results_tess_test.rtf")
        dispatch_profile_path = self._output_artifact_path("dispatch_results_tess_profile.csv")
        html_output_path = self._output_artifact_path("dispatch_results_tess_test.html")
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        graph_titles = [
            "DISPATCH PROFILE: Demand, Served, and Unmet Heat",
            "DISPATCH PROFILE: Produced Temperature and Flow Rate",
            "DISPATCH PROFILE: Runtime Fraction and Pumping Power",
            "DISPATCH PROFILE: TESS Temperature and SOC",
            "DISPATCH PROFILE: Demand, TESS Discharge, and Geothermal Charge",
            "DISPATCH PROFILE: TESS Losses and Curtailment",
        ]
        graph_paths = [self._dispatch_graph_path(html_output_path, title) for title in graph_titles]
        for artifact_path in [output_path, text_output_path, dispatch_profile_path, html_output_path, *graph_paths]:
            if artifact_path.exists():
                artifact_path.unlink()

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
            "TESS Enabled": ParameterEntry(Name="TESS Enabled", sValue="True"),
            "TESS Volume": ParameterEntry(Name="TESS Volume", sValue="10000"),
            "TESS Cost per Cubic Meter": ParameterEntry(Name="TESS Cost per Cubic Meter", sValue="750"),
            "TESS Daily Heat Loss Fraction": ParameterEntry(Name="TESS Daily Heat Loss Fraction", sValue="0"),
            "Dispatch Profile Output File": ParameterEntry(
                Name="Dispatch Profile Output File", sValue=str(dispatch_profile_path)
            ),
            "Improved Text Output File": ParameterEntry(Name="Improved Text Output File", sValue=str(text_output_path)),
            "HTML Output File": ParameterEntry(Name="HTML Output File", sValue=str(html_output_path)),
            "Generate Dispatch HTML Graphs": ParameterEntry(Name="Generate Dispatch HTML Graphs", sValue="1"),
        }

        model.read_parameters()
        model.weather_data = self._weather_data()
        model.Calculate()
        model.outputs.output_file = str(output_path)
        model.outputs.PrintOutputs(model)

        with open(output_path, encoding="UTF-8") as f:
            output_text = f.read()

        self.assertIn("***WEATHER DATA RESULTS***", output_text)
        self.assertIn("***THERMAL ENERGY STORAGE SYSTEM (TESS) RESULTS***", output_text)
        self.assertIn("***DISPATCH RESULTS***", output_text)
        self.assertIn("*  DISPATCH PROFILE  *", output_text)
        self.assertLess(output_text.index("REVENUE & CASHFLOW PROFILE"), output_text.index("DISPATCH PROFILE"))
        dispatch_profile_text = output_text.split("*  DISPATCH PROFILE  *", 1)[1]
        self.assertIn("Yr|", dispatch_profile_text)
        self.assertRegex(dispatch_profile_text, r"(Cool|Elec|Heat) Dem")
        self.assertIn("___|", dispatch_profile_text)
        self.assertIn("Maximum Flowrate per production well", output_text)
        self.assertIn("Average Pumping Power", output_text)
        self.assertNotIn("Average Direct-Use Heat Production", output_text)
        self.assertNotIn("Direct-Use heat breakeven price", output_text)
        summary_text = output_text.split("***SUMMARY OF RESULTS***", 1)[1].split("***WEATHER DATA RESULTS***", 1)[0]
        self.assertNotRegex(summary_text, r"\n\s+Flowrate per production well:")
        self.assertNotIn("***RESERVOIR SIMULATION RESULTS***", output_text)
        self.assertNotIn("Maximum Net Heat Production", output_text)
        self.assertNotIn("Average Net Heat Production", output_text)
        self.assertNotIn("Minimum Net Heat Production", output_text)
        self.assertNotIn("Initial Net Heat Production", output_text)
        self.assertNotIn("Average Annual Heat Production", output_text)
        self.assertIn("*  HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE  *", output_text)
        self.assertIn(
            "*  ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE  *",
            output_text,
        )

        result = GeophiresXResult(str(output_path))
        summary = result.result["SUMMARY OF RESULTS"]
        dispatch_results = result.result["DISPATCH RESULTS"]
        self.assertIsNone(summary["Flowrate per production well"])
        self.assertAlmostEqual(
            dispatch_results["Observed peak flow rate"]["value"],
            summary["Maximum Flowrate per production well"]["value"],
            places=1,
        )
        self.assertIn("DISPATCH PROFILE", result.result)
        self.assertEqual(8761, len(result.result["DISPATCH PROFILE"]))
        self.assertEqual("Year", result.result["DISPATCH PROFILE"][0][0])
        self.assertEqual(8760, len(result.result["DISPATCH PROFILE"]) - 1)

        tess_results = result.result["THERMAL ENERGY STORAGE SYSTEM (TESS) RESULTS"]
        self.assertEqual(10000.0, tess_results["TESS volume"]["value"])
        self.assertEqual("m3", tess_results["TESS volume"]["unit"])
        self.assertAlmostEqual(7.5, tess_results["TESS capital cost"]["value"])
        self.assertGreater(tess_results["TESS annual discharge"]["value"], 0.0)
        self.assertGreater(tess_results["Peak geothermal charge"]["value"], 0.0)
        self.assertIsNotNone(tess_results["Geothermal output smoothing ratio"])

        with open(dispatch_profile_path, encoding="UTF-8", newline="") as f:
            rows = list(DictReader(f))
        self.assertEqual(8760, len(rows))
        self.assertIn("TESS Temperature (degC)", rows[0])
        self.assertIn("TESS State of Charge (-)", rows[0])
        self.assertIn("TESS Discharge to Load (MW)", rows[0])
        self.assertIn("TESS Charge from Geothermal (MW)", rows[0])
        self.assertGreater(float(rows[0]["TESS Stored Energy (MWh)"]), 0.0)
        for graph_path in graph_paths:
            self.assertTrue(graph_path.exists())

        with open(text_output_path, encoding="ASCII") as f:
            rtf_content = f.read()
        self.assertIn("***WEATHER DATA RESULTS***", rtf_content)
        self.assertIn("*  DISPATCH PROFILE  *", rtf_content)
        self.assertIn("Dispatch analysis start year", rtf_content)
        self.assertIn("Observed peak flow rate", rtf_content)
        self.assertIn("Annual geothermal heat delivered", rtf_content)
        self.assertLess(rtf_content.index("REVENUE & CASHFLOW PROFILE"), rtf_content.index("DISPATCH PROFILE"))
        self.assertNotIn("Average Direct-Use Heat Production", rtf_content)
        self.assertNotIn("HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE", rtf_content)

        with open(html_output_path, encoding="UTF-8") as f:
            html_content = f.read()
        self.assertIn("WEATHER DATA RESULTS", html_content)
        self.assertIn("DISPATCH PROFILE", html_content)
        self.assertIn("Dispatch analysis start year", html_content)
        self.assertIn("Observed peak flow rate", html_content)
        self.assertIn("Annual geothermal heat delivered", html_content)
        self.assertLess(html_content.index("REVENUE &amp; CASHFLOW PROFILE"), html_content.index("DISPATCH PROFILE"))
        self.assertNotIn("Average Direct-Use Heat Production", html_content)
        self.assertNotIn("HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE", html_content)
        for graph_path in graph_paths:
            self.assertIn(f'<img src="{graph_path.name}"', html_content)

    def test_electric_dispatch_results_are_written_and_parseable(self):
        output_path = self._output_artifact_path("dispatch_results_electric_test.out")
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

    def test_flash_electric_dispatch_results_are_written_and_parseable(self):
        demand_csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")

        for plant_type in ["3", "4"]:
            with self.subTest(plant_type=plant_type):
                output_path = self._output_artifact_path(f"dispatch_results_flash_{plant_type}_electric_test.out")
                model = self._new_model(
                    input_file=str(Path(__file__).resolve().parents[1] / "examples" / "example1.txt")
                )
                model.InputParameters.update(
                    {
                        "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
                        "Power Plant Type": ParameterEntry(Name="Power Plant Type", sValue=plant_type),
                        "Dispatch Demand Source": ParameterEntry(
                            Name="Dispatch Demand Source", sValue="Annual Electricity Demand"
                        ),
                        "Dispatch Flow Strategy": ParameterEntry(
                            Name="Dispatch Flow Strategy", sValue="Demand Following"
                        ),
                        "Plant Lifetime": ParameterEntry(Name="Plant Lifetime", sValue="1"),
                        "Annual Electricity Demand": ParameterEntry(
                            Name="Annual Electricity Demand", sValue=demand_csv_file
                        ),
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

    def test_heat_pump_dispatch_results_are_written_and_parseable(self):
        demand_csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        output_path = self._output_artifact_path("dispatch_results_heat_pump_test.out")
        model = self._new_model(input_file=str(Path(__file__).resolve().parents[1] / "examples" / "example1.txt"))
        model.InputParameters.update(
            {
                "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
                "End-Use Option": ParameterEntry(Name="End-Use Option", sValue="2"),
                "Power Plant Type": ParameterEntry(Name="Power Plant Type", sValue="6"),
                "Dispatch Demand Source": ParameterEntry(Name="Dispatch Demand Source", sValue="Annual Heat Demand"),
                "Dispatch Flow Strategy": ParameterEntry(Name="Dispatch Flow Strategy", sValue="Demand Following"),
                "Plant Lifetime": ParameterEntry(Name="Plant Lifetime", sValue="1"),
                "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=demand_csv_file),
            }
        )

        model.read_parameters()
        model.Calculate()
        model.outputs.output_file = str(output_path)
        model.outputs.PrintOutputs(model)

        result = GeophiresXResult(str(output_path))
        dispatch_results = result.result["DISPATCH RESULTS"]
        self.assertIsNotNone(dispatch_results["Annual geothermal heat delivered"])
        self.assertIsNotNone(dispatch_results["Annual heat pump electricity consumed"])
        self.assertGreater(dispatch_results["Annual geothermal heat delivered"]["value"], 0.0)
        self.assertGreater(dispatch_results["Annual heat pump electricity consumed"]["value"], 0.0)

    def test_absorption_chiller_dispatch_results_are_written_and_parseable(self):
        cooling_csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_cooling_demand.csv")
        output_path = self._output_artifact_path("dispatch_results_absorption_chiller_test.out")
        model = self._new_model(input_file=str(Path(__file__).resolve().parents[1] / "examples" / "example1.txt"))
        model.InputParameters.update(
            {
                "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
                "End-Use Option": ParameterEntry(Name="End-Use Option", sValue="2"),
                "Power Plant Type": ParameterEntry(Name="Power Plant Type", sValue="5"),
                "Dispatch Demand Source": ParameterEntry(Name="Dispatch Demand Source", sValue="Annual Cooling Demand"),
                "Dispatch Flow Strategy": ParameterEntry(Name="Dispatch Flow Strategy", sValue="Demand Following"),
                "Plant Lifetime": ParameterEntry(Name="Plant Lifetime", sValue="1"),
                "Annual Cooling Demand": ParameterEntry(Name="Annual Cooling Demand", sValue=cooling_csv_file),
            }
        )

        model.read_parameters()
        model.Calculate()
        model.outputs.output_file = str(output_path)
        model.outputs.PrintOutputs(model)

        result = GeophiresXResult(str(output_path))
        dispatch_results = result.result["DISPATCH RESULTS"]
        self.assertIsNotNone(dispatch_results["Design cooling produced"])
        self.assertIsNotNone(dispatch_results["Annual geothermal cooling delivered"])
        self.assertIsNotNone(dispatch_results["Annual unmet cooling demand"])
        self.assertGreater(dispatch_results["Design cooling produced"]["value"], 0.0)
        self.assertGreater(dispatch_results["Annual geothermal cooling delivered"]["value"], 0.0)

    def test_new_absorption_chiller_dispatch_example_outputs_are_written_and_parseable(self):
        output_path = self._output_artifact_path("example11_new_AC_dispatch_generated.out")
        text_output_path = self._output_artifact_path("example11_new_AC_dispatch_generated_text.out")
        dispatch_profile_path = self._output_artifact_path("example11_new_AC_dispatch_generated_profile.csv")
        model = self._new_model(
            input_file=str(Path(__file__).resolve().parents[1] / "examples" / "example11_new_AC_dispatch.txt")
        )
        model.InputParameters.update(
            {
                "Dispatch Analysis Start Year": ParameterEntry(Name="Dispatch Analysis Start Year", sValue="5"),
                "Dispatch Analysis End Year": ParameterEntry(Name="Dispatch Analysis End Year", sValue="6"),
            }
        )
        model.outputs.output_file = str(output_path)
        model.outputs.text_output_file.value = str(text_output_path)
        model.outputs.text_output_file.Provided = True
        model.outputs.dispatch_profile_output_file.value = str(dispatch_profile_path)
        model.outputs.dispatch_profile_output_file.Provided = True

        model.Calculate()
        model.outputs.PrintOutputs(model)
        analysis_start_index = model.surfaceplant.dispatch_analysis_start_year.value - 1
        analysis_end_index = model.surfaceplant.dispatch_analysis_end_year.value - 1

        result = GeophiresXResult(str(output_path))
        dispatch_results = result.result["DISPATCH RESULTS"]
        self.assertGreater(dispatch_results["Annual geothermal cooling delivered"]["value"], 0.0)
        self.assertGreater(dispatch_results["Peak hourly demand"]["value"], 0.0)
        annual_cooling = model.surfaceplant.cooling_kWh_Produced.value
        report_end_index = model.surfaceplant.dispatch_analysis_end_year.value
        self.assertTrue(all(value > 0.0 for value in annual_cooling[:report_end_index]))
        self.assertTrue(all(value == 0.0 for value in annual_cooling[report_end_index:]))
        self.assertTrue(all(value > 0.0 for value in annual_cooling[analysis_start_index:analysis_end_index]))
        reported_temperatures = model.wellbores.ProducedTemperature.value[: report_end_index * 8760]
        self.assertGreater(max(reported_temperatures), min(reported_temperatures))
        first_report_year_temperatures = reported_temperatures[:8760]
        final_report_year_temperatures = reported_temperatures[(report_end_index - 1) * 8760 : report_end_index * 8760]
        self.assertGreater(
            sum(first_report_year_temperatures) / len(first_report_year_temperatures),
            sum(final_report_year_temperatures) / len(final_report_year_temperatures),
        )
        self.assertNotEqual(annual_cooling[0], annual_cooling[report_end_index - 1])
        self.assertAlmostEqual(
            annual_cooling[analysis_start_index] / 1_000_000.0,
            dispatch_results["Annual geothermal cooling delivered"]["value"],
            places=2,
        )
        with open(output_path, encoding="UTF-8") as f:
            legacy_text_output = f.read()
        legacy_profile = legacy_text_output[
            legacy_text_output.index(
                "HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE"
            ) : legacy_text_output.index("ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE")
        ]
        legacy_profile_rows = [
            line.split() for line in legacy_profile.splitlines() if line.strip() and line.strip()[0].isdigit()
        ]
        self.assertGreater(
            float(legacy_profile_rows[0][2]),
            float(legacy_profile_rows[-1][2]),
        )
        legacy_annual_profile = legacy_text_output[
            legacy_text_output.index(
                "ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE"
            ) : legacy_text_output.index("REVENUE & CASHFLOW PROFILE")
        ]
        legacy_cashflow_profile = legacy_text_output[
            legacy_text_output.index("REVENUE & CASHFLOW PROFILE") : len(legacy_text_output)
        ]
        legacy_annual_rows = [line.strip() for line in legacy_annual_profile.splitlines()]
        legacy_cashflow_rows = [line.strip() for line in legacy_cashflow_profile.splitlines()]
        for reported_year in range(1, 7):
            self.assertTrue(any(line.startswith(str(reported_year)) for line in legacy_annual_rows))
            self.assertTrue(any(line.startswith(str(reported_year)) for line in legacy_cashflow_rows))
        self.assertFalse(any(line.startswith("7") for line in legacy_annual_rows))
        self.assertFalse(any(line.startswith("7") for line in legacy_cashflow_rows))
        self.assertTrue(text_output_path.exists())
        with open(text_output_path, encoding="UTF-8") as f:
            text_output = f.read()
        self.assertIn("***DISPATCH RESULTS***", text_output)
        self.assertIn(r"\par", text_output)
        with open(dispatch_profile_path, encoding="UTF-8", newline="") as f:
            rows = list(DictReader(f))
        self.assertEqual(8760, len(rows))
        self.assertIn("Cooling Demand (MW)", rows[0])
        self.assertIn("Geothermal Cooling Output (MW)", rows[0])
        self.assertIn("Demand Served (MW)", rows[0])
        self.assertGreater(float(rows[0]["Cooling Demand (MW)"]), 0.0)
        self.assertGreaterEqual(float(rows[0]["Geothermal Cooling Output (MW)"]), 0.0)
        self.assertGreaterEqual(float(rows[0]["Demand Served (MW)"]), 0.0)

    def test_new_absorption_chiller_baseload_example_outputs_are_parseable(self):
        output_path = self._output_artifact_path("example11_new_AC_baseload_generated.out")
        model = self._new_model(
            input_file=str(Path(__file__).resolve().parents[1] / "examples" / "example11_new_AC_baseload.txt")
        )
        model.outputs.output_file = str(output_path)

        model.Calculate()
        model.outputs.PrintOutputs(model)

        result = GeophiresXResult(str(output_path))
        summary_results = result.result["SUMMARY OF RESULTS"]
        self.assertEqual(
            model.surfaceplant.plant_lifetime.value * model.economics.timestepsperyear.value,
            len(model.surfaceplant.cooling_produced.value),
        )
        self.assertGreater(summary_results["Average Cooling Production"]["value"], 0.0)
        self.assertGreater(summary_results["Average Direct-Use Heat Production"]["value"], 0.0)
        self.assertGreaterEqual(min(model.surfaceplant.HeatProduced.value), 0.0)
        self.assertGreaterEqual(min(model.surfaceplant.HeatExtracted.value), 0.0)
        self.assertGreater(
            max(model.surfaceplant.HeatProduced.value),
            min(model.surfaceplant.HeatProduced.value),
        )
        self.assertGreater(
            max(model.surfaceplant.cooling_produced.value),
            min(model.surfaceplant.cooling_produced.value),
        )
        self.assertGreater(
            max(model.wellbores.ProducedTemperature.value), min(model.wellbores.ProducedTemperature.value)
        )
        self.assertGreater(
            model.wellbores.ProducedTemperature.value[0],
            model.wellbores.ProducedTemperature.value[-1],
        )
        with open(output_path, encoding="UTF-8") as f:
            legacy_text_output = f.read()
        legacy_profile = legacy_text_output[
            legacy_text_output.index(
                "HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE"
            ) : legacy_text_output.index("ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE")
        ]
        legacy_profile_rows = [line.strip() for line in legacy_profile.splitlines()]
        self.assertTrue(any(line.startswith("0") for line in legacy_profile_rows))
        self.assertTrue(any(line.startswith("29") for line in legacy_profile_rows))

    def test_district_heating_dispatch_results_are_written_and_parseable(self):
        demand_csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        output_path = self._output_artifact_path("dispatch_results_district_heating_test.out")
        model = self._new_model(input_file=str(Path(__file__).resolve().parents[1] / "examples" / "example1.txt"))
        model.InputParameters.update(
            {
                "Operating Mode": ParameterEntry(Name="Operating Mode", sValue="Dispatchable"),
                "End-Use Option": ParameterEntry(Name="End-Use Option", sValue="2"),
                "Power Plant Type": ParameterEntry(Name="Power Plant Type", sValue="7"),
                "Dispatch Demand Source": ParameterEntry(Name="Dispatch Demand Source", sValue="Annual Heat Demand"),
                "Dispatch Flow Strategy": ParameterEntry(Name="Dispatch Flow Strategy", sValue="Demand Following"),
                "Plant Lifetime": ParameterEntry(Name="Plant Lifetime", sValue="1"),
                "Annual Heat Demand": ParameterEntry(Name="Annual Heat Demand", sValue=demand_csv_file),
            }
        )

        model.read_parameters()
        model.Calculate()
        model.outputs.output_file = str(output_path)
        model.outputs.PrintOutputs(model)

        result = GeophiresXResult(str(output_path))
        dispatch_results = result.result["DISPATCH RESULTS"]
        self.assertIsNotNone(dispatch_results["Annual geothermal heat delivered"])
        self.assertIsNotNone(dispatch_results["Annual peaking boiler heat delivered"])
        self.assertIsNotNone(dispatch_results["Peak peaking boiler demand"])
        self.assertGreater(dispatch_results["Annual geothermal heat delivered"]["value"], 0.0)

    def test_chp_heat_following_dispatch_results_are_written_and_parseable(self):
        output_path = self._output_artifact_path("dispatch_results_chp_heat_test.out")
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

        output_path = self._output_artifact_path("dispatch_profile_results_test.out")
        csv_output_path = self._output_artifact_path("dispatch_profile_results_test.csv")
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
        self.assertIn("Ambient Temperature (degC)", rows[0])

    def test_electric_dispatch_profile_csv_is_written(self):
        output_path = self._output_artifact_path("dispatch_profile_electric_results_test.out")
        csv_output_path = self._output_artifact_path("dispatch_profile_electric_results_test.csv")
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
        self.assertIn("Ambient Temperature (degC)", rows[0])

    def test_chp_electric_dispatch_profile_csv_is_written(self):
        output_path = self._output_artifact_path("dispatch_profile_chp_electric_results_test.out")
        csv_output_path = self._output_artifact_path("dispatch_profile_chp_electric_results_test.csv")
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
        self.assertIn("Ambient Temperature (degC)", rows[0])

    def test_chp_electric_dispatch_results_are_written_and_parseable(self):
        output_path = self._output_artifact_path("dispatch_results_chp_electric_test.out")
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
        input_file = self._geophires_input_parameters(
            "dispatch_summary_chp_test.out",
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
        self.assertEqual(1, dispatch_summary["schema_version"])
        self.assertEqual("thermal", dispatch_summary["demand_type"])
        self.assertEqual("chp", dispatch_summary["surfaceplant_mode"])
        self.assertGreater(dispatch_summary["summary_metrics"]["annual_served_heat_kwh"], 0.0)
        self.assertGreater(dispatch_summary["summary_metrics"]["annual_served_electricity_kwh"], 0.0)
        self.assertGreater(dispatch_summary["summary_metrics"]["design_net_electricity_produced_mw"], 0.0)
        self.assertTrue(result.json_output_file_path.exists())

    def test_heat_pump_dispatch_summary_is_written_to_json_output(self):
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        input_file = self._geophires_input_parameters(
            "dispatch_summary_heat_pump_test.out",
            from_file_path=str(Path(__file__).resolve().parents[1] / "examples" / "example1.txt"),
            params={
                "Operating Mode": "Dispatchable",
                "End-Use Option": "2",
                "Power Plant Type": "6",
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
        self.assertEqual("Annual Heat Demand", dispatch_summary["dispatch_settings"]["demand_source"])
        self.assertGreater(dispatch_summary["summary_metrics"]["annual_heat_pump_electricity_kwh"], 0.0)

    def test_absorption_chiller_dispatch_summary_is_written_to_json_output(self):
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_cooling_demand.csv")
        input_file = self._geophires_input_parameters(
            "dispatch_summary_absorption_chiller_test.out",
            from_file_path=str(Path(__file__).resolve().parents[1] / "examples" / "example1.txt"),
            params={
                "Operating Mode": "Dispatchable",
                "End-Use Option": "2",
                "Power Plant Type": "5",
                "Dispatch Demand Source": "Annual Cooling Demand",
                "Dispatch Flow Strategy": "Demand Following",
                "Plant Lifetime": "1",
                "Annual Cooling Demand": csv_file,
            },
        )
        result = GeophiresXClient().get_geophires_result(input_file)
        dispatch_summary = result.dispatch_summary_json
        self.assertIsNotNone(dispatch_summary)
        self.assertEqual("cooling", dispatch_summary["demand_type"])
        self.assertEqual("Annual Cooling Demand", dispatch_summary["dispatch_settings"]["demand_source"])
        self.assertGreater(dispatch_summary["summary_metrics"]["annual_served_cooling_kwh"], 0.0)

    def test_district_heating_dispatch_summary_is_written_to_json_output(self):
        csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")
        input_file = self._geophires_input_parameters(
            "dispatch_summary_district_heating_test.out",
            from_file_path=str(Path(__file__).resolve().parents[1] / "examples" / "example1.txt"),
            params={
                "Operating Mode": "Dispatchable",
                "End-Use Option": "2",
                "Power Plant Type": "7",
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
        self.assertGreater(dispatch_summary["summary_metrics"]["annual_district_heating_boiler_kwh"], 0.0)

    def test_dispatch_html_graphs_are_generated_when_enabled(self):
        from geophires_x.CylindricalReservoir import CylindricalReservoir

        output_path = self._output_artifact_path("dispatch_graphs_results_test.out")
        html_output_path = self._output_artifact_path("dispatch_graphs_results_test.html")
        demand_csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")

        graph_titles = [
            "DISPATCH PROFILE: Demand, Served, and Unmet Heat",
            "DISPATCH PROFILE: Produced Temperature and Flow Rate",
            "DISPATCH PROFILE: Runtime Fraction and Pumping Power",
        ]
        graph_paths = [self._dispatch_graph_path(html_output_path, title) for title in graph_titles]
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
        output_path = self._output_artifact_path("dispatch_graphs_electric_results_test.out")
        html_output_path = self._output_artifact_path("dispatch_graphs_electric_results_test.html")
        demand_csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")

        graph_titles = [
            "DISPATCH PROFILE: Demand, Served, and Unmet Electricity",
            "DISPATCH PROFILE: Produced Temperature and Flow Rate",
            "DISPATCH PROFILE: Runtime Fraction and Electric Output",
        ]
        graph_paths = [self._dispatch_graph_path(html_output_path, title) for title in graph_titles]
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
        output_path = self._output_artifact_path("dispatch_graphs_chp_heat_results_test.out")
        html_output_path = self._output_artifact_path("dispatch_graphs_chp_heat_results_test.html")
        demand_csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")

        graph_titles = [
            "DISPATCH PROFILE: Demand, Served, and Unmet Heat",
            "DISPATCH PROFILE: Produced Temperature and Flow Rate",
            "DISPATCH PROFILE: Runtime Fraction and Pumping Power",
        ]
        graph_paths = [self._dispatch_graph_path(html_output_path, title) for title in graph_titles]
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
        output_path = self._output_artifact_path("dispatch_graphs_chp_electric_results_test.out")
        html_output_path = self._output_artifact_path("dispatch_graphs_chp_electric_results_test.html")
        demand_csv_file = str(Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv")

        graph_titles = [
            "DISPATCH PROFILE: Demand, Served, and Unmet Electricity",
            "DISPATCH PROFILE: Produced Temperature and Flow Rate",
            "DISPATCH PROFILE: Runtime Fraction and Electric Output",
        ]
        graph_paths = [self._dispatch_graph_path(html_output_path, title) for title in graph_titles]
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
        demand_csv_path = Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv"
        text_output_path = self._output_artifact_path("example1_dispatchable_full_scale_generated_text.out")
        html_output_path = self._output_artifact_path("example1_dispatchable_full_scale_generated.html")
        dispatch_profile_path = self._output_artifact_path(
            "example1_dispatchable_full_scale_generated_dispatch_profile.csv"
        )
        graph_titles = [
            "DISPATCH PROFILE: Demand, Served, and Unmet Heat",
            "DISPATCH PROFILE: Produced Temperature and Flow Rate",
            "DISPATCH PROFILE: Runtime Fraction and Pumping Power",
        ]
        graph_paths = [self._dispatch_graph_path(html_output_path, title) for title in graph_titles]
        legacy_graph_paths = [
            self._legacy_example_graph_path("example1_dispatchable_full_scale", title) for title in graph_titles
        ]
        legacy_artifact_paths = [
            self._register_output_artifact(Path.cwd() / "example1_dispatchable_full_scale_text.out"),
            self._register_output_artifact(Path.cwd() / "example1_dispatchable_full_scale_dispatch_profile.csv"),
            self._register_output_artifact(
                Path(__file__).resolve().parents[1] / "example1_dispatchable_full_scale.out"
            ),
            *legacy_graph_paths,
        ]

        for artifact_path in [
            text_output_path,
            html_output_path,
            dispatch_profile_path,
            *graph_paths,
            *legacy_artifact_paths,
        ]:
            if artifact_path.exists():
                artifact_path.unlink()

        try:
            result = GeophiresXClient().get_geophires_result(
                self._geophires_input_parameters(
                    "example1_dispatchable_full_scale.out",
                    from_file_path=str(input_path),
                    params={
                        "Annual Heat Demand": str(demand_csv_path),
                        "Improved Text Output File": str(text_output_path),
                        "HTML Output File": str(html_output_path),
                        "Dispatch Profile Output File": str(dispatch_profile_path),
                    },
                )
            )
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

    def test_tess_dispatch_example_input_runs(self) -> None:
        """Verify the TESS dispatch example generates expected TESS outputs."""
        input_path = Path(__file__).resolve().parent / "example1_dispatchable_tess.txt"
        demand_csv_path = Path(__file__).resolve().parents[1] / "assets" / "params" / "annual_heat_demand.csv"
        text_output_path = self._output_artifact_path("example1_dispatchable_tess_generated_text.out")
        html_output_path = self._output_artifact_path("example1_dispatchable_tess_generated.html")
        dispatch_profile_path = self._output_artifact_path("example1_dispatchable_tess_generated_dispatch_profile.csv")
        graph_titles = [
            "DISPATCH PROFILE: Demand, Served, and Unmet Heat",
            "DISPATCH PROFILE: Produced Temperature and Flow Rate",
            "DISPATCH PROFILE: Runtime Fraction and Pumping Power",
            "DISPATCH PROFILE: TESS Temperature and SOC",
            "DISPATCH PROFILE: Demand, TESS Discharge, and Geothermal Charge",
            "DISPATCH PROFILE: TESS Losses and Curtailment",
        ]
        graph_paths = [self._dispatch_graph_path(html_output_path, title) for title in graph_titles]
        legacy_graph_paths = [
            self._legacy_example_graph_path("example1_dispatchable_tess", title) for title in graph_titles
        ]
        legacy_artifact_paths = [
            self._register_output_artifact(Path.cwd() / "example1_dispatchable_tess_text.out"),
            self._register_output_artifact(Path.cwd() / "example1_dispatchable_tess_dispatch_profile.csv"),
            self._register_output_artifact(Path(__file__).resolve().parents[1] / "example1_dispatchable_tess.out"),
            *legacy_graph_paths,
        ]

        for artifact_path in [
            text_output_path,
            html_output_path,
            dispatch_profile_path,
            *graph_paths,
            *legacy_artifact_paths,
        ]:
            if artifact_path.exists():
                artifact_path.unlink()

        try:
            result = GeophiresXClient().get_geophires_result(
                self._geophires_input_parameters(
                    "example1_dispatchable_tess.out",
                    from_file_path=str(input_path),
                    params={
                        "Annual Heat Demand": str(demand_csv_path),
                        "Improved Text Output File": str(text_output_path),
                        "HTML Output File": str(html_output_path),
                        "Dispatch Profile Output File": str(dispatch_profile_path),
                    },
                )
            )
        except RuntimeError as e:
            has_expected_error_msg = (
                "cannot unpack non-iterable NoneType object" in str(e)
                or "Can't find a usable tk.tcl" in str(e)
                or 'invalid command name "tcl_findLibrary"' in str(e)
            )
            if has_expected_error_msg and os.name == "nt" and "TOXPYTHON" in os.environ:
                _log.warning(
                    f"Ignoring error while testing TESS dispatch example HTML output "
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

        parsed_results = result.result["THERMAL ENERGY STORAGE SYSTEM (TESS) RESULTS"]
        self.assertEqual(10000.0, parsed_results["TESS volume"]["value"])
        self.assertAlmostEqual(7.5, parsed_results["TESS capital cost"]["value"])
        self.assertGreater(parsed_results["TESS annual discharge"]["value"], 0.0)
        self.assertGreater(parsed_results["Peak geothermal charge"]["value"], 0.0)

        with open(dispatch_profile_path, encoding="UTF-8", newline="") as f:
            rows = list(DictReader(f))
        self.assertEqual(8760, len(rows))
        self.assertEqual("1", rows[0]["Year"])
        self.assertAlmostEqual(13.1882, float(rows[0]["Thermal Demand (MW)"]), places=4)
        self.assertIn("TESS Temperature (degC)", rows[0])
        self.assertIn("TESS State of Charge (-)", rows[0])
        self.assertIn("TESS Charge from Geothermal (MW)", rows[0])

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
