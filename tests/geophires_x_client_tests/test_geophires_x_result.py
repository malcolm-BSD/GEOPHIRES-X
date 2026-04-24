from __future__ import annotations

import json
from typing import Any

from geophires_x_client import GeophiresXClient
from geophires_x_client import GeophiresXResult
from geophires_x_client import ImmutableGeophiresInputParameters
from tests.base_test_case import BaseTestCase


class GeophiresXResultTestCase(BaseTestCase):
    def test_get_sam_cash_flow_row_name_unit_split(self) -> None:
        cases = [
            ("Electricity to grid (kWh)", ["Electricity to grid", "kWh"]),
            ("Federal tax benefit (liability) ($)", ["Federal tax benefit (liability)", "$"]),
            ("Underwater baskets", ["Underwater baskets", ""]),
        ]

        for case in cases:
            with self.subTest(msg=case[0]):
                actual = GeophiresXResult._get_sam_cash_flow_row_name_unit_split(case[0])
                self.assertListEqual(actual, case[1])

    def test_get_lines_by_category(self) -> None:
        r: GeophiresXResult = GeophiresXResult(self._get_test_file_path("../examples/example2.out"))
        lines_by_cat = r._get_lines_by_category()
        res_params_lines = lines_by_cat["RESERVOIR PARAMETERS"]
        self.assertGreater(len(res_params_lines), 0)

    def test_reservoir_volume_calculation_note(self) -> None:
        r: GeophiresXResult = GeophiresXResult(self._get_test_file_path("../examples/example2.out"))
        field_name = "Reservoir volume calculation note"
        self.assertIn(field_name, r.result["RESERVOIR PARAMETERS"])
        self.assertEqual(
            r.result["RESERVOIR PARAMETERS"][field_name],
            "Number of fractures calculated with reservoir volume and fracture separation as input",
        )

    def test_sam_econ_model_capex_in_summary(self) -> None:
        r: GeophiresXResult = GeophiresXResult(self._get_test_file_path("../examples/example_SAM-single-owner-PPA.out"))
        field_name = "Total CAPEX"
        self.assertIn(field_name, r.result["SUMMARY OF RESULTS"])
        self.assertIn("value", r.result["SUMMARY OF RESULTS"][field_name])
        self.assertGreater(r.result["SUMMARY OF RESULTS"][field_name]["value"], 1)
        self.assertEqual(r.result["SUMMARY OF RESULTS"][field_name]["unit"], "MUSD")

    def test_xlcoe_fields_are_parsed_from_summary(self) -> None:
        r: GeophiresXResult = GeophiresXClient().get_geophires_result(
            ImmutableGeophiresInputParameters(
                from_file_path=self._get_test_file_path("../examples/example1.txt"),
                params={
                    "Do XLCO(E|H|C) Calculations": True,
                    "XLCO(E|H|C) Carbon Price": 25.0,
                    "XLCOE REC Price": 15.0,
                    "XLCOE Displaced Water Use Intensity": 1.0,
                    "XLCO(E|H|C) Water Shadow Price": 0.5,
                    "XLCO(E|H|C) Operations Jobs Per MW": 0.2,
                    "XLCO(E|H|C) Indirect Jobs Multiplier": 1.5,
                    "XLCO(E|H|C) Average Monthly Wage": 4000.0,
                },
            )
        )
        summary = r.result["SUMMARY OF RESULTS"]

        market_field = "Extended Electricity Breakeven Price (XLCOE Market)"
        social_field = "Extended Electricity Breakeven Price (XLCOE Market + Social)"
        baseline_lcoe = summary["Electricity breakeven price"]["value"]

        self.assertIn(market_field, summary)
        self.assertIn(social_field, summary)
        self.assertEqual("cents/kWh", summary[market_field]["unit"])
        self.assertEqual("cents/kWh", summary[social_field]["unit"])
        self.assertLess(summary[market_field]["value"], baseline_lcoe)
        self.assertLess(summary[social_field]["value"], summary[market_field]["value"])

    def test_xlcoh_fields_are_parsed_from_summary(self) -> None:
        r: GeophiresXResult = GeophiresXClient().get_geophires_result(
            ImmutableGeophiresInputParameters(
                from_file_path=self._get_test_file_path("../examples/example2.txt"),
                params={
                    "Do XLCO(E|H|C) Calculations": True,
                    "XLCO(E|H|C) Carbon Price": 25.0,
                    "XLCOH Thermal REC": 15.0,
                    "XLCOH Displaced Water Use Intensity": 1.0,
                    "XLCO(E|H|C) Water Shadow Price": 0.5,
                    "XLCO(E|H|C) Operations Jobs Per MW": 0.2,
                    "XLCO(E|H|C) Indirect Jobs Multiplier": 1.5,
                    "XLCO(E|H|C) Average Monthly Wage": 4000.0,
                },
            )
        )
        summary = r.result["SUMMARY OF RESULTS"]

        market_field = "Extended Heat Breakeven Price (XLCOH Market)"
        social_field = "Extended Heat Breakeven Price (XLCOH Market + Social)"
        baseline_lcoh_field = "Direct-Use heat breakeven price (LCOH)"
        baseline_lcoh = summary[baseline_lcoh_field]["value"]

        self.assertIn(market_field, summary)
        self.assertIn(social_field, summary)
        self.assertEqual(summary[baseline_lcoh_field]["unit"], summary[market_field]["unit"])
        self.assertEqual(summary[baseline_lcoh_field]["unit"], summary[social_field]["unit"])
        self.assertLess(summary[market_field]["value"], baseline_lcoh)
        self.assertLess(summary[social_field]["value"], summary[market_field]["value"])

    def test_xlcoc_fields_are_parsed_from_summary(self) -> None:
        r: GeophiresXResult = GeophiresXClient().get_geophires_result(
            ImmutableGeophiresInputParameters(
                from_file_path=self._get_test_file_path("../examples/example11_AC.txt"),
                params={
                    "Do XLCO(E|H|C) Calculations": True,
                    "XLCO(E|H|C) Carbon Price": 25.0,
                    "XLCOC Thermal REC": 15.0,
                    "XLCOC Displaced Water Use Intensity": 1.0,
                    "XLCO(E|H|C) Water Shadow Price": 0.5,
                    "XLCO(E|H|C) Operations Jobs Per MW": 0.2,
                    "XLCO(E|H|C) Indirect Jobs Multiplier": 1.5,
                    "XLCO(E|H|C) Average Monthly Wage": 4000.0,
                },
            )
        )
        summary = r.result["SUMMARY OF RESULTS"]

        market_field = "Extended Cooling Breakeven Price (XLCOC Market)"
        social_field = "Extended Cooling Breakeven Price (XLCOC Market + Social)"
        baseline_lcoc_field = "Direct-Use Cooling Breakeven Price (LCOC)"
        baseline_lcoc = summary[baseline_lcoc_field]["value"]

        self.assertIn(market_field, summary)
        self.assertIn(social_field, summary)
        self.assertEqual(summary[baseline_lcoc_field]["unit"], summary[market_field]["unit"])
        self.assertEqual(summary[baseline_lcoc_field]["unit"], summary[social_field]["unit"])
        self.assertLess(summary[market_field]["value"], baseline_lcoc)
        self.assertLess(summary[social_field]["value"], summary[market_field]["value"])

    def test_valcoe_fields_are_parsed_from_summary(self) -> None:
        r: GeophiresXResult = GeophiresXClient().get_geophires_result(
            ImmutableGeophiresInputParameters(
                from_file_path=self._get_test_file_path("../examples/example1.txt"),
                params={
                    "Do VALCO(E|H|C) Calculations": True,
                    "VALCOE System Average Energy Value": 1.0,
                    "VALCOE Technology Energy Value": 2.5,
                    "VALCOE System Average Capacity Value": 0.2,
                    "VALCOE Technology Capacity Value": 0.1,
                },
            )
        )
        summary = r.result["SUMMARY OF RESULTS"]

        valco_field = "Value-Adjusted Electricity Breakeven Price (VALCOE)"
        energy_field = "VALCOE Energy Adjustment"
        capacity_field = "VALCOE Capacity Adjustment"
        flexibility_field = "VALCOE Flexibility Adjustment"
        baseline_lcoe = summary["Electricity breakeven price"]["value"]

        self.assertIn(valco_field, summary)
        self.assertIn(energy_field, summary)
        self.assertIn(capacity_field, summary)
        self.assertIn(flexibility_field, summary)
        self.assertEqual("cents/kWh", summary[valco_field]["unit"])
        self.assertEqual("cents/kWh", summary[energy_field]["unit"])
        self.assertEqual(-1.5, summary[energy_field]["value"])
        self.assertEqual(0.1, summary[capacity_field]["value"])
        self.assertEqual(0.0, summary[flexibility_field]["value"])
        self.assertAlmostEqual(baseline_lcoe - 1.4, summary[valco_field]["value"], places=7)

    def test_valcoh_fields_are_parsed_from_summary(self) -> None:
        r: GeophiresXResult = GeophiresXClient().get_geophires_result(
            ImmutableGeophiresInputParameters(
                from_file_path=self._get_test_file_path("../examples/example2.txt"),
                params={
                    "Do VALCO(E|H|C) Calculations": True,
                    "VALCOH System Average Capacity Value": 1.0,
                    "VALCOH Technology Capacity Value": 0.25,
                },
            )
        )
        summary = r.result["SUMMARY OF RESULTS"]

        valco_field = "Value-Adjusted Heat Breakeven Price (VALCOH)"
        adjustment_field = "VALCOH Capacity Adjustment"
        baseline_lcoh_field = "Direct-Use heat breakeven price (LCOH)"

        self.assertIn(valco_field, summary)
        self.assertIn(adjustment_field, summary)
        self.assertEqual(summary[baseline_lcoh_field]["unit"], summary[valco_field]["unit"])
        self.assertEqual(0.75, summary[adjustment_field]["value"])
        self.assertAlmostEqual(
            summary[baseline_lcoh_field]["value"] + 0.75,
            summary[valco_field]["value"],
            places=7,
        )

    def test_valcoc_fields_are_parsed_from_summary(self) -> None:
        r: GeophiresXResult = GeophiresXClient().get_geophires_result(
            ImmutableGeophiresInputParameters(
                from_file_path=self._get_test_file_path("../examples/example11_AC.txt"),
                params={
                    "Do VALCO(E|H|C) Calculations": True,
                    "VALCOC System Average Flexibility Value": 0.5,
                    "VALCOC Technology Flexibility Value": 0.2,
                },
            )
        )
        summary = r.result["SUMMARY OF RESULTS"]

        valco_field = "Value-Adjusted Cooling Breakeven Price (VALCOC)"
        adjustment_field = "VALCOC Flexibility Adjustment"
        baseline_lcoc_field = "Direct-Use Cooling Breakeven Price (LCOC)"

        self.assertIn(valco_field, summary)
        self.assertIn(adjustment_field, summary)
        self.assertEqual(summary[baseline_lcoc_field]["unit"], summary[valco_field]["unit"])
        self.assertEqual(0.3, summary[adjustment_field]["value"])
        self.assertAlmostEqual(
            summary[baseline_lcoc_field]["value"] + 0.3,
            summary[valco_field]["value"],
            places=7,
        )

    def test_sam_economic_model_result_csv(self) -> None:
        r: GeophiresXResult = GeophiresXResult(self._get_test_file_path("sam-em-csv-test.out"))
        as_csv = r.as_csv()
        self.assertIsNotNone(as_csv)

    def test_multicategory_fields_only_in_case_report_category(self) -> None:
        r: GeophiresXResult = GeophiresXResult(
            self._get_test_file_path("../examples/example_SAM-single-owner-PPA-3.out")
        )
        self.assertIsNone(r.result["EXTENDED ECONOMICS"]["Total Add-on CAPEX"])
        self.assertIsNone(r.result["EXTENDED ECONOMICS"]["Total Add-on OPEX"])

        self.assertIn("Total Add-on CAPEX", r.result["CAPITAL COSTS (M$)"])
        self.assertIn("Total Add-on OPEX", r.result["OPERATING AND MAINTENANCE COSTS (M$/yr)"])

        self.assertIsNone(r.result["RESERVOIR SIMULATION RESULTS"]["Average Net Electricity Production"])
        self.assertIsNotNone(r.result["SUMMARY OF RESULTS"]["Average Net Electricity Production"])
        self.assertIsNotNone(r.result["SURFACE EQUIPMENT SIMULATION RESULTS"]["Average Net Electricity Generation"])

    def test_ags_clgs_style_output(self) -> None:
        r: GeophiresXResult = GeophiresXResult(
            self._get_test_file_path("../examples/Beckers_et_al_2023_Tabulated_Database_Uloop_sCO2_elec.out")
        )
        self.assertIsNotNone(r.result["SUMMARY OF RESULTS"]["LCOE"])

    def test_sutra_reservoir_model_in_summary(self) -> None:
        r: GeophiresXResult = GeophiresXResult(self._get_test_file_path("../examples/SUTRAExample1.out"))
        self.assertEqual("SUTRA Model", r.result["SUMMARY OF RESULTS"]["Reservoir Model"])

    def test_produced_temperature_json_output(self) -> None:
        r: GeophiresXResult = GeophiresXClient().get_geophires_result(
            ImmutableGeophiresInputParameters(from_file_path=self._get_test_file_path("client_test_input_1.txt"))
        )
        with open(r.json_output_file_path, encoding="utf-8") as f:
            r_json_obj: dict[str, Any] = json.load(f)

        prod_temp_key: str = "Produced Temperature"
        self.assertIn(prod_temp_key, r_json_obj)
        self.assertGreater(len(r_json_obj[prod_temp_key]["value"]), 100)
        self.assertTrue(all(it > 0 for it in r_json_obj[prod_temp_key]["value"]))

    def test_dispatch_summary_json_property(self) -> None:
        r: GeophiresXResult = GeophiresXClient().get_geophires_result(
            ImmutableGeophiresInputParameters(
                from_file_path=self._get_test_file_path("../examples/example1.txt"),
                params={
                    "Operating Mode": "Dispatchable",
                    "End-Use Option": "31",
                    "Dispatch Demand Source": "Annual Heat Demand",
                    "Dispatch Flow Strategy": "Demand Following",
                    "Plant Lifetime": "1",
                    "Annual Heat Demand": self._get_test_file_path("../assets/params/annual_heat_demand.csv"),
                },
            )
        )

        self.assertIsNotNone(r.json_fields)
        dispatch_summary = r.dispatch_summary_json
        self.assertIsNotNone(dispatch_summary)
        self.assertEqual(1, dispatch_summary["schema_version"])
        self.assertEqual("thermal", dispatch_summary["demand_type"])
        self.assertEqual("chp", dispatch_summary["surfaceplant_mode"])
        self.assertEqual("Annual Heat Demand", dispatch_summary["dispatch_settings"]["demand_source"])
        self.assertEqual("Demand Following", dispatch_summary["dispatch_settings"]["flow_strategy"])
        self.assertEqual(1, dispatch_summary["analysis_window"]["start_year"])
        self.assertEqual(2, dispatch_summary["analysis_window"]["end_year"])
        self.assertIn("summary_metrics", dispatch_summary)
        self.assertGreater(dispatch_summary["summary_metrics"]["annual_served_heat_kwh"], 0.0)
        self.assertGreater(dispatch_summary["summary_metrics"]["annual_served_electricity_kwh"], 0.0)

    def test_metadata_end_use_option_parses_numeric_output_value(self) -> None:
        r: GeophiresXResult = GeophiresXClient().get_geophires_result(
            ImmutableGeophiresInputParameters(
                params={
                    "Print Output to Console": 0,
                    "End-Use Option": 2,
                    "Reservoir Model": 1,
                    "Time steps per year": 1,
                    "Reservoir Depth": 3,
                    "Gradient 1": 50,
                    "Maximum Temperature": 250,
                }
            )
        )

        self.assertEqual("DIRECT_USE_HEAT", r.result["metadata"]["End-Use Option"])
