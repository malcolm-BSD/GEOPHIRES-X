"""Nonpositive heat output cannot support a project heat price."""

import json
import math
import re
from html import unescape
from pathlib import Path

import pytest

from geophires_x.levelized_costs import HEAT_COMMODITY
from geophires_x.levelized_costs import _build_basis
from geophires_x.OutputsUtils import HEAT_PRICE_UNAVAILABLE_REASON
from geophires_x.OutputsUtils import format_heat_price
from geophires_x_client import GeophiresInputParameters
from geophires_x_client import GeophiresXClient


@pytest.mark.parametrize("output", [-100.0, 0.0])
def test_nonpositive_heat_output_is_unavailable(output):
    basis = _build_basis(HEAT_COMMODITY, 10.0, output, 293100000.0)
    assert math.isnan(basis.public_value)
    assert format_heat_price(basis.public_value).strip() == "N/A"


@pytest.mark.parametrize("cost", [-10.0, 0.0, 10.0])
def test_positive_heat_output_preserves_cost_credits_and_zero_cost(cost):
    basis = _build_basis(HEAT_COMMODITY, cost, 100.0, 293100000.0)
    assert basis.public_value == pytest.approx(cost / 100.0 * 293100000.0)
    assert "N/A" not in format_heat_price(basis.public_value)


def test_sdac_heat_price_unavailable_in_reports_and_client(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples/S-DAC-GT.txt"
    html = tmp_path / "sdac.html"
    text = tmp_path / "sdac.txt"
    result = GeophiresXClient().get_geophires_result(
        GeophiresInputParameters(
            from_file_path=example,
            params={
                "Print Output to Console": False,
                "Do XLCO(E|H|C) Calculations": True,
                "Do VALCO(E|H|C) Calculations": True,
                "HTML Output File": str(html),
                "Improved Text Output File": str(text),
            },
        )
    )
    summary = result.result["SUMMARY OF RESULTS"]
    assert result.direct_use_heat_breakeven_price_USD_per_MMBTU is None
    assert summary["Direct-Use heat breakeven price (LCOH)"] == {"value": None, "unit": "USD/MMBTU"}
    assert summary["Heat price status"]["value"] == HEAT_PRICE_UNAVAILABLE_REASON
    for field in (
        "Extended Heat Breakeven Price (XLCOH Market)",
        "Extended Heat Breakeven Price (XLCOH Market + Social)",
        "Value-Adjusted Heat Breakeven Price (VALCOH)",
    ):
        assert summary[field]["value"] is None
    assert summary["Electricity breakeven price"]["value"] == pytest.approx(24.39)
    for report in (Path(result.output_file_path), text, html):
        content = report.read_text(encoding="utf-8")
        visible_text = " ".join(unescape(re.sub(r"<[^>]*>", "", content)).split())
        assert HEAT_PRICE_UNAVAILABLE_REASON in visible_text
        assert "N/A" in content
    raw = json.loads(result.json_output_file_path.read_text(encoding="utf-8"))
    assert raw["Project Heat Price"] == {
        "value": None,
        "unit": "USD/MMBTU",
        "status": "unavailable",
        "reason": "net heat output is nonpositive",
    }
    assert raw["XLCOH_Market"]["value"] is None
    assert raw["VALCOH"]["value"] is None
    # S-DAC's geothermal supply cost is a distinct, still-valid metric.
    assert result.result["S-DAC-GT ECONOMICS"]["Geothermal LCOH"]["value"] > 0
