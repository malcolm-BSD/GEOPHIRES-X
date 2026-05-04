"""Tests for catalog ILP selection using PuLP.

This test is skipped automatically if PuLP is not installed in the test
environment. It verifies that the integer-program solver chooses the least
cost combination of units to meet a required capacity when that choice is
cheaper than a single large unit.
"""

import pytest

from geophires_x.absorption.catalog import Catalog
from geophires_x.absorption.catalog import CatalogEntry


def test_catalog_ilp_selection_optimal():
    pytest.importorskip("pulp")

    cat = Catalog()
    # build a tiny synthetic catalog in-memory
    cat.entries = [
        CatalogEntry(model_id="A", nominal_cooling_kW="1000", installed_cost_USD="1000"),
        CatalogEntry(model_id="B", nominal_cooling_kW="600", installed_cost_USD="400"),
        CatalogEntry(model_id="C", nominal_cooling_kW="400", installed_cost_USD="200"),
    ]

    sel = cat.select_min_cost_set(1000)
    # optimal is B(600)+C(400) with total cost 600, not A with cost 1000
    assert sel["total_capacity_kW"] >= 1000
    assert pytest.approx(sel["estimated_cost_USD"]) == 600.0
    selected_map = {s["model_id"]: s["count"] for s in sel["selected"]}
    assert selected_map.get("B", 0) == 1
    assert selected_map.get("C", 0) == 1
