"""Behavioural tests for absorption chiller subsystem.

Covers energy balance, part-load/performance map behavior, and catalog selection.
"""

import json

import numpy as np

from geophires_x.absorption.absorption_chiller import AbsorptionChiller
from geophires_x.absorption.catalog import Catalog
from geophires_x.absorption.performance_map import PerformanceMap


def test_performance_map_partload():
    pm = PerformanceMap(rated_cop=0.8, plr_curve_params={"alpha": 0.15, "beta": 1.2})
    cop_full = pm.evaluate(1.0, 100.0, 7.0, 30.0)["cop"]
    cop_low = pm.evaluate(0.2, 100.0, 7.0, 30.0)["cop"]
    assert cop_full > cop_low


def test_chiller_energy_balance_and_mdot():
    # single unit test via AbsorptionChiller with a simple demand
    hours = 24
    cooling = np.zeros(hours)
    cooling[8:18] = 500.0  # 500 kW daytime load
    geo_t = np.full(hours, 95.0)
    ch = AbsorptionChiller()
    results = ch.evaluate_hourly(cooling, geo_t, chilled_supply_setpoint_c=7.0)

    cooling_out = results.get("cooling_produced_hourly")
    q_gen = results.get("q_gen_hourly")
    cop = results.get("COP_hourly")
    chilled_mdot = results.get("chilled_mdot_hourly")

    # energy balance: q_gen ~= cooling / cop (within tolerance)
    nonzero = cop > 0
    if np.any(nonzero):
        ratio = np.where(nonzero, cooling_out / (q_gen + 1e-12), 1.0)
        assert np.allclose(ratio[nonzero], cop[nonzero], rtol=1e-3, atol=1e-6)

    # chilled mdot: mdot = Q / (cp * dT) where Q [kW] -> W
    cp = ch.fluid_adapter.cp("Water", 7.0)
    dT = max(1.0, ch.chilled_deltaT_K)
    expected_mdot = (cooling_out * 1000.0) / (cp * dT)
    assert np.allclose(chilled_mdot, expected_mdot, rtol=1e-3, atol=1e-6)


def test_catalog_selection_meets_capacity():
    catalog = Catalog()
    sel = catalog.select_min_cost_set(2500)
    assert sel["total_capacity_kW"] >= 2500
    assert isinstance(sel["selected"], list)


def test_default_catalog_has_complete_provenance():
    catalog = Catalog()

    assert catalog.entries
    assert catalog.has_complete_provenance(), catalog.provenance_issues()


def test_remote_catalog_query_reads_json_and_caches(tmp_path):
    remote_catalog = tmp_path / "remote_catalog.json"
    cache_path = tmp_path / "remote_cache.json"
    remote_catalog.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "model_id": "REMOTE-1",
                        "manufacturer": "Remote Maker",
                        "nominal_cooling_kW": "1250",
                        "nominal_COP": "0.82",
                        "effect_type": "single",
                        "refrigerant_family": "LiBr-water",
                        "installed_cost_USD": "1000000",
                        "source": "remote test fixture",
                        "source_url": "file://remote_catalog.json",
                        "last_verified": "2026-05-07",
                        "license_note": "test fixture",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    catalog = Catalog(cache_path=str(cache_path))

    entries = catalog.query_remote_catalog({"url": remote_catalog.as_uri()})

    assert entries[0].get("model_id") == "REMOTE-1"
    assert cache_path.exists()

    remote_catalog.unlink()
    cached_entries = catalog.query_remote_catalog({"url": remote_catalog.as_uri()})
    assert cached_entries[0].get("model_id") == "REMOTE-1"
