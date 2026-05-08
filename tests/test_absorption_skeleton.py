"""Basic tests for absorption chiller skeleton modules.

These tests validate that modules import and basic flows run without error.
"""

import numpy as np

from geophires_x.absorption.absorption_chiller import AbsorptionChiller


def test_absorption_chiller_evaluate_hourly_runs():
    ch = AbsorptionChiller()
    hours = 24
    cooling = np.zeros(hours)
    cooling[8:18] = 500.0  # 500 kW daytime load
    geo_t = np.full(hours, 95.0)
    results = ch.evaluate_hourly(cooling, geo_t, chilled_supply_setpoint_c=7.0)
    assert "cooling_produced_hourly" in results
    assert len(results["cooling_produced_hourly"]) == hours
    assert "q_gen_hourly" in results
    assert "chilled_mdot_hourly" in results
