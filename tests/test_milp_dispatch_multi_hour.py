import importlib.util

import numpy as np
import pytest

from geophires_x.absorption.chiller_bank import ChillerBank
from geophires_x.absorption.chiller_unit import ChillerUnit

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pulp") is None, reason="pulp is required for MILP dispatch tests"
)


def test_milp_dispatch_respects_generator_heat():
    # Create units: two types
    uA = ChillerUnit("A1000", "Maker", 1000.0, nominal_COP=0.8, min_PLR=0.2, installed_cost_USD=200000)
    uB = ChillerUnit("B500", "Maker", 500.0, nominal_COP=0.75, min_PLR=0.2, installed_cost_USD=80000)

    # bank: 3 of A and 2 of B
    bank = ChillerBank(units=[(uA, 3), (uB, 2)], dispatch_strategy="follow_heat", n_segments=5, use_hourly_temps=True)

    # multi-hour cooling demand
    cooling = np.array([2500.0, 1500.0, 3000.0, 0.0])
    # provide tight available generator heat for hour 2 and 3 to force scaling
    # set generous for hour 0 and 1
    avail_heat = np.array([4000.0, 2000.0, 2500.0, 0.0])

    # create hourly temps for the dispatch (generator, evaporator, condenser)
    temps = {
        "t_gen": np.array([95.0, 95.0, 85.0, 95.0]),
        "t_evap": np.array([7.0, 7.0, 7.0, 7.0]),
        "t_cond": np.array([30.0, 30.0, 30.0, 30.0]),
    }

    results = bank.dispatch_hourly(cooling, generator_heat_available_kW_hourly=avail_heat, temps=temps, mode="dispatch")

    cooling_out = results["cooling_produced_hourly"]
    q_gen = results["q_gen_hourly"]

    # produced cooling must not exceed requested by large margin
    assert np.all(cooling_out <= cooling + 1e-6)

    # where available heat provided, q_gen must be <= available_heat + small tol
    mask = avail_heat > 0
    assert np.all(q_gen[mask] <= avail_heat[mask] + 1e-6)

    # sanity: unit_dispatch shape and arrays present
    assert "unit_dispatch" in results
    ud = results["unit_dispatch"]
    assert ud.shape[1] == len(cooling)


def test_milp_dispatch_prefers_lower_cost_units_when_equal_heat():
    # Create two unit types same capacity but different installed cost
    cheap = ChillerUnit("C1", "X", 1000.0, nominal_COP=0.8, min_PLR=0.2, installed_cost_USD=100000)
    expensive = ChillerUnit("C2", "Y", 1000.0, nominal_COP=0.8, min_PLR=0.2, installed_cost_USD=300000)

    bank = ChillerBank(units=[(cheap, 1), (expensive, 1)], dispatch_strategy="min_cost", n_segments=5)

    cooling = np.array([1000.0])
    # generous heat
    avail_heat = np.array([5000.0])

    results = bank.dispatch_hourly(cooling, generator_heat_available_kW_hourly=avail_heat)

    # when both can meet load, cheap unit should be chosen first (installed-cost objective)
    ud = results["unit_dispatch"]
    assert ud[0, 0] == 1
    assert ud[1, 0] == 0


def test_milp_min_cost_dispatch_respects_generator_heat():
    unit = ChillerUnit("C1", "X", 1000.0, nominal_COP=1.0, min_PLR=0.2, installed_cost_USD=100000)
    bank = ChillerBank(units=[(unit, 1)], dispatch_strategy="min_cost", n_segments=5)

    results = bank.dispatch_hourly(
        np.array([1000.0]),
        generator_heat_available_kW_hourly=np.array([500.0]),
    )

    assert results["q_gen_hourly"][0] <= 500.0 + 1e-6
    assert 0.0 < results["cooling_produced_hourly"][0] < 1000.0


def test_milp_dispatch_does_not_overproduce_when_load_below_minimum_plr():
    unit = ChillerUnit("C1", "X", 1000.0, nominal_COP=0.8, min_PLR=0.2, installed_cost_USD=100000)
    bank = ChillerBank(units=[(unit, 1)], dispatch_strategy="min_cost", n_segments=5)

    results = bank.dispatch_hourly(np.array([1.0]))

    assert results["cooling_produced_hourly"][0] <= 1.0 + 1e-6
    assert results["unit_dispatch"][0, 0] == 0


def test_milp_dispatch_can_partially_load_above_minimum_plr_without_overproduction():
    unit = ChillerUnit("C1", "X", 1000.0, nominal_COP=0.8, min_PLR=0.2, installed_cost_USD=100000)
    bank = ChillerBank(units=[(unit, 1)], dispatch_strategy="min_cost", n_segments=5)

    results = bank.dispatch_hourly(np.array([350.0]))

    assert results["cooling_produced_hourly"][0] == pytest.approx(350.0)
    assert results["unit_dispatch"][0, 0] == 1
