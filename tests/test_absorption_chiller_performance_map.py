import numpy as np

from geophires_x.absorption.chiller_bank import ChillerBank
from geophires_x.absorption.chiller_unit import ChillerUnit
from geophires_x.absorption.performance_map import PerformanceMap


def test_fallback_performance_map_uses_temperature_inputs():
    performance_map = PerformanceMap(rated_cop=0.8)

    nominal = performance_map.evaluate(1.0, t_gen_c=95.0, t_evap_c=7.0, t_cond_c=30.0)["cop"]
    hotter_generator = performance_map.evaluate(1.0, t_gen_c=105.0, t_evap_c=7.0, t_cond_c=30.0)["cop"]
    hotter_condenser = performance_map.evaluate(1.0, t_gen_c=95.0, t_evap_c=7.0, t_cond_c=40.0)["cop"]

    assert hotter_generator > nominal
    assert hotter_condenser < nominal


def test_chiller_bank_hourly_temperatures_change_generator_heat_required():
    unit = ChillerUnit("C1", "X", 1000.0, nominal_COP=0.8, min_PLR=0.2, installed_cost_USD=100000)
    bank = ChillerBank(units=[(unit, 1)], dispatch_strategy="min_cost", n_segments=5, use_hourly_temps=True)

    results = bank.dispatch_hourly(
        np.array([500.0, 500.0]),
        temps={
            "t_gen": np.array([95.0, 85.0]),
            "t_evap": np.array([7.0, 7.0]),
            "t_cond": np.array([30.0, 40.0]),
        },
        use_milp=False,
    )

    assert results["cooling_produced_hourly"].tolist() == [500.0, 500.0]
    assert results["q_gen_hourly"][1] > results["q_gen_hourly"][0]
