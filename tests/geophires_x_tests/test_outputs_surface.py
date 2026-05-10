from __future__ import annotations

from geophires_x.OptionList import EndUseOptions
from geophires_x.OptionList import PlantType
from geophires_x.OutputsSurface import has_electricity_component
from geophires_x.OutputsSurface import writes_surface_heat_results


def test_cogeneration_topping_extra_heat_writes_surface_heat_results_from_raw_values():
    assert has_electricity_component(EndUseOptions.COGENERATION_TOPPING_EXTRA_HEAT)
    assert has_electricity_component(31)
    assert has_electricity_component("31")

    assert writes_surface_heat_results(EndUseOptions.COGENERATION_TOPPING_EXTRA_HEAT)
    assert writes_surface_heat_results(31)
    assert writes_surface_heat_results("31")


def test_special_heat_plant_types_write_surface_heat_results_from_raw_values():
    assert writes_surface_heat_results(PlantType.DISTRICT_HEATING)
    assert writes_surface_heat_results(7)
    assert writes_surface_heat_results("7")
