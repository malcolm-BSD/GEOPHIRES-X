from geophires_x.absorption.absorption_chiller import AbsorptionChiller
from geophires_x.absorption.catalog import Catalog
from geophires_x.absorption.catalog import CatalogEntry
from geophires_x.Dispatch import _dispatch_output_state
from geophires_x.GeoPHIRESUtils import read_input_file
from geophires_x.Model import Model
from geophires_x.Parameter import ParameterEntry
from geophires_x.SurfacePlantAbsorptionChiller import SurfacePlantAbsorptionChiller


def test_dispatch_example_uses_canonical_temperature_profile_parameters():
    params = read_input_file(input_file_name="tests/examples/example11_new_AC_dispatch.txt")

    assert not any(name.startswith("Temps.") for name in params)
    assert "Dispatch Strategy" not in params
    assert params["Absorption Chiller Dispatch Strategy"].sValue == "min_cost"
    assert params["Absorption Chiller Generator Temperature Profile"].sValue == "95.0,95.0,85.0,95.0"
    assert params["Absorption Chiller Evaporator Temperature Profile"].sValue == "7.0,7.0,7.0,7.0"
    assert params["Absorption Chiller Condenser Temperature Profile"].sValue == "30.0,30.0,30.0,30.0"


def test_absorption_chiller_temperature_profiles_read_as_parameters():
    model = Model(enable_geophires_logging_config=False)
    surface_plant = SurfacePlantAbsorptionChiller(model)
    model.InputParameters = {
        "Absorption Chiller Generator Temperature Profile": ParameterEntry(
            Name="Absorption Chiller Generator Temperature Profile",
            sValue="95.0,95.0,85.0,95.0",
        ),
        "Absorption Chiller Evaporator Temperature Profile": ParameterEntry(
            Name="Absorption Chiller Evaporator Temperature Profile",
            sValue="7.0,7.0,7.0,7.0",
        ),
        "Absorption Chiller Condenser Temperature Profile": ParameterEntry(
            Name="Absorption Chiller Condenser Temperature Profile",
            sValue="30.0,30.0,30.0,30.0",
        ),
    }

    surface_plant.read_parameters(model)

    assert surface_plant.absorption_chiller_generator_temperature.value == [95.0, 95.0, 85.0, 95.0]
    assert surface_plant.absorption_chiller_evaporator_temperature.value == [7.0, 7.0, 7.0, 7.0]
    assert surface_plant.absorption_chiller_condenser_temperature.value == [30.0, 30.0, 30.0, 30.0]


def test_catalog_built_chiller_bank_preserves_installed_cost():
    catalog = Catalog()
    catalog.entries = [
        CatalogEntry(
            model_id="AC-1000",
            manufacturer="Example",
            nominal_cooling_kW="1000",
            nominal_COP="0.8",
            installed_cost_USD="123456",
        )
    ]
    chiller = AbsorptionChiller(catalog=catalog)

    bank = chiller.build_bank(500.0)

    assert bank.units[0][0].installed_cost_USD == 123456.0


def test_dispatch_absorption_chiller_uses_advanced_bank_for_cooling_output():
    model = Model(input_file="tests/examples/example11_new_AC_dispatch.txt", enable_geophires_logging_config=False)
    model.read_parameters()
    model.reserv.cpwater.value = 4180.0
    model.surfaceplant.CoolingDemand.value = [1000.0]

    state = _dispatch_output_state(
        model,
        produced_temperature_c=95.0,
        actual_flow_kg_per_sec=model.wellbores.prodwellflowrate.value,
        pumping_power_mw=0.0,
        cooling_demand_mw=1.0,
        timestep_index=0,
    )

    assert state["dispatch_output_mw"] == state["cooling_produced_mw"]
    assert state["cooling_produced_mw"] > 0.0
    assert getattr(model.surfaceplant, "_absorption_chiller_dispatch_bank", None) is not None


def test_geophires_dispatch_integration_uses_fast_chiller_bank_path(monkeypatch):
    model = Model(input_file="tests/examples/example11_new_AC_dispatch.txt", enable_geophires_logging_config=False)
    model.read_parameters()
    model.reserv.cpwater.value = 4180.0
    model.surfaceplant.CoolingDemand.value = [1000.0]

    calls = []
    original_build_bank = model.surfaceplant._advanced_absorption_chiller().build_bank

    def build_tracking_bank(required_capacity_kW):
        bank = original_build_bank(required_capacity_kW)
        original_dispatch = bank.dispatch_hourly

        def dispatch_tracking(*args, **kwargs):
            calls.append(kwargs.get("use_milp"))
            return original_dispatch(*args, **kwargs)

        monkeypatch.setattr(bank, "dispatch_hourly", dispatch_tracking)
        return bank

    monkeypatch.setattr(model.surfaceplant._advanced_absorption_chiller(), "build_bank", build_tracking_bank)

    model.surfaceplant.advanced_dispatch_output(
        model,
        cooling_demand_mw=1.0,
        generator_heat_available_mw=10.0,
        generator_temperature_c=95.0,
    )

    assert calls == [False]
