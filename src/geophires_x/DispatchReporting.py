from __future__ import annotations

from typing import Any


DISPATCH_PROFILE_CATEGORY_NAME = "DISPATCH PROFILE"


def is_dispatch_report(model: Any) -> bool:
    return getattr(model, "dispatch_results", None) is not None


def weather_output_rows(model: Any) -> list[tuple[str, str | float, str | None]]:
    weather_data = getattr(model, "weather_data", None)
    if weather_data is None:
        return []

    hourly_data = weather_data.hourly_data
    temperature_units = weather_data.hourly_units.get("temperature_2m", "degC")
    hourly_temperature = hourly_data["temperature_2m"]
    return [
        ("Annual average temperature (from Open-Meteo)", float(hourly_temperature.mean()), temperature_units),
        ("Minimum hourly temperature (from Open-Meteo)", float(hourly_temperature.min()), temperature_units),
        ("Maximum hourly temperature (from Open-Meteo)", float(hourly_temperature.max()), temperature_units),
    ]


def dispatch_profile_tess_columns(dispatch_results: Any) -> list[str]:
    if dispatch_results.summary_metrics.get("tess_enabled", 0.0) <= 0.0:
        return []

    return [
        "TESS Temperature (degC)",
        "TESS State of Charge (-)",
        "TESS Stored Energy (MWh)",
        "TESS Discharge to Load (MW)",
        "TESS Charge from Geothermal (MW)",
        "TESS Charge Curtailed (MW)",
        "TESS Standby Loss (MW)",
        "TESS Efficiency Loss (MW)",
        "Geothermal Charge Command (MW)",
    ]


def dispatch_profile_tess_row(dispatch_results: Any, timestep_index: int) -> list[float]:
    if dispatch_results.summary_metrics.get("tess_enabled", 0.0) <= 0.0:
        return []

    return [
        float(dispatch_results.hourly_tess_temperature[timestep_index]),
        float(dispatch_results.hourly_tess_soc[timestep_index]),
        float(dispatch_results.hourly_tess_stored_energy[timestep_index]),
        float(dispatch_results.hourly_tess_discharge_to_load[timestep_index]),
        float(dispatch_results.hourly_tess_charge_from_geothermal[timestep_index]),
        float(dispatch_results.hourly_tess_charge_curtailed[timestep_index]),
        float(dispatch_results.hourly_tess_standby_loss[timestep_index]),
        float(dispatch_results.hourly_tess_efficiency_loss[timestep_index]),
        float(dispatch_results.hourly_geothermal_charge_command[timestep_index]),
    ]


def dispatch_profile_columns(dispatch_results: Any) -> list[str]:
    demand_type = getattr(dispatch_results, "demand_type", "thermal")
    demand_column = "Electricity Demand (MW)" if demand_type == "electric" else "Thermal Demand (MW)"
    output_column = (
        "Geothermal Electric Output (MW)"
        if demand_type == "electric"
        else "Geothermal Thermal Output (MW)"
    )
    return [
        "Year",
        "Hour of Year",
        "Simulation Hour",
        demand_column,
        output_column,
        "Demand Served (MW)",
        "Unmet Demand (MW)",
        "Produced Temperature (degC)",
        "Ambient Temperature (degC)",
        "Flow Rate (kg/s)",
        "Runtime Fraction",
        "Pumping Power (MW)",
        *dispatch_profile_tess_columns(dispatch_results),
    ]


def dispatch_profile_row(dispatch_results: Any, timestep_index: int) -> list[int | float]:
    timesteps_per_year = 8760
    simulation_start_hour = getattr(dispatch_results, "simulation_start_hour", 1)
    analysis_start_year = getattr(dispatch_results, "analysis_start_year", 1)
    demand_type = getattr(dispatch_results, "demand_type", "thermal")

    return [
        analysis_start_year + (timestep_index // timesteps_per_year),
        timestep_index % timesteps_per_year + 1,
        simulation_start_hour + timestep_index,
        float(dispatch_results.hourly_thermal_demand[timestep_index]),
        float(
            dispatch_results.hourly_geothermal_electric_output[timestep_index]
            if demand_type == "electric"
            else dispatch_results.hourly_geothermal_thermal_output[timestep_index]
        ),
        float(dispatch_results.hourly_demand_served[timestep_index] / 1000.0),
        float(dispatch_results.hourly_unmet_demand[timestep_index] / 1000.0),
        float(dispatch_results.hourly_produced_temperature[timestep_index]),
        float(dispatch_results.hourly_ambient_temperature[timestep_index]),
        float(dispatch_results.hourly_flow[timestep_index]),
        float(dispatch_results.hourly_runtime_fraction[timestep_index]),
        float(dispatch_results.hourly_pumping_power[timestep_index]),
        *dispatch_profile_tess_row(dispatch_results, timestep_index),
    ]


def dispatch_profile_rows(dispatch_results: Any) -> list[list[int | float]]:
    num_timesteps = len(dispatch_results.hourly_thermal_demand)
    if num_timesteps == 0:
        return []

    return [dispatch_profile_row(dispatch_results, timestep_index) for timestep_index in range(num_timesteps)]


def dispatch_profile_table(model: Any) -> list[list[str | int | float]]:
    dispatch_results = getattr(model, "dispatch_results", None)
    if dispatch_results is None or len(dispatch_results.hourly_thermal_demand) == 0:
        return []

    return [dispatch_profile_columns(dispatch_results), *dispatch_profile_rows(dispatch_results)]


def build_dispatch_profile_json(model: Any) -> dict[str, Any] | None:
    dispatch_results = getattr(model, "dispatch_results", None)
    if dispatch_results is None or len(dispatch_results.hourly_thermal_demand) == 0:
        return None

    return {
        "schema_version": 1,
        "columns": dispatch_profile_columns(dispatch_results),
        "rows": dispatch_profile_rows(dispatch_results),
    }
