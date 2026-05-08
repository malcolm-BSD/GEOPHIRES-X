from __future__ import annotations

import csv
import io
from io import TextIOWrapper
from typing import TYPE_CHECKING, Any

from geophires_x.DispatchReporting import dispatch_profile_columns
from geophires_x.DispatchReporting import dispatch_profile_rows
from geophires_x.DispatchReporting import dispatch_profile_table
from geophires_x.DispatchReporting import dispatch_profile_tess_columns as _dispatch_profile_tess_columns
from geophires_x.DispatchReporting import dispatch_profile_tess_row as _dispatch_profile_tess_row
from geophires_x.OptionList import EndUseOptions, PlantType

if TYPE_CHECKING:
    from geophires_x.Model import Model

NL = "\n"

COGENERATION_END_USE_OPTIONS = (
    EndUseOptions.COGENERATION_TOPPING_EXTRA_HEAT,
    EndUseOptions.COGENERATION_TOPPING_EXTRA_ELECTRICITY,
    EndUseOptions.COGENERATION_BOTTOMING_EXTRA_ELECTRICITY,
    EndUseOptions.COGENERATION_BOTTOMING_EXTRA_HEAT,
    EndUseOptions.COGENERATION_PARALLEL_EXTRA_HEAT,
    EndUseOptions.COGENERATION_PARALLEL_EXTRA_ELECTRICITY,
)
ELECTRICITY_END_USE_OPTIONS = (EndUseOptions.ELECTRICITY, *COGENERATION_END_USE_OPTIONS)
HEAT_END_USE_OPTIONS = (EndUseOptions.HEAT, *COGENERATION_END_USE_OPTIONS)


def has_electricity_component(enduse_option: EndUseOptions) -> bool:
    return enduse_option in ELECTRICITY_END_USE_OPTIONS


def has_heat_component(enduse_option: EndUseOptions) -> bool:
    return enduse_option in HEAT_END_USE_OPTIONS


def dispatch_output_rows(model: Model) -> list[tuple[str, float, str]]:
    dispatch_results = getattr(model, "dispatch_results", None)
    if dispatch_results is None:
        return []

    metrics = dispatch_results.summary_metrics
    demand_type = getattr(dispatch_results, "demand_type", "thermal")
    enduse_option = model.surfaceplant.enduse_option.value
    has_heat = has_heat_component(enduse_option)
    has_electric = has_electricity_component(enduse_option)
    plant_type = model.surfaceplant.plant_type.value
    rows = [
        ("Dispatch analysis start year", metrics.get("dispatch_analysis_start_year", 1.0), "year"),
        ("Dispatch analysis end year", metrics.get("dispatch_analysis_end_year", 2.0), "year"),
        ("Dispatch capacity factor", metrics.get("dispatch_capacity_factor", 0.0) * 100.0, "%"),
        ("Average runtime fraction", metrics.get("average_runtime_fraction", 0.0) * 100.0, "%"),
        ("Peak hourly demand", metrics.get("peak_hourly_demand_mw", 0.0), "MW"),
        ("Design flow rate", metrics.get("design_flow_kg_per_sec", 0.0), "kg/s"),
        ("Observed peak flow rate", metrics.get("observed_peak_flow_kg_per_sec", 0.0), "kg/s"),
    ]
    summary_rows = []
    if demand_type == "electric":
        if has_heat:
            summary_rows.append(("Design heat produced", metrics.get("design_heat_produced_mw", 0.0), "MW"))
        if has_electric:
            summary_rows.extend([
                ("Design net electricity produced", metrics.get("design_net_electricity_produced_mw", 0.0), "MW"),
                (
                    "Annual geothermal electricity delivered",
                    metrics.get("annual_served_electricity_kwh", 0.0) / 1.0e6,
                    "GWh/year",
                ),
            ])
        if has_heat:
            summary_rows.append(
                ("Annual geothermal heat delivered", metrics.get("annual_served_heat_kwh", 0.0) / 1.0e6, "GWh/year")
            )
        summary_rows.extend([
            (
                "Annual unmet electricity demand",
                metrics.get("annual_unmet_electricity_kwh", 0.0) / 1.0e6,
                "GWh/year",
            ),
            ("Peak geothermal contribution", metrics.get("peak_served_electricity_kwh", 0.0) / 1000.0, "MW"),
            ("Peak unmet load", metrics.get("peak_unmet_electricity_kwh", 0.0) / 1000.0, "MW"),
        ])
    elif demand_type == "cooling":
        summary_rows.extend([
            ("Design heat produced", metrics.get("design_heat_produced_mw", 0.0), "MW"),
            ("Design cooling produced", metrics.get("design_cooling_produced_mw", 0.0), "MW"),
            ("Annual geothermal cooling delivered", metrics.get("annual_served_cooling_kwh", 0.0) / 1.0e6, "GWh/year"),
            ("Annual unmet cooling demand", metrics.get("annual_unmet_cooling_kwh", 0.0) / 1.0e6, "GWh/year"),
            ("Peak geothermal contribution", metrics.get("peak_served_cooling_kwh", 0.0) / 1000.0, "MW"),
            ("Peak unmet load", metrics.get("peak_unmet_cooling_kwh", 0.0) / 1000.0, "MW"),
        ])
    else:
        if has_heat:
            summary_rows.extend([
                ("Design heat produced", metrics.get("design_heat_produced_mw", 0.0), "MW"),
                ("Annual geothermal heat delivered", metrics.get("annual_served_heat_kwh", 0.0) / 1.0e6, "GWh/year"),
            ])
        if has_electric:
            summary_rows.extend([
                ("Design net electricity produced", metrics.get("design_net_electricity_produced_mw", 0.0), "MW"),
                (
                    "Annual geothermal electricity delivered",
                    metrics.get("annual_served_electricity_kwh", 0.0) / 1.0e6,
                    "GWh/year",
                ),
            ])
        summary_rows.extend([
            ("Annual unmet thermal demand", metrics.get("annual_unmet_heat_kwh", 0.0) / 1.0e6, "GWh/year"),
            ("Peak geothermal contribution", metrics.get("peak_served_heat_kwh", 0.0) / 1000.0, "MW"),
            ("Peak unmet load", metrics.get("peak_unmet_heat_kwh", 0.0) / 1000.0, "MW"),
        ])
        if plant_type == PlantType.HEAT_PUMP:
            summary_rows.append(
                (
                    "Annual heat pump electricity consumed",
                    metrics.get("annual_heat_pump_electricity_kwh", 0.0) / 1.0e6,
                    "GWh/year",
                )
            )
        if plant_type == PlantType.DISTRICT_HEATING:
            summary_rows.extend([
                (
                    "Annual peaking boiler heat delivered",
                    metrics.get("annual_district_heating_boiler_kwh", 0.0) / 1.0e6,
                    "GWh/year",
                ),
                ("Peak peaking boiler demand", metrics.get("peak_district_heating_boiler_mw", 0.0), "MW"),
            ])
    rows[3:3] = summary_rows
    return rows


def tess_output_rows(model: Model, metrics: dict[str, float]) -> list[tuple[str, float, str]]:
    """Return TESS summary rows for dispatch text and parsed output."""
    if metrics.get("tess_enabled", 0.0) <= 0.0:
        return []

    return [
        ("TESS volume", metrics.get("tess_volume_m3", 0.0), "m3"),
        ("TESS usable capacity", metrics.get("tess_usable_capacity_mwh", 0.0), "MWh"),
        ("TESS capital cost", model.economics.tess_capital_cost.value, model.economics.tess_capital_cost.CurrentUnits.value),
        ("TESS fixed O&M cost", model.economics.tess_o_and_m_cost.value, model.economics.tess_o_and_m_cost.CurrentUnits.value),
        ("TESS average SOC", metrics.get("tess_average_soc", 0.0) * 100.0, "%"),
        ("TESS minimum SOC", metrics.get("tess_min_soc", 0.0) * 100.0, "%"),
        ("TESS maximum SOC", metrics.get("tess_max_soc", 0.0) * 100.0, "%"),
        ("TESS annual charge", metrics.get("tess_annual_charge_kwh", 0.0) / 1.0e6, "GWh/year"),
        ("TESS annual discharge", metrics.get("tess_annual_discharge_kwh", 0.0) / 1.0e6, "GWh/year"),
        ("TESS annual standby loss", metrics.get("tess_annual_standby_loss_kwh", 0.0) / 1.0e6, "GWh/year"),
        ("TESS annual efficiency loss", metrics.get("tess_annual_efficiency_loss_kwh", 0.0) / 1.0e6, "GWh/year"),
        ("TESS curtailed geothermal heat", metrics.get("tess_annual_curtailed_heat_kwh", 0.0) / 1.0e6, "GWh/year"),
        ("TESS equivalent full cycles", metrics.get("tess_equivalent_full_cycles", 0.0), "cycles/year"),
        ("Peak customer demand", metrics.get("peak_customer_demand_mw", 0.0), "MW"),
        ("Peak geothermal charge", metrics.get("peak_geothermal_charge_mw", 0.0), "MW"),
        ("Customer demand standard deviation", metrics.get("customer_demand_standard_deviation_mw", 0.0), "MW"),
        ("Geothermal output standard deviation", metrics.get("geothermal_output_standard_deviation_mw", 0.0), "MW"),
        ("Geothermal output smoothing ratio", metrics.get("geothermal_output_smoothing_ratio", 0.0), "-"),
        ("Geothermal peak reduction ratio", metrics.get("geothermal_peak_reduction_fraction", 0.0), "-"),
        ("Geothermal variability reduction ratio", metrics.get("geothermal_output_variability_reduction_fraction", 0.0), "-"),
    ]


def _dispatch_profile_cell(column_name: str, value: str | int | float) -> str:
    if isinstance(value, str):
        return value

    if column_name in {"Year", "Hour of Year", "Simulation Hour"}:
        return f"{value:.0f}"

    if "Temperature" in column_name:
        return f"{value:.2f}"

    if "Runtime Fraction" in column_name:
        return f"{value:.4f}"

    return f"{value:.4f}"


def dispatch_profile_report_text(category_name: str, table: list[list[str | int | float]]) -> str:
    """Return the hourly dispatch profile as a fixed-width report table."""
    if len(table) == 0:
        return ""

    columns = [str(column_name) for column_name in table[0]]
    rows = [
        [_dispatch_profile_cell(column_name, value) for column_name, value in zip(columns, row)]
        for row in table[1:]
    ]
    widths = [
        max(len(column_name), *(len(row[column_index]) for row in rows))
        for column_index, column_name in enumerate(columns)
    ]

    buffer = io.StringIO()
    buffer.write(NL)
    buffer.write(NL)
    buffer.write("                            **********************\n")
    buffer.write(f"                            *  {category_name}  *\n")
    buffer.write("                            **********************\n")
    buffer.write("  ".join(column_name.ljust(widths[index]) for index, column_name in enumerate(columns)) + NL)
    buffer.write("  ".join("_" * width for width in widths) + NL)
    for row in rows:
        buffer.write("  ".join(value.rjust(widths[index]) for index, value in enumerate(row)) + NL)
    buffer.write(NL)
    return buffer.getvalue()


def write_dispatch_profile_report_table(model: Model, f: TextIOWrapper, category_name: str) -> None:
    table = dispatch_profile_table(model)
    if len(table) == 0:
        return

    f.write(dispatch_profile_report_text(category_name, table))


def dispatch_profile_tess_columns(dispatch_results: Any) -> list[str]:
    """Return TESS profile CSV columns when storage is active."""
    return _dispatch_profile_tess_columns(dispatch_results)


def dispatch_profile_tess_row(dispatch_results: Any, timestep_index: int) -> list[float]:
    """Return one timestep of TESS profile CSV data when storage is active."""
    return _dispatch_profile_tess_row(dispatch_results, timestep_index)


def write_dispatch_profile_output(model: Model, output_file: str) -> None:
    dispatch_results = getattr(model, "dispatch_results", None)
    if dispatch_results is None:
        return

    num_timesteps = len(dispatch_results.hourly_thermal_demand)
    if num_timesteps == 0:
        return

    with open(output_file, "w", encoding="UTF-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(dispatch_profile_columns(dispatch_results))
        writer.writerows(dispatch_profile_rows(dispatch_results))
