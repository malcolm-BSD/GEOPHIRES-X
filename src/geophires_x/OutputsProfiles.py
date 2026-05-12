from __future__ import annotations

from io import TextIOWrapper
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

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
SPECIAL_HEAT_PLANT_TYPES = (
    PlantType.ABSORPTION_CHILLER,
    PlantType.HEAT_PUMP,
    PlantType.DISTRICT_HEATING,
)


def _is_cogeneration_end_use(enduse_option: EndUseOptions) -> bool:
    return enduse_option in COGENERATION_END_USE_OPTIONS


def _thermal_drawdown_ratio(model: Model, idx: int) -> float:
    initial_temp = float(model.wellbores.ProducedTemperature.value[0])
    if initial_temp == 0.0:
        return 0.0

    return float(model.wellbores.ProducedTemperature.value[idx] / initial_temp)


def shorten_array_to_annual(array_to_shorten: pd.array, new_length: int, time_steps_per_year: int) -> pd.array:
    if len(array_to_shorten) == new_length:
        return array_to_shorten

    new_array = np.zeros(new_length)
    j = 0
    for i in range(0, len(array_to_shorten), time_steps_per_year):
        new_array[j] = array_to_shorten[i]
        j = j + 1

    return new_array


def _dispatch_report_year_count(model: Model, default: int | None = None) -> int | None:
    dispatch_results = getattr(model, "dispatch_results", None)
    if dispatch_results is None:
        return default

    analysis_end_year = int(
        getattr(dispatch_results, "analysis_end_year", model.surfaceplant.plant_lifetime.value)
    )
    return min(analysis_end_year, model.surfaceplant.plant_lifetime.value)


def _filter_dispatch_year_rows(table: pd.DataFrame, model: Model) -> pd.DataFrame:
    report_end_year = _dispatch_report_year_count(model)
    if report_end_year is None:
        return table

    year_column = next((column for column in table.columns if str(column).startswith("Year|")), None)
    if year_column is None:
        return table

    return table[table[year_column].isin(range(1, report_end_year + 1))]


def production_profile_table(model: Model, dispatch_report: bool) -> pd.DataFrame:
    if dispatch_report:
        return pd.DataFrame()

    hce = pd.DataFrame()
    hce[f"Year|:2.0f"] = [i for i in range(1, (model.surfaceplant.plant_lifetime.value + 1))]
    short_pt = shorten_array_to_annual(
        model.wellbores.ProducedTemperature.value,
        model.surfaceplant.plant_lifetime.value,
        model.economics.timestepsperyear.value,
    )
    if short_pt[0] == 0:
        hce[f"Thermal Drawdown (%)|:8.4f"] = np.zeros_like(short_pt)
    else:
        hce[f"Thermal Drawdown (%)|:8.4f"] = short_pt / short_pt[0]

    hce[
        f"Geofluid Temperature ({model.wellbores.ProducedTemperature.CurrentUnits.value})|:8.2f"
    ] = shorten_array_to_annual(
        model.wellbores.ProducedTemperature.value,
        model.surfaceplant.plant_lifetime.value,
        model.economics.timestepsperyear.value,
    )
    hce[f"Pump Power ({model.wellbores.PumpingPower.CurrentUnits.value})|:8.4f"] = shorten_array_to_annual(
        model.wellbores.PumpingPower.value,
        model.surfaceplant.plant_lifetime.value,
        model.economics.timestepsperyear.value,
    )

    if model.surfaceplant.enduse_option.value == EndUseOptions.ELECTRICITY:
        hce[
            f"Net Power ({model.surfaceplant.NetElectricityProduced.CurrentUnits.value})|:8.4f"
        ] = shorten_array_to_annual(
            model.surfaceplant.NetElectricityProduced.value,
            model.surfaceplant.plant_lifetime.value,
            model.economics.timestepsperyear.value,
        )
        hce[f"First Law Efficiency (%)|:8.4f"] = shorten_array_to_annual(
            model.surfaceplant.FirstLawEfficiency.value * 100,
            model.surfaceplant.plant_lifetime.value,
            model.economics.timestepsperyear.value,
        )
    elif (
        model.surfaceplant.enduse_option.value == EndUseOptions.HEAT
        and model.surfaceplant.plant_type.value not in SPECIAL_HEAT_PLANT_TYPES
    ):
        hce[f"Net Heat ({model.surfaceplant.HeatProduced.CurrentUnits.value})|:8.4f"] = shorten_array_to_annual(
            model.surfaceplant.HeatProduced.value,
            model.surfaceplant.plant_lifetime.value,
            model.economics.timestepsperyear.value,
        )
    elif (
        model.surfaceplant.enduse_option.value == EndUseOptions.HEAT
        and model.surfaceplant.plant_type.value == PlantType.HEAT_PUMP
    ):
        hce[f"Net Heat ({model.surfaceplant.HeatProduced.CurrentUnits.value})|:8.4f"] = shorten_array_to_annual(
            model.surfaceplant.HeatProduced.value,
            model.surfaceplant.plant_lifetime.value,
            model.economics.timestepsperyear.value,
        )
        hce[
            f"Heat Pump Electricity Used ({model.surfaceplant.heat_pump_electricity_used.CurrentUnits.value}|:8.4f"
        ] = shorten_array_to_annual(
            model.surfaceplant.heat_pump_electricity_used.value,
            model.surfaceplant.plant_lifetime.value,
            model.economics.timestepsperyear.value,
        )
    elif (
        model.surfaceplant.enduse_option.value == EndUseOptions.HEAT
        and model.surfaceplant.plant_type.value == PlantType.DISTRICT_HEATING
    ):
        hce[
            f"Geothermal Heat Output ({model.surfaceplant.HeatProduced.CurrentUnits.value})|:8.4f"
        ] = shorten_array_to_annual(
            model.surfaceplant.HeatProduced.value,
            model.surfaceplant.plant_lifetime.value,
            model.economics.timestepsperyear.value,
        )
    elif (
        model.surfaceplant.enduse_option.value == EndUseOptions.HEAT
        and model.surfaceplant.plant_type.value == PlantType.ABSORPTION_CHILLER
    ):
        hce[f"Net Heat ({model.surfaceplant.HeatProduced.CurrentUnits.value})|:8.4f"] = shorten_array_to_annual(
            model.surfaceplant.HeatProduced.value,
            model.surfaceplant.plant_lifetime.value,
            model.economics.timestepsperyear.value,
        )
        hce[f"Net Cooling ({model.surfaceplant.HeatProduced.CurrentUnits.value})|:8.4f"] = shorten_array_to_annual(
            model.surfaceplant.cooling_produced.value,
            model.surfaceplant.plant_lifetime.value,
            model.economics.timestepsperyear.value,
        )
    elif _is_cogeneration_end_use(model.surfaceplant.enduse_option.value):
        hce[
            f"Net Power ({model.surfaceplant.NetElectricityProduced.CurrentUnits.value})|:8.4f"
        ] = shorten_array_to_annual(
            model.surfaceplant.NetElectricityProduced.value,
            model.surfaceplant.plant_lifetime.value,
            model.economics.timestepsperyear.value,
        )
        hce[f"Net Heat ({model.surfaceplant.HeatProduced.CurrentUnits.value})|:8.4f"] = shorten_array_to_annual(
            model.surfaceplant.HeatProduced.value,
            model.surfaceplant.plant_lifetime.value,
            model.economics.timestepsperyear.value,
        )
        hce[f"First Law Efficiency (%)|:8.4f"] = (
            shorten_array_to_annual(
                model.surfaceplant.FirstLawEfficiency.value,
                model.surfaceplant.plant_lifetime.value,
                model.economics.timestepsperyear.value,
            )
            * 100
        )

    return _filter_dispatch_year_rows(hce, model).reset_index()


def annual_production_profile_table(model: Model, dispatch_report: bool) -> pd.DataFrame:
    if dispatch_report:
        return pd.DataFrame()

    ahce = pd.DataFrame()
    ahce[f"Year|:2.0f"] = [i for i in range(1, (model.surfaceplant.plant_lifetime.value + 1))]

    if model.surfaceplant.enduse_option.value == EndUseOptions.ELECTRICITY:
        ahce[f"Electricity Provided ({model.surfaceplant.NetkWhProduced.CurrentUnits.value})|:8.1f"] = (
            model.surfaceplant.NetkWhProduced.value / 1E6
        )
    elif model.surfaceplant.plant_type.value == PlantType.ABSORPTION_CHILLER:
        ahce[f"Cooling Provided ({model.surfaceplant.cooling_kWh_Produced.CurrentUnits.value})|:8.1f"] = (
            shorten_array_to_annual(
                model.surfaceplant.cooling_kWh_Produced.value,
                model.surfaceplant.plant_lifetime.value,
                model.economics.timestepsperyear.value,
            )
            / 1E6
        )
    elif model.surfaceplant.plant_type.value == PlantType.HEAT_PUMP:
        ahce[f"Heating Provided ({model.surfaceplant.HeatkWhProduced.CurrentUnits.value})|:8.1f"] = (
            shorten_array_to_annual(
                model.surfaceplant.HeatkWhProduced.value,
                model.surfaceplant.plant_lifetime.value,
                model.economics.timestepsperyear.value,
            )
            / 1E6
        )
        ahce[f"Reservoir Heat Extracted ({model.surfaceplant.HeatkWhExtracted.CurrentUnits.value})|:8.1f"] = (
            shorten_array_to_annual(
                model.surfaceplant.HeatkWhExtracted.value,
                model.surfaceplant.plant_lifetime.value,
                model.economics.timestepsperyear.value,
            )
            / 1E6
        )
        ahce[
            f"Heat Pump Electricity Used ({model.surfaceplant.heat_pump_electricity_kwh_used.CurrentUnits.value})|:8.1f"
        ] = (
            shorten_array_to_annual(
                model.surfaceplant.heat_pump_electricity_kwh_used.value,
                model.surfaceplant.plant_lifetime.value,
                model.economics.timestepsperyear.value,
            )
            / 1E6
        )
    elif _is_cogeneration_end_use(model.surfaceplant.enduse_option.value):
        ahce[f"Heating Provided ({model.surfaceplant.HeatkWhProduced.CurrentUnits.value})|:8.1f"] = (
            shorten_array_to_annual(
                model.surfaceplant.HeatkWhProduced.value,
                model.surfaceplant.plant_lifetime.value,
                model.economics.timestepsperyear.value,
            )
            / 1E6
        )
        ahce[f"Electricity Provided ({model.surfaceplant.NetkWhProduced.CurrentUnits.value})|:8.1f"] = (
            shorten_array_to_annual(
                model.surfaceplant.NetkWhProduced.value,
                model.surfaceplant.plant_lifetime.value,
                model.economics.timestepsperyear.value,
            )
            / 1E6
        )
    elif model.surfaceplant.plant_type.value == PlantType.DISTRICT_HEATING:
        ahce[f"Electricity Provided ({model.surfaceplant.HeatkWhProduced.CurrentUnits.value})|:8.1f"] = (
            shorten_array_to_annual(
                model.surfaceplant.HeatkWhProduced.value,
                model.surfaceplant.plant_lifetime.value,
                model.economics.timestepsperyear.value,
            )
            / 1E6
        )
        ahce[f"Peaking Boiler Heat Provided ({model.surfaceplant.annual_ng_demand.CurrentUnits.value})|:8.1f"] = (
            shorten_array_to_annual(
                model.surfaceplant.annual_ng_demand.value,
                model.surfaceplant.plant_lifetime.value,
                model.economics.timestepsperyear.value,
            )
            / 1E3
        )
    elif model.surfaceplant.enduse_option.value == EndUseOptions.HEAT:
        ahce[f"Heating Provided ({model.surfaceplant.HeatkWhProduced.CurrentUnits.value})|:8.1f"] = (
            shorten_array_to_annual(
                model.surfaceplant.HeatkWhProduced.value,
                model.surfaceplant.plant_lifetime.value,
                model.economics.timestepsperyear.value,
            )
            / 1E6
        )

    ahce[f"Heat Extracted({model.surfaceplant.HeatkWhExtracted.CurrentUnits.value})|:8.2f"] = (
        model.surfaceplant.HeatkWhExtracted.value / 1E6
    )
    ahce[f"Reservoir Heat Content ({model.surfaceplant.RemainingReservoirHeatContent.CurrentUnits.value})|:8.2f"] = (
        model.surfaceplant.RemainingReservoirHeatContent.value
    )
    ahce[f"Percentage of Total Heat Mined (%)|:8.2f"] = (
        (model.reserv.InitialReservoirHeatContent.value - model.surfaceplant.RemainingReservoirHeatContent.value)
        * 100.0
        / model.reserv.InitialReservoirHeatContent.value
    )
    return _filter_dispatch_year_rows(ahce, model).reset_index()


def write_production_profile(model: Model, f: TextIOWrapper, dispatch_report: bool) -> None:
    """Write the point-in-time production profile table."""
    profile_year_count = _dispatch_report_year_count(
        model, default=model.surfaceplant.plant_lifetime.value
    )
    if profile_year_count is None or profile_year_count <= 0:
        return

    f.write(NL)
    f.write("                            ************************************************************\n")
    f.write("                            *  HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE  *\n")
    f.write("                            ************************************************************\n")

    if model.surfaceplant.enduse_option.value == EndUseOptions.ELECTRICITY:
        _write_electricity_production_profile(model, f, profile_year_count)
    elif (
        model.surfaceplant.enduse_option.value == EndUseOptions.HEAT
        and model.surfaceplant.plant_type.value not in SPECIAL_HEAT_PLANT_TYPES
    ):
        _write_direct_use_production_profile(model, f, profile_year_count)
    elif (
        model.surfaceplant.enduse_option.value == EndUseOptions.HEAT
        and model.surfaceplant.plant_type.value == PlantType.HEAT_PUMP
    ):
        _write_heat_pump_production_profile(model, f, profile_year_count)
    elif (
        model.surfaceplant.enduse_option.value == EndUseOptions.HEAT
        and model.surfaceplant.plant_type.value == PlantType.DISTRICT_HEATING
    ):
        _write_district_heating_production_profile(model, f, profile_year_count)
    elif (
        model.surfaceplant.enduse_option.value == EndUseOptions.HEAT
        and model.surfaceplant.plant_type.value == PlantType.ABSORPTION_CHILLER
    ):
        _write_absorption_chiller_production_profile(model, f, profile_year_count)
    elif _is_cogeneration_end_use(model.surfaceplant.enduse_option.value):
        _write_cogeneration_production_profile(model, f, profile_year_count)

    f.write(NL)
    f.write(NL)


def _write_electricity_production_profile(model: Model, f: TextIOWrapper, profile_year_count: int) -> None:
    f.write("  YEAR       THERMAL               GEOFLUID               PUMP               NET               FIRST LAW\n")
    f.write("             DRAWDOWN             TEMPERATURE             POWER             POWER              EFFICIENCY\n")
    f.write(
        "                                     ("
        + model.wellbores.ProducedTemperature.CurrentUnits.value
        + ")               ("
        + model.wellbores.PumpingPower.CurrentUnits.value
        + ")              ("
        + model.surfaceplant.NetElectricityProduced.CurrentUnits.value
        + ")                  (%)\n"
    )
    for i in range(0, profile_year_count):
        idx = i * model.economics.timestepsperyear.value
        f.write(
            "  {0:2.0f}         {1:8.4f}              {2:8.2f}             {3:8.4f}          "
            "{4:8.4f}              {5:8.4f}".format(
                i + 1,
                _thermal_drawdown_ratio(model, idx),
                model.wellbores.ProducedTemperature.value[idx],
                model.wellbores.PumpingPower.value[idx],
                model.surfaceplant.NetElectricityProduced.value[idx],
                model.surfaceplant.FirstLawEfficiency.value[idx] * 100,
            )
            + NL
        )


def _write_direct_use_production_profile(model: Model, f: TextIOWrapper, profile_year_count: int) -> None:
    f.write("  YEAR       THERMAL               GEOFLUID               PUMP               NET\n")
    f.write("             DRAWDOWN             TEMPERATURE             POWER              HEAT\n")
    f.write("                                   (deg C)                (MW)               (MW)\n")
    for i in range(0, profile_year_count):
        idx = i * model.economics.timestepsperyear.value
        f.write(
            "  {0:2.0f}         {1:8.4f}              {2:8.2f}             {3:8.4f}          "
            "{4:8.4f}".format(
                i,
                _thermal_drawdown_ratio(model, idx),
                model.wellbores.ProducedTemperature.value[idx],
                model.wellbores.PumpingPower.value[idx],
                model.surfaceplant.HeatProduced.value[idx],
            )
            + NL
        )


def _write_heat_pump_production_profile(model: Model, f: TextIOWrapper, profile_year_count: int) -> None:
    f.write("  YEAR         THERMAL              GEOFLUID               PUMP               NET             HEAT PUMP\n")
    f.write("               DRAWDOWN            TEMPERATURE             POWER              HEAT         ELECTRICITY USE\n")
    f.write("                                    (deg C)                (MWe)              (MWt)             (MWe)\n")
    for i in range(0, profile_year_count):
        idx = i * model.economics.timestepsperyear.value
        f.write(
            "  {0:2.0f}          {1:8.4f}             {2:8.2f}              {3:8.4f}           "
            "{4:8.4f}          {5:8.4f}".format(
                i,
                _thermal_drawdown_ratio(model, idx),
                model.wellbores.ProducedTemperature.value[idx],
                model.wellbores.PumpingPower.value[idx],
                model.surfaceplant.HeatProduced.value[idx],
                model.surfaceplant.heat_pump_electricity_used.value[idx],
            )
            + NL
        )


def _write_district_heating_production_profile(model: Model, f: TextIOWrapper, profile_year_count: int) -> None:
    f.write("  YEAR         THERMAL              GEOFLUID               PUMP              GEOTHERMAL\n")
    f.write("               DRAWDOWN            TEMPERATURE             POWER            HEAT OUTPUT\n")
    f.write("                                    (deg C)                (MWe)               (MWt)\n")
    for i in range(0, profile_year_count):
        idx = i * model.economics.timestepsperyear.value
        f.write(
            "  {0:2.0f}          {1:8.4f}             {2:8.2f}              {3:8.4f}            "
            "{4:8.4f}".format(
                i,
                _thermal_drawdown_ratio(model, idx),
                model.wellbores.ProducedTemperature.value[idx],
                model.wellbores.PumpingPower.value[idx],
                model.surfaceplant.HeatProduced.value[idx],
            )
            + NL
        )


def _write_absorption_chiller_production_profile(model: Model, f: TextIOWrapper, profile_year_count: int) -> None:
    f.write("  YEAR         THERMAL              GEOFLUID               PUMP               NET              NET\n")
    f.write("               DRAWDOWN            TEMPERATURE             POWER              HEAT             COOLING\n")
    f.write("                                    (deg C)                (MWe)              (MWt)            (MWt)\n")
    for i in range(0, profile_year_count):
        idx = i * model.economics.timestepsperyear.value
        f.write(
            "  {0:2.0f}          {1:8.4f}             {2:8.2f}              {3:8.4f}           "
            "{4:8.4f}         {5:8.4f}".format(
                i,
                _thermal_drawdown_ratio(model, idx),
                model.wellbores.ProducedTemperature.value[idx],
                model.wellbores.PumpingPower.value[idx],
                model.surfaceplant.HeatProduced.value[idx],
                model.surfaceplant.cooling_produced.value[idx],
            )
            + NL
        )


def _write_cogeneration_production_profile(model: Model, f: TextIOWrapper, profile_year_count: int) -> None:
    f.write("  YEAR     THERMAL             GEOFLUID             PUMP             NET              NET             FIRST LAW\n")
    f.write("           DRAWDOWN           TEMPERATURE           POWER           POWER             HEAT            EFFICIENCY\n")
    f.write("                                (deg C)             (MW)            (MW)              (MW)               (%)\n")
    for i in range(0, profile_year_count):
        idx = i * model.economics.timestepsperyear.value
        f.write(
            "  {0:2.0f}       {1:8.4f}            {2:8.2f}           {3:8.4f}        "
            "{4:8.4f}            {5:8.4f}             {6:8.4f}".format(
                i,
                _thermal_drawdown_ratio(model, idx),
                model.wellbores.ProducedTemperature.value[idx],
                model.wellbores.PumpingPower.value[idx],
                model.surfaceplant.NetElectricityProduced.value[idx],
                model.surfaceplant.HeatProduced.value[idx],
                model.surfaceplant.FirstLawEfficiency.value[idx] * 100,
            )
            + NL
        )


def write_annual_production_profile(model: Model, f: TextIOWrapper, dispatch_report: bool) -> None:
    """Write the annual production profile table."""
    profile_year_count = _dispatch_report_year_count(
        model, default=model.surfaceplant.plant_lifetime.value
    )
    if profile_year_count is None or profile_year_count <= 0:
        return

    f.write("                              *******************************************************************\n")
    f.write("                              *  ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE  *\n")
    f.write("                              *******************************************************************\n")

    if model.surfaceplant.enduse_option.value == EndUseOptions.ELECTRICITY:
        _write_electricity_annual_production_profile(model, f, profile_year_count)
    elif model.surfaceplant.plant_type.value == PlantType.ABSORPTION_CHILLER:
        _write_absorption_chiller_annual_production_profile(model, f, profile_year_count)
    elif model.surfaceplant.plant_type.value == PlantType.HEAT_PUMP:
        _write_heat_pump_annual_production_profile(model, f, profile_year_count)
    elif _is_cogeneration_end_use(model.surfaceplant.enduse_option.value):
        _write_cogeneration_annual_production_profile(model, f, profile_year_count)
    elif model.surfaceplant.plant_type.value == PlantType.DISTRICT_HEATING:
        _write_district_heating_annual_production_profile(model, f, profile_year_count)
    elif model.surfaceplant.enduse_option.value == EndUseOptions.HEAT:
        _write_direct_use_annual_production_profile(model, f, profile_year_count)


def _percent_total_heat_mined(model: Model, year_index: int) -> float:
    return (
        (model.reserv.InitialReservoirHeatContent.value - model.surfaceplant.RemainingReservoirHeatContent.value[year_index])
        * 100
        / model.reserv.InitialReservoirHeatContent.value
    )


def _write_electricity_annual_production_profile(model: Model, f: TextIOWrapper, profile_year_count: int) -> None:
    f.write("  YEAR             ELECTRICITY                   HEAT                RESERVOIR            PERCENTAGE OF\n")
    f.write("                    PROVIDED                   EXTRACTED            HEAT CONTENT        TOTAL HEAT MINED\n")
    f.write("                   (GWh/year)                  (GWh/year)            (10^15 J)                 (%)\n")
    for i in range(0, profile_year_count):
        f.write(
            "  {0:2.0f}              {1:8.1f}                    {2:8.1f}              {3:8.2f}               "
            "{4:8.2f}".format(
                i + 1,
                model.surfaceplant.NetkWhProduced.value[i] / 1E6,
                model.surfaceplant.HeatkWhExtracted.value[i] / 1E6,
                model.surfaceplant.RemainingReservoirHeatContent.value[i],
                _percent_total_heat_mined(model, i),
            )
            + NL
        )


def _write_absorption_chiller_annual_production_profile(model: Model, f: TextIOWrapper, profile_year_count: int) -> None:
    f.write("  YEAR              COOLING                 HEAT                RESERVOIR            PERCENTAGE OF\n")
    f.write("                    PROVIDED              EXTRACTED            HEAT CONTENT        TOTAL HEAT MINED\n")
    f.write("                   (GWh/year)             (GWh/year)            (10^15 J)                 (%)\n")
    for i in range(0, profile_year_count):
        f.write(
            "  {0:2.0f}              {1:8.1f}               {2:8.1f}              {3:8.2f}               "
            "{4:8.2f}".format(
                i + 1,
                model.surfaceplant.cooling_kWh_Produced.value[i] / 1E6,
                model.surfaceplant.HeatkWhExtracted.value[i] / 1E6,
                model.surfaceplant.RemainingReservoirHeatContent.value[i],
                _percent_total_heat_mined(model, i),
            )
            + NL
        )


def _write_heat_pump_annual_production_profile(model: Model, f: TextIOWrapper, profile_year_count: int) -> None:
    f.write("  YEAR              HEATING             RESERVOIR HEAT          HEAT PUMP          RESERVOIR           PERCENTAGE OF\n")
    f.write("                    PROVIDED              EXTRACTED          ELECTRICITY USE      HEAT CONTENT        TOTAL HEAT MINED\n")
    f.write("                   (GWh/year)             (GWh/year)           (GWh/year)           (10^15 J)                (%)\n")
    for i in range(0, profile_year_count):
        f.write(
            "  {0:2.0f}              {1:8.1f}               {2:8.1f}             {3:8.2f}             "
            "{4:8.2f}              {5:8.2f}".format(
                i + 1,
                model.surfaceplant.HeatkWhProduced.value[i] / 1E6,
                model.surfaceplant.HeatkWhExtracted.value[i] / 1E6,
                model.surfaceplant.heat_pump_electricity_kwh_used.value[i] / 1E6,
                model.surfaceplant.RemainingReservoirHeatContent.value[i],
                _percent_total_heat_mined(model, i),
            )
            + NL
        )


def _write_cogeneration_annual_production_profile(model: Model, f: TextIOWrapper, profile_year_count: int) -> None:
    f.write("  YEAR             HEAT                 ELECTRICITY                HEAT              RESERVOIR        PERCENTAGE OF\n")
    f.write("                  PROVIDED               PROVIDED                EXTRACTED          HEAT CONTENT    TOTAL HEAT MINED\n")
    f.write("                 (GWh/year)             (GWh/year)               (GWh/year)          (10^15 J)           (%)\n")
    for i in range(0, profile_year_count):
        f.write(
            "  {0:2.0f}            {1:8.1f}               {2:8.1f}                  {3:8.2f}            "
            "{4:8.2f}             {5:8.2f}".format(
                i + 1,
                model.surfaceplant.HeatkWhProduced.value[i] / 1E6,
                model.surfaceplant.NetkWhProduced.value[i] / 1E6,
                model.surfaceplant.HeatkWhExtracted.value[i] / 1E6,
                model.surfaceplant.RemainingReservoirHeatContent.value[i],
                _percent_total_heat_mined(model, i),
            )
            + NL
        )


def _write_district_heating_annual_production_profile(model: Model, f: TextIOWrapper, profile_year_count: int) -> None:
    f.write("  YEAR           GEOTHERMAL          PEAKING BOILER       RESERVOIR HEAT          RESERVOIR         PERCENTAGE OF\n")
    f.write("              HEATING PROVIDED      HEATING PROVIDED        EXTRACTED            HEAT CONTENT     TOTAL HEAT MINED\n")
    f.write("                 (GWh/year)            (GWh/year)           (GWh/year)            (10^15 J)              (%)\n")
    for i in range(0, profile_year_count):
        f.write(
            "  {0:2.0f}            {1:8.1f}              {2:8.1f}              {3:8.2f}             "
            "{4:8.2f}            {5:8.2f}".format(
                i + 1,
                model.surfaceplant.HeatkWhProduced.value[i] / 1E6,
                model.surfaceplant.annual_ng_demand.value[i] / 1E3,
                model.surfaceplant.HeatkWhExtracted.value[i] / 1E6,
                model.surfaceplant.RemainingReservoirHeatContent.value[i],
                _percent_total_heat_mined(model, i),
            )
            + NL
        )


def _write_direct_use_annual_production_profile(model: Model, f: TextIOWrapper, profile_year_count: int) -> None:
    f.write("  YEAR               HEAT                       HEAT                RESERVOIR            PERCENTAGE OF\n")
    f.write("                    PROVIDED                   EXTRACTED            HEAT CONTENT        TOTAL HEAT MINED\n")
    f.write("                   (GWh/year)                  (GWh/year)            (10^15 J)                 (%)\n")
    for i in range(0, profile_year_count):
        f.write(
            "  {0:2.0f}              {1:8.1f}                    {2:8.1f}              {3:8.2f}               "
            "{4:8.2f}".format(
                i + 1,
                model.surfaceplant.HeatkWhProduced.value[i] / 1E6,
                model.surfaceplant.HeatkWhExtracted.value[i] / 1E6,
                model.surfaceplant.RemainingReservoirHeatContent.value[i],
                _percent_total_heat_mined(model, i),
            )
            + NL
        )
