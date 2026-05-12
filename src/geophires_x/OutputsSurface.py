from __future__ import annotations

import re
from io import StringIO
from io import TextIOWrapper
from typing import TYPE_CHECKING

import numpy as np

from geophires_x.OptionList import EndUseOptions, PlantType
from geophires_x.OutputsReport import field_label
from geophires_x.OutputsUtils import OutputTableItem

if TYPE_CHECKING:
    from geophires_x.Model import Model

NL = "\n"

SURFACE_HEAT_RESULT_OPTIONS = (
    EndUseOptions.HEAT,
    EndUseOptions.COGENERATION_TOPPING_EXTRA_HEAT,
    EndUseOptions.COGENERATION_TOPPING_EXTRA_ELECTRICITY,
    EndUseOptions.COGENERATION_BOTTOMING_EXTRA_HEAT,
    EndUseOptions.COGENERATION_BOTTOMING_EXTRA_ELECTRICITY,
    EndUseOptions.COGENERATION_PARALLEL_EXTRA_HEAT,
    EndUseOptions.COGENERATION_PARALLEL_EXTRA_ELECTRICITY,
    PlantType.ABSORPTION_CHILLER,
    PlantType.HEAT_PUMP,
    PlantType.DISTRICT_HEATING,
)


def _surface_output_option(option: object) -> object:
    if isinstance(option, (EndUseOptions, PlantType)):
        return option

    for option_type in (EndUseOptions, PlantType):
        if isinstance(option, int):
            try:
                return option_type.from_int(option)
            except ValueError:
                continue

        if isinstance(option, str):
            try:
                return option_type.from_input_string(option.strip())
            except ValueError:
                pass

            for member in option_type:
                if option in (member.value, member.name):
                    return member

    return option


def has_electricity_component(enduse_option: EndUseOptions | int | str) -> bool:
    enduse_option = _surface_output_option(enduse_option)
    return enduse_option in (
        EndUseOptions.ELECTRICITY,
        EndUseOptions.COGENERATION_TOPPING_EXTRA_HEAT,
        EndUseOptions.COGENERATION_TOPPING_EXTRA_ELECTRICITY,
        EndUseOptions.COGENERATION_BOTTOMING_EXTRA_HEAT,
        EndUseOptions.COGENERATION_BOTTOMING_EXTRA_ELECTRICITY,
        EndUseOptions.COGENERATION_PARALLEL_EXTRA_HEAT,
        EndUseOptions.COGENERATION_PARALLEL_EXTRA_ELECTRICITY,
    )


def writes_surface_heat_results(enduse_option: EndUseOptions | PlantType | int | str) -> bool:
    return _surface_output_option(enduse_option) in SURFACE_HEAT_RESULT_OPTIONS


def surface_equipment_simulation_result_output_items(
    model: Model,
    dispatch_report: bool,
) -> list[OutputTableItem]:
    section_text = StringIO()
    write_surface_equipment_simulation_results(model, section_text, dispatch_report)
    return _surface_equipment_simulation_result_output_items_from_text(section_text.getvalue())


_SURFACE_VALUE_PATTERN = re.compile(
    r"^(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?|NaN|N/A)(?:\s+(?P<units>.+))?$"
)


def _surface_equipment_simulation_result_output_items_from_text(section_text: str) -> list[OutputTableItem]:
    items = []
    for line in section_text.splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line == "***SURFACE EQUIPMENT SIMULATION RESULTS***":
            continue

        if ":" not in stripped_line:
            items.append(OutputTableItem(stripped_line))
            continue

        parameter, raw_value = stripped_line.split(":", 1)
        value = raw_value.strip()
        units = ""
        value_match = _SURFACE_VALUE_PATTERN.match(value)
        if value_match is not None:
            value = value_match.group("value")
            units = value_match.group("units") or ""

        items.append(OutputTableItem(parameter.strip(), value, units))

    return items


def write_surface_equipment_simulation_results(
    model: Model,
    f: TextIOWrapper,
    dispatch_report: bool,
) -> None:
    f.write(NL)
    f.write(NL)
    f.write("                           ***SURFACE EQUIPMENT SIMULATION RESULTS***\n")
    f.write(NL)
    if not dispatch_report and has_electricity_component(model.surfaceplant.enduse_option.value):
        f.write(f"      Initial geofluid availability:                    {model.surfaceplant.Availability.value[0]:10.2f} " + model.surfaceplant.Availability.PreferredUnits.value + NL)
        f.write(f"      Maximum Total Electricity Generation:             {np.max(model.surfaceplant.ElectricityProduced.value):10.2f} " + model.surfaceplant.ElectricityProduced.PreferredUnits.value + NL)
        f.write(f"      Average Total Electricity Generation:             {np.average(model.surfaceplant.ElectricityProduced.value):10.2f} " + model.surfaceplant.ElectricityProduced.PreferredUnits.value + NL)
        f.write(f"      Minimum Total Electricity Generation:             {np.min(model.surfaceplant.ElectricityProduced.value):10.2f} " + model.surfaceplant.ElectricityProduced.PreferredUnits.value + NL)
        f.write(f"      Initial Total Electricity Generation:             {model.surfaceplant.ElectricityProduced.value[0]:10.2f} " + model.surfaceplant.ElectricityProduced.PreferredUnits.value + NL)
        f.write(f"      Maximum Net Electricity Generation:               {np.max(model.surfaceplant.NetElectricityProduced.value):10.2f} " + model.surfaceplant.NetElectricityProduced.PreferredUnits.value + NL)
        f.write(f"      Average Net Electricity Generation:               {np.average(model.surfaceplant.NetElectricityProduced.value):10.2f} " + model.surfaceplant.NetElectricityProduced.PreferredUnits.value + NL)
        f.write(f"      Minimum Net Electricity Generation:               {np.min(model.surfaceplant.NetElectricityProduced.value):10.2f} " + model.surfaceplant.NetElectricityProduced.PreferredUnits.value + NL)
        f.write(f"      Initial Net Electricity Generation:               {model.surfaceplant.NetElectricityProduced.value[0]:10.2f} " + model.surfaceplant.NetElectricityProduced.PreferredUnits.value + NL)
        f.write(f"      Average Annual Total Electricity Generation:      {np.average(model.surfaceplant.TotalkWhProduced.value / 1E6):10.2f} GWh" + NL)
        f.write(f"      Average Annual Net Electricity Generation:        {np.average(model.surfaceplant.NetkWhProduced.value / 1E6):10.2f} GWh" + NL)

        if model.wellbores.PumpingPower.value[0] > 0.0:
            ipp_nip = model.wellbores.PumpingPower.value[0] / model.surfaceplant.NetElectricityProduced.value[0]
            f.write(f"      Initial pumping power/net installed power:        {(ipp_nip * 100):10.2f} %\n")

    if not dispatch_report and writes_surface_heat_results(model.surfaceplant.enduse_option.value):
        f.write(f"      Maximum Net Heat Production:                      {np.max(model.surfaceplant.HeatProduced.value):10.2f} " + model.surfaceplant.HeatProduced.PreferredUnits.value + NL)
        f.write(f"      Average Net Heat Production:                      {np.average(model.surfaceplant.HeatProduced.value):10.2f} " + model.surfaceplant.HeatProduced.PreferredUnits.value + NL)
        f.write(f"      Minimum Net Heat Production:                      {np.min(model.surfaceplant.HeatProduced.value):10.2f} " + model.surfaceplant.HeatProduced.PreferredUnits.value + NL)
        f.write(f"      Initial Net Heat Production:                      {model.surfaceplant.HeatProduced.value[0]:10.2f} " + model.surfaceplant.HeatProduced.PreferredUnits.value + NL)
        f.write(f"      Average Annual Heat Production:                   {np.average(model.surfaceplant.HeatkWhProduced.value / 1E6):10.2f} GWh" + NL)

    if not dispatch_report and model.surfaceplant.plant_type.value == PlantType.HEAT_PUMP:
        f.write(f"      Average Annual Heat Pump Electricity Use:         {np.average(model.surfaceplant.heat_pump_electricity_kwh_used.value / 1E6):10.2f} " + "GWh/year" + NL)
    if not dispatch_report and model.surfaceplant.plant_type.value == PlantType.ABSORPTION_CHILLER:
        f.write(f"      Maximum Cooling Production:                       {np.max(model.surfaceplant.cooling_produced.value):10.2f} " + model.surfaceplant.cooling_produced.PreferredUnits.value + NL)
        f.write(f"      Average Cooling Production:                       {np.average(model.surfaceplant.cooling_produced.value):10.2f} " + model.surfaceplant.cooling_produced.PreferredUnits.value + NL)
        f.write(f"      Minimum Cooling Production:                       {np.min(model.surfaceplant.cooling_produced.value):10.2f} " + model.surfaceplant.cooling_produced.PreferredUnits.value + NL)
        f.write(f"      Initial Cooling Production:                       {model.surfaceplant.cooling_produced.value[0]:10.2f} " + model.surfaceplant.cooling_produced.PreferredUnits.value + NL)
        f.write(f"      Average Annual Cooling Production:                {np.average(model.surfaceplant.cooling_kWh_Produced.value / 1E6):10.2f} " + "GWh/year" + NL)

    if not dispatch_report and model.surfaceplant.plant_type.value == PlantType.DISTRICT_HEATING:
        f.write(f"      Annual District Heating Demand:                   {model.surfaceplant.annual_heating_demand.value:10.2f} " + model.surfaceplant.annual_heating_demand.PreferredUnits.value + NL)
        f.write(f"      Maximum Daily District Heating Demand:            {np.max(model.surfaceplant.daily_heating_demand.value):10.2f} " + model.surfaceplant.daily_heating_demand.PreferredUnits.value + NL)
        f.write(f"      Average Daily District Heating Demand:            {np.average(model.surfaceplant.daily_heating_demand.value):10.2f} " + model.surfaceplant.daily_heating_demand.PreferredUnits.value + NL)
        f.write(f"      Minimum Daily District Heating Demand:            {np.min(model.surfaceplant.daily_heating_demand.value):10.2f} " + model.surfaceplant.daily_heating_demand.PreferredUnits.value + NL)
        f.write(f"      Maximum Geothermal Heating Production:            {np.max(model.surfaceplant.dh_geothermal_heating.value):10.2f} " + model.surfaceplant.dh_geothermal_heating.PreferredUnits.value + NL)
        f.write(f"      Average Geothermal Heating Production:            {np.average(model.surfaceplant.dh_geothermal_heating.value):10.2f} " + model.surfaceplant.dh_geothermal_heating.PreferredUnits.value + NL)
        f.write(f"      Minimum Geothermal Heating Production:            {np.min(model.surfaceplant.dh_geothermal_heating.value):10.2f} " + model.surfaceplant.dh_geothermal_heating.PreferredUnits.value + NL)
        f.write(f"      Maximum Peaking Boiler Heat Production:           {np.max(model.surfaceplant.dh_natural_gas_heating.value):10.2f} " + model.surfaceplant.dh_natural_gas_heating.PreferredUnits.value + NL)
        f.write(f"      Average Peaking Boiler Heat Production:           {np.average(model.surfaceplant.dh_natural_gas_heating.value):10.2f} " + model.surfaceplant.dh_natural_gas_heating.PreferredUnits.value + NL)
        f.write(f"      Minimum Peaking Boiler Heat Production:           {np.min(model.surfaceplant.dh_natural_gas_heating.value):10.2f} " + model.surfaceplant.dh_natural_gas_heating.PreferredUnits.value + NL)

    f.write(f"      Average Pumping Power:                            {np.average(model.wellbores.PumpingPower.value):10.2f} {model.wellbores.PumpingPower.CurrentUnits.value}{NL}")

    if not dispatch_report and model.surfaceplant.heat_to_power_conversion_efficiency.value is not None:
        hpce = model.surfaceplant.heat_to_power_conversion_efficiency
        f.write(f"      {field_label(hpce.Name, 50)}"
                f"{hpce.value:10.2f} {model.surfaceplant.heat_to_power_conversion_efficiency.CurrentUnits.value}\n")
