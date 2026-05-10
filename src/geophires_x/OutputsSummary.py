from __future__ import annotations

import re
from io import StringIO
from io import TextIOWrapper
from typing import TYPE_CHECKING

import numpy as np

from geophires_x.OptionList import EndUseOptions, PlantType
from geophires_x.OutputsReport import field_label
from geophires_x.OutputsSurface import has_electricity_component
from geophires_x.OutputsUtils import OutputTableItem

if TYPE_CHECKING:
    from geophires_x.Model import Model

NL = "\n"

VERTICAL_WELL_DEPTH_OUTPUT_NAME = "Well depth"

COGENERATION_END_USE_OPTIONS = (
    EndUseOptions.COGENERATION_TOPPING_EXTRA_HEAT,
    EndUseOptions.COGENERATION_TOPPING_EXTRA_ELECTRICITY,
    EndUseOptions.COGENERATION_BOTTOMING_EXTRA_HEAT,
    EndUseOptions.COGENERATION_BOTTOMING_EXTRA_ELECTRICITY,
    EndUseOptions.COGENERATION_PARALLEL_EXTRA_HEAT,
    EndUseOptions.COGENERATION_PARALLEL_EXTRA_ELECTRICITY,
)

SPECIAL_HEAT_PLANT_TYPES = (
    PlantType.ABSORPTION_CHILLER,
    PlantType.HEAT_PUMP,
    PlantType.DISTRICT_HEATING,
)


def is_cogeneration_end_use(enduse_option: EndUseOptions) -> bool:
    return enduse_option in COGENERATION_END_USE_OPTIONS


def summary_output_items(
    model: Model,
    dispatch_report: bool,
    is_sam_econ_model: bool,
) -> list[OutputTableItem]:
    summary_text = StringIO()
    write_summary_of_results(model, summary_text, dispatch_report, is_sam_econ_model)
    return _summary_output_items_from_text(summary_text.getvalue())


_SUMMARY_VALUE_PATTERN = re.compile(
    r"^(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?|N/A)(?:\s+(?P<units>.+))?$"
)


def _summary_output_items_from_text(summary_text: str) -> list[OutputTableItem]:
    items = []
    for line in summary_text.splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line == "***SUMMARY OF RESULTS***" or ":" not in stripped_line:
            continue

        parameter, raw_value = stripped_line.split(":", 1)
        value = raw_value.strip()
        units = ""
        value_match = _SUMMARY_VALUE_PATTERN.match(value)
        if value_match is not None:
            value = value_match.group("value")
            units = value_match.group("units") or ""

        items.append(OutputTableItem(parameter.strip(), value, units))

    return items


def write_summary_of_results(
    model: Model,
    f: TextIOWrapper,
    dispatch_report: bool,
    is_sam_econ_model: bool,
) -> None:
    econ = model.economics

    f.write(NL)
    f.write("                           ***SUMMARY OF RESULTS***\n")
    f.write(NL)
    f.write(f"      {model.surfaceplant.enduse_option_output.display_name}: "
            f"{model.surfaceplant.enduse_option.value.value}\n")
    if not dispatch_report and model.surfaceplant.plant_type.value in SPECIAL_HEAT_PLANT_TYPES:
        f.write("      Surface Application: " + str(model.surfaceplant.plant_type.value.value) + NL)
    if not dispatch_report and has_electricity_component(model.surfaceplant.enduse_option.value):
        f.write(f"      Average Net Electricity Production:               {np.average(model.surfaceplant.NetElectricityProduced.value):10.2f} " + model.surfaceplant.NetElectricityProduced.CurrentUnits.value + NL)
    if not dispatch_report and model.surfaceplant.enduse_option.value is not EndUseOptions.ELECTRICITY:
        # there is a direct-use component
        f.write(f"      Average Direct-Use Heat Production:               {np.average(model.surfaceplant.HeatProduced.value):10.2f} " + model.surfaceplant.HeatProduced.CurrentUnits.value + NL)
    if not dispatch_report and model.surfaceplant.plant_type.value == PlantType.DISTRICT_HEATING:
        f.write(f"      Annual District Heating Demand:                   {np.average(model.surfaceplant.annual_heating_demand.value):10.2f} " + model.surfaceplant.annual_heating_demand.CurrentUnits.value + NL)
        f.write(f"      Average Annual Geothermal Heat Production:        {sum(model.surfaceplant.dh_geothermal_heating.value * 24) / model.surfaceplant.plant_lifetime.value / 1e3:10.2f} " + model.surfaceplant.annual_heating_demand.CurrentUnits.value + NL)
        f.write(f"      Average Annual Peaking Fuel Heat Production:      {sum(model.surfaceplant.dh_natural_gas_heating.value * 24) / model.surfaceplant.plant_lifetime.value / 1e3:10.2f} " + model.surfaceplant.annual_heating_demand.CurrentUnits.value + NL)
    if not dispatch_report and model.surfaceplant.plant_type.value == PlantType.ABSORPTION_CHILLER:
        f.write(f"      Average Cooling Production:                       {np.average(model.surfaceplant.cooling_produced.value):10.2f} " + model.surfaceplant.cooling_produced.CurrentUnits.value + NL)

    if not dispatch_report and model.surfaceplant.enduse_option.value in [EndUseOptions.ELECTRICITY]:
        f.write(f"      {model.economics.LCOE.display_name}:                      {model.economics.LCOE.value:10.2f} {model.economics.LCOE.CurrentUnits.value}\n")
        if model.economics.DoXLCOCalculations.value:
            # XLCO and VALCO live in the same summary block as the baseline LCO output so
            # downstream parsers can treat them as parallel commodity summary metrics.
            f.write(f"      {model.economics.XLCOE_Market.display_name}: {model.economics.XLCOE_Market.value:10.2f} {model.economics.XLCOE_Market.CurrentUnits.value}\n")
            f.write(f"      {model.economics.XLCOE_MarketSocial.display_name}: {model.economics.XLCOE_MarketSocial.value:10.2f} {model.economics.XLCOE_MarketSocial.CurrentUnits.value}\n")
        if model.economics.DoVALCOCalculations.value:
            f.write(f"      {model.economics.VALCOE.display_name}: {model.economics.VALCOE.value:10.2f} {model.economics.VALCOE.CurrentUnits.value}\n")
            f.write(f"      {model.economics.VALCOE_EnergyAdjustment.display_name}: {model.economics.VALCOE_EnergyAdjustment.value:10.2f} {model.economics.VALCOE_EnergyAdjustment.CurrentUnits.value}\n")
            f.write(f"      {model.economics.VALCOE_CapacityAdjustment.display_name}: {model.economics.VALCOE_CapacityAdjustment.value:10.2f} {model.economics.VALCOE_CapacityAdjustment.CurrentUnits.value}\n")
            f.write(f"      {model.economics.VALCOE_FlexibilityAdjustment.display_name}: {model.economics.VALCOE_FlexibilityAdjustment.value:10.2f} {model.economics.VALCOE_FlexibilityAdjustment.CurrentUnits.value}\n")
    elif not dispatch_report and model.surfaceplant.enduse_option.value in [EndUseOptions.HEAT] and \
            model.surfaceplant.plant_type.value not in [PlantType.ABSORPTION_CHILLER]:
        f.write(f"      {model.economics.LCOH.display_name}:            {model.economics.LCOH.value:10.2f} {model.economics.LCOH.CurrentUnits.value}\n")
        if model.economics.DoXLCOCalculations.value:
            f.write(f"      {model.economics.XLCOH_Market.display_name}: {model.economics.XLCOH_Market.value:10.2f} {model.economics.XLCOH_Market.CurrentUnits.value}\n")
            f.write(f"      {model.economics.XLCOH_MarketSocial.display_name}: {model.economics.XLCOH_MarketSocial.value:10.2f} {model.economics.XLCOH_MarketSocial.CurrentUnits.value}\n")
        if model.economics.DoVALCOCalculations.value:
            f.write(f"      {model.economics.VALCOH.display_name}: {model.economics.VALCOH.value:10.2f} {model.economics.VALCOH.CurrentUnits.value}\n")
            f.write(f"      {model.economics.VALCOH_EnergyAdjustment.display_name}: {model.economics.VALCOH_EnergyAdjustment.value:10.2f} {model.economics.VALCOH_EnergyAdjustment.CurrentUnits.value}\n")
            f.write(f"      {model.economics.VALCOH_CapacityAdjustment.display_name}: {model.economics.VALCOH_CapacityAdjustment.value:10.2f} {model.economics.VALCOH_CapacityAdjustment.CurrentUnits.value}\n")
            f.write(f"      {model.economics.VALCOH_FlexibilityAdjustment.display_name}: {model.economics.VALCOH_FlexibilityAdjustment.value:10.2f} {model.economics.VALCOH_FlexibilityAdjustment.CurrentUnits.value}\n")
    elif not dispatch_report and model.surfaceplant.enduse_option.value in [EndUseOptions.HEAT] and model.surfaceplant.plant_type.value == PlantType.ABSORPTION_CHILLER:
        f.write(f"      {model.economics.LCOC.display_name}:         {model.economics.LCOC.value:10.2f} {model.economics.LCOC.CurrentUnits.value}\n")
        if model.economics.DoXLCOCalculations.value:
            f.write(f"      {model.economics.XLCOC_Market.display_name}: {model.economics.XLCOC_Market.value:10.2f} {model.economics.XLCOC_Market.CurrentUnits.value}\n")
            f.write(f"      {model.economics.XLCOC_MarketSocial.display_name}: {model.economics.XLCOC_MarketSocial.value:10.2f} {model.economics.XLCOC_MarketSocial.CurrentUnits.value}\n")
        if model.economics.DoVALCOCalculations.value:
            f.write(f"      {model.economics.VALCOC.display_name}: {model.economics.VALCOC.value:10.2f} {model.economics.VALCOC.CurrentUnits.value}\n")
            f.write(f"      {model.economics.VALCOC_EnergyAdjustment.display_name}: {model.economics.VALCOC_EnergyAdjustment.value:10.2f} {model.economics.VALCOC_EnergyAdjustment.CurrentUnits.value}\n")
            f.write(f"      {model.economics.VALCOC_CapacityAdjustment.display_name}: {model.economics.VALCOC_CapacityAdjustment.value:10.2f} {model.economics.VALCOC_CapacityAdjustment.CurrentUnits.value}\n")
            f.write(f"      {model.economics.VALCOC_FlexibilityAdjustment.display_name}: {model.economics.VALCOC_FlexibilityAdjustment.value:10.2f} {model.economics.VALCOC_FlexibilityAdjustment.CurrentUnits.value}\n")
    elif not dispatch_report and is_cogeneration_end_use(model.surfaceplant.enduse_option.value):
        # Cogeneration writes both electricity and heat competitiveness outputs because
        # XLCO/VALCO are tracked independently per active commodity.
        f.write(f"      {model.economics.LCOE.display_name}:                      {model.economics.LCOE.value:10.2f} {model.economics.LCOE.CurrentUnits.value}\n")
        if model.economics.DoXLCOCalculations.value:
            f.write(f"      {model.economics.XLCOE_Market.display_name}: {model.economics.XLCOE_Market.value:10.2f} {model.economics.XLCOE_Market.CurrentUnits.value}\n")
            f.write(f"      {model.economics.XLCOE_MarketSocial.display_name}: {model.economics.XLCOE_MarketSocial.value:10.2f} {model.economics.XLCOE_MarketSocial.CurrentUnits.value}\n")
        if model.economics.DoVALCOCalculations.value:
            f.write(f"      {model.economics.VALCOE.display_name}: {model.economics.VALCOE.value:10.2f} {model.economics.VALCOE.CurrentUnits.value}\n")
            f.write(f"      {model.economics.VALCOE_EnergyAdjustment.display_name}: {model.economics.VALCOE_EnergyAdjustment.value:10.2f} {model.economics.VALCOE_EnergyAdjustment.CurrentUnits.value}\n")
            f.write(f"      {model.economics.VALCOE_CapacityAdjustment.display_name}: {model.economics.VALCOE_CapacityAdjustment.value:10.2f} {model.economics.VALCOE_CapacityAdjustment.CurrentUnits.value}\n")
            f.write(f"      {model.economics.VALCOE_FlexibilityAdjustment.display_name}: {model.economics.VALCOE_FlexibilityAdjustment.value:10.2f} {model.economics.VALCOE_FlexibilityAdjustment.CurrentUnits.value}\n")
        f.write(f"      {model.economics.LCOH.display_name}:           {model.economics.LCOH.value:10.2f} {model.economics.LCOH.CurrentUnits.value}\n")
        if model.economics.DoXLCOCalculations.value:
            f.write(f"      {model.economics.XLCOH_Market.display_name}: {model.economics.XLCOH_Market.value:10.2f} {model.economics.XLCOH_Market.CurrentUnits.value}\n")
            f.write(f"      {model.economics.XLCOH_MarketSocial.display_name}: {model.economics.XLCOH_MarketSocial.value:10.2f} {model.economics.XLCOH_MarketSocial.CurrentUnits.value}\n")
        if model.economics.DoVALCOCalculations.value:
            f.write(f"      {model.economics.VALCOH.display_name}: {model.economics.VALCOH.value:10.2f} {model.economics.VALCOH.CurrentUnits.value}\n")
            f.write(f"      {model.economics.VALCOH_EnergyAdjustment.display_name}: {model.economics.VALCOH_EnergyAdjustment.value:10.2f} {model.economics.VALCOH_EnergyAdjustment.CurrentUnits.value}\n")
            f.write(f"      {model.economics.VALCOH_CapacityAdjustment.display_name}: {model.economics.VALCOH_CapacityAdjustment.value:10.2f} {model.economics.VALCOH_CapacityAdjustment.CurrentUnits.value}\n")
            f.write(f"      {model.economics.VALCOH_FlexibilityAdjustment.display_name}: {model.economics.VALCOH_FlexibilityAdjustment.value:10.2f} {model.economics.VALCOH_FlexibilityAdjustment.CurrentUnits.value}\n")

    if not dispatch_report and is_sam_econ_model:
        f.write(f"      {field_label(econ.capex_total.display_name, 50)}{econ.capex_total.value:10.2f} {econ.capex_total.CurrentUnits.value}\n")
        f.write(f"      {field_label(econ.capex_total_per_kw.display_name, 50)}{econ.capex_total_per_kw.value:10.0f} {econ.capex_total_per_kw.CurrentUnits.value}\n")

    f.write(f"      Number of production wells:                    {model.wellbores.nprod.value:10.0f}" + NL)
    f.write(f"      Number of injection wells:                     {model.wellbores.ninj.value:10.0f}" + NL)
    if dispatch_report:
        maximum_flowrate = model.dispatch_results.summary_metrics.get("observed_peak_flow_kg_per_sec", 0.0)
        f.write(f"      Maximum Flowrate per production well:            {maximum_flowrate:10.1f} kg/s" + NL)
    else:
        f.write(f"      Flowrate per production well:                    {model.wellbores.prodwellflowrate.value:10.1f} " + model.wellbores.prodwellflowrate.CurrentUnits.value + NL)
    f.write(f"      {field_label(VERTICAL_WELL_DEPTH_OUTPUT_NAME, 49)}{model.reserv.depth.value:10.1f} " + model.reserv.depth.CurrentUnits.value + NL)

    if model.reserv.numseg.value == 1:
        f.write(f"      Geothermal gradient:                             {model.reserv.gradient.value[0]:10.4g} " + model.reserv.gradient.CurrentUnits.value + NL)
    else:
        for i in range(1, model.reserv.numseg.value):
            f.write(f"      Segment {str(i):s}   Geothermal gradient:                    {model.reserv.gradient.value[i - 1]:10.4g} " + model.reserv.gradient.CurrentUnits.value + NL)
            f.write(f"      Segment {str(i):s}   Thickness:                         {round(model.reserv.layerthickness.value[i - 1], 10)} {model.reserv.layerthickness.CurrentUnits.value}\n")
        f.write(f"      Segment {str(i + 1):s}   Geothermal gradient:                    {model.reserv.gradient.value[i]:10.4g} " + model.reserv.gradient.CurrentUnits.value + NL)
    if not dispatch_report and model.economics.DoCarbonCalculations.value:
        f.write(f"      {model.economics.CarbonThatWouldHaveBeenProducedTotal.display_name}:"
                f"                       {model.economics.CarbonThatWouldHaveBeenProducedTotal.value:10.2f}"
                f" {model.economics.CarbonThatWouldHaveBeenProducedTotal.CurrentUnits.value}\n")
