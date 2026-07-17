from __future__ import annotations

import math
import re
from io import StringIO
from io import TextIOWrapper
from typing import TYPE_CHECKING

from geophires_x.Economics import Economics
from geophires_x.OptionList import EconomicModel, EndUseOptions, PlantType
from geophires_x.OutputsReport import field_label
from geophires_x.OutputsSummary import is_cogeneration_end_use
from geophires_x.OutputsUtils import OutputTableItem
from geophires_x.Parameter import Parameter
from geophires_x.Parameter import OutputParameter

if TYPE_CHECKING:
    from geophires_x.Model import Model

NL = "\n"


def economic_parameter_output_items(model: Model, is_sam_econ_model: bool) -> list[OutputTableItem]:
    section_text = StringIO()
    write_economic_parameters(model, section_text, is_sam_econ_model)
    return _economic_output_items_from_text(section_text.getvalue(), include_text_rows=False)


_ECONOMIC_VALUE_PATTERN = re.compile(
    r"^(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?|NaN|N/A)(?:\s+(?P<units>.+))?$"
)


def _economic_parameter_output_items_from_text(section_text: str) -> list[OutputTableItem]:
    return _economic_output_items_from_text(section_text, include_text_rows=False)


def capital_cost_output_items(model: Model, is_sam_econ_model: bool) -> list[OutputTableItem]:
    section_text = StringIO()
    write_capital_costs(model, section_text, is_sam_econ_model)
    return _economic_output_items_from_text(section_text.getvalue(), include_text_rows=True)


def operation_and_maintenance_cost_output_items(model: Model, is_sam_econ_model: bool) -> list[OutputTableItem]:
    section_text = StringIO()
    write_operation_and_maintenance_costs(model, section_text, is_sam_econ_model)
    return _economic_output_items_from_text(section_text.getvalue(), include_text_rows=True)


def _economic_output_items_from_text(section_text: str, *, include_text_rows: bool) -> list[OutputTableItem]:
    items = []
    for line in section_text.splitlines():
        stripped_line = line.strip()
        if not stripped_line or (stripped_line.startswith("***") and stripped_line.endswith("***")):
            continue

        if " = " in stripped_line and ":" not in stripped_line.split(" = ", 1)[0]:
            parameter, value = stripped_line.split(" = ", 1)
            items.append(OutputTableItem(parameter.strip(), value.strip()))
            continue

        if ":" not in stripped_line:
            if include_text_rows:
                items.append(OutputTableItem(stripped_line))
            continue

        parameter, raw_value = stripped_line.split(":", 1)
        value = raw_value.strip()
        units = ""
        value_match = _ECONOMIC_VALUE_PATTERN.match(value)
        if value_match is not None:
            value = value_match.group("value")
            units = value_match.group("units") or ""

        items.append(OutputTableItem(parameter.strip(), value, units))

    return items


def write_economic_parameters(model: Model, f: TextIOWrapper, is_sam_econ_model: bool) -> None:
    econ: Economics = model.economics

    f.write(NL)
    f.write(NL)
    f.write("                           ***ECONOMIC PARAMETERS***\n")
    f.write(NL)
    if model.economics.econmodel.value == EconomicModel.FCR:
        f.write(f"      Economic Model = {model.economics.econmodel.value.value}\n")
        f.write(f"      Fixed Charge Rate (FCR):                          {model.economics.FCR.value * 100.0:10.2f} {model.economics.FCR.CurrentUnits.value}\n")
    elif model.economics.econmodel.value == EconomicModel.STANDARDIZED_LEVELIZED_COST:
        f.write(f"      Economic Model = {model.economics.econmodel.value.value}\n")
        # TODO disambiguate interest rate for all economic models - see
        #  https://github.com/softwareengineerprogrammer/GEOPHIRES/commit/535c02d4adbeeeca553b61e9b996fccf00016529
        f.write(f"      {model.economics.interest_rate.Name}:                                    {model.economics.interest_rate.value:10.2f} {model.economics.interest_rate.CurrentUnits.value}\n")

    elif is_sam_econ_model or model.economics.econmodel.value == EconomicModel.BICYCLE:
        f.write(f"      Economic Model = {model.economics.econmodel.value.value}\n")

    if is_sam_econ_model:
        sam_econ_fields: list[OutputParameter] = [
            econ.real_discount_rate,
            econ.nominal_discount_rate,
            econ.wacc,
        ]

        for field in sam_econ_fields:
            label = field_label(field.Name, 49)
            f.write(f"      {label}{field.value:10.2f} {field.CurrentUnits.value}\n")

    if econ.RITCValue.value and is_sam_econ_model:
        # Non-SAM-EMs (inaccurately) treat ITC as a capital cost and thus are displayed in the capital
        # costs category rather than here.
        f.write(
            f"      {econ.RITCValue.display_name}:                           {abs(econ.RITCValue.value):10.2f} {econ.RITCValue.CurrentUnits.value}\n")

    if not is_sam_econ_model:  # (parameter is ambiguous to the point of meaninglessness for SAM-EM)
        acf: OutputParameter = econ.accrued_financing_during_construction_percentage
        acf_label = field_label(acf.display_name, 49)
        f.write(f"      {acf_label}{acf.value:10.2f} {acf.CurrentUnits.value}\n")

    display_inflation_costs_in_economic_parameters: bool = (
        econ.econmodel.value in [EconomicModel.BICYCLE,
                                 EconomicModel.FCR,
                                 EconomicModel.STANDARDIZED_LEVELIZED_COST]
        and
        econ.inflation_cost_during_construction.value != 0.
    )
    if display_inflation_costs_in_economic_parameters:
        # Inflation cost is displayed here for economic models that don't treat inflation cost as a
        # capital cost
        icc: OutputParameter = econ.inflation_cost_during_construction
        icc_label = field_label(icc.display_name, 49)
        f.write(f"      {icc_label}{icc.value:10.2f} {icc.CurrentUnits.value}\n")

    f.write(f"      Project lifetime:                              {model.surfaceplant.plant_lifetime.value:10.0f} {model.surfaceplant.plant_lifetime.CurrentUnits.value}\n")
    f.write(f"      Capacity factor:                                 {model.surfaceplant.utilization_factor.value * 100:10.1f} %\n")

    e_npv: OutputParameter = model.economics.ProjectNPV
    npv_field_label = field_label(e_npv.display_name, 49)
    # TODO should use CurrentUnits instead of PreferredUnits
    f.write(f"      {npv_field_label}{e_npv.value:10.2f} {e_npv.PreferredUnits.value}\n")

    irr_output_param: OutputParameter = econ.ProjectIRR \
        if not is_sam_econ_model else econ.after_tax_irr
    irr_field_label = field_label(irr_output_param.display_name, 49)
    irr_display_value = f"{irr_output_param.value:10.2f}" \
        if not math.isnan(irr_output_param.value) else "NaN"
    f.write(f"      {irr_field_label}{irr_display_value} {irr_output_param.CurrentUnits.value}\n")

    f.write(f"      {econ.ProjectVIR.display_name}:                              {econ.ProjectVIR.value:10.2f}\n")
    f.write(f"      {econ.ProjectMOIC.display_name}:                                    {econ.ProjectMOIC.value:10.2f}\n")

    payback_period_val = model.economics.ProjectPaybackPeriod.value
    project_payback_period_display = (f"{payback_period_val:10.2f} "
                                      f"{econ.ProjectPaybackPeriod.PreferredUnits.value}") \
        if payback_period_val > 0.0 else "N/A"
    project_payback_period_label = field_label(model.economics.ProjectPaybackPeriod.display_name, 56)
    f.write(f"      {project_payback_period_label}{project_payback_period_display}\n")

    if is_cogeneration_end_use(model.surfaceplant.enduse_option.value):
        f.write(f"      CHP: Percent cost allocation for electrical plant: {model.economics.CAPEX_heat_electricity_plant_ratio.value * 100.0:10.2f} %\n")

    if model.surfaceplant.enduse_option.value in [EndUseOptions.ELECTRICITY]:
        f.write(f"      Estimated Jobs Created:                                 {model.economics.jobs_created.value}\n")


def write_stimulation_costs_outputs(econ: Economics, f: TextIOWrapper) -> None:
    f.write(
        f"         {econ.Cstim.display_name}:                             "
        f"{econ.Cstim.value:10.2f} {econ.Cstim.CurrentUnits.value}\n"
    )

    def write_output(stimulation_cost_output: OutputParameter) -> None:
        if stimulation_cost_output.value is None:
            return

        label = field_label(stimulation_cost_output.display_name, 43)
        f.write(
            f"             {label}{stimulation_cost_output.value:10.2f} "
            f"{stimulation_cost_output.CurrentUnits.value}\n"
        )

    if econ.cstim_per_well.value is not None:
        write_output(econ.cstim_per_well)
    else:
        for stimulation_cost_output in (
            econ.cstim_per_production_well,
            econ.cstim_per_injection_well,
        ):
            write_output(stimulation_cost_output)


def write_capital_costs(model: Model, f: TextIOWrapper, is_sam_econ_model: bool) -> None:
    econ: Economics = model.economics

    f.write("\n\n                          ***CAPITAL COSTS (M$)***\n\n")
    if not model.economics.totalcapcost.Valid:
        f.write(f"         {econ.Cexpl.display_name}:                             {econ.Cexpl.value:10.2f} {econ.Cexpl.CurrentUnits.value}\n")

        f.write(f"         {model.economics.Cwell.display_name}:                 {model.economics.Cwell.value:10.2f} {model.economics.Cwell.CurrentUnits.value}\n")

        if econ.cost_lateral_section.value > 0.0:
            f.write(f"             Drilling and completion costs per vertical production well:   {econ.cost_one_production_well.value:10.2f} " + econ.cost_one_production_well.CurrentUnits.value + NL)
            f.write(f"             Drilling and completion costs per vertical injection well:    {econ.cost_one_injection_well.value:10.2f} " + econ.cost_one_injection_well.CurrentUnits.value + NL)
            f.write(f"             {econ.cost_per_lateral_section.Name}:       {econ.cost_per_lateral_section.value:10.2f} {econ.cost_lateral_section.CurrentUnits.value}\n")
        elif round(econ.cost_one_production_well.value, 4) != round(econ.cost_one_injection_well.value, 4) \
            and model.economics.cost_one_injection_well.value != -1:
            f.write(f"             {econ.cost_one_production_well.display_name}:   {econ.cost_one_production_well.value:10.2f} {econ.cost_one_production_well.CurrentUnits.value}\n")
            f.write(f"             {econ.cost_one_injection_well.display_name}:    {econ.cost_one_injection_well.value:10.2f} {econ.cost_one_injection_well.CurrentUnits.value}\n")
        else:
            cpw_label = field_label(econ.drilling_and_completion_costs_per_well.display_name, 47)
            f.write(f"         {cpw_label}{econ.drilling_and_completion_costs_per_well.value:10.2f} {econ.Cwell.CurrentUnits.value}\n")

        write_stimulation_costs_outputs(econ, f)

        f.write(f"         {econ.Cplant.display_name}:                     {econ.Cplant.value:10.2f} {econ.Cplant.CurrentUnits.value}\n")
        if is_cogeneration_end_use(model.surfaceplant.enduse_option.value):
            f.write(
                f"            {econ.CAPEX_cost_electrical_plant.display_name}:"
                f"                {econ.CAPEX_cost_electrical_plant.value:10.2f} {econ.CAPEX_cost_electrical_plant.CurrentUnits.value}\n"
            )
            f.write(
                f"            {econ.CAPEX_cost_heat_plant.display_name}:"
                f"                      {econ.CAPEX_cost_heat_plant.value:10.2f} {econ.CAPEX_cost_heat_plant.CurrentUnits.value}\n"
            )
        if model.surfaceplant.plant_type.value == PlantType.ABSORPTION_CHILLER:
            f.write(f"            of which Absorption Chiller Cost:           {model.economics.chillercapex.value:10.2f} " + model.economics.Cplant.CurrentUnits.value + NL)
        if model.surfaceplant.plant_type.value == PlantType.HEAT_PUMP:
            f.write(f"            of which Heat Pump Cost:                    {model.economics.heatpumpcapex.value:10.2f} " + model.economics.Cplant.CurrentUnits.value + NL)
        if model.surfaceplant.plant_type.value == PlantType.DISTRICT_HEATING:
            f.write(f"            of which Peaking Boiler Cost:               {model.economics.peakingboilercost.value:10.2f} " + model.economics.peakingboilercost.CurrentUnits.value + NL)
        f.write(f"         {model.economics.Cgath.display_name}:                  {model.economics.Cgath.value:10.2f} {model.economics.Cgath.CurrentUnits.value}\n")

        if model.surfaceplant.piping_length.value > 0:
            f.write(f"         {model.economics.Cpiping.display_name}:                    {model.economics.Cpiping.value:10.2f} {model.economics.Cpiping.CurrentUnits.value}\n")

        if model.surfaceplant.plant_type.value == PlantType.DISTRICT_HEATING:
            f.write(f"         District Heating System Cost:                  {model.economics.dhdistrictcost.value:10.2f} {model.economics.dhdistrictcost.CurrentUnits.value}\n")

        f.write(
            f"         {model.economics.surface_equipment_costs_total.Name}:"
            f"                 {model.economics.surface_equipment_costs_total.value:10.2f} "
            f"{model.economics.surface_equipment_costs_total.CurrentUnits.value}\n"
        )
        if Economics._tess_enabled(model):
            f.write(
                f"         {econ.tess_capital_cost.display_name}:"
                f"                                 {econ.tess_capital_cost.value:10.2f} "
                f"{econ.tess_capital_cost.CurrentUnits.value}\n"
            )

    if model.economics.totalcapcost.Valid and model.wellbores.redrill.value > 0:
        f.write(f"         Drilling and completion costs (for redrilling):{econ.Cwell.value:10.2f} {econ.Cwell.CurrentUnits.value}\n")
        f.write(f"      Drilling and completion costs per redrilled well: {(econ.Cwell.value / (model.wellbores.nprod.value + model.wellbores.ninj.value)):10.2f} {econ.Cwell.CurrentUnits.value}\n")
        f.write(f"         Stimulation costs (for redrilling):            {econ.Cstim.value:10.2f} {econ.Cstim.CurrentUnits.value}\n")

    if model.economics.RITCValue.value and not is_sam_econ_model:
        # Note ITC is in ECONOMIC PARAMETERS category for SAM-EM (not capital costs)
        f.write(f"         {econ.RITCValue.display_name}:                         {-1 * econ.RITCValue.value:10.2f} {econ.RITCValue.CurrentUnits.value}\n")

    def render_additional_capital_cost_modifiers(modifiers: list[tuple[Parameter, int]]) -> None:
        for parameter, multiplier in modifiers:
            if parameter.Provided:
                label = field_label(parameter.Name, 47)
                f.write(
                    f"         {label}{parameter.value * multiplier:10.2f} "
                    f"{parameter.CurrentUnits.value}\n"
                )

    additional_occ_modifiers: list[tuple[Parameter, int]] = [(econ.FlatLicenseEtc, 1)]
    if not is_sam_econ_model:
        additional_occ_modifiers.extend(
            [
                (econ.OtherIncentives, -1),
                (econ.TotalGrant, -1),
            ]
        )
    render_additional_capital_cost_modifiers(additional_occ_modifiers)

    if is_sam_econ_model and econ.DoAddOnCalculations.value:
        # Non-SAM econ models print this in Extended Economics profile
        aoc_label = field_label(model.addeconomics.AddOnCAPEXTotal.display_name, 47)
        f.write(f"         {aoc_label}{model.addeconomics.AddOnCAPEXTotal.value:10.2f} {model.addeconomics.AddOnCAPEXTotal.CurrentUnits.value}\n")

    display_occ_and_inflation_during_construction_in_capital_costs = is_sam_econ_model
    if display_occ_and_inflation_during_construction_in_capital_costs:
        occ_label = field_label(econ.overnight_capital_cost.display_name, 47)
        f.write(f"         {occ_label}{econ.overnight_capital_cost.value:10.2f} {econ.overnight_capital_cost.CurrentUnits.value}\n")

        icc_label = field_label(econ.inflation_cost_during_construction.display_name, 47)
        f.write(f"         {icc_label}{econ.inflation_cost_during_construction.value:10.2f} {econ.inflation_cost_during_construction.CurrentUnits.value}\n")

    if econ.royalty_supplemental_payments.Provided:
        rsp_label = field_label(econ.royalty_supplemental_payments_cost_during_construction.display_name, 41)
        f.write(f"         {rsp_label}   {econ.royalty_supplemental_payments_cost_during_construction.value:.2f} {econ.royalty_supplemental_payments_cost_during_construction.CurrentUnits.value}\n")

    display_idc_in_capital_costs = is_sam_econ_model and model.surfaceplant.construction_years.value > 1
    if display_idc_in_capital_costs:
        idc_label = field_label(econ.interest_during_construction.display_name, 47)
        f.write(f"         {idc_label}{econ.interest_during_construction.value:10.2f} {econ.interest_during_construction.CurrentUnits.value}\n")

    if is_sam_econ_model:
        render_additional_capital_cost_modifiers(
            [
                (econ.OtherIncentives, -1),
                (econ.TotalGrant, -1),
            ]
        )

    capex_param = econ.CCap if not is_sam_econ_model else econ.capex_total
    capex_label = field_label(capex_param.display_name, 50)
    f.write(f"      {capex_label}{capex_param.value:10.2f} {capex_param.CurrentUnits.value}\n")

    if model.economics.econmodel.value == EconomicModel.FCR:
        f.write(f"      Annualized capital costs:                         {(model.economics.CCap.value * (1 + model.economics.inflrateconstruction.value) * model.economics.FCR.value):10.2f} " + model.economics.CCap.CurrentUnits.value + NL)


def write_operation_and_maintenance_costs(model: Model, f: TextIOWrapper, is_sam_econ_model: bool) -> None:
    econ: Economics = model.economics

    f.write(NL)
    f.write(NL)
    f.write("                ***OPERATING AND MAINTENANCE COSTS (M$/yr)***\n")
    f.write(NL)
    if not model.economics.oamtotalfixed.Valid:
        f.write(f"         {model.economics.Coamwell.display_name}:                   {model.economics.Coamwell.value:10.2f} {model.economics.Coamwell.CurrentUnits.value}\n")
        f.write(f"         {model.economics.Coamplant.display_name}:                 {model.economics.Coamplant.value:10.2f} {model.economics.Coamplant.CurrentUnits.value}\n")
        f.write(f"         {model.economics.Coamwater.display_name}:                                   {model.economics.Coamwater.value:10.2f} {model.economics.Coamwater.CurrentUnits.value}\n")
        if model.surfaceplant.plant_type.value in [PlantType.INDUSTRIAL, PlantType.ABSORPTION_CHILLER, PlantType.HEAT_PUMP, PlantType.DISTRICT_HEATING]:
            f.write(f"         Average Reservoir Pumping Cost:                {model.economics.averageannualpumpingcosts.value:10.2f} {model.economics.averageannualpumpingcosts.CurrentUnits.value}\n")
        if model.surfaceplant.plant_type.value == PlantType.ABSORPTION_CHILLER:
            f.write(f"         Absorption Chiller O&M Cost:                   {model.economics.chilleropex.value:10.2f} {model.economics.chilleropex.CurrentUnits.value}\n")
        if Economics._tess_enabled(model):
            f.write(
                f"         {econ.tess_o_and_m_cost.display_name}:"
                f"                             {econ.tess_o_and_m_cost.value:10.2f} "
                f"{econ.tess_o_and_m_cost.CurrentUnits.value}\n"
            )
        if model.surfaceplant.plant_type.value == PlantType.HEAT_PUMP:
            f.write(f"         Average Heat Pump Electricity Cost:            {model.economics.averageannualheatpumpelectricitycost.value:10.2f} {model.economics.averageannualheatpumpelectricitycost.CurrentUnits.value}\n")
        if model.surfaceplant.plant_type.value == PlantType.DISTRICT_HEATING:
            f.write(f"         Annual District Heating O&M Cost:              {model.economics.dhdistrictoandmcost.value:10.2f} {model.economics.dhdistrictoandmcost.CurrentUnits.value}\n")
            f.write(f"         Average Annual Peaking Fuel Cost:              {model.economics.averageannualngcost.value:10.2f} {model.economics.averageannualngcost.CurrentUnits.value}\n")

        if model.wellbores.redrill.value > 0:
            redrill_label = field_label(econ.redrilling_annual_cost.display_name, 47)
            f.write(f"         {redrill_label}{econ.redrilling_annual_cost.value:10.2f} {econ.redrilling_annual_cost.CurrentUnits.value}\n")

        if econ.DoAddOnCalculations.value and is_sam_econ_model:
            # Non-SAM econ models print this in Extended Economics profile
            aoc_label = field_label(model.addeconomics.AddOnOPEXTotalPerYear.display_name, 47)
            f.write(f"         {aoc_label}{model.addeconomics.AddOnOPEXTotalPerYear.value:10.2f} {model.addeconomics.AddOnOPEXTotalPerYear.CurrentUnits.value}\n")

        if econ.has_production_based_royalties:
            royalties_label = field_label(econ.royalties_average_annual_cost.display_name, 47)
            f.write(f"         {royalties_label}{econ.royalties_average_annual_cost.value:10.2f} {econ.royalties_average_annual_cost.CurrentUnits.value}\n")

        f.write(f"      {econ.Coam.display_name}:            {(econ.Coam.value + econ.averageannualpumpingcosts.value + econ.averageannualheatpumpelectricitycost.value):10.2f} {econ.Coam.CurrentUnits.value}\n")
    else:
        f.write(f"      {econ.Coam.display_name}:            {econ.Coam.value:10.2f} {econ.Coam.CurrentUnits.value}\n")
