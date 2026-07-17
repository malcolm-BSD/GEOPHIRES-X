from __future__ import annotations

import copy
from io import TextIOWrapper
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from geophires_x.EconomicsSam import get_sam_cash_flow_profile_tabulated_output

if TYPE_CHECKING:
    from geophires_x.Model import Model

NL = "\n"


def _carbon_revenue_outputs(model: Model):
    econ = model.economics
    carbon_price = copy.deepcopy(econ.CarbonPrice)
    carbon_revenue = copy.deepcopy(econ.CarbonRevenue)
    cumulative_carbon_revenue = copy.deepcopy(econ.CarbonCummCashFlow)

    if econ.DoSDACGTCalculations.value:
        construction_years = model.surfaceplant.construction_years.value

        carbon_revenue = copy.deepcopy(model.sdacgteconomics.CarbonRevenue)
        carbon_revenue.value = [*([0] * construction_years), *carbon_revenue.value]

        cumulative_carbon_revenue = copy.deepcopy(model.sdacgteconomics.CarbonCummCashFlow)
        cumulative_carbon_revenue.value = [
            *([0] * construction_years),
            *cumulative_carbon_revenue.value,
        ]

        for sdac_output, economics_output in (
            (carbon_revenue, econ.CarbonRevenue),
            (cumulative_carbon_revenue, econ.CarbonCummCashFlow),
        ):
            sdac_output.value = sdac_output.quantity().to(economics_output.CurrentUnits).magnitude
            sdac_output.CurrentUnits = economics_output.CurrentUnits

    return carbon_price, carbon_revenue, cumulative_carbon_revenue


def _dispatch_report_year_count(model: Model, default: int | None = None) -> int | None:
    dispatch_results = getattr(model, "dispatch_results", None)
    if dispatch_results is None:
        return default

    analysis_end_year = int(
        getattr(dispatch_results, "analysis_end_year", model.surfaceplant.plant_lifetime.value)
    )
    return min(analysis_end_year, model.surfaceplant.plant_lifetime.value)


def _filter_dispatch_cashflow_rows(table: pd.DataFrame, model: Model) -> pd.DataFrame:
    report_end_year = _dispatch_report_year_count(model)
    if report_end_year is None:
        return table

    construction_years = int(model.surfaceplant.construction_years.value)
    start_index = construction_years
    end_index = construction_years + report_end_year
    filtered_table = table.iloc[start_index:end_index].copy()
    year_column = next((column for column in filtered_table.columns if str(column).startswith("Year|")), None)
    if year_column is not None:
        filtered_table[year_column] = list(range(1, report_end_year + 1))
    return filtered_table


def revenue_and_cashflow_profile_table(model: Model) -> pd.DataFrame:
    econ = model.economics
    carbon_price, carbon_revenue, cumulative_carbon_revenue = _carbon_revenue_outputs(model)
    coam = np.zeros(model.surfaceplant.construction_years.value + model.surfaceplant.plant_lifetime.value)
    for ii in range(model.surfaceplant.construction_years.value, model.surfaceplant.plant_lifetime.value + 1):
        coam[ii] = econ.Coam.value

    cashflow = pd.DataFrame()
    cashflow[f"Year|:3.0f"] = [
        i
        for i in range(
            1,
            model.surfaceplant.plant_lifetime.value + model.surfaceplant.construction_years.value + 1,
        )
    ]
    cashflow[f"Electricity:Price ({econ.ElecPrice.CurrentUnits.value})|:7.4f"] = econ.ElecPrice.value
    cashflow[f"Electricity:Ann. Rev. ({econ.ElecRevenue.CurrentUnits.value})|:5.2f"] = econ.ElecRevenue.value
    cashflow[
        f"Electricity:Cumm. Rev. ({econ.ElecCummRevenue.CurrentUnits.value})|:5.2f"
    ] = econ.ElecCummRevenue.value
    cashflow[f"Heat:Price ({econ.HeatPrice.CurrentUnits.value})|:7.4f"] = econ.HeatPrice.value
    cashflow[f"Heat:Ann. Rev. ({econ.HeatRevenue.CurrentUnits.value})|:5.2f"] = econ.HeatRevenue.value
    cashflow[f"Heat:Cumm. Rev. ({econ.HeatCummRevenue.CurrentUnits.value})|:5.2f"] = (
        econ.HeatCummRevenue.value
    )
    cashflow[f"Cooling:Price ({econ.CoolingPrice.CurrentUnits.value})|:7.4f"] = econ.CoolingPrice.value
    cashflow[f"Cooling:Ann. Rev. ({econ.CoolingRevenue.CurrentUnits.value})|:5.2f"] = (
        econ.CoolingRevenue.value
    )
    cashflow[
        f"Cooling:Cumm. Rev. ({econ.CoolingCummRevenue.CurrentUnits.value})|:5.2f"
    ] = econ.CoolingCummRevenue.value
    cashflow[f"Carbon:Price ({carbon_price.CurrentUnits.value})|:7.4f"] = carbon_price.value
    cashflow[f"Carbon:Ann. Rev. ({carbon_revenue.CurrentUnits.value})|:5.2f"] = (
        carbon_revenue.value
    )
    cashflow[
        f"Carbon:Cumm. Rev. ({cumulative_carbon_revenue.CurrentUnits.value})|:5.2f"
    ] = cumulative_carbon_revenue.value
    cashflow[f"Project:OPEX ({econ.Coam.CurrentUnits.value})|:5.2f"] = coam
    cashflow[f"Project:Net Rev. ({econ.TotalRevenue.CurrentUnits.value})|:5.2f"] = econ.TotalRevenue.value
    cashflow[f"Project:Net Cashflow ({econ.TotalCummRevenue.CurrentUnits.value})|:5.2f"] = (
        econ.TotalCummRevenue.value
    )
    return _filter_dispatch_cashflow_rows(cashflow, model).reset_index()


def write_revenue_and_cashflow_profile_output(model: Model, f: TextIOWrapper) -> None:
    """Write the legacy revenue and cashflow profile table."""
    econ = model.economics

    f.write(NL)
    f.write(NL)
    f.write("                             ********************************\n")
    f.write("                             *  REVENUE & CASHFLOW PROFILE  *\n")
    f.write("                             ********************************\n")
    f.write(
        "Year            Electricity             |            Heat                  |           Cooling                 |         Carbon                    |          Project"
        + NL
    )
    f.write(
        "Since     Price   Ann. Rev.  Cumm. Rev. |   Price   Ann. Rev.   Cumm. Rev. |  Price   Ann. Rev.   Cumm. Rev.   |   Price   Ann. Rev.   Cumm. Rev.  | OPEX    Net Rev.      Net Cashflow"
        + NL
    )

    def o(output_param):
        # TODO generalize this and/or FIXME make it unnecessary
        if output_param.Name in econ.OutputParameterDict:
            return econ.OutputParameterDict[output_param.Name]
        else:
            return output_param

    carbon_price, carbon_revenue, cumulative_carbon_revenue = _carbon_revenue_outputs(model)

    f.write(
        "Start    ("
        + o(econ.ElecPrice).CurrentUnits.value
        + ")("
        + o(econ.ElecRevenue).CurrentUnits.value
        + ") ("
        + o(econ.ElecCummRevenue).CurrentUnits.value
        + ")    |("
        + o(econ.HeatPrice).CurrentUnits.value
        + ") ("
        + o(econ.HeatRevenue).CurrentUnits.value
        + ")    ("
        + o(econ.HeatCummRevenue).CurrentUnits.value
        + ")   |("
        + o(econ.CoolingPrice).CurrentUnits.value
        + ") ("
        + o(econ.CoolingRevenue).CurrentUnits.value
        + ")    ("
        + o(econ.CoolingCummRevenue).CurrentUnits.value
        + ")    |("
        + o(carbon_price).CurrentUnits.value
        + ")    ("
        + o(carbon_revenue).CurrentUnits.value
        + ")    ("
        + o(cumulative_carbon_revenue).CurrentUnits.value
        + ")    |("
        + o(econ.Coam).CurrentUnits.value
        + ") ("
        + o(econ.TotalRevenue).CurrentUnits.value
        + ")    ("
        + o(econ.TotalCummRevenue).CurrentUnits.value
        + ")\n"
    )
    f.write(
        "________________________________________________________________________________________________________________________________________________________________________________________"
        + NL
    )
    dispatch_report_year_count = _dispatch_report_year_count(model, default=None)
    if dispatch_report_year_count is None:
        cashflow_indices = range(
            0,
            model.surfaceplant.construction_years.value + model.surfaceplant.plant_lifetime.value,
        )
    else:
        construction_years = model.surfaceplant.construction_years.value
        cashflow_indices = range(construction_years, construction_years + dispatch_report_year_count)

    # running years...
    for ii in cashflow_indices:
        if ii < model.surfaceplant.construction_years.value:
            opex = 0.0  # zero out the OPEX during construction years
        else:
            opex = o(econ.Coam).value
        display_year = (
            ii
            if dispatch_report_year_count is None
            else ii - model.surfaceplant.construction_years.value + 1
        )
        f.write(
            f"{display_year:3.0f}     {o(econ.ElecPrice).value[ii]:5.2f}          {o(econ.ElecRevenue).value[ii]:5.2f}  {o(econ.ElecCummRevenue).value[ii]:5.2f}     |   {o(econ.HeatPrice).value[ii]:5.2f}    {o(econ.HeatRevenue).value[ii]:5.2f}        {o(econ.HeatCummRevenue).value[ii]:5.2f}    |   {o(econ.CoolingPrice).value[ii]:5.2f}    {o(econ.CoolingRevenue).value[ii]:5.2f}        {o(econ.CoolingCummRevenue).value[ii]:5.2f}     |   {o(carbon_price).value[ii]:5.2f}    {o(carbon_revenue).value[ii]:5.2f}        {o(cumulative_carbon_revenue).value[ii]:5.2f}     | {opex:5.2f}     {o(econ.TotalRevenue).value[ii]:5.2f}     {o(econ.TotalCummRevenue).value[ii]:5.2f}\n"
        )
    f.write(NL)


def get_sam_cash_flow_profile_output(model: Model) -> str:
    ret = "\n"
    ret += "                            ***************************\n"
    ret += "                            *  SAM CASH FLOW PROFILE  *\n"
    ret += "                            ***************************\n"

    cfp_o: str = get_sam_cash_flow_profile_tabulated_output(model)

    # Ideally the separator line would be exactly the print width of the widest column, but the actual print width
    # of tabs varies (at least according to https://stackoverflow.com/a/7643592). 4 spaces seems to be the minimum
    # number that results in a separator line at least as wide as the table (narrower would be unsightly).
    spaces_per_tab = 4

    # The tabulate library has native separating line functionality (per https://pypi.org/project/tabulate/) but
    # I wasn't able to get it to replicate the formatting as coded below.
    separator_line = len(cfp_o.split("\n")[0].replace("\t", " " * spaces_per_tab)) * "-"

    ret += separator_line + "\n"
    ret += cfp_o
    ret += "\n" + separator_line

    ret += "\n"

    return ret
