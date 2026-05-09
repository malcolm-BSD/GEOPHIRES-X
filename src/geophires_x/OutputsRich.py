import datetime
import csv
import io
import string
import time
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rich.console import Console
from rich.table import Table

import geophires_x
from geophires_x import Model as Model
from geophires_x.DispatchReporting import (
    DISPATCH_PROFILE_CATEGORY_NAME,
    dispatch_profile_table,
    is_dispatch_report,
    weather_output_rows,
)
from geophires_x.GeoPHIRESUtils import InsertImagesIntoHTML, UpgradeSymbologyOfUnits, render_default
from geophires_x.MatplotlibUtils import plt_subplots
from geophires_x.OptionList import EndUseOptions, PlantType, EconomicModel
from geophires_x.OutputsCashFlow import revenue_and_cashflow_profile_table
from geophires_x.OutputsDispatch import dispatch_output_rows, dispatch_profile_report_text, tess_output_rows
from geophires_x.OutputsEconomics import (
    capital_cost_output_items,
    economic_parameter_output_items,
    operation_and_maintenance_cost_output_items,
)
from geophires_x.OutputsEngineering import engineering_parameter_output_items
from geophires_x.OutputsProfiles import (
    annual_production_profile_table,
    production_profile_table,
)
from geophires_x.OutputsReservoir import (
    pumping_power_profile_table,
    reservoir_parameter_output_items,
    reservoir_simulation_result_output_items,
)
from geophires_x.OutputsResource import resource_characteristic_output_items
from geophires_x.OutputsSurface import surface_equipment_simulation_result_output_items
from geophires_x.OutputsSummary import summary_output_items
from geophires_x.OutputsUtils import OutputTableItem

from geophires_x.Parameter import intParameter, strParameter

NL = '\n'
validFilenameChars = "-_.() %s%s" % (string.ascii_letters, string.digits)

_GRAPH_FIGSIZE = (12, 6)


def _set_plot_xlim(ax, x: pd.array) -> None:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_min == x_max:
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
    else:
        ax.set_xlim(x_min, x_max)


def print_outputs_rich(
        text_output_file: strParameter,
        html_output_file: strParameter,
        model: Model,
        sdac_results: list,
        addon_results: list,
        sdac_df: pd.DataFrame,
        addon_df: pd.DataFrame):
    """
    TODO Implementation of rich output in this method/file is duplicative of Outputs.PrintOutputs. This adds undue
      code complexity, maintenance overhead, inconsistency, and potential for bugs. Rich output should instead be
      generated in a module that uses GeophiresXClient or an equivalent pattern which maintains Outputs.PrintOutputs
      as the ultimate source of truth/authority for output logic.

    The XLCO and VALCO summary entries below intentionally mirror Outputs.PrintOutputs so the
    text report, rich report, client parser, and generated schema all expose the same
    competitiveness metrics for each active commodity.
    """

    weather_results = []
    tess_results = []
    dispatch_results = []
    dispatch_report = is_dispatch_report(model)
    dispatch_profile_rows_table = dispatch_profile_table(model)
    is_sam_econ_model = model.economics.econmodel.value == EconomicModel.SAM_SINGLE_OWNER_PPA
    # addon_results = []

    simulation_metadata = [
        OutputTableItem('GEOPHIRES Version', geophires_x.__version__),
        OutputTableItem('Simulation Date', datetime.datetime.now().strftime('%Y-%m-%d')),
        OutputTableItem('Simulation Time', datetime.datetime.now().strftime('%H:%M')),
        OutputTableItem('Calculation Time', '{0:10.3f}'.format((time.time() - model.tic)) + ' sec'),
    ]

    if dispatch_report:
        for field_name, value, units in weather_output_rows(model):
            weather_results.append(
                OutputTableItem(field_name, value if isinstance(value, str) else '{0:10.2f}'.format(value), units)
            )

    summary = summary_output_items(
        model,
        dispatch_report,
        is_sam_econ_model,
    )

    if getattr(model, 'dispatch_results', None) is not None:
        dispatch_results.extend(
            OutputTableItem(field_name, '{0:10.2f}'.format(value), units)
            for field_name, value, units in dispatch_output_rows(model)
        )
        tess_results.extend(
            OutputTableItem(field_name, '{0:10.2f}'.format(value), units)
            for field_name, value, units in tess_output_rows(model, model.dispatch_results.summary_metrics)
        )

    economic_parameters = economic_parameter_output_items(model, is_sam_econ_model)
    engineering_parameters = engineering_parameter_output_items(model)
    resource_characteristics = resource_characteristic_output_items(model)
    reservoir_parameters = reservoir_parameter_output_items(model)
    reservoir_stimulation_results = reservoir_simulation_result_output_items(model, dispatch_report)
    capex = capital_cost_output_items(model, is_sam_econ_model)
    opex = operation_and_maintenance_cost_output_items(model, is_sam_econ_model)
    surface_equipment_results = surface_equipment_simulation_result_output_items(model, dispatch_report)
    hce = production_profile_table(model, dispatch_report)
    ahce = annual_production_profile_table(model, dispatch_report)
    cashflow = revenue_and_cashflow_profile_table(model)
    pumping_power_profiles = pumping_power_profile_table(model)

    if text_output_file.Provided:
        Write_RTF_Output(text_output_file.value, simulation_metadata, summary, economic_parameters,
                         engineering_parameters, resource_characteristics, reservoir_parameters,
                         reservoir_stimulation_results, capex, opex, surface_equipment_results, weather_results,
                         tess_results, dispatch_results, dispatch_profile_rows_table, sdac_results, addon_results, hce,
                         ahce, cashflow, pumping_power_profiles, sdac_df, addon_df)

    if html_output_file.Provided:
        Write_HTML_Output(html_output_file.value, simulation_metadata, summary, economic_parameters,
                          engineering_parameters, resource_characteristics, reservoir_parameters,
                          reservoir_stimulation_results, capex, opex, surface_equipment_results, weather_results,
                          tess_results, dispatch_results, dispatch_profile_rows_table, sdac_results, addon_results, hce,
                          ahce, cashflow, pumping_power_profiles, sdac_df, addon_df)

        if not dispatch_report:
            Plot_Tables_Into_HTML(model.surfaceplant.enduse_option, model.surfaceplant.plant_type,
                                  html_output_file.value, hce, ahce, cashflow, pumping_power_profiles, sdac_df,
                                  addon_df)
        if getattr(model.outputs, 'generate_dispatch_html_graphs', None) is not None and \
                model.outputs.generate_dispatch_html_graphs.value and getattr(model, 'dispatch_results', None) is not None:
            Plot_Dispatch_Graphs_Into_HTML(model, html_output_file.value)
        # make district heating plot
        if model.surfaceplant.plant_type.value == PlantType.DISTRICT_HEATING:
            MakeDistrictHeatingPlot(html_output_file.value, model.surfaceplant.dh_geothermal_heating.value,
                                    model.surfaceplant.daily_heating_demand.value)


def remove_disallowed_filename_chars(filename: str) -> str:
    """
     This function removes disallowed filename characters
     :param filename: the filename
     :type filename: str
     :return: the cleaned filename
     :rtype: str
     """
    cleaned_filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore')
    return ''.join(chr(c) for c in cleaned_filename if chr(c) in validFilenameChars)


def removeDisallowedFilenameChars(filename):
    return remove_disallowed_filename_chars(filename)


def Write_Simple_Text_Table(title: str, items: list, f) -> None:
    """
    This function writes out the simple tables as text
    :param title: the title of the table
    :type title: str
    :param items: the list of items to be written out
    :type items: list
    :param f: the file object
    :type f: file
    """
    f.write(f'{NL}')
    f.write(f'                           ***{title}***{NL}')
    f.write(f'{NL}')
    for item in items:
        f.write(f'      {item.parameter:<45}: {item.value:^10} {item.units}{NL}')


def Write_Complex_Text_table(title: str, df_table: pd.DataFrame, time_steps_per_year: int, f) -> None:
    """
    This function writes out the complex tables as text
    :param title: the title of the table
    :type title: str
    :param df_table: the dataframe to be written out
    :type df_table: pd.DataFrame
    :param time_steps_per_year: the number of time steps per year
    :type time_steps_per_year: int
    :param f: the file object
    :type f: file
    """
    f.write(f'{NL}')
    f.write(f'                            ***************************************************************{NL}')
    f.write(f'                            *{title:^58}*{NL}')
    f.write(f'                            ***************************************************************{NL}')
    column_fmt = []
    for col in df_table.columns:
        if col != 'index':
            pair = col.split('|')
            column_name = pair[0]
            column_fmt.append(pair[1])
            f.write(f'  {UpgradeSymbologyOfUnits(column_name):^29}   ')

    f.write(f'{NL}')
    for index, row in df_table.iterrows():
        # only print the number of rows implied by time_steps_per_year
        if int(index) % time_steps_per_year == 0:
            for i in range(1, len(row)):
                f.write(f'{render_default((df_table.at[index, row.index[i]]) / time_steps_per_year, "", column_fmt[i - 1]):^33} ')
            f.write(f'{NL}')


def _csv_table_to_text(title: str, table_rows: list[list]) -> str:
    if title == DISPATCH_PROFILE_CATEGORY_NAME:
        return dispatch_profile_report_text(title, table_rows)

    buffer = io.StringIO()
    buffer.write(f'{NL}')
    buffer.write(f'                            **********************{NL}')
    buffer.write(f'                            *  {title}  *{NL}')
    buffer.write(f'                            **********************{NL}')
    writer = csv.writer(buffer, lineterminator=NL)
    writer.writerows(table_rows)
    buffer.write(f'{NL}')
    return buffer.getvalue()


def Write_CSV_Text_Table(title: str, table_rows: list[list], f) -> None:
    f.write(_csv_table_to_text(title, table_rows))


def Write_CSV_HTML_Table(title: str, table_rows: list[list], console: Console) -> None:
    console.print(_csv_table_to_text(title, table_rows))


def Write_Text_Output(output_path: str, simulation_metadata: list, summary: list, economic_parameters: list,
                      engineering_parameters: list, resource_characteristics: list, reservoir_parameters: list,
                      reservoir_stimulation_results: list, capex: list, opex: list, surface_equipment_results: list,
                      weather_results: list, tess_results: list, dispatch_results: list, dispatch_profile_rows_table: list,
                      sdac_results: list, addon_results: list, hce: pd.DataFrame, ahce: pd.DataFrame,
                      cashflow: pd.DataFrame, pumping_power_profiles: pd.DataFrame, sdac_df: pd.DataFrame,
                      addon_df: pd.DataFrame) -> None:
    """
    This function writes out the text output
    :param output_path: the path to the output file
    :type output_path: str
    :param simulation_metadata: the simulation metadata
    :type simulation_metadata: list
    :param summary: the summary of results
    :type summary: list
    :param economic_parameters: the economic parameters
    :type economic_parameters: list
    :param engineering_parameters: the engineering parameters
    :type engineering_parameters: list
    :param resource_characteristics: the resource characteristics
    :type resource_characteristics: list
    :param reservoir_parameters: the reservoir parameters
    :type reservoir_parameters: list
    :param reservoir_stimulation_results: the reservoir stimulation results
    :type reservoir_stimulation_results: list
    :param capex: the capital costs
    :type capex: list
    :param opex: the operating and maintenance costs
    :type opex: list
    :param surface_equipment_results: the surface equipment simulation results
    :type surface_equipment_results: list
    :param dispatch_results: the dispatch results
    :type dispatch_results: list
    :param sdac_results: the sdac results
    :type sdac_results: list
    :param hce: the heating, cooling and/or electricity production profile
    :type hce: pd.DataFrame
    :param cashflow: the revenue & cashflow profile
    :type cashflow: pd.DataFrame
    :param pumping_power_profiles: the pumping power profiles
    :type pumping_power_profiles: pd.DataFrame
    :param sdac_df: the sdac dataframe
    :type sdac_df: pd.DataFrame
    :return: None
    """
    with open(output_path, 'w', encoding='UTF-8') as f:
        f.write(Build_Text_Output(
            simulation_metadata, summary, economic_parameters, engineering_parameters, resource_characteristics,
            reservoir_parameters, reservoir_stimulation_results, capex, opex, surface_equipment_results,
            weather_results, tess_results, dispatch_results, dispatch_profile_rows_table, sdac_results, addon_results,
            hce, ahce, cashflow, pumping_power_profiles, sdac_df, addon_df
        ))


def Build_Text_Output(simulation_metadata: list, summary: list, economic_parameters: list, engineering_parameters: list,
                      resource_characteristics: list, reservoir_parameters: list,
                      reservoir_stimulation_results: list, capex: list, opex: list,
                      surface_equipment_results: list, weather_results: list, tess_results: list,
                      dispatch_results: list, dispatch_profile_rows_table: list, sdac_results: list,
                      addon_results: list, hce: pd.DataFrame, ahce: pd.DataFrame, cashflow: pd.DataFrame,
                      pumping_power_profiles: pd.DataFrame, sdac_df: pd.DataFrame, addon_df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    buffer.write(f'                               *****************{NL}')
    buffer.write(f'                               ***CASE REPORT***{NL}')
    buffer.write(f'                               *****************{NL}')
    buffer.write(f'{NL}')

    buffer.write(f'Simulation Metadata{NL}')
    buffer.write(f'----------------------{NL}')
    for item in simulation_metadata:
        buffer.write(f'{item.parameter}: {item.value} {item.units}{NL}')

    Write_Simple_Text_Table('SUMMARY OF RESULTS', summary, buffer)
    Write_Simple_Text_Table('ECONOMIC PARAMETERS', economic_parameters, buffer)
    Write_Simple_Text_Table('ENGINEERING PARAMETERS', engineering_parameters, buffer)
    Write_Simple_Text_Table('RESOURCE CHARACTERISTICS', resource_characteristics, buffer)
    Write_Simple_Text_Table('RESERVOIR PARAMETERS', reservoir_parameters, buffer)
    if len(reservoir_stimulation_results) > 0:
        Write_Simple_Text_Table('RESERVOIR STIMULATION RESULTS', reservoir_stimulation_results, buffer)
    Write_Simple_Text_Table('CAPITAL COSTS', capex, buffer)
    Write_Simple_Text_Table('OPERATING AND MAINTENANCE COSTS', opex, buffer)
    Write_Simple_Text_Table('SURFACE EQUIPMENT SIMULATION RESULTS', surface_equipment_results, buffer)
    if len(weather_results) > 0:
        Write_Simple_Text_Table('WEATHER DATA RESULTS', weather_results, buffer)
    if len(tess_results) > 0:
        Write_Simple_Text_Table('THERMAL ENERGY STORAGE SYSTEM (TESS) RESULTS', tess_results, buffer)
    if len(dispatch_results) > 0:
        Write_Simple_Text_Table('DISPATCH RESULTS', dispatch_results, buffer)
    if len(addon_results) > 0:
        Write_Simple_Text_Table('ADD-ON ECONOMICS', addon_results, buffer)
    if len(sdac_results) > 0:
        Write_Simple_Text_Table('S_DAC_GT ECONOMICS', sdac_results, buffer)

    if len(hce) > 0:
        Write_Complex_Text_table('HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE', hce, 1, buffer)
    if len(ahce) > 0:
        Write_Complex_Text_table('ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE', ahce, 1, buffer)
    Write_Complex_Text_table('REVENUE & CASHFLOW PROFILE', cashflow, 1, buffer)
    if len(pumping_power_profiles) > 0:
        Write_Complex_Text_table('PUMPING POWER PROFILES', pumping_power_profiles, 1, buffer)
    if len(addon_df) > 0:
        Write_Complex_Text_table('ADD-ON PROFILE', addon_df, 1, buffer)
    if len(sdac_df) > 0:
        Write_Complex_Text_table('S_DAC_GT PROFILE', sdac_df, 1, buffer)
    if len(dispatch_profile_rows_table) > 0:
        Write_CSV_Text_Table(DISPATCH_PROFILE_CATEGORY_NAME, dispatch_profile_rows_table, buffer)

    return buffer.getvalue()


def _escape_rtf(text: str) -> str:
    return text.replace('\\', r'\\').replace('{', r'\{').replace('}', r'\}')


def Write_RTF_Output(output_path: str, simulation_metadata: list, summary: list, economic_parameters: list,
                     engineering_parameters: list, resource_characteristics: list, reservoir_parameters: list,
                     reservoir_stimulation_results: list, capex: list, opex: list, surface_equipment_results: list,
                     weather_results: list, tess_results: list, dispatch_results: list,
                     dispatch_profile_rows_table: list, sdac_results: list, addon_results: list,
                     hce: pd.DataFrame, ahce: pd.DataFrame, cashflow: pd.DataFrame,
                     pumping_power_profiles: pd.DataFrame, sdac_df: pd.DataFrame, addon_df: pd.DataFrame) -> None:
    plain_text = Build_Text_Output(
        simulation_metadata, summary, economic_parameters, engineering_parameters, resource_characteristics,
        reservoir_parameters, reservoir_stimulation_results, capex, opex, surface_equipment_results,
        weather_results, tess_results, dispatch_results, dispatch_profile_rows_table, sdac_results, addon_results,
        hce, ahce, cashflow, pumping_power_profiles, sdac_df, addon_df
    )

    with open(output_path, 'w', encoding='ASCII', errors='ignore') as f:
        f.write(r'{\rtf1\ansi\deff0{\fonttbl{\f0\fmodern Courier New;}}' + '\n')
        f.write(r'\viewkind4\uc1\pard\f0\fs20' + '\n')
        for line in plain_text.splitlines():
            f.write(f'{_escape_rtf(line)}\\par\n')
        f.write('}\n')


def Write_Simple_HTML_Table(title: str, items: list, console: Console) -> None:
    """
    This function writes out the simple tables as HTML. The console object is used to write out the HTML.
    :param title: the title of the table
    :type title: str
    :param items: the list of items to be written out
    :type items: list
    :param console: the console object
    :type console: rich.Console
    """
    table = Table(title=title)
    table.add_column('Parameter', style='bold', no_wrap=True, justify='center')
    table.add_column('Value', style='bold', no_wrap=True, justify='center')
    table.add_column('Units', style='bold', no_wrap=True, justify='center')
    for item in items:
        table.add_row(str(item.parameter), str(item.value), str(item.units))
    console.print(table)


def Write_Complex_HTML_Table(
        title: str,
        df_table: pd.DataFrame,
        time_steps_per_year: int,
        console: Console) -> None:
    """
    This function writes out the complex tables
    :param title: the title of the table
    :type title: str
    :param df_table: the dataframe to be written out
    :type df_table: pd.DataFrame
    :param time_steps_per_year: the number of time steps per year
    :type time_steps_per_year: int
    :param console: the console object
    :type console: rich.Console
    """
    table = Table(title=title)
    column_fmt = []
    for col in df_table.columns:
        if col != 'index':
            pair = col.split('|')
            column_name=pair[0]
            column_fmt.append(pair[1])
            table.add_column(UpgradeSymbologyOfUnits(column_name), style='bold', no_wrap=True, justify='center')
    for index, row in df_table.iterrows():
        # only print the number of rows implied by time_steps_per_year
        if time_steps_per_year == 0 or int(index) % time_steps_per_year == 0:
            table.add_row(*[render_default((df_table.at[index, row.index[i]]) / time_steps_per_year, '', column_fmt[i - 1]) for i in range(1, len(row))])
    console.print(table)


def _new_html_console() -> Console:
    return Console(
        style='bold black on white',
        force_terminal=True,
        record=True,
        width=500,
        file=io.StringIO(),
    )


def _write_html_case_report_header(console: Console, simulation_metadata: list) -> None:
    console.print('*****************')
    console.print('***CASE REPORT***')
    console.print('*****************')
    console.print('Simulation Metadata')
    console.print('----------------------')

    for item in simulation_metadata:
        console.print(f'{str(item.parameter)}: {str(item.value)} {str(item.units)}')


def _simple_html_sections(
        summary: list,
        economic_parameters: list,
        engineering_parameters: list,
        resource_characteristics: list,
        reservoir_parameters: list,
        reservoir_stimulation_results: list,
        capex: list,
        opex: list,
        surface_equipment_results: list,
        weather_results: list,
        tess_results: list,
        dispatch_results: list,
        addon_results: list,
        sdac_results: list) -> list[tuple[str, list, bool]]:
    return [
        ('SUMMARY OF RESULTS', summary, False),
        ('ECONOMIC PARAMETERS', economic_parameters, False),
        ('ENGINEERING PARAMETERS', engineering_parameters, False),
        ('RESOURCE CHARACTERISTICS', resource_characteristics, False),
        ('RESERVOIR PARAMETERS', reservoir_parameters, False),
        ('RESERVOIR STIMULATION RESULTS', reservoir_stimulation_results, True),
        ('CAPITAL COSTS', capex, False),
        ('OPERATING AND MAINTENANCE COSTS', opex, False),
        ('SURFACE EQUIPMENT SIMULATION RESULTS', surface_equipment_results, False),
        ('WEATHER DATA RESULTS', weather_results, True),
        ('THERMAL ENERGY STORAGE SYSTEM (TESS) RESULTS', tess_results, True),
        ('DISPATCH RESULTS', dispatch_results, True),
        ('ADD-ON ECONOMICS', addon_results, True),
        ('S_DAC_GT ECONOMICS', sdac_results, True),
    ]


def _write_html_simple_sections(console: Console, sections: list[tuple[str, list, bool]]) -> None:
    for title, items, skip_empty in sections:
        if skip_empty and len(items) == 0:
            continue
        Write_Simple_HTML_Table(title, items, console)


def _complex_html_sections(
        hce: pd.DataFrame,
        ahce: pd.DataFrame,
        cashflow: pd.DataFrame,
        pumping_power_profiles: pd.DataFrame,
        addon_df: pd.DataFrame,
        sdac_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame, int, bool]]:
    return [
        ('HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE', hce, 1, True),
        ('ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE', ahce, 1, True),
        ('REVENUE & CASHFLOW PROFILE', cashflow, 1, False),
        ('PUMPING POWER PROFILES', pumping_power_profiles, 1, True),
        ('ADD-ON PROFILE', addon_df, 1, True),
        ('S_DAC_GT PROFILE', sdac_df, 1, True),
    ]


def _write_html_complex_sections(console: Console, sections: list[tuple[str, pd.DataFrame, int, bool]]) -> None:
    for title, table, time_steps_per_year, skip_empty in sections:
        if skip_empty and len(table) == 0:
            continue
        Write_Complex_HTML_Table(title, table, time_steps_per_year, console)


def _write_html_dispatch_profile_section(
        console: Console,
        dispatch_profile_rows_table: list[list]) -> None:
    if len(dispatch_profile_rows_table) == 0:
        return
    Write_CSV_HTML_Table(DISPATCH_PROFILE_CATEGORY_NAME, dispatch_profile_rows_table, console)


def Write_HTML_Output(html_path: Optional[str], simulation_metadata: list, summary: list, economic_parameters: list,
                      engineering_parameters: list, resource_characteristics: list, reservoir_parameters: list,
                      reservoir_stimulation_results: list, capex: list, opex: list, surface_equipment_results: list,
                      weather_results: list, tess_results: list, dispatch_results: list,
                      dispatch_profile_rows_table: list, sdac_results: list, addon_results: list,
                      hce: pd.DataFrame, ahce: pd.DataFrame, cashflow: pd.DataFrame,
                      pumping_power_profiles: pd.DataFrame, sdac_df: pd.DataFrame, addon_df: pd.DataFrame) -> None:
    """
    This function writes out the HTML output
    :param html_path: the path to the HTML output file.
    :type html_path: str | None
    :param simulation_metadata: the simulation metadata
    :type simulation_metadata: list
    :param summary: the summary of results
    :type summary: list
    :param economic_parameters: the economic parameters
    :type economic_parameters: list
    :param engineering_parameters: the engineering parameters
    :type engineering_parameters: list
    :param resource_characteristics: the resource characteristics
    :type resource_characteristics: list
    :param reservoir_parameters: the reservoir parameters
    :type reservoir_parameters: list
    :param reservoir_stimulation_results: the reservoir stimulation results
    :type reservoir_stimulation_results: list
    :param capex: the capital costs
    :type capex: list
    :param opex: the operating and maintenance costs
    :type opex: list
    :param surface_equipment_results: the surface equipment simulation results
    :type surface_equipment_results: list
    :param dispatch_results: the dispatch results
    :type dispatch_results: list
    :param sdac_results: the sdac results
    :type sdac_results: list
    :param addon_results: the addon results
    :type addon_results: list
    :param hce: the heating, cooling and/or electricity production profile
    :type hce: pd.DataFrame
    :param ahce: the annual heating, cooling and/or electricity production profile
    :type ahce: pd.DataFrame
    :param cashflow: the revenue & cashflow profile
    :type cashflow: pd.DataFrame
    :param pumping_power_profiles: the pumping power profiles
    :type pd.DataFrame
    :param sdac_df: the sdac dataframe
    :type sdac_df: pd.DataFrame
    :param addon_df: the addon dataframe

    """

    console = _new_html_console()
    _write_html_case_report_header(console, simulation_metadata)
    _write_html_simple_sections(
        console,
        _simple_html_sections(
            summary,
            economic_parameters,
            engineering_parameters,
            resource_characteristics,
            reservoir_parameters,
            reservoir_stimulation_results,
            capex,
            opex,
            surface_equipment_results,
            weather_results,
            tess_results,
            dispatch_results,
            addon_results,
            sdac_results,
        )
    )
    _write_html_complex_sections(
        console,
        _complex_html_sections(hce, ahce, cashflow, pumping_power_profiles, addon_df, sdac_df),
    )
    _write_html_dispatch_profile_section(console, dispatch_profile_rows_table)

    if html_path is not None:
        console.save_html(html_path)


def profile_title_adjusted_for_figure(title: str) -> str:
    return title.replace('PROFILE: ', 'PROFILE:\n').replace('PROFILES: ', 'PROFILES:\n')


def _graph_file_stem(title: str, filename_prefix: Optional[str] = None) -> str:
    title_stem = remove_disallowed_filename_chars(title.replace(' ', '_'))
    if filename_prefix in [None, '']:
        return title_stem
    return f'{filename_prefix}_{title_stem}'


def Plot_Twin_Graph(title: str, html_path: str, x: pd.array, y1: pd.array, y2: pd.array,
                    x_label: str, y1_label: str, y2_label: str, filename_prefix: Optional[str] = None) -> None:
    """
    This function plots the twin graph
    :param title: the title of the graph
    :type title: str
    :param html_path: the path to the HTML output file
    :type html_path: str
    :param x: the x values
    :type x: pd.array
    :param y1: the y1 values
    :type y1: pd.array
    :param y2: the y2 values
    :type y2: pd.array
    :param x_label: the x label
    :type x_label: str
    :param y1_label: the y1 label
    :type y1_label: str
    :param y2_label: the y2 label
    :type y2_label: str
    """
    COLOR_TEMPERATURE = "#69b3a2"
    COLOR_PRICE = "#3399e6"

    fig, ax1 = plt_subplots(figsize=_GRAPH_FIGSIZE)

    ax1.plot(x, y1, label=UpgradeSymbologyOfUnits(y1_label), color=COLOR_PRICE, lw=3)
    ax1.set_xlabel(UpgradeSymbologyOfUnits(x_label), color=COLOR_PRICE, fontsize=14)
    ax1.set_ylabel(UpgradeSymbologyOfUnits(y1_label), color=COLOR_PRICE, fontsize=14)
    ax1.tick_params(axis="y", labelcolor=COLOR_PRICE)
    _set_plot_xlim(ax1, x)
    ax1.legend(loc='lower left')

    ax2 = ax1.twinx()
    ax2.plot(x, y2, label=UpgradeSymbologyOfUnits(y2_label), color=COLOR_TEMPERATURE, lw=4)
    ax2.set_ylabel(UpgradeSymbologyOfUnits(y2_label), color=COLOR_TEMPERATURE, fontsize=14)
    ax2.tick_params(axis="y", labelcolor=COLOR_TEMPERATURE)
    ax2.legend(loc='best')

    fig.suptitle(profile_title_adjusted_for_figure(title), fontsize=20)

    full_names: set = set()
    short_names: set = set()
    file_stem = _graph_file_stem(title, filename_prefix)
    save_path = Path(Path(html_path).parent, f'{file_stem}.png')
    plt.savefig(save_path)
    short_names.add(file_stem)
    full_names.add(save_path)
    plt.close(fig)

    InsertImagesIntoHTML(html_path, short_names, full_names)


def Plot_Single_Graph(title: str, html_path: str, x: pd.array, y: pd.array, x_label: str, y_label: str,
                      filename_prefix: Optional[str] = None) -> None:
    """
    This function plots the single graph
    :param title: the title of the graph
    :type title: str
    :param html_path: the path to the HTML output file
    :type html_path: str
    :param x: the x values
    :type x: pd.array
    :param y: the y values
    :type y: pd.array
    :param x_label: the x label
    :type x_label: str
    :param y_label: the y label
    :type y_label: str
    """
    COLOR_PRICE = "#3399e6"

#    plt.plot(x, y, color=COLOR_PRICE)
    fig, ax = plt_subplots(figsize=_GRAPH_FIGSIZE)
    ax.plot(x, y, label=UpgradeSymbologyOfUnits(y_label), color=COLOR_PRICE)
    ax.set_xlabel(UpgradeSymbologyOfUnits(x_label), color=COLOR_PRICE, fontsize=14)
    ax.set_ylabel(UpgradeSymbologyOfUnits(y_label), color=COLOR_PRICE, fontsize=14)
    ax.tick_params(axis="y", labelcolor=COLOR_PRICE)
    _set_plot_xlim(ax, x)
    ax.legend(loc='best')
    #plt.ylim(y.min(), y.max())
    #plt.gca().legend((UpgradeSymbologyOfUnits(x_label), UpgradeSymbologyOfUnits(y_label)), loc='best')
    fig.suptitle(profile_title_adjusted_for_figure(title), fontsize=20)

    full_names: set = set()
    short_names: set = set()
    file_stem = _graph_file_stem(title, filename_prefix)
    save_path = Path(Path(html_path).parent, f'{file_stem}.png')
    plt.savefig(save_path)
    short_names.add(file_stem)
    full_names.add(save_path)
    plt.close(fig)

    InsertImagesIntoHTML(html_path, short_names, full_names)


def Plot_Multi_Graph(title: str, html_path: str, x: pd.array, ys: list[pd.array], x_label: str,
                     y_label: str, series_labels: list[str], filename_prefix: Optional[str] = None) -> None:
    """
    Plot multiple series sharing the same y-axis.
    """
    colors = ["#3399e6", "#69b3a2", "#c04b37", "#d4a017"]
    fig, ax = plt_subplots(figsize=_GRAPH_FIGSIZE)

    for idx, y in enumerate(ys):
        ax.plot(x, y, label=UpgradeSymbologyOfUnits(series_labels[idx]), color=colors[idx % len(colors)], lw=2.5)

    ax.set_xlabel(UpgradeSymbologyOfUnits(x_label), color=colors[0], fontsize=14)
    ax.set_ylabel(UpgradeSymbologyOfUnits(y_label), color=colors[0], fontsize=14)
    ax.tick_params(axis="y", labelcolor=colors[0])
    _set_plot_xlim(ax, x)
    ax.legend(loc='best')
    fig.suptitle(profile_title_adjusted_for_figure(title), fontsize=20)

    full_names: set = set()
    short_names: set = set()
    file_stem = _graph_file_stem(title, filename_prefix)
    save_path = Path(Path(html_path).parent, f'{file_stem}.png')
    plt.savefig(save_path)
    short_names.add(file_stem)
    full_names.add(save_path)
    plt.close(fig)

    InsertImagesIntoHTML(html_path, short_names, full_names)


def Plot_Dispatch_Graphs_Into_HTML(model: Model, html_path: str) -> None:
    dispatch_results = getattr(model, 'dispatch_results', None)
    if dispatch_results is None:
        return

    hours = np.arange(1, len(dispatch_results.hourly_thermal_demand) + 1, dtype=float)
    if hours.size == 0:
        return

    demand_type = getattr(dispatch_results, 'demand_type', 'thermal')
    filename_prefix = remove_disallowed_filename_chars(Path(html_path).stem)
    if demand_type == 'electric':
        profile_title = 'DISPATCH PROFILE: Demand, Served, and Unmet Electricity'
        y_label = 'Electric Power (MW)'
        legend = ['Electricity Demand (MW)', 'Demand Served (MW)', 'Unmet Demand (MW)']
        geothermal_output = dispatch_results.hourly_geothermal_electric_output
    elif demand_type == 'cooling':
        profile_title = 'DISPATCH PROFILE: Demand, Served, and Unmet Cooling'
        y_label = 'Cooling Power (MW)'
        legend = ['Cooling Demand (MW)', 'Demand Served (MW)', 'Unmet Demand (MW)']
        geothermal_output = dispatch_results.hourly_cooling_output
    else:
        profile_title = 'DISPATCH PROFILE: Demand, Served, and Unmet Heat'
        y_label = 'Thermal Power (MW)'
        legend = ['Thermal Demand (MW)', 'Demand Served (MW)', 'Unmet Demand (MW)']
        geothermal_output = dispatch_results.hourly_geothermal_thermal_output

    Plot_Multi_Graph(
        profile_title,
        html_path,
        hours,
        [
            dispatch_results.hourly_thermal_demand,
            dispatch_results.hourly_demand_served / 1000.0,
            dispatch_results.hourly_unmet_demand / 1000.0,
        ],
        'Simulation Hour',
        y_label,
        legend,
        filename_prefix=filename_prefix,
    )
    Plot_Twin_Graph(
        'DISPATCH PROFILE: Produced Temperature and Flow Rate',
        html_path,
        hours,
        dispatch_results.hourly_produced_temperature,
        dispatch_results.hourly_flow,
        'Simulation Hour',
        'Produced Temperature (degC)',
        'Flow Rate (kg/s)',
        filename_prefix=filename_prefix,
    )
    Plot_Twin_Graph(
        'DISPATCH PROFILE: Runtime Fraction and Electric Output' if demand_type == 'electric'
        else ('DISPATCH PROFILE: Runtime Fraction and Cooling Output' if demand_type == 'cooling'
              else 'DISPATCH PROFILE: Runtime Fraction and Pumping Power'),
        html_path,
        hours,
        dispatch_results.hourly_runtime_fraction,
        geothermal_output if demand_type in ['electric', 'cooling'] else dispatch_results.hourly_pumping_power,
        'Simulation Hour',
        'Runtime Fraction',
        'Geothermal Electric Output (MW)' if demand_type == 'electric'
        else ('Cooling Output (MW)' if demand_type == 'cooling' else 'Pumping Power (MW)'),
        filename_prefix=filename_prefix,
    )
    if dispatch_results.summary_metrics.get('tess_enabled', 0.0) > 0.0:
        Plot_Twin_Graph(
            'DISPATCH PROFILE: TESS Temperature and SOC',
            html_path,
            hours,
            dispatch_results.hourly_tess_temperature,
            dispatch_results.hourly_tess_soc * 100.0,
            'Simulation Hour',
            'TESS Temperature (degC)',
            'TESS State of Charge (%)',
            filename_prefix=filename_prefix,
        )
        Plot_Multi_Graph(
            'DISPATCH PROFILE: Demand, TESS Discharge, and Geothermal Charge',
            html_path,
            hours,
            [
                dispatch_results.hourly_thermal_demand,
                dispatch_results.hourly_tess_discharge_to_load,
                dispatch_results.hourly_tess_charge_from_geothermal,
            ],
            'Simulation Hour',
            'Thermal Power (MW)',
            ['Customer Demand (MW)', 'TESS Discharge (MW)', 'Geothermal Charge (MW)'],
            filename_prefix=filename_prefix,
        )
        Plot_Multi_Graph(
            'DISPATCH PROFILE: TESS Losses and Curtailment',
            html_path,
            hours,
            [
                dispatch_results.hourly_tess_standby_loss,
                dispatch_results.hourly_tess_efficiency_loss,
                dispatch_results.hourly_tess_charge_curtailed,
            ],
            'Simulation Hour',
            'Thermal Power (MW)',
            ['Standby Loss (MW)', 'Efficiency Loss (MW)', 'Curtailed Heat (MW)'],
            filename_prefix=filename_prefix,
        )


def Plot_Tables_Into_HTML(enduse_option: intParameter, plant_type: intParameter, html_path: str,
                          hce: pd.DataFrame, ahce: pd.DataFrame, cashflow: pd.DataFrame, pumping_power_profiles: pd.DataFrame,
                          sdac_df: pd.DataFrame, addon_df: pd.DataFrame) -> None:
    """
    This function plots the tables into the HTML
    :param enduse_option: the end use option
    :type enduse_option: intParameter
    :param html_path: the path to the HTML output file
    :type html_path: str
    :param plant_type: the plant type
    :type plant_type: intParameter
    :param hce: the heating, cooling and/or electricity production profile
    :type hce: pd.DataFrame
    :param ahce: the annual heating, cooling and/or electricity production profile
    :type ahce: pd.DataFrame
    :param cashflow: the revenue & cashflow profile
    :type cashflow: pd.DataFrame
    :param pumping_power_profiles: The pumping power profiles
    :type pd.DataFrame
    :param sdac_df: the sdac dataframe
    :type sdac_df: pd.DataFrame
    :param addon_df: the addon dataframe
    :type addon_df: pd.DataFrame
    """

    # HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILES
    # Plot the three that appear for all end uses
    Plot_Single_Graph('HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILES: Thermal Drawdown',
                      html_path, hce.values[0:, 0], hce.values[0:, 1], hce.columns[0].split('|')[0], hce.columns[1].split('|')[0])
    Plot_Single_Graph('HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILES: Geofluid Temperature',
                      html_path, hce.values[0:, 0], hce.values[0:, 2], hce.columns[0].split('|')[0], hce.columns[2].split('|')[0])
    Plot_Single_Graph('HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILES: Pump Power',
                      html_path, hce.values[0:, 0], hce.values[0:, 3], hce.columns[0].split('|')[0], hce.columns[3].split('|')[0])
    if enduse_option.value == EndUseOptions.ELECTRICITY:
        # only electricity
        Plot_Single_Graph('HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILES: First Law Efficiency',
                        html_path, hce.values[0:, 0], hce.values[0:, 5], hce.columns[0].split('|')[0], hce.columns[5].split('|')[0])
        Plot_Single_Graph('HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILES: Net Power',
                        html_path, hce.values[0:, 0], hce.values[0:, 4], hce.columns[0].split('|')[0], hce.columns[4].split('|')[0])
    elif enduse_option.value == EndUseOptions.HEAT and plant_type.value not in [PlantType.HEAT_PUMP, PlantType.DISTRICT_HEATING, PlantType.ABSORPTION_CHILLER]:
        # only direct-use
        Plot_Single_Graph('HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILES: Net Heat',
                        html_path, hce.values[0:, 0], hce.values[0:, 4], hce.columns[0].split('|')[0], hce.columns[4].split('|')[0])
    elif enduse_option.value == EndUseOptions.HEAT and plant_type.value == PlantType.HEAT_PUMP:
        # heat pump
        Plot_Twin_Graph('HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILES: Net Heat & Heat Pump Electricity Use',
                        html_path, hce.values[0:, 0], hce.values[0:, 4], hce.values[0:, 5],
                        hce.columns[0].split('|')[0], hce.columns[4].split('|')[0], hce.columns[5].split('|')[0])
    elif enduse_option.value == EndUseOptions.HEAT and plant_type.value == PlantType.DISTRICT_HEATING:
        # district heating
        Plot_Single_Graph('HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILES: Geothermal Heat Output',
                        html_path, hce.values[0:, 0], hce.values[0:, 4], hce.columns[0].split('|')[0], hce.columns[4].split('|')[0])
    elif enduse_option.value == EndUseOptions.HEAT and plant_type.value == PlantType.ABSORPTION_CHILLER:
        # absorption chiller
        Plot_Twin_Graph('HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILES: Net Heat & Net Cooling',
                        html_path, hce.values[0:, 0], hce.values[0:, 4], hce.values[0:, 5],
                        hce.columns[0].split('|')[0], hce.columns[4].split('|')[0], hce.columns[5].split('|')[0])
    elif enduse_option.value in [EndUseOptions.COGENERATION_TOPPING_EXTRA_HEAT, EndUseOptions.COGENERATION_TOPPING_EXTRA_ELECTRICITY,
                                EndUseOptions.COGENERATION_BOTTOMING_EXTRA_ELECTRICITY, EndUseOptions.COGENERATION_BOTTOMING_EXTRA_HEAT,
                                EndUseOptions.COGENERATION_PARALLEL_EXTRA_HEAT, EndUseOptions.COGENERATION_PARALLEL_EXTRA_ELECTRICITY]:
        # co-gen
        Plot_Twin_Graph('HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILES: Net Power & Net Heat',
                        html_path, hce.values[0:, 0], hce.values[0:, 4], hce.values[0:, 5],
                        hce.columns[0].split('|')[0], hce.columns[4].split('|')[0], hce.columns[5].split('|')[0])
        Plot_Single_Graph('HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILES: First Law Efficiency',
                        html_path, hce.values[0:, 0], hce.values[0:, 6], hce.columns[0].split('|')[0], hce.columns[6].split('|')[0])

    # ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE
    # plot the common graphs
    Plot_Twin_Graph('ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE: Heat Extracted & Reservoir Heat Content',
                    html_path, ahce.values[0:, 0], ahce.values[0:, 2], ahce.values[0:, 3],
                    ahce.columns[0].split('|')[0], ahce.columns[2].split('|')[0], ahce.columns[3].split('|')[0])
    if plant_type.value in [PlantType.DISTRICT_HEATING]:
        # columns are in a different place for district heating
        Plot_Single_Graph('ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE: Percentage of Total Heat Mined',
                          html_path, ahce.values[0:, 0], ahce.values[0:, 5], ahce.columns[0].split('|')[0], ahce.columns[5].split('|')[0])
    else:
        Plot_Single_Graph('ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE: Percentage of Total Heat Mined',
                          html_path, ahce.values[0:, 0], ahce.values[0:, 4], ahce.columns[0].split('|')[0], ahce.columns[4].split('|')[0])

    if enduse_option.value == EndUseOptions.ELECTRICITY:
        # only electricity
        Plot_Single_Graph('ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE: Electricity Provided',
                          html_path, ahce.values[0:, 0], ahce.values[0:, 1], ahce.columns[0].split('|')[0], ahce.columns[1].split('|')[0])
    elif plant_type.value == PlantType.ABSORPTION_CHILLER:
        # absorption chiller
        Plot_Single_Graph('ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE: Cooling Provided',
                          html_path, ahce.values[0:, 0], ahce.values[0:, 1], ahce.columns[0].split('|')[0], ahce.columns[1].split('|')[0])
    elif plant_type.value in [PlantType.DISTRICT_HEATING]:
        # district-heating
        Plot_Twin_Graph('ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE: Geothermal Heating Provided & Peaking Boiler Heating Provided',
                        html_path, ahce.values[0:, 0], ahce.values[0:, 1], ahce.values[0:, 2],
                        ahce.columns[0].split('|')[0], ahce.columns[1].split('|')[0], ahce.columns[2].split('|')[0])
    elif plant_type.value == PlantType.HEAT_PUMP:
        # heat pump
        Plot_Single_Graph('ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE: Heating Provided',
                          html_path, ahce.values[0:, 0], ahce.values[0:, 1], ahce.columns[0].split('|')[0], ahce.columns[1].split('|')[0])
        Plot_Single_Graph('ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE: Heat Pump Electricity Use',
                          html_path, ahce.values[0:, 0], ahce.values[0:, 3], ahce.columns[0].split('|')[0], ahce.columns[3].split('|')[0])
    elif enduse_option.value in [EndUseOptions.COGENERATION_TOPPING_EXTRA_HEAT, EndUseOptions.COGENERATION_TOPPING_EXTRA_ELECTRICITY,
                                                    EndUseOptions.COGENERATION_BOTTOMING_EXTRA_ELECTRICITY, EndUseOptions.COGENERATION_BOTTOMING_EXTRA_HEAT,
                                                    EndUseOptions.COGENERATION_PARALLEL_EXTRA_HEAT, EndUseOptions.COGENERATION_PARALLEL_EXTRA_ELECTRICITY]:
        # co-gen
        Plot_Twin_Graph('ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE: Heat Provided & Electricity Provided',
                        html_path, ahce.values[0:, 0], ahce.values[0:, 1], ahce.values[0:, 2],
                        ahce.columns[0].split('|')[0], ahce.columns[1].split('|')[0], ahce.columns[2].split('|')[0])
    elif enduse_option.value == EndUseOptions.HEAT:
        # only direct-use
        Plot_Single_Graph('ANNUAL HEATING, COOLING AND/OR ELECTRICITY PRODUCTION PROFILE: Heat Provided',
                          html_path, ahce.values[0:, 0], ahce.values[0:, 1], ahce.columns[0].split('|')[0], ahce.columns[1].split('|')[0])

    # Cashflow Graphs
    Plot_Twin_Graph('REVENUE & CASHFLOW PROFILE: Electricity: Price & Cumulative Revenue',
                    html_path, cashflow.values[0:, 0], cashflow.values[0:, 1], cashflow.values[0:, 3],
                    cashflow.columns[0].split('|')[0], cashflow.columns[1].split('|')[0], cashflow.columns[3].split('|')[0])
    Plot_Twin_Graph('REVENUE & CASHFLOW PROFILE: Heat: Price & Cumulative Revenue',
                    html_path, cashflow.values[0:, 0], cashflow.values[0:, 4], cashflow.values[0:, 6],
                    cashflow.columns[0].split('|')[0], cashflow.columns[4].split('|')[0], cashflow.columns[6].split('|')[0])
    Plot_Twin_Graph('REVENUE & CASHFLOW PROFILE: Cooling: Price & Cumulative Revenue',
                    html_path, cashflow.values[0:, 0], cashflow.values[0:, 7], cashflow.values[0:, 9],
                    cashflow.columns[0].split('|')[0], cashflow.columns[7].split('|')[0], cashflow.columns[9].split('|')[0])
    Plot_Twin_Graph('REVENUE & CASHFLOW PROFILE: Carbon: Price & Cumulative Revenue',
                    html_path, cashflow.values[0:, 0], cashflow.values[0:, 10], cashflow.values[0:, 12],
                    cashflow.columns[0].split('|')[0], cashflow.columns[10].split('|')[0], cashflow.columns[12].split('|')[0])
    Plot_Twin_Graph('REVENUE & CASHFLOW PROFILE: Project: Net Revenue and cashflow',
                    html_path, cashflow.values[0:, 0], cashflow.values[0:, 14], cashflow.values[0:, 15],
                    cashflow.columns[0].split('|')[0], cashflow.columns[14].split('|')[0], cashflow.columns[15].split('|')[0])

    # Pumping Power Profiles Graphs
    if len(pumping_power_profiles) > 0:
        Plot_Twin_Graph('PUMPING POWER PROFILES: Production Pumping Power & Injection Pumping Power', html_path,
                        pumping_power_profiles.values[0:, 0], pumping_power_profiles.values[0:, 1], pumping_power_profiles.values[0:, 2],
                        pumping_power_profiles.columns[0].split('|')[0], pumping_power_profiles.columns[1].split('|')[0], pumping_power_profiles.columns[2].split('|')[0])
        Plot_Single_Graph('PUMPING POWER PROFILES: Pumping Power', html_path,
                            pumping_power_profiles.values[0:, 0], pumping_power_profiles.values[0:, 3], pumping_power_profiles.columns[0].split('|')[0],
                            pumping_power_profiles.columns[3].split('|')[0])

    if len(addon_df) > 0:
        Plot_Twin_Graph('ADD-ON PROFILE: Electricity Annual Price vs. Revenue',
                        html_path, addon_df.values[0:, 0], addon_df.values[0:, 1], addon_df.values[0:, 2],
                        addon_df.columns[0].split('|')[0], addon_df.columns[1].split('|')[0], addon_df.columns[2].split('|')[0])
        Plot_Twin_Graph('ADD-ON PROFILE: Heat Annual Price vs. Revenue',
                        html_path, addon_df.values[0:, 0], addon_df.values[0:, 3], addon_df.values[0:, 4],
                        addon_df.columns[0].split('|')[0], addon_df.columns[3].split('|')[0], addon_df.columns[4].split('|')[0])
        Plot_Twin_Graph('ADD-ON PROFILE: Add-On Net Revenue & Annual Cashflow',
                        html_path, addon_df.values[0:, 0], addon_df.values[0:, 5], addon_df.values[0:, 6],
                        addon_df.columns[0].split('|')[0], addon_df.columns[5].split('|')[0], addon_df.columns[6].split('|')[0])
        Plot_Single_Graph('ADD-ON PROFILE: Add-On Cumulative Cashflow',
                        html_path, addon_df.values[0:, 0], addon_df.values[0:, 7],  addon_df.columns[0].split('|')[0],
                          addon_df.columns[7].split('|')[0])
        Plot_Twin_Graph('ADD-ON PROFILE: Project Cashflow vs. Cumulative Cashflow',
                        html_path, addon_df.values[0:, 0], addon_df.values[0:, 8], addon_df.values[0:, 9],
                        addon_df.columns[0].split('|')[0], addon_df.columns[8].split('|')[0], addon_df.columns[9].split('|')[0])
    if len(sdac_df) > 0:
        Plot_Twin_Graph('S_DAC_GT PROFILE: Annual vs Cumulative Carbon Captured',
                        html_path, sdac_df.values[0:, 0], sdac_df.values[0:, 1], sdac_df.values[0:, 2],
                        sdac_df.columns[0].split('|')[0], sdac_df.columns[1].split('|')[0], sdac_df.columns[2].split('|')[0])
        Plot_Twin_Graph('S_DAC_GT PROFILE: Annual Cost vs Cumulative Cost',
                        html_path, sdac_df.values[0:, 0], sdac_df.values[0:, 3], sdac_df.values[0:, 4],
                        sdac_df.columns[0].split('|')[0], sdac_df.columns[3].split('|')[0], sdac_df.columns[4].split('|')[0])
        Plot_Single_Graph('S_DAC_GT PROFILE: Cumulative Capture Cost per Tonne',
                        html_path, sdac_df.values[0:, 0], sdac_df.values[0:, 5], sdac_df.columns[0].split('|')[0],
                          sdac_df.columns[5].split('|')[0])


def MakeDistrictHeatingPlot(html_path: str, dh_geothermal_heating: pd.array, daily_heating_demand: pd.array) -> None:
    """"
    Make a plot of the district heating system
    :param html_path: the path to the HTML output file
    :type html_path: str
    :param dh_geothermal_heating: the geothermal heating
    :type dh_geothermal_heating: pd.array
    :param daily_heating_demand: the daily heating demand
    :type daily_heating_demand: pd.array
    """
    plt.close('all')
    year_day = np.arange(1, 366, 1)  # make an array of days for plot x-axis
    plt.plot(year_day, daily_heating_demand, label='District Heating Demand')
    plt.fill_between(year_day, 0, dh_geothermal_heating[0:365] * 24, color='g', alpha=0.5,
                     label='Geothermal Heat Supply')
    plt.fill_between(year_day, dh_geothermal_heating[0:365] * 24,
                     daily_heating_demand, color='r', alpha=0.5,
                     label='Natural Gas Heat Supply')
    plt.xlabel('Ordinal Day')
    plt.ylabel('Heating Demand/Supply [MWh/day]')
    plt.ylim([0, max(daily_heating_demand) * 1.05])
    plt.legend()
    plt.title('Geothermal district heating system with peaking boilers')
    full_names: set = set()
    short_names: set = set()
    title = remove_disallowed_filename_chars('Geothermal district heating system with peaking boilers'.replace(' ', '_'))
    save_path = Path(Path(html_path).parent, f'{title}.png')
    plt.savefig(save_path)
    short_names.add(title)
    full_names.add(save_path)
    plt.close()

    InsertImagesIntoHTML(html_path, short_names, full_names)


