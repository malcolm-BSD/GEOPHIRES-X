from __future__ import annotations

import datetime
import time
import sys
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import pandas as pd

import geophires_x
from geophires_x.DispatchReporting import DISPATCH_PROFILE_CATEGORY_NAME, is_dispatch_report, weather_output_rows
import geophires_x.Model as Model
from geophires_x.Economics import Economics
from geophires_x.OutputsCashFlow import get_sam_cash_flow_profile_output, write_revenue_and_cashflow_profile_output
from geophires_x.OutputsDispatch import dispatch_output_rows
from geophires_x.OutputsDispatch import dispatch_profile_tess_columns
from geophires_x.OutputsDispatch import dispatch_profile_tess_row
from geophires_x.OutputsDispatch import tess_output_rows
from geophires_x.OutputsDispatch import write_dispatch_profile_output
from geophires_x.OutputsDispatch import write_dispatch_profile_report_table
from geophires_x.OutputsEconomics import (
    write_capital_costs,
    write_economic_parameters,
    write_operation_and_maintenance_costs,
)
from geophires_x.OutputsEngineering import write_engineering_parameters
from geophires_x.OutputsProfiles import write_annual_production_profile, write_production_profile
from geophires_x.OutputsReport import write_scalar_section
from geophires_x.OutputsResource import write_resource_characteristics
from geophires_x.OutputsReservoir import (
    write_reservoir_parameters,
    write_reservoir_power_required_profiles,
    write_reservoir_simulation_results,
)
from geophires_x.OutputsRich import print_outputs_rich
from geophires_x.OutputsSurface import write_surface_equipment_simulation_results
from geophires_x.OutputsSummary import write_summary_of_results
from geophires_x.Parameter import ConvertUnitsBack, ConvertOutputUnits, LookupUnits, strParameter, boolParameter, \
    OutputParameter, ReadParameter, ParameterEntry
from geophires_x.OptionList import EndUseOptions, EconomicModel, OperatingMode, PlantType
from geophires_x.Parameter import Parameter
from geophires_x.Units import EnergyUnit

NL = '\n'


def _resolve_output_path(output_path: str, model: Model) -> str:
    path = Path(output_path).expanduser()
    if path.is_absolute():
        return str(path)

    candidate_bases = [Path.cwd(), Path(__file__).resolve().parents[2]]
    input_file_path = getattr(model, 'input_file_path', None)
    if input_file_path is not None:
        candidate_bases.append(Path(input_file_path).resolve().parent)

    parent = path.parent
    for base in candidate_bases:
        candidate = base / path
        if parent == Path('.') or candidate.parent.exists():
            return str(candidate.resolve())

    return str((Path.cwd() / path).resolve())


class Outputs:
    """
    This class handles all the outputs for the GEOPHIRESv3 model.
    """

    VERTICAL_WELL_DEPTH_OUTPUT_NAME = 'Well depth'
    WEATHER_DATA_RESULTS_CATEGORY_NAME = 'WEATHER DATA RESULTS'
    TESS_RESULTS_CATEGORY_NAME = 'THERMAL ENERGY STORAGE SYSTEM (TESS) RESULTS'
    DISPATCH_RESULTS_CATEGORY_NAME = 'DISPATCH RESULTS'
    DISPATCH_PROFILE_CATEGORY_NAME = DISPATCH_PROFILE_CATEGORY_NAME
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
    SURFACE_HEAT_RESULT_OPTIONS = (
        EndUseOptions.HEAT,
        PlantType.ABSORPTION_CHILLER,
        PlantType.HEAT_PUMP,
        *COGENERATION_END_USE_OPTIONS,
    )
    SPECIAL_HEAT_PLANT_TYPES = (
        PlantType.ABSORPTION_CHILLER,
        PlantType.HEAT_PUMP,
        PlantType.DISTRICT_HEATING,
    )

    def __init__(self, model: Model, output_file: str = 'HDR.out'):
        model.logger.info(f'Init {__class__!s}: {__name__}')

        self.output_file = output_file

        self.ParameterDict = {}
        self.OutputParameterDict = {}
        self.filepath_parameter_names = []

        def filepath_parameter(p: Parameter) -> Parameter:
            self.filepath_parameter_names.append(p.Name)
            return p

        self.text_output_file = self.ParameterDict[self.text_output_file.Name] = filepath_parameter(strParameter(
                'Improved Text Output File',
                DefaultValue='GEOPHIRES_Text.rtf',
                Required=False,
                Provided=False,
                ErrMessage='assume no rich text output',
                ToolTipText='Provide any value to enable rich text output written next to the main .out file as .rtf',
        ))

        self.html_output_file = self.ParameterDict[self.html_output_file.Name] = filepath_parameter(strParameter(
                'HTML Output File',
                DefaultValue='GEOPHIRES.html',
                Required=False,
                Provided=False,
                ErrMessage='assume no HTML output',
                ToolTipText='Provide any value to enable HTML output written next to the main .out file as .html',
        ))

        self.dispatch_profile_output_file = self.ParameterDict[
            self.dispatch_profile_output_file.Name
        ] = filepath_parameter(
            strParameter(
                'Dispatch Profile Output File',
                DefaultValue='dispatch_profile.csv',
                Required=False,
                Provided=False,
                ErrMessage='assume no dispatch profile csv output',
                ToolTipText=(
                    'Provide a CSV output filename if you want hourly dispatch profile export '
                    '(no dispatch profile CSV output if not provided)'
                ),
            )
        )

        # noinspection SpellCheckingInspection
        self.printoutput = self.ParameterDict[self.printoutput.Name] = boolParameter(
                'Print Output to Console',
                DefaultValue=True,
                Required=False,
                Provided=False,
                ErrMessage='assume no output to console',
                ToolTipText='Provide a 0 if you do not want to print output to the console',
            )

        self.generate_dispatch_html_graphs = self.ParameterDict[self.generate_dispatch_html_graphs.Name] = boolParameter(
                'Generate Dispatch HTML Graphs',
                DefaultValue=False,
                Required=False,
                Provided=False,
                ErrMessage='assume no dispatch html graphs',
                ToolTipText='Provide a 1 to generate optional dispatch graphs in HTML output',
            )

        model.logger.info(f'Complete {__class__!s}: {__name__}')

    def __str__(self):
        return 'Outputs'

    @staticmethod
    def _has_electricity_component(enduse_option: EndUseOptions) -> bool:
        return enduse_option in Outputs.ELECTRICITY_END_USE_OPTIONS

    @staticmethod
    def _has_heat_component(enduse_option: EndUseOptions) -> bool:
        return enduse_option in Outputs.HEAT_END_USE_OPTIONS

    @staticmethod
    def _writes_surface_heat_results(enduse_option: EndUseOptions) -> bool:
        return enduse_option in Outputs.SURFACE_HEAT_RESULT_OPTIONS

    @staticmethod
    def _is_cogeneration_end_use(enduse_option: EndUseOptions) -> bool:
        return enduse_option in Outputs.COGENERATION_END_USE_OPTIONS

    @staticmethod
    def _thermal_drawdown_ratio(model: Model, idx: int) -> float:
        initial_temp = float(model.wellbores.ProducedTemperature.value[0])
        if initial_temp == 0.0:
            return 0.0
        return float(model.wellbores.ProducedTemperature.value[idx] / initial_temp)

    def _sibling_output_path(self, suffix: str) -> str:
        output_path = Path(self.output_file)
        return str(output_path.with_suffix(suffix))

    def read_parameters(self, model: Model, default_output_path: Path = None) -> None:
        """
        The read_parameters function reads in the parameters from a dictionary and stores them in the parameters.
        It also handles special cases that need to be handled after a value has been read in and checked.
        If you choose to subclass this master class, you can also choose to override this method (or not), and if you do
        Deal with all the parameter values that the user has provided.  They should really only provide values that
        they want to change from the default values, but they can provide a value that is already set because it is a
        default value set in __init__.  It will ignore those.
        This also deals with all the special cases that need to be taken care of after a value has been read in
        and checked.
        If you choose to subclass this master class, you can also choose to override this method (or not),
        and if you do, do it before or after you call you own version of this method.  If you do, you can also choose
        to call this method from you class, which can effectively modify all these superclass parameters in your class.
        :param model: The container class of the application, giving access to everything else, including the logger
        :type model: :class:`~geophires_x.Model.Model`
        :param default_output_path: Relative path for non-absolute output path parameters
        :type default_output_path: pathlib.Path
        :return: None
        """
        model.logger.info(f'Init {__class__!s}: {__name__}')
        if len(model.InputParameters) > 0:
            # loop through all the parameters that the user wishes to set, looking for parameters that match this object
            for item in self.ParameterDict.items():
                ParameterToModify = item[1]
                key = ParameterToModify.Name.strip()
                if key in model.InputParameters:
                    ParameterReadIn: ParameterEntry = model.InputParameters[key]

                    if key in self.filepath_parameter_names:
                        if not Path(ParameterReadIn.sValue).is_absolute() and default_output_path is not None:
                            original_val = ParameterReadIn.sValue
                            output_path = Path(ParameterReadIn.sValue)
                            resolved_output_path = default_output_path.joinpath(output_path).absolute()
                            repo_output_path = Path(__file__).resolve().parents[2].joinpath(output_path).absolute()
                            if (
                                output_path.parent != Path('.')
                                and not resolved_output_path.parent.exists()
                                and repo_output_path.parent.exists()
                            ):
                                resolved_output_path = repo_output_path
                            ParameterReadIn.sValue = str(resolved_output_path)
                            model.logger.info(f'Adjusted {key} path to {ParameterReadIn.sValue} because original value '
                                              f'({original_val}) was not an absolute path.')

                    # this should handle all the non-special cases
                    ReadParameter(ParameterReadIn, ParameterToModify, model)

        # handle the special cases
        if len(model.InputParameters) > 0:
            # loop through all the parameters that the user wishes to set, looking for parameters that contain the
            # prefix "Units:" - that means we want to set a special case for converting this
            # output parameter to new units
            for key in model.InputParameters.keys():
                if key.startswith('Units:'):
                    self.ParameterDict[key.replace('Units:', '')] = LookupUnits(model.InputParameters[key].sValue)[0]

        model.logger.info(f'Complete {__class__!s}: {__name__}')

    def _convert_units(self, model: Model):
        # Deal with converting Units back to PreferredUnits, if required.
        # before we write the outputs, we go thru all the parameters for all of the objects and set the values back
        # to the units that the user entered the data in
        # We do this because the value may be displayed in the output, and we want the user to recginze their value,
        # not some converted value
        for obj in [model.reserv, model.wellbores, model.surfaceplant, model.economics]:
            for key in obj.ParameterDict:
                param = obj.ParameterDict[key]
                if not param.UnitsMatch:
                    ConvertUnitsBack(param, model)

        # now we need to loop through all the output parameters to update their units to
        # whatever units the user has specified.
        # i.e., they may have specified that all LENGTH results must be in feet, so we need to convert those
        # from whatever LENGTH unit they are to feet.
        # same for all the other classes of units (TEMPERATURE, DENSITY, etc).

        for obj in [model.reserv, model.wellbores, model.surfaceplant, model.economics]:
            for key in obj.OutputParameterDict:
                output_param:OutputParameter = obj.OutputParameterDict[key]
                if key in self.ParameterDict:
                    if self.ParameterDict[key] != output_param.CurrentUnits:
                        ConvertOutputUnits(output_param, self.ParameterDict[key], model)
                elif not output_param.UnitsMatch:
                    obj.OutputParameterDict[key] = output_param.with_preferred_units()

    @staticmethod
    def _write_case_report_header(f: TextIOWrapper) -> None:
        f.write('                               *****************\n')
        f.write('                               ***CASE REPORT***\n')
        f.write('                               *****************\n')
        f.write(NL)

    @staticmethod
    def _write_simulation_metadata(model: Model, f: TextIOWrapper) -> None:
        f.write('Simulation Metadata\n')
        f.write('----------------------\n')
        f.write(f' GEOPHIRES Version: {geophires_x.__version__}\n')
        f.write(' Simulation Date: '+ datetime.datetime.now().strftime('%Y-%m-%d\n'))
        f.write(' Simulation Time:  '+ datetime.datetime.now().strftime('%H:%M\n'))
        f.write(' Calculation Time: '+'{0:10.3f}'.format((time.time()-model.tic)) + ' sec\n')

    @staticmethod
    def _write_summary_of_results(
        model: Model,
        f: TextIOWrapper,
        dispatch_report: bool,
        is_sam_econ_model: bool,
    ) -> None:
        write_summary_of_results(model, f, dispatch_report, is_sam_econ_model)

    @staticmethod
    def _write_economic_parameters(model: Model, f: TextIOWrapper, is_sam_econ_model: bool) -> None:
        write_economic_parameters(model, f, is_sam_econ_model)

    @staticmethod
    def _write_engineering_parameters(model: Model, f: TextIOWrapper) -> None:
        write_engineering_parameters(model, f)

    @staticmethod
    def _write_resource_characteristics(model: Model, f: TextIOWrapper) -> None:
        write_resource_characteristics(model, f)

    @staticmethod
    def _write_reservoir_parameters(model: Model, f: TextIOWrapper) -> None:
        write_reservoir_parameters(model, f)

    @staticmethod
    def _write_reservoir_simulation_results(model: Model, f: TextIOWrapper, dispatch_report: bool) -> None:
        write_reservoir_simulation_results(model, f, dispatch_report)

    @staticmethod
    def _write_capital_costs(model: Model, f: TextIOWrapper, is_sam_econ_model: bool) -> None:
        write_capital_costs(model, f, is_sam_econ_model)

    @staticmethod
    def _write_operation_and_maintenance_costs(model: Model, f: TextIOWrapper, is_sam_econ_model: bool) -> None:
        write_operation_and_maintenance_costs(model, f, is_sam_econ_model)

    @staticmethod
    def _write_surface_equipment_simulation_results(
        model: Model,
        f: TextIOWrapper,
        dispatch_report: bool,
    ) -> None:
        write_surface_equipment_simulation_results(model, f, dispatch_report)

    @staticmethod
    def _dispatch_report_year_count(model: Model, default: int | None = None) -> int | None:
        dispatch_results = getattr(model, 'dispatch_results', None)
        if dispatch_results is None:
            return default

        analysis_end_year = int(getattr(dispatch_results, 'analysis_end_year',
                                        model.surfaceplant.plant_lifetime.value))
        return min(analysis_end_year, model.surfaceplant.plant_lifetime.value)

    @staticmethod
    def _write_production_profile(model: Model, f: TextIOWrapper, dispatch_report: bool) -> None:
        write_production_profile(model, f, dispatch_report)

    @staticmethod
    def _write_annual_production_profile(model: Model, f: TextIOWrapper, dispatch_report: bool) -> None:
        write_annual_production_profile(model, f, dispatch_report)

    def _write_cashflow_profile_sections(
        self,
        model: Model,
        f: TextIOWrapper,
        is_sam_econ_model: bool,
    ) -> None:
        if not is_sam_econ_model:
            self.write_revenue_and_cashflow_profile_output(model, f)

        if is_sam_econ_model:
            f.write(self.get_sam_cash_flow_profile_output(model))

    @staticmethod
    def _write_reservoir_power_required_profiles(model: Model, f: TextIOWrapper) -> None:
        write_reservoir_power_required_profiles(model, f)

    @staticmethod
    def _is_sam_econ_model(model: Model) -> bool:
        return model.economics.econmodel.value == EconomicModel.SAM_SINGLE_OWNER_PPA

    def _write_primary_report(
        self,
        model: Model,
        f: TextIOWrapper,
        is_sam_econ_model: bool,
        dispatch_report: bool,
    ) -> None:
        self._write_case_report_header(f)
        self._write_simulation_metadata(model, f)

        self._write_summary_of_results(model, f, dispatch_report, is_sam_econ_model)

        self._write_weather_data_results(model, f)
        self._write_tess_results(model, f)
        self._write_dispatch_results(model, f)

        self._write_economic_parameters(model, f, is_sam_econ_model)
        self._write_engineering_parameters(model, f)
        self._write_resource_characteristics(model, f)
        self._write_reservoir_parameters(model, f)
        self._write_reservoir_simulation_results(model, f, dispatch_report)

        self._write_capital_costs(model, f, is_sam_econ_model)
        self._write_operation_and_maintenance_costs(model, f, is_sam_econ_model)

        self._write_surface_equipment_simulation_results(model, f, dispatch_report)
        self._write_production_profile(model, f, dispatch_report)
        self._write_annual_production_profile(model, f, dispatch_report)
        self._write_cashflow_profile_sections(model, f, is_sam_econ_model)
        self._write_reservoir_power_required_profiles(model, f)
        if dispatch_report:
            self._write_dispatch_profile_report_table(model, f)

    @staticmethod
    def _write_addon_outputs(model: Model, is_sam_econ_model: bool) -> tuple[pd.DataFrame, list, bool]:
        addon_df = pd.DataFrame()
        addon_results = []
        extended_economics_header_printed = False
        if model.economics.DoAddOnCalculations.value and not is_sam_econ_model:
            # SAM econ models incorporate add-on economics into main economics, not as separate extended economics.
            addon_df, addon_results = model.addoutputs.PrintOutputs(model)
            extended_economics_header_printed = True

        return addon_df, addon_results, extended_economics_header_printed

    def _write_royalty_holder_outputs(
        self,
        model: Model,
        extended_economics_header_printed: bool,
    ) -> None:
        econ: Economics = model.economics

        if not econ.has_royalties:
            return

        with open(self.output_file, 'a', encoding='UTF-8') as f_:
            if not extended_economics_header_printed:
                self._print_extended_economics_header(f_)

            for royalty_output in [
                econ.royalty_holder_npv,
                econ.royalty_holder_annual_revenue,
                econ.royalty_holder_total_revenue
            ]:
                label = Outputs._field_label(royalty_output.display_name, 49)
                f_.write(
                    f'      {label}{royalty_output.value:10.2f} {royalty_output.CurrentUnits.value}\n')

    @staticmethod
    def _write_sdac_outputs(model: Model) -> tuple[pd.DataFrame, list]:
        sdac_df = pd.DataFrame()
        sdac_results = []
        if model.economics.DoSDACGTCalculations.value:
            sdac_df, sdac_results = model.sdacgtoutputs.PrintOutputs(model)

        return sdac_df, sdac_results

    def _write_rich_outputs(
        self,
        model: Model,
        sdac_results: list,
        addon_results: list,
        sdac_df: pd.DataFrame,
        addon_df: pd.DataFrame,
    ) -> None:
        if self.text_output_file.Provided:
            self.text_output_file.Valid = True
            self.text_output_file.value = _resolve_output_path(self.text_output_file.value, model)

        if self.html_output_file.Provided:
            self.html_output_file.Valid = True
            self.html_output_file.value = _resolve_output_path(self.html_output_file.value, model)

        print_outputs_rich(
            self.text_output_file,
            self.html_output_file,
            model,
            sdac_results,
            addon_results,
            sdac_df,
            addon_df
        )

    def PrintOutputs(self, model: Model):
        """
        PrintOutputs writes the standard outputs to the output file.
        :param model: The container class of the application, giving access to everything else, including the logger
        :type model: :class:`~geophires_x.Model.Model`
        :return: None
        """
        model.logger.info(f'Init {str(__class__)}: {sys._getframe().f_code.co_name}')

        self._convert_units(model)

        # write results to output file and screen
        try:
            is_sam_econ_model = self._is_sam_econ_model(model)
            dispatch_report = is_dispatch_report(model)
            with open(self.output_file, 'w', encoding='UTF-8') as f:
                self._write_primary_report(model, f, is_sam_econ_model, dispatch_report)

            addon_df, addon_results, extended_economics_header_printed = self._write_addon_outputs(
                model,
                is_sam_econ_model,
            )
            self._write_royalty_holder_outputs(model, extended_economics_header_printed)
            sdac_df, sdac_results = self._write_sdac_outputs(model)

        except BaseException as ex:
            tb = sys.exc_info()[2]
            msg = 'Error: GEOPHIRES Failed to write the output file. Exiting....Line %i' % tb.tb_lineno
            print(str(ex))
            print(msg)
            model.logger.critical(str(ex))
            model.logger.critical(msg)
            raise RuntimeError(msg) from ex

        self._write_rich_outputs(model, sdac_results, addon_results, sdac_df, addon_df)
        self._write_dispatch_profile_output(model)

        model.logger.info(f'Complete {__class__!s}: {sys._getframe().f_code.co_name}')

    # noinspection PyMethodMayBeStatic
    def write_revenue_and_cashflow_profile_output(self, model, f):
        write_revenue_and_cashflow_profile_output(model, f)

    # noinspection PyMethodMayBeStatic
    def get_sam_cash_flow_profile_output(self, model):
        return get_sam_cash_flow_profile_output(model)

    def _print_extended_economics_header(self, f_output_file: TextIOWrapper | None = None) -> None:
        """
        Header may be printed by either OutputsAddOns, or parent class if royalties are calculated and add-ons are not.
        """

        close_f = False
        if f_output_file is None:
            f_output_file = open(self.output_file, 'a', encoding='UTF-8')
            close_f = True

        f_output_file.write(NL)
        f_output_file.write(NL)
        f_output_file.write("                                ***EXTENDED ECONOMICS***\n")
        f_output_file.write(NL)

        if close_f:
            f_output_file.close()

    @staticmethod
    def _field_label(field_name: str, print_width_before_value: int) -> str:
        return f'{field_name}:{" " * (print_width_before_value - len(field_name) - 1)}'

    @staticmethod
    def _weather_output_rows(model: Model) -> list[tuple[str, str | float, str | None]]:
        return weather_output_rows(model)

    def _write_weather_data_results(self, model: Model, f) -> None:
        write_scalar_section(f, self.WEATHER_DATA_RESULTS_CATEGORY_NAME, self._weather_output_rows(model))

    @staticmethod
    def _dispatch_output_rows(model: Model) -> list[tuple[str, float, str]]:
        return dispatch_output_rows(model)

    @staticmethod
    def _tess_output_rows(model: Model, metrics: dict[str, float]) -> list[tuple[str, float, str]]:
        """Return TESS summary rows for dispatch text and parsed output."""
        return tess_output_rows(model, metrics)

    def _write_tess_results(self, model: Model, f) -> None:
        dispatch_results = getattr(model, 'dispatch_results', None)
        if dispatch_results is None:
            return

        tess_rows = self._tess_output_rows(model, dispatch_results.summary_metrics)
        write_scalar_section(f, self.TESS_RESULTS_CATEGORY_NAME, tess_rows)

    def _write_dispatch_results(self, model: Model, f) -> None:
        dispatch_rows = self._dispatch_output_rows(model)
        write_scalar_section(f, self.DISPATCH_RESULTS_CATEGORY_NAME, dispatch_rows)

    @staticmethod
    def _dispatch_profile_tess_columns(dispatch_results: Any) -> list[str]:
        """Return TESS profile CSV columns when storage is active."""
        return dispatch_profile_tess_columns(dispatch_results)

    @staticmethod
    def _dispatch_profile_tess_row(dispatch_results: Any, timestep_index: int) -> list[float]:
        """Return one timestep of TESS profile CSV data when storage is active."""
        return dispatch_profile_tess_row(dispatch_results, timestep_index)

    def _write_dispatch_profile_report_table(self, model: Model, f) -> None:
        write_dispatch_profile_report_table(model, f, self.DISPATCH_PROFILE_CATEGORY_NAME)

    def _write_dispatch_profile_output(self, model: Model) -> None:
        dispatch_results = getattr(model, 'dispatch_results', None)
        if dispatch_results is None or not self.dispatch_profile_output_file.Provided:
            return
        dispatch_profile_output_file = _resolve_output_path(self.dispatch_profile_output_file.value, model)
        write_dispatch_profile_output(model, dispatch_profile_output_file)

