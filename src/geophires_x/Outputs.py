from __future__ import annotations

import datetime
import math
import time
import sys
from io import TextIOWrapper
from pathlib import Path
from typing import Any

# noinspection PyPackageRequirements
import numpy as np
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
from geophires_x.OutputsEngineering import write_engineering_parameters
from geophires_x.OutputsProfiles import write_annual_production_profile, write_production_profile
from geophires_x.OutputsReport import write_scalar_section
from geophires_x.OutputsResource import write_resource_characteristics
from geophires_x.OutputsReservoir import write_reservoir_parameters, write_reservoir_simulation_results
from geophires_x.OutputsRich import print_outputs_rich
from geophires_x.Parameter import ConvertUnitsBack, ConvertOutputUnits, LookupUnits, strParameter, boolParameter, \
    OutputParameter, ReadParameter, ParameterEntry
from geophires_x.OptionList import EndUseOptions, EconomicModel, OperatingMode, PlantType
from geophires_x.Parameter import Parameter

NL = '\n'


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
                            ParameterReadIn.sValue = str(
                                default_output_path.joinpath(Path(ParameterReadIn.sValue)).absolute())
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
        econ: Economics = model.economics

        f.write(NL)
        f.write('                           ***SUMMARY OF RESULTS***\n')
        f.write(NL)
        f.write(f'      {model.surfaceplant.enduse_option_output.display_name}: '
                f'{model.surfaceplant.enduse_option.value.value}\n')
        if not dispatch_report and model.surfaceplant.plant_type.value in Outputs.SPECIAL_HEAT_PLANT_TYPES:
            f.write('      Surface Application: ' + str(model.surfaceplant.plant_type.value.value) + NL)
        if not dispatch_report and Outputs._has_electricity_component(model.surfaceplant.enduse_option.value):
            f.write(f'      Average Net Electricity Production:               {np.average(model.surfaceplant.NetElectricityProduced.value):10.2f} ' + model.surfaceplant.NetElectricityProduced.CurrentUnits.value + NL)
        if not dispatch_report and model.surfaceplant.enduse_option.value is not EndUseOptions.ELECTRICITY:
            # there is a direct-use component
            f.write(f'      Average Direct-Use Heat Production:               {np.average(model.surfaceplant.HeatProduced.value):10.2f} '+ model.surfaceplant.HeatProduced.CurrentUnits.value + NL)
        if not dispatch_report and model.surfaceplant.plant_type.value == PlantType.DISTRICT_HEATING:
            f.write(f'      Annual District Heating Demand:                   {np.average(model.surfaceplant.annual_heating_demand.value):10.2f} ' + model.surfaceplant.annual_heating_demand.CurrentUnits.value + NL)
            f.write(f'      Average Annual Geothermal Heat Production:        {sum(model.surfaceplant.dh_geothermal_heating.value * 24) / model.surfaceplant.plant_lifetime.value / 1e3:10.2f} ' + model.surfaceplant.annual_heating_demand.CurrentUnits.value + NL)
            f.write(f'      Average Annual Peaking Fuel Heat Production:      {sum(model.surfaceplant.dh_natural_gas_heating.value * 24) / model.surfaceplant.plant_lifetime.value / 1e3:10.2f} ' + model.surfaceplant.annual_heating_demand.CurrentUnits.value + NL)
        if not dispatch_report and model.surfaceplant.plant_type.value == PlantType.ABSORPTION_CHILLER:
            f.write(f'      Average Cooling Production:                       {np.average(model.surfaceplant.cooling_produced.value):10.2f} ' + model.surfaceplant.cooling_produced.CurrentUnits.value + NL)

        if not dispatch_report and model.surfaceplant.enduse_option.value in [EndUseOptions.ELECTRICITY]:
            f.write(f'      {model.economics.LCOE.display_name}:                      {model.economics.LCOE.value:10.2f} {model.economics.LCOE.CurrentUnits.value}\n')
            if model.economics.DoXLCOCalculations.value:
                # XLCO and VALCO live in the same summary block as the baseline LCO output so
                # downstream parsers can treat them as parallel commodity summary metrics.
                f.write(f'      {model.economics.XLCOE_Market.display_name}: {model.economics.XLCOE_Market.value:10.2f} {model.economics.XLCOE_Market.CurrentUnits.value}\n')
                f.write(f'      {model.economics.XLCOE_MarketSocial.display_name}: {model.economics.XLCOE_MarketSocial.value:10.2f} {model.economics.XLCOE_MarketSocial.CurrentUnits.value}\n')
            if model.economics.DoVALCOCalculations.value:
                f.write(f'      {model.economics.VALCOE.display_name}: {model.economics.VALCOE.value:10.2f} {model.economics.VALCOE.CurrentUnits.value}\n')
                f.write(f'      {model.economics.VALCOE_EnergyAdjustment.display_name}: {model.economics.VALCOE_EnergyAdjustment.value:10.2f} {model.economics.VALCOE_EnergyAdjustment.CurrentUnits.value}\n')
                f.write(f'      {model.economics.VALCOE_CapacityAdjustment.display_name}: {model.economics.VALCOE_CapacityAdjustment.value:10.2f} {model.economics.VALCOE_CapacityAdjustment.CurrentUnits.value}\n')
                f.write(f'      {model.economics.VALCOE_FlexibilityAdjustment.display_name}: {model.economics.VALCOE_FlexibilityAdjustment.value:10.2f} {model.economics.VALCOE_FlexibilityAdjustment.CurrentUnits.value}\n')
        elif not dispatch_report and model.surfaceplant.enduse_option.value in [EndUseOptions.HEAT] and \
                model.surfaceplant.plant_type.value not in [PlantType.ABSORPTION_CHILLER]:
            f.write(f'      {model.economics.LCOH.display_name}:            {model.economics.LCOH.value:10.2f} {model.economics.LCOH.CurrentUnits.value}\n')
            if model.economics.DoXLCOCalculations.value:
                f.write(f'      {model.economics.XLCOH_Market.display_name}: {model.economics.XLCOH_Market.value:10.2f} {model.economics.XLCOH_Market.CurrentUnits.value}\n')
                f.write(f'      {model.economics.XLCOH_MarketSocial.display_name}: {model.economics.XLCOH_MarketSocial.value:10.2f} {model.economics.XLCOH_MarketSocial.CurrentUnits.value}\n')
            if model.economics.DoVALCOCalculations.value:
                f.write(f'      {model.economics.VALCOH.display_name}: {model.economics.VALCOH.value:10.2f} {model.economics.VALCOH.CurrentUnits.value}\n')
                f.write(f'      {model.economics.VALCOH_EnergyAdjustment.display_name}: {model.economics.VALCOH_EnergyAdjustment.value:10.2f} {model.economics.VALCOH_EnergyAdjustment.CurrentUnits.value}\n')
                f.write(f'      {model.economics.VALCOH_CapacityAdjustment.display_name}: {model.economics.VALCOH_CapacityAdjustment.value:10.2f} {model.economics.VALCOH_CapacityAdjustment.CurrentUnits.value}\n')
                f.write(f'      {model.economics.VALCOH_FlexibilityAdjustment.display_name}: {model.economics.VALCOH_FlexibilityAdjustment.value:10.2f} {model.economics.VALCOH_FlexibilityAdjustment.CurrentUnits.value}\n')
        elif not dispatch_report and model.surfaceplant.enduse_option.value in [EndUseOptions.HEAT] and model.surfaceplant.plant_type.value == PlantType.ABSORPTION_CHILLER:
            f.write(f'      {model.economics.LCOC.display_name}:         {model.economics.LCOC.value:10.2f} {model.economics.LCOC.CurrentUnits.value}\n')
            if model.economics.DoXLCOCalculations.value:
                f.write(f'      {model.economics.XLCOC_Market.display_name}: {model.economics.XLCOC_Market.value:10.2f} {model.economics.XLCOC_Market.CurrentUnits.value}\n')
                f.write(f'      {model.economics.XLCOC_MarketSocial.display_name}: {model.economics.XLCOC_MarketSocial.value:10.2f} {model.economics.XLCOC_MarketSocial.CurrentUnits.value}\n')
            if model.economics.DoVALCOCalculations.value:
                f.write(f'      {model.economics.VALCOC.display_name}: {model.economics.VALCOC.value:10.2f} {model.economics.VALCOC.CurrentUnits.value}\n')
                f.write(f'      {model.economics.VALCOC_EnergyAdjustment.display_name}: {model.economics.VALCOC_EnergyAdjustment.value:10.2f} {model.economics.VALCOC_EnergyAdjustment.CurrentUnits.value}\n')
                f.write(f'      {model.economics.VALCOC_CapacityAdjustment.display_name}: {model.economics.VALCOC_CapacityAdjustment.value:10.2f} {model.economics.VALCOC_CapacityAdjustment.CurrentUnits.value}\n')
                f.write(f'      {model.economics.VALCOC_FlexibilityAdjustment.display_name}: {model.economics.VALCOC_FlexibilityAdjustment.value:10.2f} {model.economics.VALCOC_FlexibilityAdjustment.CurrentUnits.value}\n')
        elif not dispatch_report and Outputs._is_cogeneration_end_use(model.surfaceplant.enduse_option.value):
            # Cogeneration writes both electricity and heat competitiveness outputs because
            # XLCO/VALCO are tracked independently per active commodity.
            f.write(f'      {model.economics.LCOE.display_name}:                      {model.economics.LCOE.value:10.2f} {model.economics.LCOE.CurrentUnits.value}\n')
            if model.economics.DoXLCOCalculations.value:
                f.write(f'      {model.economics.XLCOE_Market.display_name}: {model.economics.XLCOE_Market.value:10.2f} {model.economics.XLCOE_Market.CurrentUnits.value}\n')
                f.write(f'      {model.economics.XLCOE_MarketSocial.display_name}: {model.economics.XLCOE_MarketSocial.value:10.2f} {model.economics.XLCOE_MarketSocial.CurrentUnits.value}\n')
            if model.economics.DoVALCOCalculations.value:
                f.write(f'      {model.economics.VALCOE.display_name}: {model.economics.VALCOE.value:10.2f} {model.economics.VALCOE.CurrentUnits.value}\n')
                f.write(f'      {model.economics.VALCOE_EnergyAdjustment.display_name}: {model.economics.VALCOE_EnergyAdjustment.value:10.2f} {model.economics.VALCOE_EnergyAdjustment.CurrentUnits.value}\n')
                f.write(f'      {model.economics.VALCOE_CapacityAdjustment.display_name}: {model.economics.VALCOE_CapacityAdjustment.value:10.2f} {model.economics.VALCOE_CapacityAdjustment.CurrentUnits.value}\n')
                f.write(f'      {model.economics.VALCOE_FlexibilityAdjustment.display_name}: {model.economics.VALCOE_FlexibilityAdjustment.value:10.2f} {model.economics.VALCOE_FlexibilityAdjustment.CurrentUnits.value}\n')
            f.write(f'      {model.economics.LCOH.display_name}:           {model.economics.LCOH.value:10.2f} {model.economics.LCOH.CurrentUnits.value}\n')
            if model.economics.DoXLCOCalculations.value:
                f.write(f'      {model.economics.XLCOH_Market.display_name}: {model.economics.XLCOH_Market.value:10.2f} {model.economics.XLCOH_Market.CurrentUnits.value}\n')
                f.write(f'      {model.economics.XLCOH_MarketSocial.display_name}: {model.economics.XLCOH_MarketSocial.value:10.2f} {model.economics.XLCOH_MarketSocial.CurrentUnits.value}\n')
            if model.economics.DoVALCOCalculations.value:
                f.write(f'      {model.economics.VALCOH.display_name}: {model.economics.VALCOH.value:10.2f} {model.economics.VALCOH.CurrentUnits.value}\n')
                f.write(f'      {model.economics.VALCOH_EnergyAdjustment.display_name}: {model.economics.VALCOH_EnergyAdjustment.value:10.2f} {model.economics.VALCOH_EnergyAdjustment.CurrentUnits.value}\n')
                f.write(f'      {model.economics.VALCOH_CapacityAdjustment.display_name}: {model.economics.VALCOH_CapacityAdjustment.value:10.2f} {model.economics.VALCOH_CapacityAdjustment.CurrentUnits.value}\n')
                f.write(f'      {model.economics.VALCOH_FlexibilityAdjustment.display_name}: {model.economics.VALCOH_FlexibilityAdjustment.value:10.2f} {model.economics.VALCOH_FlexibilityAdjustment.CurrentUnits.value}\n')

        if not dispatch_report and is_sam_econ_model:
            f.write(f'      {Outputs._field_label(econ.capex_total.display_name, 50)}{econ.capex_total.value:10.2f} {econ.capex_total.CurrentUnits.value}\n')
            f.write(f'      {Outputs._field_label(econ.capex_total_per_kw.display_name, 50)}{econ.capex_total_per_kw.value:10.0f} {econ.capex_total_per_kw.CurrentUnits.value}\n')

        f.write(f'      Number of production wells:                    {model.wellbores.nprod.value:10.0f}'+NL)
        f.write(f'      Number of injection wells:                     {model.wellbores.ninj.value:10.0f}'+NL)
        if dispatch_report:
            maximum_flowrate = model.dispatch_results.summary_metrics.get('observed_peak_flow_kg_per_sec', 0.0)
            f.write(f'      Maximum Flowrate per production well:            {maximum_flowrate:10.1f} kg/s' + NL)
        else:
            f.write(f'      Flowrate per production well:                    {model.wellbores.prodwellflowrate.value:10.1f} '  + model.wellbores.prodwellflowrate.CurrentUnits.value + NL)
        f.write(f'      {Outputs._field_label(Outputs.VERTICAL_WELL_DEPTH_OUTPUT_NAME, 49)}{model.reserv.depth.value:10.1f} ' + model.reserv.depth.CurrentUnits.value + NL)

        if model.reserv.numseg.value == 1:
            f.write(f'      Geothermal gradient:                             {model.reserv.gradient.value[0]:10.4g} ' + model.reserv.gradient.CurrentUnits.value + NL)
        else:
            for i in range(1, model.reserv.numseg.value):
                f.write(f'      Segment {str(i):s}   Geothermal gradient:                    {model.reserv.gradient.value[i-1]:10.4g} ' + model.reserv.gradient.CurrentUnits.value +NL)
                f.write(f'      Segment {str(i):s}   Thickness:                         {round(model.reserv.layerthickness.value[i-1], 10)} {model.reserv.layerthickness.CurrentUnits.value}\n')
            f.write(f'      Segment {str(i+1):s}   Geothermal gradient:                    {model.reserv.gradient.value[i]:10.4g} ' + model.reserv.gradient.CurrentUnits.value + NL)
        if not dispatch_report and model.economics.DoCarbonCalculations.value:
            f.write(f'      {model.economics.CarbonThatWouldHaveBeenProducedTotal.display_name}:'
                    f'                       {model.economics.CarbonThatWouldHaveBeenProducedTotal.value:10.2f}'
                    f' {model.economics.CarbonThatWouldHaveBeenProducedTotal.CurrentUnits.value}\n')

    @staticmethod
    def _write_economic_parameters(model: Model, f: TextIOWrapper, is_sam_econ_model: bool) -> None:
        econ: Economics = model.economics

        f.write(NL)
        f.write(NL)
        f.write('                           ***ECONOMIC PARAMETERS***\n')
        f.write(NL)
        if model.economics.econmodel.value == EconomicModel.FCR:
            f.write(f'      Economic Model = {model.economics.econmodel.value.value}\n')
            f.write(f'      Fixed Charge Rate (FCR):                          {model.economics.FCR.value*100.0:10.2f} {model.economics.FCR.CurrentUnits.value}\n')
        elif model.economics.econmodel.value == EconomicModel.STANDARDIZED_LEVELIZED_COST:
            f.write(f'      Economic Model = {model.economics.econmodel.value.value}\n')
            # TODO disambiguate interest rate for all economic models - see
            #  https://github.com/softwareengineerprogrammer/GEOPHIRES/commit/535c02d4adbeeeca553b61e9b996fccf00016529
            f.write(f'      {model.economics.interest_rate.Name}:                                    {model.economics.interest_rate.value:10.2f} {model.economics.interest_rate.CurrentUnits.value}\n')

        elif is_sam_econ_model or model.economics.econmodel.value == EconomicModel.BICYCLE:
            f.write(f'      Economic Model = {model.economics.econmodel.value.value}\n')

        if is_sam_econ_model:
            sam_econ_fields: list[OutputParameter] = [
                econ.real_discount_rate,
                econ.nominal_discount_rate,
                econ.wacc,
            ]

            for field in sam_econ_fields:
                label = Outputs._field_label(field.Name, 49)
                f.write(f'      {label}{field.value:10.2f} {field.CurrentUnits.value}\n')

        if econ.RITCValue.value and is_sam_econ_model:
            # Non-SAM-EMs (inaccurately) treat ITC as a capital cost and thus are displayed in the capital
            # costs category rather than here.
            f.write(
                f'      {econ.RITCValue.display_name}:                           {abs(econ.RITCValue.value):10.2f} {econ.RITCValue.CurrentUnits.value}\n')

        if not is_sam_econ_model:  # (parameter is ambiguous to the point of meaninglessness for SAM-EM)
            acf: OutputParameter = econ.accrued_financing_during_construction_percentage
            acf_label = Outputs._field_label(acf.display_name, 49)
            f.write(f'      {acf_label}{acf.value:10.2f} {acf.CurrentUnits.value}\n')

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
            icc_label = Outputs._field_label(icc.display_name, 49)
            f.write(f'      {icc_label}{icc.value:10.2f} {icc.CurrentUnits.value}\n')

        f.write(f'      Project lifetime:                              {model.surfaceplant.plant_lifetime.value:10.0f} {model.surfaceplant.plant_lifetime.CurrentUnits.value}\n')
        f.write(f'      Capacity factor:                                 {model.surfaceplant.utilization_factor.value * 100:10.1f} %\n')

        e_npv: OutputParameter = model.economics.ProjectNPV
        npv_field_label = Outputs._field_label(e_npv.display_name, 49)
        # TODO should use CurrentUnits instead of PreferredUnits
        f.write(f'      {npv_field_label}{e_npv.value:10.2f} {e_npv.PreferredUnits.value}\n')

        irr_output_param: OutputParameter = econ.ProjectIRR \
            if not is_sam_econ_model else econ.after_tax_irr
        irr_field_label = Outputs._field_label(irr_output_param.display_name, 49)
        irr_display_value = f'{irr_output_param.value:10.2f}' \
            if not math.isnan(irr_output_param.value) else 'NaN'
        f.write(f'      {irr_field_label}{irr_display_value} {irr_output_param.CurrentUnits.value}\n')

        f.write(f'      {econ.ProjectVIR.display_name}:                              {econ.ProjectVIR.value:10.2f}\n')
        f.write(f'      {econ.ProjectMOIC.display_name}:                                    {econ.ProjectMOIC.value:10.2f}\n')

        payback_period_val = model.economics.ProjectPaybackPeriod.value
        project_payback_period_display = (f'{payback_period_val:10.2f} '
                                          f'{econ.ProjectPaybackPeriod.PreferredUnits.value}') \
            if payback_period_val > 0.0 else 'N/A'
        project_payback_period_label = Outputs._field_label(model.economics.ProjectPaybackPeriod.display_name, 56)
        f.write(f'      {project_payback_period_label}{project_payback_period_display}\n')

        if Outputs._is_cogeneration_end_use(model.surfaceplant.enduse_option.value):
            f.write(f'      CHP: Percent cost allocation for electrical plant: {model.economics.CAPEX_heat_electricity_plant_ratio.value*100.0:10.2f} %\n')

        if model.surfaceplant.enduse_option.value in [EndUseOptions.ELECTRICITY]:
            f.write(f'      Estimated Jobs Created:                                 {model.economics.jobs_created.value}\n')

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
        econ: Economics = model.economics

        f.write('\n\n                          ***CAPITAL COSTS (M$)***\n\n')
        if not model.economics.totalcapcost.Valid:
            f.write(f'         {econ.Cexpl.display_name}:                             {econ.Cexpl.value:10.2f} {econ.Cexpl.CurrentUnits.value}\n')

            f.write(f'         {model.economics.Cwell.display_name}:                 {model.economics.Cwell.value:10.2f} {model.economics.Cwell.CurrentUnits.value}\n')

            if econ.cost_lateral_section.value > 0.0:
                f.write(f'             Drilling and completion costs per vertical production well:   {econ.cost_one_production_well.value:10.2f} ' + econ.cost_one_production_well.CurrentUnits.value + NL)
                f.write(f'             Drilling and completion costs per vertical injection well:    {econ.cost_one_injection_well.value:10.2f} ' + econ.cost_one_injection_well.CurrentUnits.value + NL)
                f.write(f'             {econ.cost_per_lateral_section.Name}:       {econ.cost_per_lateral_section.value:10.2f} {econ.cost_lateral_section.CurrentUnits.value}\n')
            elif round(econ.cost_one_production_well.value, 4) != round(econ.cost_one_injection_well.value, 4) \
                and model.economics.cost_one_injection_well.value != -1:
                f.write(f'             Drilling and completion costs per production well:   {econ.cost_one_production_well.value:10.2f} ' + econ.cost_one_production_well.CurrentUnits.value + NL)
                f.write(f'             Drilling and completion costs per injection well:    {econ.cost_one_injection_well.value:10.2f} ' + econ.cost_one_injection_well.CurrentUnits.value + NL)
            else:
                cpw_label = Outputs._field_label(econ.drilling_and_completion_costs_per_well.display_name, 47)
                f.write(f'         {cpw_label}{econ.drilling_and_completion_costs_per_well.value:10.2f} {econ.Cwell.CurrentUnits.value}\n')

            f.write(f'         {econ.Cstim.display_name}:                             {econ.Cstim.value:10.2f} {econ.Cstim.CurrentUnits.value}\n')

            f.write(f'         {econ.Cplant.display_name}:                     {econ.Cplant.value:10.2f} {econ.Cplant.CurrentUnits.value}\n')
            if model.surfaceplant.plant_type.value == PlantType.ABSORPTION_CHILLER:
                f.write(f'            of which Absorption Chiller Cost:           {model.economics.chillercapex.value:10.2f} ' + model.economics.Cplant.CurrentUnits.value + NL)
            if model.surfaceplant.plant_type.value == PlantType.HEAT_PUMP:
                f.write(f'            of which Heat Pump Cost:                    {model.economics.heatpumpcapex.value:10.2f} ' + model.economics.Cplant.CurrentUnits.value + NL)
            if model.surfaceplant.plant_type.value == PlantType.DISTRICT_HEATING:
                f.write(f'            of which Peaking Boiler Cost:               {model.economics.peakingboilercost.value:10.2f} ' + model.economics.peakingboilercost.CurrentUnits.value + NL)
            f.write(f'         {model.economics.Cgath.display_name}:                  {model.economics.Cgath.value:10.2f} {model.economics.Cgath.CurrentUnits.value}\n')

            if model.surfaceplant.piping_length.value > 0:
                f.write(f'         {model.economics.Cpiping.display_name}:                    {model.economics.Cpiping.value:10.2f} {model.economics.Cpiping.CurrentUnits.value}\n')

            if model.surfaceplant.plant_type.value == PlantType.DISTRICT_HEATING:
                f.write(f'         District Heating System Cost:                  {model.economics.dhdistrictcost.value:10.2f} {model.economics.dhdistrictcost.CurrentUnits.value}\n')

            f.write(f'         Total surface equipment costs:                 {(model.economics.Cplant.value+model.economics.Cgath.value):10.2f} ' + model.economics.Cplant.CurrentUnits.value + NL)

        if model.economics.totalcapcost.Valid and model.wellbores.redrill.value > 0:
            f.write(f'         Drilling and completion costs (for redrilling):{econ.Cwell.value:10.2f} {econ.Cwell.CurrentUnits.value}\n')
            f.write(f'      Drilling and completion costs per redrilled well: {(econ.Cwell.value/(model.wellbores.nprod.value+model.wellbores.ninj.value)):10.2f} {econ.Cwell.CurrentUnits.value}\n')
            f.write(f'         Stimulation costs (for redrilling):            {econ.Cstim.value:10.2f} {econ.Cstim.CurrentUnits.value}\n')

        if model.economics.RITCValue.value and not is_sam_econ_model:
            # Note ITC is in ECONOMIC PARAMETERS category for SAM-EM (not capital costs)
            f.write(f'         {econ.RITCValue.display_name}:                         {-1 * econ.RITCValue.value:10.2f} {econ.RITCValue.CurrentUnits.value}\n')

        additional_capex_modifiers: list[tuple[Parameter, int]] = [
            (econ.FlatLicenseEtc, 1),
            (econ.OtherIncentives, -1),
            (econ.TotalGrant, -1)
        ]
        for additional_capex_modifier_entry in additional_capex_modifiers:
            additional_capex_modifier_param: Parameter = additional_capex_modifier_entry[0]
            additional_capex_modifier_multiplier: int = additional_capex_modifier_entry[1]

            acm_render_value = additional_capex_modifier_param.value * additional_capex_modifier_multiplier

            if additional_capex_modifier_param.Provided:
                acm_label = Outputs._field_label(additional_capex_modifier_param.Name, 47)
                f.write(
                    f'         {acm_label}{acm_render_value:10.2f} {additional_capex_modifier_param.CurrentUnits.value}\n')

        if is_sam_econ_model and econ.DoAddOnCalculations.value:
            # Non-SAM econ models print this in Extended Economics profile
            aoc_label = Outputs._field_label(model.addeconomics.AddOnCAPEXTotal.display_name, 47)
            f.write(f'         {aoc_label}{model.addeconomics.AddOnCAPEXTotal.value:10.2f} {model.addeconomics.AddOnCAPEXTotal.CurrentUnits.value}\n')

        display_occ_and_inflation_during_construction_in_capital_costs = is_sam_econ_model
        if display_occ_and_inflation_during_construction_in_capital_costs:
            occ_label = Outputs._field_label(econ.overnight_capital_cost.display_name, 47)
            f.write(
                f'         {occ_label}{econ.overnight_capital_cost.value:10.2f} {econ.overnight_capital_cost.CurrentUnits.value}\n')

            icc_label = Outputs._field_label(econ.inflation_cost_during_construction.display_name, 47)
            f.write(f'         {icc_label}{econ.inflation_cost_during_construction.value:10.2f} {econ.inflation_cost_during_construction.CurrentUnits.value}\n')

        if econ.royalty_supplemental_payments.Provided:
            rsp_label = Outputs._field_label(econ.royalty_supplemental_payments_cost_during_construction.display_name, 41)
            f.write(
                f'         {rsp_label}   {econ.royalty_supplemental_payments_cost_during_construction.value:.2f} {econ.royalty_supplemental_payments_cost_during_construction.CurrentUnits.value}\n')

        display_idc_in_capital_costs = is_sam_econ_model \
                                               and model.surfaceplant.construction_years.value > 1
        if display_idc_in_capital_costs:
            idc_label = Outputs._field_label(econ.interest_during_construction.display_name, 47)
            f.write(
                f'         {idc_label}{econ.interest_during_construction.value:10.2f} {econ.interest_during_construction.CurrentUnits.value}\n')

        capex_param = econ.CCap if not is_sam_econ_model else econ.capex_total
        capex_label = Outputs._field_label(capex_param.display_name, 50)
        f.write(f'      {capex_label}{capex_param.value:10.2f} {capex_param.CurrentUnits.value}\n')

        if model.economics.econmodel.value == EconomicModel.FCR:
            f.write(f'      Annualized capital costs:                         {(model.economics.CCap.value*(1+model.economics.inflrateconstruction.value)*model.economics.FCR.value):10.2f} ' + model.economics.CCap.CurrentUnits.value + NL)

    @staticmethod
    def _write_operation_and_maintenance_costs(model: Model, f: TextIOWrapper, is_sam_econ_model: bool) -> None:
        econ: Economics = model.economics

        f.write(NL)
        f.write(NL)
        f.write('                ***OPERATING AND MAINTENANCE COSTS (M$/yr)***\n')
        f.write(NL)
        if not model.economics.oamtotalfixed.Valid:
            f.write(f'         {model.economics.Coamwell.display_name}:                   {model.economics.Coamwell.value:10.2f} {model.economics.Coamwell.CurrentUnits.value}\n')
            f.write(f'         {model.economics.Coamplant.display_name}:                 {model.economics.Coamplant.value:10.2f} {model.economics.Coamplant.CurrentUnits.value}\n')
            f.write(f'         {model.economics.Coamwater.display_name}:                                   {model.economics.Coamwater.value:10.2f} {model.economics.Coamwater.CurrentUnits.value}\n')
            if model.surfaceplant.plant_type.value in [PlantType.INDUSTRIAL, PlantType.ABSORPTION_CHILLER, PlantType.HEAT_PUMP, PlantType.DISTRICT_HEATING]:
                f.write(f'         Average Reservoir Pumping Cost:                {model.economics.averageannualpumpingcosts.value:10.2f} {model.economics.averageannualpumpingcosts.CurrentUnits.value}\n')
            if model.surfaceplant.plant_type.value == PlantType.ABSORPTION_CHILLER:
                f.write(f'         Absorption Chiller O&M Cost:                   {model.economics.chilleropex.value:10.2f} {model.economics.chilleropex.CurrentUnits.value}\n')
            if model.surfaceplant.plant_type.value == PlantType.HEAT_PUMP:
                f.write(f'         Average Heat Pump Electricity Cost:            {model.economics.averageannualheatpumpelectricitycost.value:10.2f} {model.economics.averageannualheatpumpelectricitycost.CurrentUnits.value}\n')
            if model.surfaceplant.plant_type.value == PlantType.DISTRICT_HEATING:
                f.write(f'         Annual District Heating O&M Cost:              {model.economics.dhdistrictoandmcost.value:10.2f} {model.economics.dhdistrictoandmcost.CurrentUnits.value}\n')
                f.write(f'         Average Annual Peaking Fuel Cost:              {model.economics.averageannualngcost.value:10.2f} {model.economics.averageannualngcost.CurrentUnits.value}\n')

            if model.wellbores.redrill.value > 0:
                redrill_label = Outputs._field_label(econ.redrilling_annual_cost.display_name, 47)
                f.write(f'         {redrill_label}{econ.redrilling_annual_cost.value:10.2f} {econ.redrilling_annual_cost.CurrentUnits.value}\n')

            if econ.DoAddOnCalculations.value and is_sam_econ_model:
                # Non-SAM econ models print this in Extended Economics profile
                aoc_label = Outputs._field_label(model.addeconomics.AddOnOPEXTotalPerYear.display_name, 47)
                f.write(f'         {aoc_label}{model.addeconomics.AddOnOPEXTotalPerYear.value:10.2f} {model.addeconomics.AddOnOPEXTotalPerYear.CurrentUnits.value}\n')

            if econ.has_production_based_royalties:
                royalties_label = Outputs._field_label(econ.royalties_average_annual_cost.display_name, 47)
                f.write(f'         {royalties_label}{econ.royalties_average_annual_cost.value:10.2f} {econ.royalties_average_annual_cost.CurrentUnits.value}\n')

            f.write(f'      {econ.Coam.display_name}:            {(econ.Coam.value + econ.averageannualpumpingcosts.value + econ.averageannualheatpumpelectricitycost.value):10.2f} {econ.Coam.CurrentUnits.value}\n')
        else:
            f.write(f'      {econ.Coam.display_name}:            {econ.Coam.value:10.2f} {econ.Coam.CurrentUnits.value}\n')

    @staticmethod
    def _write_surface_equipment_simulation_results(
        model: Model,
        f: TextIOWrapper,
        dispatch_report: bool,
    ) -> None:
        f.write(NL)
        f.write(NL)
        f.write('                           ***SURFACE EQUIPMENT SIMULATION RESULTS***\n')
        f.write(NL)
        if not dispatch_report and Outputs._has_electricity_component(model.surfaceplant.enduse_option.value):
            f.write(f'      Initial geofluid availability:                    {model.surfaceplant.Availability.value[0]:10.2f} ' + model.surfaceplant.Availability.PreferredUnits.value + NL)
            f.write(f'      Maximum Total Electricity Generation:             {np.max(model.surfaceplant.ElectricityProduced.value):10.2f} ' + model.surfaceplant.ElectricityProduced.PreferredUnits.value + NL)
            f.write(f'      Average Total Electricity Generation:             {np.average(model.surfaceplant.ElectricityProduced.value):10.2f} ' + model.surfaceplant.ElectricityProduced.PreferredUnits.value + NL)
            f.write(f'      Minimum Total Electricity Generation:             {np.min(model.surfaceplant.ElectricityProduced.value):10.2f} ' + model.surfaceplant.ElectricityProduced.PreferredUnits.value + NL)
            f.write(f'      Initial Total Electricity Generation:             {model.surfaceplant.ElectricityProduced.value[0]:10.2f} ' + model.surfaceplant.ElectricityProduced.PreferredUnits.value + NL)
            f.write(f'      Maximum Net Electricity Generation:               {np.max(model.surfaceplant.NetElectricityProduced.value):10.2f} ' + model.surfaceplant.NetElectricityProduced.PreferredUnits.value + NL)
            f.write(f'      Average Net Electricity Generation:               {np.average(model.surfaceplant.NetElectricityProduced.value):10.2f} ' + model.surfaceplant.NetElectricityProduced.PreferredUnits.value + NL)
            f.write(f'      Minimum Net Electricity Generation:               {np.min(model.surfaceplant.NetElectricityProduced.value):10.2f} ' + model.surfaceplant.NetElectricityProduced.PreferredUnits.value + NL)
            f.write(f'      Initial Net Electricity Generation:               {model.surfaceplant.NetElectricityProduced.value[0]:10.2f} ' + model.surfaceplant.NetElectricityProduced.PreferredUnits.value + NL)
            f.write(f'      Average Annual Total Electricity Generation:      {np.average(model.surfaceplant.TotalkWhProduced.value/1E6):10.2f} GWh' + NL)
            f.write(f'      Average Annual Net Electricity Generation:        {np.average(model.surfaceplant.NetkWhProduced.value/1E6):10.2f} GWh' + NL)

            if model.wellbores.PumpingPower.value[0] > 0.0:
                ipp_nip = model.wellbores.PumpingPower.value[0] / model.surfaceplant.NetElectricityProduced.value[0]
                f.write(f'      Initial pumping power/net installed power:        {(ipp_nip*100):10.2f} %\n')

        if not dispatch_report and Outputs._writes_surface_heat_results(model.surfaceplant.enduse_option.value):
            f.write(f'      Maximum Net Heat Production:                      {np.max(model.surfaceplant.HeatProduced.value):10.2f} ' + model.surfaceplant.HeatProduced.PreferredUnits.value + NL)
            f.write(f'      Average Net Heat Production:                      {np.average(model.surfaceplant.HeatProduced.value):10.2f} ' + model.surfaceplant.HeatProduced.PreferredUnits.value + NL)
            f.write(f'      Minimum Net Heat Production:                      {np.min(model.surfaceplant.HeatProduced.value):10.2f} ' + model.surfaceplant.HeatProduced.PreferredUnits.value + NL)
            f.write(f'      Initial Net Heat Production:                      {model.surfaceplant.HeatProduced.value[0]:10.2f} ' + model.surfaceplant.HeatProduced.PreferredUnits.value + NL)
            f.write(f'      Average Annual Heat Production:                   {np.average(model.surfaceplant.HeatkWhProduced.value/1E6):10.2f} GWh' + NL)

        if not dispatch_report and model.surfaceplant.plant_type.value == PlantType.HEAT_PUMP:
            f.write(f'      Average Annual Heat Pump Electricity Use:         {np.average(model.surfaceplant.heat_pump_electricity_kwh_used.value / 1E6):10.2f} ' + 'GWh/year' + NL)
        if not dispatch_report and model.surfaceplant.plant_type.value == PlantType.ABSORPTION_CHILLER:
            f.write(f'      Maximum Cooling Production:                       {np.max(model.surfaceplant.cooling_produced.value):10.2f} ' + model.surfaceplant.cooling_produced.PreferredUnits.value + NL)
            f.write(f'      Average Cooling Production:                       {np.average(model.surfaceplant.cooling_produced.value):10.2f} ' + model.surfaceplant.cooling_produced.PreferredUnits.value + NL)
            f.write(f'      Minimum Cooling Production:                       {np.min(model.surfaceplant.cooling_produced.value):10.2f} ' + model.surfaceplant.cooling_produced.PreferredUnits.value + NL)
            f.write(f'      Initial Cooling Production:                       {model.surfaceplant.cooling_produced.value[0]:10.2f} ' + model.surfaceplant.cooling_produced.PreferredUnits.value + NL)
            f.write(f'      Average Annual Cooling Production:                {np.average(model.surfaceplant.cooling_kWh_Produced.value / 1E6):10.2f} ' + 'GWh/year' + NL)

        if not dispatch_report and model.surfaceplant.plant_type.value == PlantType.DISTRICT_HEATING:
            f.write(f'      Annual District Heating Demand:                   {model.surfaceplant.annual_heating_demand.value:10.2f} ' + model.surfaceplant.annual_heating_demand.PreferredUnits.value + NL)
            f.write(f'      Maximum Daily District Heating Demand:            {np.max(model.surfaceplant.daily_heating_demand.value):10.2f} ' + model.surfaceplant.daily_heating_demand.PreferredUnits.value + NL)
            f.write(f'      Average Daily District Heating Demand:            {np.average(model.surfaceplant.daily_heating_demand.value):10.2f} ' + model.surfaceplant.daily_heating_demand.PreferredUnits.value + NL)
            f.write(f'      Minimum Daily District Heating Demand:            {np.min(model.surfaceplant.daily_heating_demand.value):10.2f} ' + model.surfaceplant.daily_heating_demand.PreferredUnits.value + NL)
            f.write(f'      Maximum Geothermal Heating Production:            {np.max(model.surfaceplant.dh_geothermal_heating.value):10.2f} ' + model.surfaceplant.dh_geothermal_heating.PreferredUnits.value + NL)
            f.write(f'      Average Geothermal Heating Production:            {np.average(model.surfaceplant.dh_geothermal_heating.value):10.2f} ' + model.surfaceplant.dh_geothermal_heating.PreferredUnits.value + NL)
            f.write(f'      Minimum Geothermal Heating Production:            {np.min(model.surfaceplant.dh_geothermal_heating.value):10.2f} ' + model.surfaceplant.dh_geothermal_heating.PreferredUnits.value + NL)
            f.write(f'      Maximum Peaking Boiler Heat Production:           {np.max(model.surfaceplant.dh_natural_gas_heating.value):10.2f} ' + model.surfaceplant.dh_natural_gas_heating.PreferredUnits.value + NL)
            f.write(f'      Average Peaking Boiler Heat Production:           {np.average(model.surfaceplant.dh_natural_gas_heating.value):10.2f} ' + model.surfaceplant.dh_natural_gas_heating.PreferredUnits.value + NL)
            f.write(f'      Minimum Peaking Boiler Heat Production:           {np.min(model.surfaceplant.dh_natural_gas_heating.value):10.2f} ' + model.surfaceplant.dh_natural_gas_heating.PreferredUnits.value + NL)

        f.write(f'      Average Pumping Power:                            {np.average(model.wellbores.PumpingPower.value):10.2f} {model.wellbores.PumpingPower.CurrentUnits.value}{NL}')

        if not dispatch_report and model.surfaceplant.heat_to_power_conversion_efficiency.value is not None:
            hpce = model.surfaceplant.heat_to_power_conversion_efficiency
            f.write(f'      {Outputs._field_label(hpce.Name, 50)}'
                    f'{hpce.value:10.2f} {model.surfaceplant.heat_to_power_conversion_efficiency.CurrentUnits.value}\n')

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
        # if we are dealing with overpressure and two different reservoirs, show a table reporting the values
        if model.wellbores.overpressure_percentage.Provided:
            f.write(NL)
            f.write('                            ***************************************\n')
            f.write('                            *  RESERVOIR POWER REQUIRED PROFILES  *\n')
            f.write('                            ***************************************\n')
            f.write('  YEAR     PROD PUMP     INJECT PUMP     TOTAL PUMP\n')
            f.write('             POWER          POWER           POWER\n')
            f.write(
                '             (' + model.wellbores.PumpingPowerProd.CurrentUnits.value + ')           (' + model.wellbores.PumpingPowerInj.CurrentUnits.value + ')            (' + model.surfaceplant.NetElectricityProduced.CurrentUnits.value + ')                  \n')
            for i in range(0, model.surfaceplant.plant_lifetime.value):
                f.write('  {0:2.0f}     {1:8.4f}        {2:8.4f}       {3:8.4f}'.format(i + 1,
                                                                                model.wellbores.PumpingPowerProd.value[
                                                                                    i * model.economics.timestepsperyear.value],
                                                                                model.wellbores.PumpingPowerInj.value[
                                                                                    i * model.economics.timestepsperyear.value],
                                                                                model.wellbores.PumpingPower.value[
                                                                                    i * model.economics.timestepsperyear.value]))
                f.write(NL)
            f.write(NL)

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

        if self.html_output_file.Provided:
            self.html_output_file.Valid = True

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
        write_dispatch_profile_output(model, self.dispatch_profile_output_file.value)

