from __future__ import annotations

import math
from io import TextIOWrapper
from typing import TYPE_CHECKING

import numpy as np

from geophires_x.OptionList import FractureShape, ReservoirModel, ReservoirVolume
from geophires_x.OutputsReport import field_label

if TYPE_CHECKING:
    from geophires_x.Model import Model

NL = "\n"


def write_reservoir_parameters(model: Model, f: TextIOWrapper) -> None:
    f.write(NL)
    f.write(NL)
    f.write("                           ***RESERVOIR PARAMETERS***\n")
    f.write(NL)
    if model.wellbores.IsAGS.value:
        f.write("The AGS models contain an intrinsic reservoir model that doesn't expose values that can be used in extensive reporting.\n")
    else:
        f.write(f"      Reservoir Model = {model.reserv.resoption.value.display_name}\n")
        if model.reserv.resoption.value is ReservoirModel.SINGLE_FRACTURE:
            f.write(f"      m/A Drawdown Parameter:                                 {model.reserv.drawdp.value:.5f} " + model.reserv.drawdp.CurrentUnits.value + NL)
        elif model.reserv.resoption.value is ReservoirModel.ANNUAL_PERCENTAGE:
            f.write(f"      Annual Thermal Drawdown:                                {model.reserv.drawdp.value * 100:.3f} " + model.reserv.drawdp.CurrentUnits.value + NL)
        f.write(f"      Bottom-hole temperature:                          {model.reserv.Trock.value:10.2f} {model.reserv.Trock.CurrentUnits.value}\n")
        if model.reserv.resoption.value in [
            ReservoirModel.ANNUAL_PERCENTAGE,
            ReservoirModel.USER_PROVIDED_PROFILE,
            ReservoirModel.TOUGH2_SIMULATOR,
        ]:
            f.write("      Warning: the reservoir dimensions and thermo-physical properties \n")
            f.write("               listed below are default values if not provided by the user.   \n")
            f.write("               They are only used for calculating remaining heat content.  \n")
            # TODO parse this note in GeophiresXResult

        if model.reserv.resoption.value in [ReservoirModel.MULTIPLE_PARALLEL_FRACTURES, ReservoirModel.LINEAR_HEAT_SWEEP]:
            f.write(f"      Fracture model = {model.reserv.fracshape.value.value}\n")
            if model.reserv.fracshape.value == FractureShape.CIRCULAR_AREA:
                f.write(f"      Well separation: fracture diameter:               {model.reserv.fracheightcalc.value:10.2f} " + model.reserv.fracheight.CurrentUnits.value + NL)
            elif model.reserv.fracshape.value == FractureShape.CIRCULAR_DIAMETER:
                f.write(f"      Well separation: fracture diameter:               {model.reserv.fracheightcalc.value:10.2f} " + model.reserv.fracheight.CurrentUnits.value + NL)
            elif model.reserv.fracshape.value == FractureShape.SQUARE:
                f.write(f"      Well separation: fracture height:                 {model.reserv.fracheightcalc.value:10.2f} " + model.reserv.fracheight.CurrentUnits.value + NL)
            elif model.reserv.fracshape.value == FractureShape.RECTANGULAR:
                f.write(f"      Well separation: fracture height:                 {model.reserv.fracheightcalc.value:10.2f} " + model.reserv.fracheight.CurrentUnits.value + NL)
                f.write(f"      {model.reserv.fracwidthcalc.display_name}:                                             {model.reserv.fracwidthcalc.value:10.2f} {model.reserv.fracwidth.CurrentUnits.value }\n")
            f.write(f"      {model.reserv.fracareacalc.display_name}:                                    {model.reserv.fracareacalc.value:10.2f} {model.reserv.fracarea.CurrentUnits.value}\n")
        if model.reserv.resvoloption.value == ReservoirVolume.FRAC_NUM_SEP:
            f.write("      Reservoir volume calculated with fracture separation and number of fractures as input\n")
        elif model.reserv.resvoloption.value == ReservoirVolume.RES_VOL_FRAC_SEP:
            f.write("      Number of fractures calculated with reservoir volume and fracture separation as input\n")
        elif model.reserv.resvoloption.value == ReservoirVolume.FRAC_NUM_SEP:
            f.write("      Fracture separation calculated with reservoir volume and number of fractures as input\n")
        elif model.reserv.resvoloption.value == ReservoirVolume.RES_VOL_ONLY:
            f.write("      Reservoir volume provided as input\n")
        if model.reserv.resvoloption.value in [ReservoirVolume.FRAC_NUM_SEP, ReservoirVolume.RES_VOL_FRAC_SEP, ReservoirVolume.FRAC_NUM_SEP]:
            frac_num_label = field_label(model.reserv.fracnumbcalc.display_name, 56)
            f.write(f"      {frac_num_label}{math.ceil(model.reserv.fracnumbcalc.value)}\n")
            f.write(f"      {model.reserv.fracsepcalc.display_name}:                              {model.reserv.fracsepcalc.value:10.2f} {model.reserv.fracsep.CurrentUnits.value}\n")
        f.write(f"      Reservoir volume:                              {model.reserv.resvolcalc.value:10.0f} {model.reserv.resvol.CurrentUnits.value}\n")

        if model.wellbores.impedancemodelused.value:
            # See note re: unit conversion:
            # https://github.com/NREL/GEOPHIRES-X/blob/d51eb8d1dc8b21c7a79c4d35f296d740347658e0/src/geophires_x/WellBores.py#L1280-L1282
            f.write(f"      Reservoir impedance:                              {model.wellbores.impedance.value / 1000:10.4f} {model.wellbores.impedance.CurrentUnits.value}\n")
        else:
            if model.wellbores.overpressure_percentage.Provided:
                # write the reservoir pressure as an average in the overpressure case
                f.write(f"      {model.wellbores.average_production_reservoir_pressure.display_name}:                       {model.wellbores.average_production_reservoir_pressure.value:10.2f} {model.wellbores.average_production_reservoir_pressure.CurrentUnits.value}\n")
            else:
                # write the reservoir pressure as a single value
                f.write(f"      Reservoir hydrostatic pressure:                       {model.wellbores.production_reservoir_pressure.value[0]:10.2f} " + model.wellbores.production_reservoir_pressure.CurrentUnits.value + NL)
            f.write(f"      Plant outlet pressure:                            {model.surfaceplant.plant_outlet_pressure.value:10.2f} " + model.surfaceplant.plant_outlet_pressure.CurrentUnits.value + NL)
            if model.wellbores.productionwellpumping.value:
                f.write(f"      Production wellhead pressure:                     {model.wellbores.Pprodwellhead.value:10.2f} " + model.wellbores.Pprodwellhead.CurrentUnits.value + NL)
                f.write(f"      Productivity Index:                               {model.wellbores.PI.value:10.2f} " + model.wellbores.PI.CurrentUnits.value + NL)
            f.write(f"      Injectivity Index:                                {model.wellbores.II.value:10.2f} " + model.wellbores.II.CurrentUnits.value + NL)

        f.write(f"      Reservoir density:                                {model.reserv.rhorock.value:10.2f} " + model.reserv.rhorock.CurrentUnits.value + NL)
        if model.wellbores.rameyoptionprod.value or model.reserv.resoption.value in [
            ReservoirModel.MULTIPLE_PARALLEL_FRACTURES,
            ReservoirModel.LINEAR_HEAT_SWEEP,
            ReservoirModel.SINGLE_FRACTURE,
            ReservoirModel.TOUGH2_SIMULATOR,
        ]:
            f.write(f"      Reservoir thermal conductivity:                   {model.reserv.krock.value:10.2f} {model.reserv.krock.CurrentUnits.value}{NL}")
        f.write(f"      Reservoir heat capacity:                          {model.reserv.cprock.value:10.2f} " + model.reserv.cprock.CurrentUnits.value + NL)
        if model.reserv.resoption.value is ReservoirModel.LINEAR_HEAT_SWEEP or (
            model.reserv.resoption.value is ReservoirModel.TOUGH2_SIMULATOR and model.reserv.usebuiltintough2model
        ):
            f.write(f"      Reservoir porosity:                               {model.reserv.porrock.value * 100:10.2f} " + model.reserv.porrock.CurrentUnits.value + NL)
        if model.reserv.resoption.value is ReservoirModel.TOUGH2_SIMULATOR and model.reserv.usebuiltintough2model:
            f.write(f"      Reservoir permeability:                           {model.reserv.permrock.value:10.2E} " + model.reserv.permrock.CurrentUnits.value + NL)
            f.write(f"      Reservoir thickness:                              {model.reserv.resthickness.value:10.2f} " + model.reserv.resthickness.CurrentUnits.value + NL)
            f.write(f"      Reservoir width:                                  {model.reserv.reswidth.value:10.2f} " + model.reserv.reswidth.CurrentUnits.value + NL)
            f.write(f"      Well separation:                                  {model.wellbores.wellsep.value:10.2f} " + model.wellbores.wellsep.CurrentUnits.value + NL)


def write_reservoir_simulation_results(model: Model, f: TextIOWrapper, dispatch_report: bool) -> None:
    if not dispatch_report:
        f.write(NL)
        f.write(NL)
        f.write("                           ***RESERVOIR SIMULATION RESULTS***\n")
        f.write(NL)
        f.write(f"      Maximum Production Temperature:                  {np.max(model.wellbores.ProducedTemperature.value):10.1f} " + model.wellbores.ProducedTemperature.PreferredUnits.value + NL)
        f.write(f"      Average Production Temperature:                  {np.average(model.wellbores.ProducedTemperature.value):10.1f} " + model.wellbores.ProducedTemperature.PreferredUnits.value + NL)
        f.write(f"      Minimum Production Temperature:                  {np.min(model.wellbores.ProducedTemperature.value):10.1f} " + model.wellbores.ProducedTemperature.PreferredUnits.value + NL)
        f.write(f"      Initial Production Temperature:                  {model.wellbores.ProducedTemperature.value[0]:10.1f} " + model.wellbores.ProducedTemperature.PreferredUnits.value + NL)
        if model.wellbores.IsAGS.value:
            f.write("The AGS models contain an intrinsic reservoir model that doesn't expose values that can be used in extensive reporting.\n")
        else:
            f.write(f"      Average Reservoir Heat Extraction:                {np.average(model.surfaceplant.HeatExtracted.value):10.2f} " + model.surfaceplant.HeatExtracted.PreferredUnits.value + NL)
            if model.wellbores.rameyoptionprod.value:
                f.write("      Production Wellbore Heat Transmission Model = Ramey Model\n")
                f.write(f"      Average Production Well Temperature Drop:        {np.average(model.wellbores.ProdTempDrop.value):10.1f} " + model.wellbores.ProdTempDrop.PreferredUnits.value + NL)
            else:
                f.write(f"      Wellbore Heat Transmission Model = Constant Temperature Drop:{model.wellbores.tempdropprod.value:10.1f} " + model.wellbores.tempdropprod.PreferredUnits.value + NL)
            if model.wellbores.impedancemodelused.value:
                f.write(f"      Total Average Pressure Drop:                     {np.average(model.wellbores.DPOverall.value):10.1f} " + model.wellbores.DPOverall.PreferredUnits.value + NL)
                f.write(f"      Average Injection Well Pressure Drop:            {np.average(model.wellbores.DPInjWell.value):10.1f} " + model.wellbores.DPInjWell.PreferredUnits.value + NL)
                f.write(f"      Average Reservoir Pressure Drop:                 {np.average(model.wellbores.DPReserv.value):10.1f} " + model.wellbores.DPReserv.PreferredUnits.value + NL)
                f.write(f"      Average Production Well Pressure Drop:           {np.average(model.wellbores.DPProdWell.value):10.1f} " + model.wellbores.DPProdWell.PreferredUnits.value + NL)
                f.write(f"      Average Buoyancy Pressure Drop:                  {np.average(model.wellbores.DPBouyancy.value):10.1f} " + model.wellbores.DPBouyancy.PreferredUnits.value + NL)
            else:
                f.write(f"      Average Injection Well Pump Pressure Drop:       {np.average(model.wellbores.DPInjWell.value):10.1f} " + model.wellbores.DPInjWell.PreferredUnits.value + NL)
                if model.wellbores.productionwellpumping.value:
                    f.write(f"      Average Production Well Pump Pressure Drop:      {np.average(model.wellbores.DPProdWell.value):10.1f} " + model.wellbores.DPProdWell.PreferredUnits.value + NL)


def write_reservoir_power_required_profiles(model: Model, f: TextIOWrapper) -> None:
    # if we are dealing with overpressure and two different reservoirs, show a table reporting the values
    if model.wellbores.overpressure_percentage.Provided:
        f.write(NL)
        f.write("                            ***************************************\n")
        f.write("                            *  RESERVOIR POWER REQUIRED PROFILES  *\n")
        f.write("                            ***************************************\n")
        f.write("  YEAR     PROD PUMP     INJECT PUMP     TOTAL PUMP\n")
        f.write("             POWER          POWER           POWER\n")
        f.write(
            "             (" + model.wellbores.PumpingPowerProd.CurrentUnits.value + ")           (" + model.wellbores.PumpingPowerInj.CurrentUnits.value + ")            (" + model.surfaceplant.NetElectricityProduced.CurrentUnits.value + ")                  \n")
        for i in range(0, model.surfaceplant.plant_lifetime.value):
            f.write("  {0:2.0f}     {1:8.4f}        {2:8.4f}       {3:8.4f}".format(i + 1,
                                                                            model.wellbores.PumpingPowerProd.value[
                                                                                i * model.economics.timestepsperyear.value],
                                                                            model.wellbores.PumpingPowerInj.value[
                                                                                i * model.economics.timestepsperyear.value],
                                                                            model.wellbores.PumpingPower.value[
                                                                                i * model.economics.timestepsperyear.value]))
            f.write(NL)
        f.write(NL)
