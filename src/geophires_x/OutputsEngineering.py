from __future__ import annotations

from io import TextIOWrapper
from typing import TYPE_CHECKING

import numpy as np

from geophires_x.OutputsDispatch import has_electricity_component
from geophires_x.OutputsReport import field_label

if TYPE_CHECKING:
    from geophires_x.Model import Model

NL = "\n"

VERTICAL_WELL_DEPTH_OUTPUT_NAME = "Well depth"


def write_engineering_parameters(model: Model, f: TextIOWrapper) -> None:
    f.write(NL)
    f.write("                          ***ENGINEERING PARAMETERS***\n")
    f.write(NL)
    f.write(f"      Number of Production Wells:                    {model.wellbores.nprod.value:10.0f}" + NL)
    f.write(f"      Number of Injection Wells:                     {model.wellbores.ninj.value:10.0f}" + NL)
    f.write(
        f"      {field_label(VERTICAL_WELL_DEPTH_OUTPUT_NAME, 49)}"
        f"{model.reserv.depth.value:10.1f} {model.reserv.depth.CurrentUnits.value}{NL}"
    )
    f.write(f"      Water loss rate:                                 {model.reserv.waterloss.value:10.1f} {model.reserv.waterloss.CurrentUnits.value}\n")
    f.write(
        f"      Pump efficiency:                                 {model.surfaceplant.pump_efficiency.value:10.1f} "
        + model.surfaceplant.pump_efficiency.CurrentUnits.value
        + NL
    )
    f.write(
        f"      Injection temperature:                           {model.wellbores.Tinj.value:10.1f} "
        + model.wellbores.Tinj.CurrentUnits.value
        + NL
    )
    if model.wellbores.rameyoptionprod.value:
        f.write("      Production Wellbore heat transmission calculated with Ramey's model\n")
        f.write(
            f"      Average production well temperature drop:        {np.average(model.wellbores.ProdTempDrop.value):10.1f} "
            + model.wellbores.ProdTempDrop.PreferredUnits.value
            + NL
        )
    else:
        f.write("      User-provided production well temperature drop\n")
        f.write(
            f"      Constant production well temperature drop:       {model.wellbores.tempdropprod.value:10.1f} "
            + model.wellbores.tempdropprod.PreferredUnits.value
            + NL
        )
    f.write(
        f"      Flowrate per production well:                    {model.wellbores.prodwellflowrate.value:10.1f} "
        + model.wellbores.prodwellflowrate.CurrentUnits.value
        + NL
    )
    f.write(
        f"      {model.wellbores.injection_well_casing_inner_diameter.display_name}:                          "
        f"{model.wellbores.injection_well_casing_inner_diameter.value:10.3f} "
        f"{model.wellbores.injection_well_casing_inner_diameter.CurrentUnits.value}\n"
    )
    f.write(
        f"      {model.wellbores.production_well_casing_inner_diameter.display_name}:                         "
        f"{model.wellbores.production_well_casing_inner_diameter.value:10.3f} "
        f"{model.wellbores.production_well_casing_inner_diameter.CurrentUnits.value}\n"
    )
    if model.wellbores.IsAGS.value and model.wellbores.tot_vert_m.value > 0:
        f.write(
            f"      Vertical length of wellbore (per vertical):         "
            f"{(model.wellbores.tot_vert_m.value / (model.wellbores.nprod.value + model.wellbores.ninj.value)):10.1f} "
            + model.wellbores.tot_vert_m.CurrentUnits.value
            + NL
        )
    if model.wellbores.IsAGS.value and model.wellbores.tot_lateral_m.value > 0:
        f.write(
            f"      Lateral length of wellbore (per lateral):           "
            f"{(model.wellbores.tot_lateral_m.value / model.wellbores.numnonverticalsections.value):10.1f} "
            + model.wellbores.tot_lateral_m.CurrentUnits.value
            + NL
        )
    f.write(f"      {model.wellbores.redrill.display_name}:                    {model.wellbores.redrill.value:10.0f}\n")
    if has_electricity_component(model.surfaceplant.enduse_option.value):
        f.write("      Power plant type:                                       " + str(model.surfaceplant.plant_type.value.value) + NL)
