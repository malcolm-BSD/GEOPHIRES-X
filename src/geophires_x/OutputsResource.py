from __future__ import annotations

from io import TextIOWrapper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geophires_x.Model import Model

NL = "\n"


def write_resource_characteristics(model: Model, f: TextIOWrapper) -> None:
    f.write(NL)
    f.write(NL)
    f.write("                         ***RESOURCE CHARACTERISTICS***\n")
    f.write(NL)
    f.write(
        f"      Maximum reservoir temperature:                   {model.reserv.Tmax.value:10.1f} {model.reserv.Tmax.CurrentUnits.value}\n"
    )
    f.write(f"      Number of segments:                            {model.reserv.numseg.value:10.0f}\n")
    if model.reserv.numseg.value == 1:
        f.write(
            f"      Geothermal gradient:                                {model.reserv.gradient.value[0]:10.4g} {model.reserv.gradient.CurrentUnits.value}\n"
        )
    else:
        for i in range(1, model.reserv.numseg.value):
            f.write(
                f"      Segment {str(i):s}   Geothermal gradient:                    {model.reserv.gradient.value[i - 1]:10.4g} {model.reserv.gradient.CurrentUnits.value}\n"
            )
            f.write(
                f"      Segment {str(i):s}   Thickness:                         {round(model.reserv.layerthickness.value[i - 1], 10)} {model.reserv.layerthickness.CurrentUnits.value}\n"
            )
        f.write(
            f"      Segment {str(i + 1):s}   Geothermal gradient:                    {model.reserv.gradient.value[i]:10.4g} {model.reserv.gradient.CurrentUnits.value}\n"
        )
