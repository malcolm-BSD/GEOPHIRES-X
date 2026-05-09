from __future__ import annotations

import re
from io import StringIO
from io import TextIOWrapper
from typing import TYPE_CHECKING

from geophires_x.OutputsUtils import OutputTableItem

if TYPE_CHECKING:
    from geophires_x.Model import Model

NL = "\n"


def resource_characteristic_output_items(model: Model) -> list[OutputTableItem]:
    section_text = StringIO()
    write_resource_characteristics(model, section_text)
    return _resource_characteristic_output_items_from_text(section_text.getvalue())


_RESOURCE_VALUE_PATTERN = re.compile(
    r"^(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?|NaN|N/A)(?:\s+(?P<units>.+))?$"
)


def _resource_characteristic_output_items_from_text(section_text: str) -> list[OutputTableItem]:
    items = []
    for line in section_text.splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line == "***RESOURCE CHARACTERISTICS***":
            continue

        if ":" not in stripped_line:
            items.append(OutputTableItem(stripped_line))
            continue

        parameter, raw_value = stripped_line.split(":", 1)
        value = raw_value.strip()
        units = ""
        value_match = _RESOURCE_VALUE_PATTERN.match(value)
        if value_match is not None:
            value = value_match.group("value")
            units = value_match.group("units") or ""

        items.append(OutputTableItem(parameter.strip(), value, units))

    return items


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
