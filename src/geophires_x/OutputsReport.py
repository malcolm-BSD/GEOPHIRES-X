from __future__ import annotations

from io import TextIOWrapper
from typing import Tuple, Union

NL = "\n"

ScalarRow = Tuple[str, Union[str, float], Union[str, None]]


def field_label(field_name: str, print_width_before_value: int) -> str:
    return f'{field_name}:{" " * (print_width_before_value - len(field_name) - 1)}'


def write_scalar_section(
    f: TextIOWrapper,
    category_name: str,
    rows: list[ScalarRow],
    *,
    label_width: int = 49,
) -> None:
    if len(rows) == 0:
        return

    f.write(NL)
    f.write(NL)
    f.write(f"                           ***{category_name}***\n")
    f.write(NL)
    for field_name, value, units in rows:
        if isinstance(value, str):
            f.write(f"      {field_label(field_name, label_width)}{value}\n")
        else:
            f.write(f"      {field_label(field_name, label_width)}{value:10.2f} {units}\n")
