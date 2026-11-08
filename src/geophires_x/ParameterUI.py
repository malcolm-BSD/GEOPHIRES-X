"""Validation and defaults for :class:`geophires_x.Parameter.Parameter` UI metadata.

UI placement and calculation dependencies are declared alongside each parameter. This module deliberately contains
no parameter-name-specific metadata: it only fills backwards-compatible defaults and validates the common structure.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


PARAMETER_UI_FIELDS = {"tab", "group", "order", "dependency"}
PARAMETER_UI_DEFAULTS: dict[str, Any] = {
    "tab": "Project",
    "group": "",
    "order": 0,
    "dependency": None,
}


def complete_parameter_ui(name: str, ui: dict[str, Any] | None) -> dict[str, Any]:
    """Fill omitted defaults without overriding metadata declared by the parameter."""
    completed = {**deepcopy(PARAMETER_UI_DEFAULTS), **deepcopy(ui or {})}
    unknown_fields = set(completed) - PARAMETER_UI_FIELDS
    if unknown_fields:
        raise ValueError(f"UI metadata for {name} has unsupported fields: {sorted(unknown_fields)}")
    if not isinstance(completed["tab"], str):
        raise TypeError(f"UI metadata tab for {name} must be a string")
    if not isinstance(completed["group"], str):
        raise TypeError(f"UI metadata group for {name} must be a string")
    order = completed["order"]
    if not isinstance(order, int) or isinstance(order, bool) or order < 0:
        raise ValueError(f"UI metadata order for {name} must be a non-negative integer")
    dependency = completed["dependency"]
    if dependency is not None and not isinstance(dependency, dict):
        raise TypeError(f"UI metadata dependency for {name} must be a dictionary or None")
    return completed
