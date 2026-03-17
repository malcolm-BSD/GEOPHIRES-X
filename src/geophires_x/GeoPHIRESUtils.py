from __future__ import annotations

import os
import math
import requests
import csv
import re
import logging
import sys
from enum import Enum
from os.path import exists
import dataclasses
import json
import numbers
import io
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
import ast
import scipy
from pint.facets.plain import PlainQuantity
from scipy.interpolate import interp1d
import numpy as np
from typing import Any, Optional, Union, Literal, Tuple, List, Sequence, Iterable
import CoolProp.CoolProp as cp

from geophires_x.Units import get_unit_registry, convertible_unit, get_unit_from_string

SingleSeries = List[float]
DualSeries = List[Tuple[float, float]]
ParsedSeries = Union[SingleSeries, DualSeries]
_PARENS_RE = re.compile(r"\([^)]*\)")  # matches one (...) group
Number = Union[int, float, np.number]
ContainerKind = Literal["list", "tuple", "set", "ndarray"]

_logger = logging.getLogger('root')  # TODO use __name__ instead of root

_T = np.array(
    [
        0.01,
        10.0,
        20.0,
        25.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
        110.0,
        120.0,
        140.0,
        160.0,
        180.0,
        200.0,
        220.0,
        240.0,
        260.0,
        280.0,
        300.0,
        320.0,
        340.0,
        360.0,
        373.946,
        600.0,
    ]
)

# TODO needs citation
_UtilEff = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0057,
        0.0337,
        0.0617,
        0.0897,
        0.1177,
        0.13,
        0.16,
        0.19,
        0.22,
        0.26,
        0.29,
        0.32,
        0.35,
        0.38,
        0.40,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.5, # Extrapolate from fig 2 in https://geothermal-energy-journal.springeropen.com/articles/10.1186/s40517-019-0119-6
    ]
)

_interp_util_eff_func = interp1d(_T, _UtilEff)

_ureg = get_unit_registry()


def InsertImagesIntoHTML(html_path: str, short_names=None, full_names: set = None) -> None:
    if full_names is None:
        full_names = short_names

    # If still None/empty, nothing to insert
    if not full_names:
        return

    # Write a reference to the image(s) into the HTML file by inserting before the "</body>" tag
    insert_string = ''
    for _ in range(len(full_names)):
        full_name = full_names.pop()
        name_to_use = full_name.name.replace('_', ' ').replace('.png', '')
        insert_string = insert_string + f'<img src="{full_name.name}" alt="{name_to_use}">\n<br>'

    match_string = '</body>'
    with open(html_path, 'r+', encoding='UTF-8') as html_file:
        contents = html_file.readlines()
        if match_string in contents[-1]:  # Handle last line to prevent IndexError
            pass
        else:
            for index, line in enumerate(contents):
                if match_string in line and insert_string not in contents[index + 1]:
                    contents.insert(index, insert_string)
                    break
        html_file.seek(0)
        html_file.writelines(contents)


def UpgradeSymbologyOfUnits(unit: str) -> str:
    """
    UpgradeSymbologyOfUnits is a function that takes a string that represents a unit and replaces the **2 and **3
    with the appropriate Unicode characters for superscript 2 and 3, and replaces "deg" with the Unicode character
    for degrees.
    :param unit: a string that represents a unit
    :return: a string that represents a unit with the appropriate Unicode characters for superscript 2 and 3, and
    replaces "deg" with the Unicode character for degrees.
    """

    return unit.replace('**2', '\u00b2').replace('**3', '\u00b3').replace('deg', '\u00b0')


def render_default(p: float, unit: str = '', fmt: str = '') -> str:
    """
    RenderDefault - render a float as a string with 2 decimal place by default, or whatever format the user specifies,
     or in scientific notation if it is greater than 10,000
     with the unit appended to it if it is not an empty string (the default)
    :param p: the float to render
    :type p: float
    :param unit: the unit to append to the string
    :type unit: str
    :param fmt: the format to use for the string representation of the float
    :type fmt: str
    :return: the string representation of the float
    """
    if not is_float(p):
        raise ValueError(f'Parameter ({p}) must be a float or convertible to float.')

    unit = UpgradeSymbologyOfUnits(unit)
    # if the number is greater than 10,000, render it in scientific notation
    if p > 10_000:
        return render_scientific(p, unit)
    # otherwise, render it with 2 decimal places
    else:
        if not fmt:
            return f'{p:10.2f} {unit}'.strip()
        else:
            if ':' in fmt:
                fmt = fmt.split(':')[1]
            fmt = '{0:' + fmt + '}{1:s}'
            return fmt.format(p, unit.strip())


def render_scientific(
    p: Any,
    unit: str = "",
    fmt: str = "",
    *,
    default_fmt: str = "10.2e",
    allow_nan: bool = True,
    allow_inf: bool = True,
) -> str:
    """
    Render a number as a string (scientific notation by default) with an optional unit.

    :param p: Value to render. Must be float-like (convertible via float()).
    :param unit: Unit string to append (after UpgradeSymbologyOfUnits()).
    :param fmt: Optional Python format *spec* (e.g., "10.2e", ".3g", "12.4e"). If empty, uses `default_fmt`.
    :param default_fmt: Format spec used when `fmt` is empty. Defaults to "10.2e".
        :param allow_nan: If False, raise ValueError when p is NaN.
        :param allow_inf: If False, raise ValueError when p is infinite.

    :return: Formatted number with unit appended (if provided).
    """

    # Convert early; this is simpler and more idiomatic than a custom is_float() check
    try:
        x = float(p)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Parameter ({p}) must be a float or convertible to float.") from e

    if (not allow_nan) and math.isnan(x):
        raise ValueError(f"Parameter ({p}) is NaN, which is not allowed.")
    if (not allow_inf) and math.isinf(x):
        raise ValueError(f"Parameter ({p}) is infinite, which is not allowed.")

    unit_str = UpgradeSymbologyOfUnits(unit or "")
    spec = fmt.strip() or default_fmt

    # Use format() to apply a dynamic format spec
    try:
        num_str = format(x, spec)
    except ValueError as e:
        raise ValueError(f"Invalid format specifier '{spec}'.") from e

    return f"{num_str} {unit_str}".strip()


def render_parameter_default(p) -> str:
    """
    Render a Parameter's value using the default formatter and append its current unit.
    :param p: Parameter instance with .value and .CurrentUnits.value attributes
    :return: Formatted string of the parameter's value with unit appended
    :raises ValueError: If `p` is None or lacks expected attributes.
    """
    if p is None:
        raise ValueError("Parameter must not be None.")

    try:
        value: Any = p.value
        unit: str = (p.CurrentUnits.value if p.CurrentUnits is not None else "") or ""
    except AttributeError as e:
        raise ValueError(f"Invalid Parameter object; missing expected attributes: {e}") from e

    # Let render_default() handle float conversion/validation
    return render_default(value, unit)


def render_parameter_scientific(p) -> str:
    """
    Render a Parameter's value in scientific notation and append its current unit.
    :param p: Parameter instance with .value and .CurrentUnits.value attributes
    :return: Formatted string of the parameter's value in scientific notation with unit appended
    :raises ValueError: If `p.value` is not float-like (convertible to float), or if `p` lacks expected attributes.
    """
    if p is None:
        raise ValueError("Parameter must not be None.")

    # Be defensive about attribute presence (helps catch wiring errors cleanly)
    try:
        value: Any = p.value
        unit: str = (p.CurrentUnits.value if p.CurrentUnits is not None else "") or ""
    except AttributeError as e:
        raise ValueError(f"Invalid Parameter object; missing expected attributes: {e}") from e

    # Delegate conversion/validation + unit symbology to render_scientific()
    return render_scientific(value, unit)

def quantity(value: float, unit: str) -> PlainQuantity:
    """
    :rtype: pint.registry.Quantity - note type annotation uses PlainQuantity due to issues with python 3.8 failing
        to import the Quantity TypeAlias
    """
    return _ureg.Quantity(value, convertible_unit(unit))


@lru_cache
def density_water_kg_per_m3(Twater_degC: float, pressure: Optional[PlainQuantity] = None) -> float:
    """
    Calculate the density of water as a function of temperature.

    Args:
        Twater_degC: The temperature of water in degrees C.
        pressure: Pressure - should be provided as a Pint quantity that knows its units
    Returns:
        The density of water in kg/m³.
    Raises:
        ValueError: If Twater_degC is not a float or convertible to float.
    """
    if not is_float(Twater_degC):
        raise ValueError(f'Twater_degC ({Twater_degC}) must be a float or convertible to float.')

    try:
        if pressure is not None:
            return cp.PropsSI('D', 'T', celsius_to_kelvin(Twater_degC), 'P', pressure.to('Pa').magnitude, 'Water')
        else:
            _logger.warning(f'density_water: No pressure provided, using vapor quality=0 instead')
            return cp.PropsSI('D', 'T', celsius_to_kelvin(Twater_degC), 'Q', 0, 'Water')

    except (NotImplementedError, ValueError) as e:
        raise ValueError(f'Input temperature & pressure ({Twater_degC}, {pressure}) '
                         f'are out of range or otherwise could not be used to calculate water density.') from e


def celsius_to_kelvin(celsius: float) -> float:
    """
    Convert temperature from Celsius to Kelvin.

    Args:
        celsius: Temperature in degrees Celsius.
    Returns:
        Temperature in Kelvin.
    Raises:
        ValueError: If celsius is not a float or convertible to float.
    """
    if not isinstance(celsius, (int, float)):
        raise ValueError(f"Invalid input for celsius ({celsius}). celsius must be a number.")

    CELSIUS_TO_KELVIN_CONSTANT = 273.15
    return celsius + CELSIUS_TO_KELVIN_CONSTANT


@lru_cache
def viscosity_water_Pa_sec(
    Twater_degC: float,
    pressure: Optional[PlainQuantity] = None) -> float:
    """
    Calculate the dynamic viscosity of water as a function of temperature and pressure.

    Args:
        Twater_degC: the temperature of water in degrees C
        pressure: Pressure - should be provided
    Returns:
        Viscosity of water in Pa·s (Ns/m2)
    Raises:
        ValueError: If Twater_degC is not a float or convertible to float.
    """

    try:
        if pressure is not None:
            return cp.PropsSI('V', 'T', celsius_to_kelvin(Twater_degC), 'P', pressure.to('Pa').magnitude, 'Water')
        else:
            _logger.warning(f'viscosity_water: No pressure provided, using vapor quality=0 instead')
            return cp.PropsSI('V', 'T', celsius_to_kelvin(Twater_degC), 'Q', 0, 'Water')

    except (NotImplementedError, ValueError) as e:
        raise ValueError(f'Input temperature & pressure ({Twater_degC}, {pressure}) '
                         f'are out of range or otherwise could not be used to calculate water viscosity.') from e


@lru_cache
def heat_capacity_water_J_per_kg_per_K(
    Twater_degC: float,
    pressure: Optional[PlainQuantity] = None,
) -> float:
    """
    Calculate the isobaric specific heat capacity (c_p) of water as a function of temperature.

    Args:
        Twater_degC: The temperature of water in degrees C.
        pressure: Pressure - should be provided
    Returns:
        The isobaric specific heat capacity of water as a function of temperature in J/(kg·K).
    Raises:
        ValueError: If Twater_degC is not a float or convertible to float.
    """
    max_allowed_temp_degC = 600
    if not isinstance(Twater_degC, numbers.Real) or Twater_degC < 0 or Twater_degC > max_allowed_temp_degC:
        raise ValueError(
            f'Invalid input for Twater_degC. '
            f'Twater_degC must be a non-negative number and must be within the range of 0 to {max_allowed_temp_degC} '
            f'degrees Celsius. The input value was: {Twater_degC}'
        )

    try:
        if pressure is not None:
            return cp.PropsSI('C', 'T', celsius_to_kelvin(Twater_degC), 'P', pressure.to('Pa').magnitude, 'Water')
        else:
            _logger.warning(f'heat_capacity_water: No pressure provided, using vapor quality=0 instead')
            return cp.PropsSI('C', 'T', celsius_to_kelvin(Twater_degC), 'Q', 0, 'Water')

    except (NotImplementedError, ValueError) as e:
        raise ValueError(f'Input temperature & pressure ({Twater_degC}, {pressure}) '
                         f'are out of range or otherwise could not be used to calculate heat capacity of water.') from e


@lru_cache
def RecoverableHeat(Twater_degC: float) -> float:
    """
    the RecoverableHeat function is used to calculate the recoverable heat fraction as a function of temperature

    Args:
        Twater_degC: the temperature of water in degrees C
    Returns:
        the recoverable heat fraction as a function of temperature
    Raises:
        ValueError: If Twater is not a float or convertible to float.
        ValueError: If DefaultRecoverableHeat is not a float or convertible to float.
    """
    LOW_TEMP_THRESHOLD = 90.0
    HIGH_TEMP_THRESHOLD = 150.0
    LOW_TEMP_RECOVERABLE_HEAT = 0.43
    HIGH_TEMP_RECOVERABLE_HEAT = 0.66

    if not isinstance(Twater_degC, (int, float)):
        raise ValueError(f'Twater_degC {Twater_degC} must be a number')

    if Twater_degC <= LOW_TEMP_THRESHOLD:
        recoverable_heat = LOW_TEMP_RECOVERABLE_HEAT
    elif Twater_degC >= HIGH_TEMP_THRESHOLD:
        recoverable_heat = HIGH_TEMP_RECOVERABLE_HEAT
    else:
        recoverable_heat = 0.0038 * Twater_degC + 0.085

    return recoverable_heat


@lru_cache
def vapor_pressure_water_kPa(temperature_degC: float) -> float:
    """
    Calculate the vapor pressure of water as a function of temperature.

    Args:
        temperature_degC: the temperature of water in degrees C
    Returns:
        The vapor pressure of water as a function of temperature in kPa
    Raises:
        ValueError: If temperature_degC is not a float or convertible to float.
        ValueError: If temperature_degC is below 0.
    """

    if not isinstance(temperature_degC, (int, float)):
        raise ValueError(f'Input temperature ({temperature_degC}) must be a number')
    if temperature_degC < 0:
        raise ValueError(f'Input temperature ({temperature_degC}C) must be greater than or equal to 0')

    try:
        return (quantity(cp.PropsSI('P', 'T', celsius_to_kelvin(temperature_degC), 'Q', 0, 'Water'), 'Pa')
                .to('kPa').magnitude)

    except (NotImplementedError, ValueError) as e:
        raise ValueError(f'Input temperature ({temperature_degC}C) is out of range or otherwise not implemented') from e


@lru_cache
def entropy_water_kJ_per_kg_per_K(temperature_degC: float, pressure: Optional[PlainQuantity] = None) -> float:
    """
    Calculate the entropy of water as a function of temperature

    Args:
        temperature_degC: the temperature of water in degrees C
        pressure: Pressure - should be provided as a Pint quantity that knows its units
    Returns:
        the entropy of water as a function of temperature in kJ/(kg·K)
    Raises:
        TypeError: If temperature is not a float or convertible to float.
        ValueError: If temperature and pressure combination are not within lookup range
    """

    try:
        temperature_degC = float(temperature_degC)
    except ValueError:
        raise TypeError(f'Input temperature ({temperature_degC}) must be a float')

    try:
        if pressure is not None:
            return cp.PropsSI('S', 'T', celsius_to_kelvin(temperature_degC),
                              'P', pressure.to('Pa').magnitude, 'Water') * 1e-3
        else:
            return cp.PropsSI('S', 'T', celsius_to_kelvin(temperature_degC), 'Q', 0, 'Water') * 1e-3
    except (NotImplementedError, ValueError) as e:
        raise ValueError(f'Input temperature {temperature_degC} is out of range or otherwise not implemented') from e


@lru_cache
def enthalpy_water_kJ_per_kg(temperature_degC: float, pressure: Optional[PlainQuantity] = None) -> float:
    """
    Calculate the enthalpy of water as a function of temperature

    Args:
        temperature_degC: the temperature of water in degrees C (float)
        pressure: Pressure - should be provided as a Pint quantity that knows its units
    Returns:
        the enthalpy of water as a function of temperature in kJ/kg
    Raises:
        TypeError: If temperature is not a float or convertible to float.
        ValueError: If temperature and pressure combination are not within lookup range
    """

    try:
        temperature_degC = float(temperature_degC)
    except ValueError:
        raise TypeError(f'Input temperature ({temperature_degC}) must be a float')

    try:
        if pressure is not None:
            return cp.PropsSI('H', 'T', celsius_to_kelvin(temperature_degC),
                              'P', pressure.to('Pa').magnitude, 'Water') * 1e-3
        else:
            return cp.PropsSI('H', 'T', celsius_to_kelvin(temperature_degC), 'Q', 0, 'Water') * 1e-3

    except (NotImplementedError, ValueError) as e:
        raise ValueError(f'Input temperature {temperature_degC} is out of range or otherwise not implemented') from e


@lru_cache
def UtilEff_func(temperature_degC: float) -> float:
    """
    the UtilEff_func function is used to calculate the utilization efficiency of the system as a function of temperature
    Args:
        temperature_degC: the temperature of water in degrees C
    Returns:
         the utilization efficiency of the system as a function of temperature
    Raises:
        ValueError: If x is not a float or convertible to float.
        ValueError: If x is not within the range of 0 to 373.946 degrees C.
    """

    if not isinstance(temperature_degC, (int, float)):
        raise ValueError(f'Input temperature ({temperature_degC}) must be a number')

    if temperature_degC < _T[0] or temperature_degC > _T[-1]:
        raise ValueError(f'Temperature ({temperature_degC}) must be within the range of {_T[0]} to {_T[-1]} degrees C.')

    util_eff = _interp_util_eff_func(temperature_degC)
    return util_eff


def read_input_file(return_dict_1, logger=None, input_file_name=None):
    """
    Read input file and return a dictionary of parameters
    :param return_dict_1: dictionary of parameters
    :param logger: logger object
    :param input_file_name: name of input file
    :return: dictionary of parameters
    :rtype: dict

    FIXME modifies dict instead of returning it - it should do what the doc says it does and return a dict instead,
      relying on mutation of parameters is Bad
    """
    from geophires_x.Parameter import ParameterEntry

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f'Init {__name__}')

    # Specify path of input file - it will always be the first command line argument.
    # If it doesn't exist, simply run the default model without any inputs

    # read input data (except input from optional filenames)
    if input_file_name is None:
        logger.warning('Input file name not provided, checking sys.argv')
        if len(sys.argv) > 1:
            input_file_name = sys.argv[1]
            logger.warning(f'Using input file from sys.argv: {input_file_name}')

    if input_file_name:
        if not exists(input_file_name):
            raise FileNotFoundError(
                f'Unable to read input file: File {input_file_name} not found'
            )

        logger.info(
            f'Found filename: {input_file_name}. Proceeding with run using input parameters from that file'
        )

        with open(input_file_name, encoding='utf-8') as f:
            content = f.readlines()

        # successful read of data into list.  Now make a dictionary with all the parameter entries.
        # Index will be the unique name of the parameter.
        # The value will be a "ParameterEntry" structure, with name, value (optionally with units), optional comment
        for raw_line in content:
            line = raw_line.strip()
            if any([line.startswith(x) for x in ['#', '--', '*']]):
                # skip any line that starts with "#" - # will be the comment parameter
                continue

            # now deal with the comma-delimited parameters
            # split on a comma - that should give us major divisions,
            # Could be:
            # 1) Desc and Val (2 elements),
            # 2) Desc and Val with Unit (2 elements, Unit split from Val by space),
            # 3) Desc, Val, and comment (3 elements),
            # 4) Desc, Val with Unit, Comment (3 elements, Unit split from Val by space)
            # If there are more than 3 commas, we are going to assume it is parseable,
            # and that the commas are in the comment
            elements = parse_param_line(line)

            # Skip blank/comment/invalid lines (parser returns empty strings)
            if not elements[0]:
                continue

            if len(elements) < 2:
                # not enough commas, so must not be data to parse
                continue

                # we have good data, so make initial assumptions
            description = elements[0].strip()
            s_val = elements[1].strip()
            comment = ""  # cases 1 & 2 - no comment
            if len(elements) == 3:  # cases 3 & 4
                comment = elements[2].strip()

            if len(elements) > 3:
                # too many commas, so assume they are in comments
                for i in range(2, len(elements), 1):
                    comment = comment + elements[i]

            # done with parsing, now create the object and add to the dictionary
            p_entry = ParameterEntry(description, s_val, comment, line)
            return_dict_1[description] = p_entry  # make the dictionary element

    else:
        logger.warning(
            'No input parameter file specified on the command line. '
            'Proceeding with default parameter run...'
        )

    logger.info(f'Complete {__name__}: {sys._getframe().f_code.co_name}')


class _EnhancedJSONEncoder(json.JSONEncoder):
    """
    Enhanced JSON encoder that can handle dataclasses
    :param json.JSONEncoder: JSON encoder
    :return: JSON encoder
    :rtype: json.JSONEncoder
    """

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        if issubclass(o, Enum):
            def get_entry(member) -> dict[str, Any]:
                d = {
                    'name': member.name,
                    'value': member.value
                }

                if hasattr(member, 'int_value'):
                    d['int_value'] = member.int_value

                return d

            ret = [get_entry(member) for member in o]
            return ret

        return super().default(o)


def json_dumpse(obj) -> str:
    return json.dumps(obj, cls=_EnhancedJSONEncoder)


def static_pressure_MPa(rho_kg_per_m3: float, depth_m: float) -> float:
    """
    Calculate static pressure in a reservoir (i.e. lithostatic pressure or hydrostatic pressure).

    Args:
        rho_kg_per_m3 (float): Density of the fluid in kg/m^3.
        depth_m (float): Depth of the reservoir in meters.
    Returns:
        float: Static pressure in megapascals (MPa).
    """

    g = scipy.constants.g  # Acceleration due to gravity (m/s^2)

    # Calculate lithostatic pressure
    pressure_Pa = rho_kg_per_m3 * g * depth_m

    pressure_mpa = quantity(pressure_Pa, 'Pa').to('MPa').magnitude

    return pressure_mpa


def is_int(o: Any) -> bool:
    try:
        float_n = float(o)
        int_n = int(float_n)
    except ValueError:
        return False
    else:
        return float_n == int_n


def is_float(o: Any) -> bool:
    try:
        float(o)
    except ValueError:
        return False
    except TypeError:
        return False
    else:
        return True

def sig_figs(val: float | list | tuple, num_sig_figs: int) -> float:
    if val is None:
        return 0

    if isinstance(val, list) or isinstance(val, tuple):
        return [sig_figs(v, num_sig_figs) for v in val]

    try:
        return float('%s' % float(f'%.{num_sig_figs}g' % val))  # pylint: disable=consider-using-f-string
    except TypeError:
        # TODO warn
        return val


def _sig_figs_scalar(x: float, n: int) -> float:
    """
    Fast significant-figures rounding for a Python float.
    String-free; handles 0, NaN, Inf.
    :param x: The number to round.
    :param n: The number of significant figures (must be >= 1).
    :return: The rounded number.
    """
    if n < 1:
        raise ValueError("num_sig_figs must be >= 1")

    if not math.isfinite(x) or x == 0.0:
        return x

    # decimals = n - 1 - floor(log10(abs(x)))
    decimals = n - 1 - int(math.floor(math.log10(abs(x))))
    return round(x, decimals)


def _sig_figs_numpy(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Vectorized significant-figures rounding for NumPy arrays.
    String-free; preserves shape; leaves NaN/Inf unchanged.
    :param arr: The array to round.
    :param n: The number of significant figures (must be >= 1).
    :return: The rounded array.
    """
    if n < 1:
        raise ValueError("num_sig_figs must be >= 1")

    a = np.asarray(arr)
    # Work in float64 for consistent behavior
    out = a.astype(np.float64, copy=True)

    finite = np.isfinite(out)
    nonzero = out != 0.0
    mask = finite & nonzero

    if not np.any(mask):
        return out

    ax = np.abs(out[mask])
    # decimals = n - 1 - floor(log10(abs(x)))
    decimals = (n - 1 - np.floor(np.log10(ax))).astype(np.int32)

    # np.round supports per-element decimals
    out[mask] = np.round(out[mask], decimals)

    return out


def sig_figs_hp(
    val: Any,
    num_sig_figs: int,
    *,
    strict: bool = False,
    prefer_numpy: bool = True,
) -> Any:
    """
    High-performance significant-figures rounding.

    Supports:
      - scalars: int/float/np.number
      - numpy arrays: np.ndarray (vectorized)
      - lists/tuples: returns same structure, optionally accelerated via NumPy

    Non-numeric entries in sequences are left unchanged by default (GEOPHIRES-friendly), but can raise if strict=True.
    :param val: Scalar, list, tuple, numpy array, or None.
    :param num_sig_figs: Significant figures (must be >= 1).
    :param strict: If True, raise on non-numeric entries in sequences. If False
        (default), leave non-numeric entries unchanged (GEOPHIRES-friendly).
    :param prefer_numpy: If True (default), attempt to accelerate lists/tuples by converting to NumPy when feasible.
    :return: Same shape/type as input (roughly):
      - scalar -> float
      - list -> list
      - tuple -> tuple
      - np.ndarray -> np.ndarray (float64)
      - None -> None
    """
    if val is None:
        return None

    if num_sig_figs < 1:
        raise ValueError("num_sig_figs must be >= 1")

    # --- NumPy arrays: fastest path ---
    if isinstance(val, np.ndarray):
        return _sig_figs_numpy(val, num_sig_figs)

    # --- Scalar numbers ---
    if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
        return _sig_figs_scalar(float(val), num_sig_figs)

    # --- Sequences (lists/tuples), but not strings/bytes ---
    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        # Optional acceleration: try numeric conversion to NumPy
        if prefer_numpy:
            try:
                arr = np.asarray(val, dtype=np.float64)  # will fail if ragged/non-numeric
                rounded = _sig_figs_numpy(arr, num_sig_figs)

                # Preserve list vs tuple
                if isinstance(val, tuple):
                    return tuple(rounded.tolist())
                return rounded.tolist()
            except (TypeError, ValueError):
                # Fall back to element-wise handling (supports ragged / mixed content)
                pass

        # Element-wise fallback (supports nested, ragged, mixed content)
        def convert_item(item: Any) -> Any:
            if item is None:
                return None
            if isinstance(item, np.ndarray):
                return _sig_figs_numpy(item, num_sig_figs)
            if isinstance(item, (int, float, np.number)) and not isinstance(item, bool):
                return _sig_figs_scalar(float(item), num_sig_figs)
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                # Preserve nesting structure
                if isinstance(item, tuple):
                    return tuple(convert_item(v) for v in item)
                return [convert_item(v) for v in item]

            if strict:
                raise TypeError(f"Non-numeric value encountered: {item!r}")
            return item

        if isinstance(val, tuple):
            return tuple(convert_item(v) for v in val)
        return [convert_item(v) for v in val]

    # --- Non-numeric non-sequence ---
    if strict:
        raise TypeError(f"Value is not numeric or a supported container: {val!r}")
    return val

def is_numeric_sequence_string(s: str) -> bool:
    """
    Return True if `s` can be parsed as a Python/JSON sequence (list/tuple) and the list has a length of > 1

    Accepts:
      - Python literals: "[1, 2, 3]", "(1, 2, 3)", "[1e-3, -4.2]"
      - JSON arrays:    "[1, 2, 3]", "[1.2, -3, 4e2]"
      - CSV lists:      "1,2,3", "1.2, -3, 4e2"

    Rules:
      - Numeric means int/float by default (bool excluded unless allow_bool=True)
      - If allow_nested=True, nested lists/tuples are allowed and must be numeric throughout
      - Rejects dict/set, strings, None, NaN/Infinity if they appear as strings, etc.
        (Note: JSON "NaN"/"Infinity" are non-standard; Python literal float('nan') won't parse via literal_eval)

    Examples:
      True:  "[1, 2, 3]"
      True:  "(1, 2.5, -3)"
      True:  "[[1,2],[3,4]]"  (if allow_nested=True)
      False: "['1', 2]"
      False: "[1, None]"
      False: "{'a': 1}"
      True: "1,2,3"
    """
    if not isinstance(s, str):
        return False
    text = s.strip()
    if not text:
        return False
    obj: Any = None

    # 1) Try JSON first (covers strict JSON format cleanly)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback to Python literal or CSV
    if obj is None:
        try:
            obj = ast.literal_eval(text)
            return True # if literal_eval succeeded, we know it's a valid Python literal
        except (ValueError, SyntaxError):
            pass
    if obj is None:   #Try to see if it is a valid CSV list, ignoring comment lines and blank lines
        COMMENT_PREFIXES = ("#", "//", ";", "%")
        def is_data_line(line: str) -> bool:
            stripped = line.strip()
            if not stripped:  # blank line
                return False
            if stripped.startswith(COMMENT_PREFIXES):
                return False
            return True

        clean_lines = [
            line for line in text.splitlines()
            if is_data_line(line)
        ]

        reader = csv.reader(io.StringIO("\n".join(clean_lines)))
        obj = list(reader)

        if obj is None or len(obj) == 0:
            return False

    # Must be a sequence (list/tuple).
    if not isinstance(obj, (list, tuple)):
        return False
    return True

def is_existing_file_path(path_str: str) -> bool:
    """
    Return True if `path_str` is a valid path to an existing *regular file*.
    - Rejects empty/whitespace-only strings.
    - Expands ~ and environment variables.
    - Works on Windows/Unix.
    """
    if not isinstance(path_str, str):
        return False

    # Reject URLs (we want file paths here)
    parsed = urlparse(path_str)
    if parsed.scheme in ['http', 'https']:
        return False

    s = path_str.strip()
    if not s:
        return False

    # Expand ~ and %VAR% / $VAR
    expanded = os.path.expandvars(os.path.expanduser(s))
    p = Path(expanded)

    #first check: is it a file path (not a directory, etc.)?  This also implicitly checks that the path is syntactically valid.
    #if not p.is_file():
    #    return False

    #second check: does it exist at all?
    #if not p.exists():
    #    return False

    #the ultimate test: can we open it for reading without an OSError?
    try:
        with p.open("r") as f:
            pass
    except OSError as e:
        return False

    return True


def url_returns_content(
    url_str: str,
    *,
    timeout: float = 5.0,
    user_agent: str = "url_returns_content/1.0",
    allow_redirects: bool = True,
    session: Optional[requests.Session] = None,
) -> bool:
    """
    Return True if `url_str` is a valid http(s) URL and appears to return >= 1 byte.

    Strategy:
    1) Validate scheme/netloc.
    2) HEAD request:
        - If status is OK and Content-Length >= 1 => True
        - If Content-Length missing/0/unknown => try Range GET for 1 byte
    3) Range GET (bytes=0-0):
        - If status 206 and body length >= 1 => True
        - If 200 and body length >= 1 => True
    Otherwise => False

    Notes:
    - Some servers block HEAD; we fall back to GET.
    - For safety/perf, GET uses stream=True and only reads 1 byte.
    """
    if not isinstance(url_str, str):
        return False

    s = url_str.strip()
    if not s:
        return False

    parsed = urlparse(s)
    if parsed.scheme not in ("http", "https"):
        return False
    if not parsed.netloc:
        return False

    headers = {
        "User-Agent": user_agent,
        "Accept": "*/*",
    }

    sess = session or requests.Session()

    def _content_length_is_positive(resp: requests.Response) -> bool:
        cl = resp.headers.get("Content-Length")
        if cl is None:
            return False
        try:
            return int(cl) >= 1
        except ValueError:
            return False

    try:
        # --- 1) Try HEAD ---
        try:
            head = sess.head(
                s,
                headers=headers,
                timeout=timeout,
                allow_redirects=allow_redirects,
            )
            if 200 <= head.status_code < 400:
                if _content_length_is_positive(head):
                    return True
                # Some resources are chunked / no content-length; try 1-byte GET.
            else:
                # Non-success HEAD doesn't prove absence; fall back to GET.
                pass
        except requests.RequestException:
            # HEAD may be blocked; fall back to GET.
            pass

        # --- 2) Try 1-byte Range GET ---
        range_headers = {**headers, "Range": "bytes=0-0"}
        get = sess.get(
            s,
            headers=range_headers,
            timeout=timeout,
            allow_redirects=allow_redirects,
            stream=True,
        )

        # Accept partial content (206) or normal (200)
        if get.status_code in (200, 206) and get.ok:
            # Read up to 1 byte without downloading the whole thing
            for chunk in get.iter_content(chunk_size=1):
                if chunk:
                    return True

        return False

    except requests.RequestException:
        return False
    finally:
        if session is None:
            sess.close()


def parse_container_string_simple(
    text: str,
    param_to_modify,
    model,
    *,
    out: ContainerKind = "list",
    ndarray_dtype: Any = np.float64,
) -> Any:
    """
    Parse a string containing a JSON/Python container literal OR a comma-separated "bare list"
    and return it as the requested container type (default: list).
        Accepted examples:
      - A URL or a file path, if which case we will open that stream and parse it
      - JSON arrays:           "[1, 2, 3]"
      - Python literals:       "[1, 2, 3]", "(1, 2, 3)", "{1, 2, 3}"
      - Bare comma lists:      "1,2,3"   or  "1, 2, 3"
      - Bare comma w/ quotes:  "a,b,c"   or  "'a', 'b', 'c'"   or  '"a","b","c"'

    Notes:
      - Assumes `text` has already been validated upstream (non-empty string, etc.).
      - Does not enforce numeric-only here.
      - Does not attempt to parse expressions like "np.array([1,2])" (only JSON/literals).

    :param text: The input string to parse.
    :param param_to_modify: The parameter being modified (for error messages).
    :param model: The model object (for error messages).
    :param out: The desired output container type: "list", "tuple", "set", or "ndarray".
    :param ndarray_dtype: If out="ndarray", the dtype to use for the NumPy array (default: np.float64).
    :return: The parsed container in the requested type.
    """
    s = text.strip()
    obj: Any = None

    # 0) check to see if we need to read text from file or URL (strip BOM via utf-8-sig)
    ###if s.startswith('http') or is_existing_file_path(s):

        # If it looks like a URL or file path, try to read from it and parse the content instead.
    ###    obj = get_data_from_file_or_url(s, param_to_modify, model)

        # if it is a list, return it.
    ###    if isinstance(obj, (list, tuple, set)):
    ###        return list(obj)
    ###    else:
            # If it's not a list/tuple/set, we will try to parse it as a literal or a bare comma list.
            # We will convert it to a string and continue with the parsing below.
            # We want to allow the file/URL to contain a JSON array or Python literal, but if it is already a list/tuple/set, we can skip that step.
    ###        s = str(obj).strip()

    # 1) Try JSON first (fast + strict for JSON arrays)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        obj = None

    # 2) Try Python literal
    if obj is None:
        try:
            obj = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            obj = None

    # 3) Bare comma-separated list fallback: treat as a Python list literal
    #    Only do this if there's at least one comma or newline; avoids turning "123" into [123].
    if obj is None and any(d in s for d in (",", "\n", "\r", "\n\r", "\r\n")):
        # Wrap as list literal and parse items safely via literal_eval
        try:
            COMMENT_PREFIXES = ("#", "//", ";", "%")
            clean_lines = [
                line for line in s.splitlines()
                if line.strip() and not line.strip().startswith(COMMENT_PREFIXES)
            ]

            def auto_cast(value):
                try:
                    if "." in value:
                        return float(value)
                    return int(value)
                except ValueError:
                    return value

            reader = csv.reader(io.StringIO("\n".join(clean_lines)))
            rows = list(reader)
            header = rows[0]
            data = [[auto_cast(cell) for cell in row] for row in rows[1:]]
            obj = [header] + data
        except (ValueError, SyntaxError) as e:
            obj = None

    if obj is None:
        return None

    # Reject dicts and scalars; accept list/tuple/set from Python, and list from JSON.
    if isinstance(obj, dict) or isinstance(obj, (str, bytes)) or not isinstance(obj, (list, tuple, set)):
        raise TypeError(
            f"Parsed value is {type(obj).__name__}; expected list/tuple/set (or JSON array / bare comma list)."
        )

    # Convert to requested output
    if out == "list":
        return list(obj) if not isinstance(obj, list) else obj
    if out == "tuple":
        return tuple(obj)
    if out == "set":
        return set(obj)
    if out == "ndarray":
        return np.asarray(list(obj), dtype=ndarray_dtype)

    raise ValueError(f"Invalid 'out' value: {out!r}")


def looks_like_html(text: str) -> bool:
        t = text.lstrip().lower()
        if t.startswith("<!doctype html") or t.startswith("<html"):
            return True
        # quick sniff for typical head/body tags near top
        return "<head" in t[:500] or "<body" in t[:500]


def get_header_from_source(first_line: str) -> str:
    """
    Reads the text and returns the header string if it can be detected, or "" if no header is found.

    Comments / blanks:
      - Blank lines are ignored
      - Comment lines begin with '#', '--', or '*' after leading whitespace is stripped

    Header detection (after filtering comments/blanks):
      - If entire first line contains exactly ONE '(...)' group => 1D header; parse col 1 only
      - Else if first two CSV cells each contain '(...)' => 2D header; parse first 2 cols
      - Else: no header => header_str=""

    :param first_line: First row of the data source after it has been filtered, cleaned, and otherwise prepared.
    :type first_line: str
    :return: header_str, as in "depth (feet)" OR "Time (hour),Temperature (degF)" OR ""
    :rtype: str or Tuple[str, str]
    """

    def parse_row(line: str) -> list[str]:
        # Robust CSV parsing (quotes, commas, whitespace)
        return next(csv.reader([line], skipinitialspace=True))

    def cell_has_parens(cell: str) -> bool:
        return bool(_PARENS_RE.search(cell))

    # --- Decide header ---
    header_str = ""
    first_row = parse_row(first_line)

    parens_count_line = len(_PARENS_RE.findall(first_line))
    if parens_count_line == 1:
        header_str = (first_row[0].strip() if first_row else "").strip()
    elif len(first_row) >= 2 and cell_has_parens(first_row[0]) and cell_has_parens(first_row[1]):
        h0 = first_row[0].strip()
        h1 = first_row[1].strip()
        header_str = f"{h0},{h1}"
    else:
        header_str = ""

    return header_str


def parse_header_units(header: str) -> Union[str, Tuple[str, str]]:
    """
    Parse a header string like:
      - ""                               -> ""
      - "depth (feet)"                   -> "feet"
      - "Time (hour),Temperature (degF)" -> ("hour", "degF")

    Rules:
      - If header is blank/whitespace => return ""
      - Extract text inside each '(...)'
      - If exactly 1 set of parens => return that inner text (str)
      - If exactly 2 sets of parens => return both inner texts (tuple[str, str])
      - If there are 0 parens on a non-blank header, returns "" (no units found)
      - If there are >2 parens, returns the first two (consistent with your file parser)
    """
    if header is None:
        return ""

    # trimming / blank
    h = str(header).strip()
    if not h:
        return ""

    # if multiline, pick the most plausible "header line" but keep compatibility.
    # - Prefer the first non-empty line with parens, if it exists.
    # - If that line has no parens, and there are later lines with parens,
    #   use the first non-comment line containing parens.
    lines = [ln.strip() for ln in h.splitlines() if ln.strip() != ""]
    if not lines:
        return ""

    header_line = lines[0]

    # If the first line doesn't contain units but later lines do, skip leading metadata.
    if "(" not in header_line or ")" not in header_line:
        for ln in lines[1:]:
            if ln.startswith("#"):
                continue
            if "(" in ln and ")" in ln:
                header_line = ln
                break

    # Now run the original parsing logic on the chosen header line
    matches = re.findall(r"\(([^)]*)\)", header_line)
    if not matches:
        return ""

    # Normalize whitespace inside units
    units = [m.strip() for m in matches if m.strip() != ""]

    if len(units) == 0:
        return ""

    if len(units) == 1:
        return units[0]

    # If 2+ found, return first two
    return units[0], units[1]


def get_data_from_file_or_url_as_string(source: str) -> str:
    """
    Read and parse a file/URL into a string containing the raw text content.
    :param source: pre-validated File path or URL to read from.
    :return: text, as string of the raw text content from the file or URL, with BOM stripped if present, or "".
    """
    # --- Read text from file or URL (strip BOM via utf-8-sig) ---
    raw_text = ""
    if url_returns_content(source):
        headers = {"User-Agent": "geophires-x/1.0", "Accept": "*/*"}
        resp = requests.get(source, headers=headers, timeout=10.0, allow_redirects=True)
        resp.raise_for_status()

        raw_text = resp.content.decode("utf-8-sig", errors="replace")

        ct = resp.headers.get("Content-Type", "")
        if "text/html" in ct.lower() or looks_like_html(raw_text):
            raise ValueError(
                f"URL did not return text/numeric/CSV data; it returned HTML instead. "
                f"URL={source!r}, status={resp.status_code}, content-type={ct!r}. "
                f"Likely a wrong URL (e.g., GitHub 'blob' page), auth wall, redirect, or error page."
            )

    else:
        # Read the whole file as text, split into lines, and strip BOM if present (utf-8-sig)
        if not source.startswith("http"):
            raw_text = Path(source).read_text(encoding="utf-8-sig")
        else:
            return ""

    return raw_text

def _try_parse_multiline(lines: list[str],
                         ParamToModify,
                         model,
                         *,
                         assume_two_column: Optional[bool] = False,
                         ndarray_dtype: Any = np.float64) -> None | list[tuple[float, float]] | list[float]:
    """
    Function to process the header and units from the input lines, and return the data as a list of tuples (for 2D) or a list of floats (for 1D), with units converted if necessary.
    :param lines: list of strings, each representing a line from the input file or URL content.
    :type lines: list[str]
    :param ParamToModify: The parameter being modified (for error messages and unit preferences).
    :param model: The model object (for error messages and context).
    :param assume_two_column: If True, assume the data has two columns even if the header doesn't clearly indicate it. Default is False.
    :type assume_two_column: bool, optional
    :param ndarray_dtype: If out="ndarray", the dtype to use for the NumPy array (default: np.float64).
    :type ndarray_dtype: Any, optional
    :return: list of tuples (for 2D) or list of floats (for 1D), with units converted if necessary, or None if no data found.
    :return type: None | list[tuple[float, float]] | list[float]
    """

    # if it has new lines in it, then convert it into a single line.
    candidate = ''
    if len(__import__('re').findall(r'\r\n|\n\r|\r|\n', lines)) > 0:
        candidate = str([float(x) if __import__('re').fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', x) else x
                  for x in (l.strip() for l in __import__('re').split(r'\r\n|\n|\r', lines)) if x])

    # If bracket-wrapped, unwrap (still keep single-line semantics)
    if candidate.startswith('[') and candidate.endswith(']'):
        candidate = candidate[1:-1].strip()

    first_from = second_from = first_to = second_to = None
    is_two_col = assume_two_column

    # Find out if the input has a header:
    file_header = get_header_from_source(lines[0])
    if file_header:
        # If we have a header line, then there is a chance that the units need to be converted, so we will attempt to do that safely.
        # First, choose destination units:
        # 1) PreferredUnits if set
        # 3) else no conversion target (None) and let convert_content_inplace no-op or handle it
        # This will get us the units required for the second column, if there are two columns, and if not,
        # it will just get us the units for the parameter in general.
        if getattr(ParamToModify, "PreferredUnits", None) is not None:
            first_to = second_to = get_unit_from_string(str(ParamToModify.PreferredUnits.value))

        # we need to figure out what the header value is, because it tells us what the units are,
        # and if they don't match, that we need to convert the data
        units = parse_header_units(file_header)
        if units:
            if isinstance(units, tuple):
                # if there are two units, it will be a list,
                # and the second one is the one for the data (the first one is usually for the index, like time)
                first_from = get_unit_from_string(units[0]).value
                second_from = get_unit_from_string(units[1]).value
                first_to = "hours"  # hardcode hours for now, because that is the most common index unit, and it is what we use in our historical data - we can make this more flexible in the future if needed
                is_two_col = True
            else:
                ParamToModify.CurrentUnits = get_unit_from_string(units)
                first_from = get_unit_from_string(units).value
                is_two_col = False
    else:
        # We will assume 1 column with no units for now,
        # but we should make this more robust in the future, perhaps by looking at the content and trying to infer it.
        is_two_col = False
        if getattr(ParamToModify, "PreferredUnits", None) is not None:
            first_to = get_unit_from_string(str(ParamToModify.PreferredUnits.value))
        if getattr(ParamToModify, "CurrentUnits", None) is not None:
            first_from = get_unit_from_string(str(ParamToModify.CurrentUnits.value))

    def parse_row(line: str) -> list[str]:
        # Robust CSV parsing (quotes, commas, whitespace)
        return next(csv.reader([line], skipinitialspace=True))

    def cell_has_parens(cell: str) -> bool:
        return bool(_PARENS_RE.search(cell))

    def to_float(cell: str) -> Optional[float]:
        # Skip blank cells
        s = cell.strip()
        if s == "":
            return None
        return float(s)

    # --- Decide to where to start ---
    if file_header:
        data_start_idx = 1
    else:
        data_start_idx = 0

    # --- Parse data ---
    single: SingleSeries = []
    dual: DualSeries = []

    for idx in range(data_start_idx, len(lines)):
        row = parse_row(lines[idx])
        if not row or all(not c.strip() for c in row):
            continue

        if not is_two_col:
            if len(row) < 1:
                continue
            try:
                v = to_float(row[0])
            except ValueError as e:
                raise ValueError(
                    f"Non-numeric value in first column on line {idx + 1}: {row[0]!r}"
                ) from e
            if v is None:
                continue
            single.append(v)

        else:
            if len(row) < 2:
                # Not enough columns; skip or raise—here we raise because it's structurally wrong for 2D
                raise ValueError(f"Expected at least 2 columns on line {idx + 1}, got: {row!r}")
            try:
                x = to_float(row[0])
                y = to_float(row[1])
            except ValueError as e:
                raise ValueError(
                    f"Non-numeric value in first two columns on line {idx + 1}: {row[0]!r}, {row[1]!r}"
                ) from e
            if x is None or y is None:
                continue
            dual.append((x, y))

    #parsing complete, now we have either single or dual populated based on is_two_col,
    # and we have the from/to units if applicable. Now we can do the conversion if needed.
    content = dual if is_two_col else single

    # Do the unit conversion if it is required
    if (
        (is_two_col and (not first_from == first_to or not second_from == second_to))
        or
        (not is_two_col and not first_from == first_to)
    ):
        Q_ = _ureg.Quantity
        if not is_two_col:
            arr = np.asarray(content, dtype=ndarray_dtype)
            converted = Q_(arr, first_from).to(first_to).magnitude
            # mutate original list in place
            content[:] = converted.tolist()
            return content
        else:
            # Convert via arrays for speed, then write back into the list in place
            x = np.asarray([a for a, _ in content], dtype=ndarray_dtype)
            y = np.asarray([b for _, b in content], dtype=ndarray_dtype)

            x_conv = Q_(x, first_from).to(first_to).magnitude
            y_conv = Q_(y, second_from).to(second_to).magnitude

            # Replace tuples in-place (list mutated; tuples replaced)
            content[:] = list(zip(x_conv.tolist(), y_conv.tolist()))

def _try_parse_multiline_list(lines: list[Any],
                         header: str = "",
                         ParamToModify = None,
                         *,
                         assume_two_column: Optional[bool] = False,
                         ndarray_dtype: Any = np.float64) -> None | list[tuple[float, float]] | list[float]:
    """
    Function to process the header and units from the input lines, and return the data as a list of tuples (for 2D) or a list of floats (for 1D), with units converted if necessary.
    :param lines: list of strings, each representing a line from the input file or URL content.
    :type lines: list[str]
    :param header: Optional header string to use for unit parsing; if not provided, will attempt to detect from lines[0].
    :type header: str, optional
    :param ParamToModify: The parameter being modified (for error messages and unit preferences).
    :param model: The model object (for error messages and context).
    :param assume_two_column: If True, assume the data has two columns even if the header doesn't clearly indicate it. Default is False.
    :type assume_two_column: bool, optional
    :param ndarray_dtype: If out="ndarray", the dtype to use for the NumPy array (default: np.float64).
    :type ndarray_dtype: Any, optional
    :return: list of tuples (for 2D) or list of floats (for 1D), with units converted if necessary, or None if no data found.
    :return type: None | list[tuple[float, float]] | list[float]
    """

    first_from = second_from = first_to = second_to = None
    is_two_col = assume_two_column

    #figure out if we are processing a Time Series (TS) or a regular list parameter,
    # because that will affect how we interpret the header and units.
    # We will use the presence of PreferredXUnits and PreferredYUnits to determine if it is a TS.
    isTS = hasattr(ParamToModify, "PreferredXUnits") and hasattr(ParamToModify, "PreferredYUnits")

    if header:
        # If we have a header line, then there is a chance that the units need to be converted, so we will attempt to do that safely.
        # First, choose destination units:
        # 1) PreferredUnits if set
        # 3) else no conversion target (None) and let convert_content_inplace no-op or handle it
        # This will get us the units required for the second column, if there are two columns, and if not,
        # it will just get us the units for the parameter in general.
        if not isTS and getattr(ParamToModify, "PreferredUnits", None) is not None:
            first_to = second_to = get_unit_from_string(str(ParamToModify.PreferredUnits.value))
        else:
            first_to = get_unit_from_string(str(ParamToModify.PreferredXUnits))
            second_to = get_unit_from_string(str(ParamToModify.PreferredYUnits))

        # we need to figure out what the header value is, because it tells us what the units are,
        # and if they don't match, that we need to convert the data
        units = parse_header_units(header)
        if units:
            if isinstance(units, tuple):
                # if there are two units, it will be a list,
                # and the second one is the one for the data (the first one is usually for the index, like time)
                first_from = get_unit_from_string(units[0]).value
                second_from = get_unit_from_string(units[1]).value
                if not isTS:
                    first_to = first_from  # hardcode to no conversion for now, we can make this more flexible in the future if needed
                is_two_col = True
            else:
                ParamToModify.CurrentUnits = get_unit_from_string(units)
                first_from = get_unit_from_string(units).value
                is_two_col = False
        data_start_idx = 1
    else:
        # We will assume 1 column with no units for now,
        # but we should make this more robust in the future, perhaps by looking at the content and trying to infer it.
        is_two_col = False
        if getattr(ParamToModify, "PreferredUnits", None) is not None:
            first_to = get_unit_from_string(str(ParamToModify.PreferredUnits.value))
        if getattr(ParamToModify, "CurrentUnits", None) is not None:
            first_from = get_unit_from_string(str(ParamToModify.CurrentUnits.value))
        data_start_idx = 0

    def parse_row(line: str) -> list[str]:
        # Robust CSV parsing (quotes, commas, whitespace)
        return next(csv.reader([line], skipinitialspace=True))

    def cell_has_parens(cell: str) -> bool:
        return bool(_PARENS_RE.search(cell))

    def to_float(cell: str) -> Optional[float]:
        # Skip blank cells
        s = cell.strip()
        if s == "":
            return None
        return float(s)

    # --- Parse data ---
    single: SingleSeries = []
    dual: DualSeries = []

    for idx in range(data_start_idx, len(lines)):
        if not is_two_col:
            row = str(lines[idx][0])
        else:
            row = [str(c) for c in lines[idx][:2]]  # take first two columns if 2D, else first column
        if not row or all(not c.strip() for c in row):
            continue

        if not is_two_col:
            if len(row) < 1:
                continue
            try:
                v = to_float(row)
            except ValueError as e:
                raise ValueError(
                    f"Non-numeric value in first column on line {idx + 1}: {row[0]!r}"
                ) from e
            if v is None:
                continue
            single.append(v)

            #units converted, so update unit status for the parameter
            ParamToModify.CurrentUnits = first_to

        else:
            if len(row) < 2:
                # Not enough columns; skip or raise—here we raise because it's structurally wrong for 2D
                raise ValueError(f"Expected at least 2 columns on line {idx + 1}, got: {row!r}")
            try:
                x = to_float(row[0])
                y = to_float(row[1])
            except ValueError as e:
                raise ValueError(
                    f"Non-numeric value in first two columns on line {idx + 1}: {row[0]!r}, {row[1]!r}"
                ) from e
            if x is None or y is None:
                continue
            dual.append((x, y))

    #parsing complete, now we have either single or dual populated based on is_two_col,
    # and we have the from/to units if applicable. Now we can do the conversion if needed.
    content = dual if is_two_col else single

    # Do the unit conversion if it is required
    if (
        (is_two_col and (not first_from == first_to or not second_from == second_to))
        or
        (not is_two_col and not first_from == first_to)
    ):
        Q_ = _ureg.Quantity
        if not is_two_col:
            arr = np.asarray(content, dtype=ndarray_dtype)
            converted = Q_(arr, first_from).to(first_to).magnitude
            # mutate original list in place
            content[:] = converted.tolist()
        else:
            # Convert via arrays for speed, then write back into the list in place
            x = np.asarray([a for a, _ in content], dtype=ndarray_dtype)
            y = np.asarray([b for _, b in content], dtype=ndarray_dtype)

            x_conv = Q_(x, first_from).to(first_to).magnitude
            y_conv = Q_(y, second_from).to(second_to).magnitude

            # Replace tuples in-place (list mutated; tuples replaced)
            content[:] = list(zip(x_conv.tolist(), y_conv.tolist()))

        # units converted, so update unit status for the parameter
        ParamToModify.CurrentXUnits = first_to
        ParamToModify.CurrentYUnits = second_to

        return content
    return content

def get_data_from_file_or_url(
    source: str,
    ParamToModify,
    model,
) -> ParsedSeries:
    """
        Read and parse a file/URL into either:
      - 1D series: [v1, v2, ...] from the first column, OR
      - 2D series: [(x1, y1), (x2, y2), ...] from the first two columns.

    Returns:
      data list[float] (1D) OR list[tuple[float,float]] (2D) or [] if no valid data rows

    Source:
      - pre-validated file path (default, from_url=False)
      - pre-validated URL (from_url=True)

    Comments / blanks:
      - Blank lines are ignored
      - Comment lines begin with '#', '--', or '*' after leading whitespace is stripped

    Data:
      - Extra columns are ignored
      - Blank numeric cells in required columns cause that row to be skipped
      - If URL returns HTML (e.g., GitHub blob page / auth wall), raise a clear ValueError

    :param source: File path or URL to read from.
    :param ParamToModify: The parameter object being modified, used to determine preferred units for
        conversion if needed.
    :param model: The model object, used for logging.
    :param assume_two_column: If True, treat as 2-column data even if header doesn't clearly indicate it. If False, treat as 1-column. If None (default), auto-detect based on header.
    :param ndarray_dtype: The dtype to use for NumPy arrays during conversion (default: np.float64).

    :return: data, where data is either a list of floats (1D) or a list of (float, float) tuples (2D), or [].
    """
    # --- Read text from file or URL (strip BOM via utf-8-sig) ---
    content = raw_lines = []
    if url_returns_content(source):
        headers = {"User-Agent": "geophires-x/1.0", "Accept": "*/*"}
        resp = requests.get(source, headers=headers, timeout=10.0, allow_redirects=True)
        resp.raise_for_status()

        text = resp.content.decode("utf-8-sig", errors="replace")

        ct = resp.headers.get("Content-Type", "")
        if "text/html" in ct.lower() or looks_like_html(text):
            raise ValueError(
                f"URL did not return numeric/CSV data; it returned HTML instead. "
                f"URL={source!r}, status={resp.status_code}, content-type={ct!r}. "
                f"Likely a wrong URL (e.g., GitHub 'blob' page), auth wall, redirect, or error page."
            )

        raw_lines = text.splitlines()
    else:
        # Read the whole file as text, split into lines, and strip BOM if present (utf-8-sig)
        raw_lines = Path(source).read_text(encoding="utf-8-sig").splitlines()

    # --- Filter out comments and blanks ---
    lines: list[str] = []
    for line in raw_lines:
        stripped = line.lstrip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("*") or stripped.startswith("--"):
            continue
        lines.append(line)

    if not lines:
        model.logger.warning(f"No data lines found in {'URL' if {source} else 'file'}: {source}")
        return []

    content = _try_parse_multiline(lines, ParamToModify, model)
    return content

DEFAULT_LINE_COMMENT_PREFIXES: tuple[str, ...] = (
    "#",
    "//",
    ";",
    "--",
    "/*",
    "*",   # covers **** headers and lines of ********
    "`",
)


DEFAULT_INLINE_COMMENT_PREFIXES: tuple[str, ...] = (
    "#",
    "//",
    ";",
    "--",
    "/*",
    "*",
    "`",
)


def _unquote_and_unescape(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        q = s[0]
        return s[1:-1].replace(q + q, q).strip()
    return s


def _looks_like_comment(s: str, *, inline_prefixes: Sequence[str]) -> bool:
    """
    Heuristic: treat as comment if it begins with a comment marker or contains letters.
    This keeps numeric multi-values as value (e.g., '3,2, 1') while capturing
    'comments galore' or '# another comment'.
    """
    t = s.strip()
    if not t:
        return False
    if any(t.startswith(pfx) for pfx in inline_prefixes):
        return True
    return any(ch.isalpha() for ch in t)


def parse_param_line(
    line: str,
    *,
    line_comment_prefixes: Sequence[str] = DEFAULT_LINE_COMMENT_PREFIXES,
    inline_comment_prefixes: Sequence[str] = DEFAULT_INLINE_COMMENT_PREFIXES,
) -> tuple[str, str, str]:
    """
    Returns (name, value, comment). For blank/comment-only/invalid lines returns ("", "", "").

    Supports:
      - "..." and '...' quoting
      - escaped quotes by doubling: "" or ''
      - inline '--' comments outside quotes
      - value lists like: Thicknesses, 3,2, 1
      - comments after last comma even without '--' (heuristic)
    """
    raw = line.rstrip("\n")
    if not raw.strip():
        return "", "", ""

    stripped = raw.lstrip()
    if any(stripped.startswith(pfx) for pfx in line_comment_prefixes):
        return "", "", ""

    first_comma = -1
    second_comma = -1
    dashdash = -1

    quote: str | None = None
    i = 0
    n = len(raw)

    while i < n:
        ch = raw[i]

        if quote is None:
            if ch in ('"', "'"):
                quote = ch
            elif ch == ",":
                if first_comma < 0:
                    first_comma = i
                elif second_comma < 0:
                    second_comma = i
            elif ch == "-" and i + 1 < n and raw[i + 1] == "-":
                dashdash = i
                break
        else:
            if ch == quote:
                # doubled-quote escape: "" or ''
                if i + 1 < n and raw[i + 1] == quote:
                    i += 1
                else:
                    quote = None

        i += 1

    if first_comma < 0:
        # Not enough commas for it to be a name,value pair -> treat as comment/skip
        return "", "", ""

    name = raw[:first_comma].strip()
    if not name:
        return "", "", ""

    # Highest priority: explicit '--' comment marker
    if dashdash >= 0:
        value_region = raw[first_comma + 1 : dashdash]
        comment = raw[dashdash:].strip()
    else:
        # No '--'. If we have a second comma, decide if tail is comment by heuristic.
        if second_comma >= 0:
            tail = raw[second_comma + 1 :].strip()
            if _looks_like_comment(tail, inline_prefixes=inline_comment_prefixes):
                value_region = raw[first_comma + 1 : second_comma]
                comment = tail
            else:
                value_region = raw[first_comma + 1 :]
                comment = ""
        else:
            value_region = raw[first_comma + 1 :]
            comment = ""

    value_region = value_region.strip().rstrip(",").strip()
    value = _unquote_and_unescape(value_region)

    return name, value, comment

import re
from typing import Any, List

_NUMERIC_RE = re.compile(r"^\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\s*$")

def _looks_numeric_string(s: str) -> bool:
    return bool(_NUMERIC_RE.match(s))

def header_is_present(rows: List[List[Any]]) -> str:
    """
    If rows[0][0] looks like a header line, return it as a string.
    Otherwise return "".

    Assumes rows is like:
      [[<first cell>], [<data>], ...]
    or
      [[<header string possibly with commas>], [<data row list>], ...]

    Heuristic for header detection:
        - Must be a non-empty string in rows[0][0]
        - If it looks like a numeric string, treat as data, not header
        - If it has letters, punctuation like commas/units, and not too digit-heavy,
            then it's likely a header. We use a simple scoring system based on these features.
    :param: rows: List of rows, where each row is a list of cell values. We check rows[0][0] for header.
    :return: header string if detected, else "".
    """
    if not rows or not rows[0]:
        return ""

    first = second = None
    one_col = None
    if not rows:
        return ""
    if isinstance(rows[0], list) and len(rows[0]) == 2:
        first = str(rows[0][0])
        second = str(rows[0][1])
        one_col = False
    else:
        first = str(rows[0])
        second = ""
        one_col = True

    # Non-strings are treated as data (e.g., 0.16, 1, 60.08)
    if one_col and not isinstance(first, str):
        return ""
    if not one_col and (not isinstance(second, str) or second == ""):
        return ""

    s1 = first.strip()
    s2 = second.strip() if isinstance(second, str) else ""
    if not s1 and not s2:
        return ""

    # If it's a numeric string, treat as data
    if _looks_numeric_string(s1) and (not s2 or _looks_numeric_string(s2)):
        return ""

    # Character counts
    s = s1 + s2
    letters = sum(ch.isalpha() for ch in s)
    digits  = sum(ch.isdigit() for ch in s)
    total_alnum = letters + digits

    # If it has no letters at all, it's almost certainly not a header
    if letters == 0:
        return ""

    # Heuristic features
    comma_count = s.count(",")
    has_units_or_fields_punct = any(ch in s for ch in ("(", ")", "_", "/", "%"))
    has_spaces = " " in s

    # Letter dominance ratio (avoid div by 0)
    letter_digit_ratio = letters / (digits + 1)

    # A simple scoring model (tunable)
    score = 0.0
    score += 1.5 if comma_count >= 1 else 0.0          # "a,b,c" style header line
    score += 0.8 if comma_count >= 3 else 0.0          # many columns
    score += 0.6 if has_units_or_fields_punct else 0.0 # units like (degF/ft) or underscores
    score += 0.3 if has_spaces else 0.0
    score += 1.2 if letter_digit_ratio >= 2.0 else 0.0 # letters dominate digits
    score += 0.6 if letter_digit_ratio >= 4.0 else 0.0 # very header-like

    # Guardrail: if it’s extremely digit-heavy, likely data
    if total_alnum > 0:
        digit_fraction = digits / total_alnum
        if digit_fraction > 0.65:
            score -= 2.0

    # Threshold chosen to match your examples; adjust if needed
    if s2 == "":
        s = s1
    else:
        s = f"{s1},{s2}"
    return s if score >= 1.5 else ""


def resample_to_hourly_year(x_values: np.ndarray, y_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Parse XY series from CSV content, detect units, convert to canonical units, and optionally resample to hourly data.

    :param x_values: The raw X values as a numpy array, which may represent time, distance, or other dimensions depending on the context.
    :type x_values: np.ndarray
    :param y_values: The raw Y values as a numpy array, which may represent temperature, cost rate, or other dimensions depending on the context.
    :type y_values: np.ndarray
    return: A tuple containing the resampled X values (as a numpy array), the resampled Y values (as a numpy array),
            and a list of notes describing any normalization steps taken during resampling.
    """
    notes: List[str] = []
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if len(x) != len(y):
        raise ValueError('X and Y lengths must match')
    if len(x) < 2:
        raise ValueError('Need at least two XY samples to resample')

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if x[0] != 0.0:
        notes.append('Shifted source X values so first sample begins at 0 hour')
        x = x - x[0]

    x_target = np.arange(0, 8760, 1.0)
    if x[-1] < x_target[-1]:
        notes.append('Input time span shorter than 8760h; using edge hold extrapolation')
    if x[-1] > x_target[-1]:
        notes.append('Input time span longer than 8760h; truncating to first 8760h')

    y_target = np.interp(x_target, x, y, left=y[0], right=y[-1])
    return x_target, y_target, notes
