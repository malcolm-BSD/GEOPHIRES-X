
from __future__ import annotations

import copy
import dataclasses
import csv
import math
from typing import Optional
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
import numpy as np
import ast
from forex_python.converter import CurrencyRates, CurrencyCodes, get_rate
from abc import ABC
from pint.facets.plain import PlainQuantity
from pint.errors import UndefinedUnitError

from geophires_x.OptionList import GeophiresInputEnum
from geophires_x.Units import *
from geophires_x.GeoPHIRESUtils import is_numeric_sequence_string, is_existing_file_path, url_returns_content, \
    parse_container_string_simple, _try_parse_multiline_list
from geophires_x.GeoPHIRESUtils import _looks_like_comment, get_data_from_file_or_url, get_data_from_file_or_url_as_string
from geophires_x.GeoPHIRESUtils import _try_parse_multiline, header_is_present, resample_to_hourly_year

SCHEDULE_DSL_MULTIPLIER_SYMBOL = '*'

_ureg = get_unit_registry()
_DISABLE_FOREX_API = True  # See https://github.com/NREL/GEOPHIRES-X/issues/236#issuecomment-2414681434

_JSON_PARAMETER_TYPE_STRING = 'string'
_JSON_PARAMETER_TYPE_INTEGER = 'integer'
_JSON_PARAMETER_TYPE_NUMBER = 'number'
_JSON_PARAMETER_TYPE_ARRAY = 'array'
_JSON_PARAMETER_TYPE_BOOLEAN = 'boolean'
_JSON_PARAMETER_TYPE_OBJECT = 'object'

class HasQuantity(ABC):

    def quantity(self) -> PlainQuantity:
        """
        :rtype: pint.registry.Quantity - note type annotation uses PlainQuantity due to issues with python 3.8 failing
            to import the Quantity TypeAlias
        """
        return _ureg.Quantity(self.value, str(self.CurrentUnits.value))


@dataclass
class ParameterEntry:
    """A dataclass that contains the three fields that are being read from the user-provided file

    Attributes:
        Name (str): The official name of the parameter that the user wants to set
        sValue (str): The value that the user wants it to be set to, as a string.
        Comment (str): The optional comment that the user provided with that parameter in the text file
    """

    Name: str
    sValue: str
    Comment: Optional[str] = None
    raw_entry: Optional[str] = None


@dataclass
class OutputParameter(HasQuantity):
    """A dataclass that is the holder values that are provided to the user as output
     but are calculated internally by GEOPHIRES

    Attributes:
        Name (str): The official name of that output
        value: (any): the value of this parameter - can be int, float, text, bool, list, etc...
        ToolTipText (str): Text to place in a ToolTip in a UI
        UnitType (IntEnum): The class of units that parameter falls in (i.e., "length", "time", "area"...)
        PreferredUnits (Enum): The units as required by GEOPHIRES (or your algorithms)
        CurrentUnits (Enum): The units that the parameter is provided in (usually the same PreferredUnits)
        UnitsMatch (boolean): Internal flag set when units are different
    """

    Name: str = ""
    display_name: str = None
    value: Any = 0
    ToolTipText: str = ""
    UnitType: IntEnum = Units.NONE
    PreferredUnits: Enum = Units.NONE
    # set to PreferredUnits by default assuming that the current units are the preferred units -
    # they will only change if the read function reads a different unit associated with a parameter
    CurrentUnits: Enum = PreferredUnits
    json_parameter_type: str = None

    @property
    def UnitsMatch(self) -> bool:
        return self.CurrentUnits == self.PreferredUnits

    def with_preferred_units(self) -> Any:  # Any is a proxy for Self
        ret: OutputParameter = dataclasses.replace(self)
        ret.value = ret.quantity().to(convertible_unit(ret.PreferredUnits)).magnitude
        ret.CurrentUnits = ret.PreferredUnits
        return ret

    def __post_init__(self):
        if self.display_name is None:
            self.display_name: str = self.Name

        if self.json_parameter_type is None:
            # Note that this is sensitive to order of comparison; unit test ensures correct behavior:
            # test_parameter.ParameterTestCase.test_output_parameter_json_types
            if isinstance(self.value, str):
                self.json_parameter_type = _JSON_PARAMETER_TYPE_STRING
            elif isinstance(self.value, bool):
                self.json_parameter_type = _JSON_PARAMETER_TYPE_BOOLEAN
            elif isinstance(self.value, float) or isinstance(self.value, int):
                # Default number values may not be representative of whether calculated values are integer-only,
                # so we specify number type even if value is int.
                self.json_parameter_type = _JSON_PARAMETER_TYPE_NUMBER
            elif isinstance(self.value, dict):
                self.json_parameter_type = _JSON_PARAMETER_TYPE_OBJECT
            elif isinstance(self.value, Iterable):
                self.json_parameter_type = _JSON_PARAMETER_TYPE_ARRAY
            else:
                self.json_parameter_type = _JSON_PARAMETER_TYPE_OBJECT

@dataclass
class Parameter(HasQuantity):
    """
     A dataclass that is the holder values that are provided (optionally) by the user.  These are all the inout values
     to the model.  They all must have a default value that is reasonable and will
     provide a reasonable result if not changed.

    Attributes:
        Name (str): The official name of that output
        Required (bool, False): Is this parameter required to be set?  See user manual.
        Provided (bool, False): Has this value been provided by the user?
        Valid (bool, True): has this value been successfully validated?
        ErrMessage (str): the error message that the user sees if the value they provide does not pass validation -
              by default, it is: "assuming default value (see manual)"
        InputComment (str): The optional comment that the user provided with that parameter in the text file
        ToolTipText (str): Text to place in a ToolTip in a UI
        UnitType (IntEnum): The class of units that parameter falls in (i.e., "length", "time", "area"...)
        PreferredUnits (Enum): The units as required by GEOPHIRES (or your algorithms)
        CurrentUnits (Enum): The units that the parameter is provided in (usually the same PreferredUnits)
        UnitsMatch (boolean): Internal flag set when units are different
        AllowExtendedInput (boolean, False): Whether to allow extended input for this parameter, such as a list of values,
             a filename, or a URL, instead of a single value.  If True, the code will check if the user provided value is
             a list of values (in the form of a string), a filename, or a URL, and if so, it will try to read the content
             and use that as the value for this parameter instead of the original value.  This allows for more flexible input
             for parameters that may need to be provided as a list of values or as a file or URL containing the values,
             but it also bypasses the normal validation and conversion of the value, so it should be used with caution
             and only for parameters where this kind of input is appropriate. Note that the code that uses this parameter
             MUST know how to handle the values that were in the list, file, or URL correctly!
    """

    Name: str = ""
    Required: bool = False
    Provided: bool = False
    Valid: bool = True
    ErrMessage: str = "assume default value (see manual)"
    InputComment: str = ""
    ToolTipText: str = Name
    UnitType: IntEnum = Units.NONE
    PreferredUnits: Enum = None
    AllowExtendedInput: bool = False

    # set to PreferredUnits assuming that the current units are the preferred units
    # - they will only change if the read function reads a different unit associated with a parameter
    CurrentUnits: Enum = PreferredUnits

    @property
    def UnitsMatch(self) -> bool:
        return self.PreferredUnits == self.CurrentUnits

    parameter_category: str = None
    ValuesEnum:GeophiresInputEnum = None
    auto_raise_exception_on_invalid_read: bool = False


    def __post_init__(self):
        if self.PreferredUnits is None:
            self.PreferredUnits = self.CurrentUnits

@dataclass
class boolParameter(Parameter):
    """
    boolParameter: a dataclass that stores the values for a Boolean value.  Includes the default value and the
    validation values (if appropriate).  Child of Parameter, so it gets all the Attributes of that class.

    Attributes:
        value (bool): The value of that parameter
        DefaultValue (bool, True):  The default value of that parameter
    """

    def __post_init__(self):
        if self.value is None:
            self.value: bool = self.DefaultValue
        super().__post_init__()

    value: bool = None
    DefaultValue: bool = value
    json_parameter_type: str = _JSON_PARAMETER_TYPE_BOOLEAN


@dataclass
class intParameter(Parameter):
    """
    intParameter: a dataclass that stores the values for an Integer value.  Includes the default value and the
    validation values (if appropriate).  Child of Parameter, so it gets all the Attributes of that class.

    Attributes:
        value (int): The value of that parameter
        DefaultValue (int, 0):  The default value of that parameter
        AllowableRange (list): A list of the valid values
    """

    def __post_init__(self):
        if self.value is None:
            self.value:int = self.DefaultValue
        super().__post_init__()

    value: int = None
    DefaultValue: int = value
    AllowableRange: List[int] = field(default_factory=list)
    json_parameter_type: str = _JSON_PARAMETER_TYPE_INTEGER

    def coerce_value_to_enum(self):
        if self.ValuesEnum is not None:
            if not isinstance(self.value, self.ValuesEnum):
                self.value = self.ValuesEnum.from_int(self.value)


@dataclass
class floatParameter(Parameter):
    """
    floatParameter: a dataclass that stores the values for a Float value.  Includes the default value and the
    validation values (if appropriate).  Child of Parameter, so it gets all the Attributes of that class.

    Attributes:
        value (float): The value of that parameter
        DefaultValue (float, 0.0):  The default value of that parameter
        Min (float, -1.8e308): minimum valid value - not that it is set to a very small value,
                which means that any value is valid by default
        Min (float, 1.8e308): maximum valid value - not that it is set to a very large value,
                which means that any value is valid by default
    """

    def __post_init__(self):
        if self.value is None:
            self.value = self.DefaultValue
        super().__post_init__()

    value: float = None
    DefaultValue: float = 0.0
    Min: float = -1.8e30
    Max: float = 1.8e30
    json_parameter_type: str = _JSON_PARAMETER_TYPE_NUMBER


@dataclass
class strParameter(Parameter):
    """
    strParameter: a dataclass that stores the values for a String value.  Includes the default value and the
    validation values (if appropriate).  Child of Parameter, so it gets all the Attributes of that class.

    Attributes:
        value (str): The value of that parameter
        DefaultValue (str, ""):  The default value of that parameter
    """
    def __post_init__(self):
        if self.value is None:
            self.value: str = self.DefaultValue
        super().__post_init__()

    value: str = None
    DefaultValue: str = value
    json_parameter_type: str = _JSON_PARAMETER_TYPE_STRING


@dataclass
class listParameter(Parameter):
    """
    listParameter: a dataclass that stores the values for a List of values.  Includes the default value and the
    validation values (if appropriate).  Child of Parameter, so it gets all the Attributes of that class.

    Attributes:
        value (list): The value of that parameter
        DefaultValue (list, []):  The default value of that parameter
        Min (float, -1.8e308): minimum valid value of each value in the list - not that it is set to a very small value,
              which means that any value is valid by default
        Min (float, 1.8e308): maximum valid value of each value in the list - not that it is set to a very large value,
            which means that any value is valid by default
    """

    def __post_init__(self):
        if self.value is None:
            self.value: str = self.DefaultValue
        super().__post_init__()

    value: List[float] = None
    DefaultValue: List[float] = field(default_factory=list)
    Min: float = -1.8e308
    Max: float = 1.8e308
    json_parameter_type: str = _JSON_PARAMETER_TYPE_ARRAY


@dataclass
class filenameParameter(Parameter):
    """
    filenameParameter: a dataclass that stores the values for a filename value.  Includes the default value and the
    validation values (if appropriate).  Child of Parameter, so it gets all the Attributes of that class.

    Attributes:
        value (str): The value of that parameter
        DefaultValue (str, ""):  The default value of that parameter
    """
    def __post_init__(self):
        if self.value is None:
            self.value: str = self.DefaultValue
        super().__post_init__()

    value: str = None
    DefaultValue: str = value
    json_parameter_type: str = _JSON_PARAMETER_TYPE_STRING


@dataclass
class TimeSeriesParameter(listParameter):
    """
    TimeSeriesParameter: a dataclass that stores the values for a Historical List of values (time, and value).
    Derived from listParameter because it is still a list of values, but it has some additional attributes that are specific to historical lists.
    Includes the default value and the validation values (if appropriate).
    Child of listParameter, and thus Parameter, so it gets all the Attributes of that class.

    Attributes:
        PairVectorAsNumpyArray (bool, True): Whether to store the pair vector as a numpy array (if False, will store as list of lists)
        PreferredXUnits (Optional[str], None): Specifies the name of the preferred x dimension units, default = "hours"
        PreferredYUnits (Optional[str], None): Specifies the name of the preferred y dimension units
        CurrentXUnits (Optional[str], None): Holds the current units of the x dimension (determined from file)
        CurrentYUnits (Optional[str], None): Holds the current units of the y dimension (determined from file)
        ResampleToHourlyYear (bool, False): specify whether to resample the historical array to an hourly year (8760 hours)
    """

    def __post_init__(self):
        if self.value is None:
            self.value: str = self.DefaultValue
        super().__post_init__()

    PairVectorAsNumpyArray: bool = True
    PreferredXUnits: Optional[str] = r"hours"
    PreferredYUnits: Optional[str] = r"Temperature"
    CurrentXUnits: Optional[str] = PreferredXUnits
    CurrentYUnits: Optional[str] = PreferredYUnits
    ResampleToHourlyYear: bool = False
    XMax: float = 1.8e30
    XMin: float = -1.8e30
    YMax: float = 1.8e30
    YMin: float = -1.8e30

def ReadParameter(ParameterReadIn: ParameterEntry, ParamToModify, model) -> None:
    """
    ReadParameter: A method to take a single ParameterEntry object and use it to update the associated Parameter.
    Does validation as well as Unit and Currency conversion
    :param ParameterReadIn: The value the user wants to change and the value they want to change it to (as a string)
     and  any comment they provided with it (as a string) - all in one object (ParameterEntry) that is passed in
      to this method as a parameter itself (ParameterReadIn) - see ParameterEntry class for details on the fields in it
    :type ParameterReadIn: :class:`~geophires_x.Parameter.ParameterEntry`
    :param ParamToModify: The Parameter that will be modified (assuming it passes validation and conversion) - this is
      the object that will be modified by this method - see Parameter class for details on the fields in it
    :type ParamToModify: :class:`~geophires_x.Parameter.Parameter`
    :param model: The container class of the application, giving access to everything else, including the logger
    :type model: :class:`~geophires_x.Model.Model`
    :return: None
    """
    model.logger.info(f'Init {str(__name__)}: {sys._getframe().f_code.co_name} for {ParamToModify.Name}')

    #First, check to see if they are trying to set something to its default value.
    # If so, notify and return without doing any of the rest of the processing, since we know that the default value
    # is valid and in the correct units, so there is no need to do any of the validation or conversion that we do for other values.
    if ParameterReadIn.sValue == str(ParamToModify.DefaultValue):
        # let the user know then have provided a value that is the same as the default value or the existing value,
        # and that they can remove it from the input file if they want to use the default value
        model.logger.info(default_parameter_value_message(ParameterReadIn.sValue, ParamToModify.Name, ParamToModify.DefaultValue))
        ParamToModify.value = ParamToModify.DefaultValue
        # Preserve the fact that the user explicitly supplied this parameter even if it matches the default.
        ParamToModify.Provided = True
        ParamToModify.Valid = True
        return

    # If the ParamToModify allows ExtendedInput
    # then the first thing to do is to see if the user has specified a list, URL or a File,
    # and if so, the first thing we need to do is read the content of that file or URL and use it as the value for this parameter
    # instead of the original value, which is just a string that is the filename or URL
    if ParamToModify.AllowExtendedInput:
        seems_like_file = seems_like_URL = False
        content = ""
        if ParameterReadIn.sValue.startswith('http'):
            content = get_data_from_file_or_url_as_string(ParameterReadIn.sValue)
            if content:
                seems_like_URL = True
                ParameterReadIn.sValue = content
        elif is_existing_file_path(ParameterReadIn.sValue):
            content = get_data_from_file_or_url_as_string(ParameterReadIn.sValue)
            if content:
                seems_like_file = True
                ParameterReadIn.sValue = content
            else:
                seems_like_file = False
        # if the file or URL is provided but not valid, log an error and raise an exception
        if (seems_like_file or seems_like_URL) and not content:
            err_msg = f'Error: Provided value ({ParameterReadIn.sValue}) for {ParamToModify.Name} is not a valid file path.'
            print(err_msg)
            model.logger.fatal(err_msg)
            model.logger.info(f'Complete {str(__name__)}: {sys._getframe().f_code.co_name}')
            raise ValueError(err_msg)

    # Validate the type of Parameter read in, and set The Parameter they wish to modify if it is valid -
    # we have to do this separately for each type because of the different validation rules and the different ways of setting the value

    # Boolean and String Parameter Types don't have units and don't do anything fancy with the value,
    # so we can just deal with them right here, without needing to worry about any of the unit conversion or other validation
    # that we do for the other parameter types.
    # just do the conversion for bools with multiple options for when true and false are more flexible for the user,
    # but for strings, just take the value as is and assign it to the parameter
    if isinstance(ParamToModify, strParameter):
            #must be a string, so just assign it - any string is a valid string, so no validation needed
            ParamToModify.value = str(ParameterReadIn.sValue)
            ParamToModify.Provided = True  # set provided to true because we are using a user provide value now
            ParamToModify.Valid = True  # set Valid to true because it passed the validation tests
            model.logger.info(f'Complete {str(__name__)}: {sys._getframe().f_code.co_name}')
            return

    elif isinstance(ParamToModify, boolParameter):
        if ParameterReadIn.sValue.strip() in ['0', 'false', 'False', 'f', 'F', 'no', 'No', 'n', 'N']:
            ParamToModify.value = False
        elif ParameterReadIn.sValue.strip() in ['1', 'true', 'True', 't', 'T', 'yes', 'Yes', 'y', 'Y']:
            ParamToModify.value = True
        else:
            ParamToModify.value = ParamToModify.DefaultValue
        ParamToModify.Provided = True  # set provided to true because we are using a user provide value now
        ParamToModify.Valid = True  # set Valid to true because it passed the validation tests
        model.logger.info(f'Complete {str(__name__)}: {sys._getframe().f_code.co_name}')
        return

    # Int Parameter
    if isinstance(ParamToModify, intParameter):
        process_int_or_float_parameter(ParameterReadIn, ParamToModify, model)

    # Float Parameter
    elif isinstance(ParamToModify, floatParameter):
        process_int_or_float_parameter(ParameterReadIn, ParamToModify, model)

    # List Parameter and Timeseries List (since it is a child of list)
    elif isinstance(ParamToModify, listParameter):
        # if it is a list, replace the list is a text for with a list as a python list
        pair_vector = parse_container_string_simple(ParameterReadIn.sValue, ParamToModify, model)
        if pair_vector is None:
            pair_vector = _try_read_pair_vector(ParameterReadIn, param_to_modify=ParamToModify, model=model)
        if pair_vector is None:
            err_msg = f'Error: Provided value ({ParameterReadIn.sValue}) for {ParamToModify.Name} is not a list.'
            print(err_msg)
            model.logger.fatal(err_msg)
            model.logger.info(f'Complete {str(__name__)}: {sys._getframe().f_code.co_name}')
            raise ValueError(err_msg)

        # If there is a header in the pair_vector, and it has units, then we need to do the unit conversion
        header = header_is_present(pair_vector)
        if header:
            pair_vector = _try_parse_multiline_list(pair_vector, header, ParamToModify=ParamToModify)

        ParamToModify.value = pair_vector
        ParamToModify.Provided = True
        ParamToModify.Valid = True

        # if this is a Time Series, and the length of the series is reuired to be the same as the number of hours in a year,
        # then check that and interpolate it if not
        if isinstance(ParamToModify, TimeSeriesParameter) and ParamToModify.ResampleToHourlyYear:
            if len(ParamToModify.value) != 8760:
                data = list(map(list, ParamToModify.value))
                x, y = zip(*data)
                x,y,notes = resample_to_hourly_year(x,y)
                ParamToModify.value = list(zip(x, y))
                model.logger.info(f'Resampling {ParamToModify.Name} with comment {notes} to hourly year (8760 hours) since it has {len(ParamToModify.value)} entries and ResampleToHourlyYear is True')


    # Filename Parameter (new)
    elif isinstance(ParamToModify, filenameParameter):
        # for a filename, make sure the file exists. The user can also provide a URL, so make sure that is valid.
        # If they are OK, then just assign the input text string to the Parameter.
        # When the Parameter is used, the code is responsible for reading the content of the file or URL and using it correctly.
        if is_existing_file_path(ParameterReadIn.sValue) or url_returns_content(ParameterReadIn.sValue):
            ParamToModify.value = ParameterReadIn.sValue
            ParamToModify.Provided = True
            ParamToModify.Valid = True
            model.logger.info(f'Validated filename input for {ParamToModify.Name}: {ParameterReadIn.sValue}')
        else:
            if ParamToModify.Name == "Reservoir Output File Name":
                ParamToModify.value = ParameterReadIn.sValue
                ParamToModify.Provided = True
                ParamToModify.Valid = True
                model.logger.warning(
                    f'Deferred validation of {ParamToModify.Name} until reservoir output file parsing: '
                    f'{ParameterReadIn.sValue}'
                )
                model.logger.info(f'Complete {str(__name__)}: {sys._getframe().f_code.co_name}')
                return

            err_msg = f'Error: Provided value ({ParameterReadIn.sValue}) for {ParamToModify.Name} is not a valid file path or URL.'
            print(err_msg)
            model.logger.fatal(err_msg)
            model.logger.info(f'Complete {str(__name__)}: {sys._getframe().f_code.co_name}')
            raise ValueError(err_msg)

    #Make sure that the units are set to something - if the user provided a value with units,
    # then the CurrentUnits will have been updated by the ConvertUnits function, but if they provided a value without units,
    # then we want to make sure that the PreferredUnits is still set to something reasonable (i.e., the default PreferredUnits for that parameter)
    if ParamToModify.PreferredUnits is None:
        ParamToModify.PreferredUnits = ParamToModify.CurrentUnits

    model.logger.info(f'Complete {str(__name__)}: {sys._getframe().f_code.co_name}')


_PAIR_VECTOR_MAX_BYTES = 1_000_000


def _parse_csv_pair_line(line: str, param_to_modify=None, model=None) -> Optional[np.ndarray]:
    """
    Function to parse a single line of text as a pair vector, which is a common format for historical data input.
    The line should contain two values separated by a comma, which represent the x and y values of the pair vector.
    The function will try to convert the values to floats and return them as a numpy array. If the line is not in the correct format,
    or if the values cannot be converted to floats, or if they are not finite numbers, the function will return None.

    :param line: The line of text to parse as a pair vector
    :type line: str
    :param param_to_modify: The Parameter that will be modified (assuming it passes validation and conversion) -
        this is the object that will be modified by this method - see Parameter class for details on the fields in it
    :type param_to_modify: :class:`~geophires_x.Parameter.Parameter`
    :param model: The container class of the application, giving access to everything else, including the logger
    :type model: :class:`~geophires_x.Model.Model`
    :return: A numpy array containing the x and y values of the pair vector if successful, or None if any step fails
    :rtype: Optional[np.ndarray]
    """
    row = next(csv.reader([line], quotechar='"'), None)
    if row is None or len(row) != 2:
        return None

    converted_row: list[str] = []
    for raw_component in row:
        component = raw_component.strip()
        if param_to_modify is not None and model is not None and ' ' in component:
            try:
                component = ConvertUnits(param_to_modify, component, model)
            except Exception:
                # Fall back to scalar parsing behavior if conversion fails.
                return None
        converted_row.append(component)

    try:
        x = float(converted_row[0])
        y = float(converted_row[1])
    except ValueError:
        return None

    if not math.isfinite(x) or not math.isfinite(y):
        return None

    return np.array([x, y], dtype=float)


import csv
from typing import Optional
import numpy as np

def _try_parse_pair_vector_inline(raw_value: str, param_to_modify=None, model=None) -> Optional[np.ndarray]:
    candidate = raw_value.strip()

    # if blank, parsing fails.
    if candidate == '':
        return None

    # Inline parsing is for *one CSV row*. If it contains newlines, it's not inline.
    # Let caller fall through to file/url parsing, and ultimately scalar parsing.
    if '\n' in candidate or '\r' in candidate:
        return None

    try:
        return _parse_csv_pair_line(candidate, param_to_modify=param_to_modify, model=model)
    except csv.Error:
        # Not a valid single-line CSV row
        return None

def _pair_vector_from_csv_text(csv_text: str, param_to_modify=None, model=None) -> Optional[np.ndarray]:
    for line in csv_text.splitlines():
        stripped = line.strip()
        if stripped == '' or stripped.startswith('#') or stripped.startswith('--') or stripped.startswith('*'):
            continue
        return _parse_csv_pair_line(stripped, param_to_modify=param_to_modify, model=model)

    return None


def _try_parse_pair_vector_csv_file(path_str: str, param_to_modify=None, model=None) -> Optional[np.ndarray]:
    """
    A function to try to read a pair vector from a CSV file, given the path to the file as a string.
    It checks that the file exists and is not too large, then reads the content of the file and tries to parse it as a pair vector.
    If any of these steps fail, it returns None.
    :param path_str: The path to the CSV file as a string
    :type path_str: str
    :param param_to_modify: The Parameter that will be modified (assuming it passes validation and conversion) -
        this is the object that will be modified by this method - see Parameter class for details on the fields in it
    :type param_to_modify: :class:`~geophires_x.Parameter.Parameter`
    :param model: The container class of the application, giving access to everything else, including the logger
    :type model: :class:`~geophires_x.Model.Model`
    :return: A numpy array containing the pair vector if successful, or None if any step fails
    :rtype: Optional[np.ndarray]
    """
    if path_str.startswith('http'):
        return None

    path = Path(path_str)
    if not path.is_file():
        return None

    if path.stat().st_size > _PAIR_VECTOR_MAX_BYTES:
        return None

    with path.open(encoding='UTF-8') as f:
        text = f.read(_PAIR_VECTOR_MAX_BYTES + 1)
        if len(text) > _PAIR_VECTOR_MAX_BYTES:
            return None

    return _pair_vector_from_csv_text(text, param_to_modify=param_to_modify, model=model)


def _try_parse_pair_vector_csv_url(url_str: str, param_to_modify=None, model=None) -> Optional[np.ndarray]:
    parsed = urlparse(url_str)
    if parsed.scheme not in ['http', 'https']:
        return None

    with urlopen(url_str, timeout=5) as response:
        data = response.read(_PAIR_VECTOR_MAX_BYTES + 1)

    if len(data) > _PAIR_VECTOR_MAX_BYTES:
        return None

    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return None

    return _pair_vector_from_csv_text(text, param_to_modify=param_to_modify, model=model)


def _parse_numeric_list_tokens(tokens: list[str], param_to_modify, model) -> Optional[list[float]]:
    values: list[float] = []

    for raw_token in tokens:
        token = raw_token.strip()
        if token == '':
            continue

        if token.startswith('['):
            token = token[1:]
        if token.endswith(']'):
            token = token[:-1]
        token = token.strip()
        if token == '':
            continue

        if ' ' in token:
            try:
                token = ConvertUnits(param_to_modify, token, model)
            except Exception:
                return None

        try:
            value = float(token)
        except ValueError:
            return None

        if not math.isfinite(value):
            return None

        values.append(value)

    return values


def _parse_numeric_list_text(csv_text: str, param_to_modify, model) -> Optional[list[float]]:
    values: list[float] = []

    for raw_line in csv_text.splitlines():
        line = raw_line.strip()
        if line == '' or line.startswith('#') or line.startswith('--') or line.startswith('*'):
            continue

        row = next(csv.reader([line]), None)
        if row is None:
            continue

        parsed_row = _parse_numeric_list_tokens(row, param_to_modify, model)
        if parsed_row is None:
            return None

        values.extend(parsed_row)

    return values if len(values) > 0 else None


def _try_read_numeric_list_from_source(parameter_read_in: ParameterEntry, param_to_modify, model) -> Optional[list[float]]:
    candidates: list[str] = []

    rhs = _raw_input_rhs(parameter_read_in.raw_entry)
    if rhs is not None:
        candidates.append(rhs)

    if parameter_read_in.sValue is not None:
        candidates.append(parameter_read_in.sValue.strip())

    for candidate in candidates:
        if candidate == '':
            continue

        parsed_url = urlparse(candidate)
        if parsed_url.scheme in ['http', 'https']:
            try:
                with urlopen(candidate, timeout=5) as response:
                    data = response.read(_PAIR_VECTOR_MAX_BYTES + 1)
                if len(data) > _PAIR_VECTOR_MAX_BYTES:
                    continue
                text = data.decode('utf-8')
            except Exception:
                continue

            parsed_values = _parse_numeric_list_text(text, param_to_modify, model)
            if parsed_values is not None:
                return parsed_values
            continue

        file_path = Path(candidate)
        if file_path.is_file():
            if file_path.stat().st_size > _PAIR_VECTOR_MAX_BYTES:
                continue

            try:
                with file_path.open(encoding='UTF-8') as f:
                    text = f.read(_PAIR_VECTOR_MAX_BYTES + 1)
                if len(text) > _PAIR_VECTOR_MAX_BYTES:
                    continue
            except Exception:
                continue

            parsed_values = _parse_numeric_list_text(text, param_to_modify, model)
            if parsed_values is not None:
                return parsed_values

    return None


def _raw_input_rhs(raw_entry: Optional[str]) -> Optional[str]:
    if raw_entry is None or ',' not in raw_entry:
        return None

    # keep behavior aligned with list-parameter parsing: text after '--' is comment
    entry = raw_entry.split('--')[0]
    if ',' not in entry:
        return None

    # Drop incidental trailing delimiters (e.g. "Parameter, 8 degC, -- comment") so scalar
    # values with comments are not misclassified as CSV-like pair/historical arrays.
    return entry.split(',', 1)[1].strip().rstrip(',').strip()


def _is_pair_vector_candidate(parameter_read_in: ParameterEntry, param_to_modify=None) -> bool:
    if param_to_modify is None or not getattr(param_to_modify, 'AllowPairVectorInput', False):
        return False

    candidates = []
    rhs = _raw_input_rhs(parameter_read_in.raw_entry)
    if rhs is not None:
        candidates.append(rhs)
    if parameter_read_in.sValue is not None:
        candidates.append(parameter_read_in.sValue.strip())

    for candidate in candidates:
        lowered = candidate.lower()
        if ',' in candidate or candidate.startswith('[') or lowered.endswith('.csv'):
            return True
        parsed = urlparse(candidate)
        if parsed.scheme in ['http', 'https']:
            return True

    return False


def _is_historical_array_candidate(parameter_read_in: ParameterEntry, param_to_modify=None) -> bool:
    if param_to_modify is None or not getattr(param_to_modify, 'AllowHistoricalArrayInput', False):
        return False

    candidates = []
    rhs = _raw_input_rhs(parameter_read_in.raw_entry)
    if rhs is not None:
        candidates.append(rhs)
    if parameter_read_in.sValue is not None:
        candidates.append(parameter_read_in.sValue.strip())

    for candidate in candidates:
        lowered = candidate.lower()
        if ',' in candidate or '\n' in candidate or lowered.endswith('.csv'):
            return True
        parsed = urlparse(candidate)
        if parsed.scheme in ['http', 'https']:
            return True

    return False


def _try_read_pair_vector(parameter_read_in: ParameterEntry, param_to_modify=None, model=None) -> Optional[np.ndarray]:
    candidates = []

    #rhs = _raw_input_rhs(parameter_read_in.raw_entry)
    #if rhs is not None:
    #    candidates.append(rhs)

    # If the string, make a list out of it
    if isinstance(parameter_read_in.sValue, str):
        if len(__import__('re').findall(r'\r\n|\n\r|\r|\n', parameter_read_in.sValue.strip())) > 0:
            parameter_read_in.sValue = str([float(x) if __import__('re').fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', x) else x
                      for x in (l.strip() for l in __import__('re').split(r'\r\n|\n|\r', parameter_read_in.sValue.strip())) if x])
    if parameter_read_in.sValue is not None:
        candidates.append(parameter_read_in.sValue.strip())

    for candidate in candidates:
        if candidate == '':
            continue

        # Try inline parsing first, which is the most restrictive format and least expensive to parse.
        # that looks like a single line of CSV text, which is the format we expect for inline pair vector input, so try to parse it as that first.
        parsed = _try_parse_pair_vector_inline(candidate, param_to_modify=param_to_modify, model=model)
        if parsed is not None:
            return parsed

        # This option is a multiline string
        parsed = _try_parse_multiline(candidate, ParamToModify=param_to_modify, model=model)
        if parsed is not None:
            return parsed

        parsed = _try_parse_pair_vector_csv_file(candidate, param_to_modify=param_to_modify, model=model)
        if parsed is not None:
            return parsed

        try:
            parsed = _try_parse_pair_vector_csv_url(candidate, param_to_modify=param_to_modify, model=model)
            if parsed is not None:
                return parsed
        except Exception:
            # Fall back to scalar parsing if URL retrieval/parsing fails.
            continue

    return None


def ConvertUnits(ParamToModify, strUnit: str, model) -> str:
    """
    ConvertUnits gets called if a unit version is needed: either currency or standard units like F to C or m to ft
    :param ParamToModify: The Parameter that will be modified (assuming it passes validation and conversion) - this is
        the object that will be modified by this method - see Parameter class for details on the fields in it
    :type ParamToModify: :class:`~geophires_x.Parameter.Parameter`
    :param strUnit: A string containing the value to be converted along with the units it is currently in.
        The units to convert to are set by the PreferredUnits of ParamToModify
    :type strUnit: str
    :param model: The container class of the application, giving access to everything else, including the logger
    :type model: :class:`~geophires_x.Model.Model`
    :return: The new value as a string (without the units, because they are already held in PreferredUnits of ParamToModify)
    :rtype: str
    """
    model.logger.info(f'Init {str(__name__)}: {sys._getframe().f_code.co_name} for {ParamToModify.Name}')

    # deal with the currency case
    if ParamToModify.UnitType in [Units.CURRENCY, Units.CURRENCYFREQUENCY, Units.COSTPERMASS, Units.ENERGYCOST]:
        prefType = ParamToModify.PreferredUnits.value
        parts = strUnit.split(' ')
        val = parts[0].strip()
        currType = parts[1].strip()
        # user has provided a currency that is the currency expected, so just strip off the currency
        if prefType == currType:
            strUnit = str(val)
            ParamToModify.CurrentUnits = currType
            return strUnit

        # First we need to deal the possibility that there is a suffix on the units (like /yr, kwh, or /tonne)
        # that will make it not be recognized by the currency conversion engine.
        # generally, we will just strip the suffix off of a copy of the string that represents the units,
        # then allow the conversion to happen. For now, we ignore the suffix.
        # this has the consequence that we don't do any conversion based on that suffix, so units like EUR/MMBTU
        # will trigger a conversion to USD/MMBTU, where MMBY+TU doesn't get converted to KW (or whatever)
        currSuff = prefSuff = ""
        elements = currType.split("/")
        if len(elements) > 1:
            currType = elements[0]  # strip off the suffix, but save it
            currSuff = "/" + elements[1]
        elements = prefType.split("/")
        if len(elements) > 1:
            prefType = elements[0]  # strip off the suffix, but save it
            prefSuff = "/" + elements[1]

        # Let's try to deal with first the simple conversion where the required units have a prefix like M (m) or K (k)
        # that means a "million" or a "thousand", like MUSD (or KUSD), and the user provided USD (or KUSD) or KEUR, MEUR
        # we have to deal with the case that the M, m, K, or k are NOT prefixes, but rather are a part of the currency name.
        cc = CurrencyCodes()
        currFactor = prefFactor = 1.0
        currPrefix = prefPrefix = False
        Factor = 1.0
        prefShort = prefType
        currShort = currType
        # if either of these returns a symbol, then we must have prefixes we need to deal with
        symbol = cc.get_symbol(prefType[1:])
        symbol2 = cc.get_symbol(currType[1:])
        if symbol is not None:
            prefPrefix = True
        if symbol2 is not None:
            currPrefix = True
        if prefPrefix and prefType[0] in ['M', 'm']:
            prefFactor = prefFactor * 1_000_000.0
        elif prefPrefix and prefType[0] in ['K', 'k']:
            prefFactor = prefFactor * 1000.0
        if currPrefix and currType[0] in ['M', 'm']:
            currFactor = currFactor / 1_000_000.0
        elif currPrefix and currType[0] in ['K', 'k']:
            currFactor = currFactor / 1000.0
        Factor = currFactor * prefFactor
        Factor *= _ratio_suffix_conversion_factor(prefSuff, currSuff)
        if prefPrefix:
            prefShort = prefType[1:]
        if currPrefix:
            currShort = currType[1:]

        if prefShort == currShort:
            # this is true, then we just have a conversion between KUSD and USD, MUSD to KUSD, MUER to EUR, etc.,
            # so just do the simple factor conversion

            val = float(val) * Factor
            strUnit = str(val)
            ParamToModify.CurrentUnits = currType
            return strUnit

        try:
            # if we come here, we have a currency conversion to do (USD->EUR, etc.).

            if _DISABLE_FOREX_API:
                raise RuntimeError('Forex API disabled')

            cr = CurrencyRates()
            conv_rate = cr.get_rate(currShort, prefShort)
        except BaseException as ex:
            print(str(ex))
            msg = (
                f'Error: GEOPHIRES failed to convert your currency for {ParamToModify.Name} to something it understands. '
                f'You gave {strUnit} - conversion may be affected by https://github.com/NREL/GEOPHIRES-X/issues/236. '
                f'Please change your units to {ParamToModify.PreferredUnits.value} '
                f'to continue. Cannot continue unless you do. Exiting.'
            )
            print(msg)
            model.logger.critical(str(ex))
            model.logger.critical(msg)

            raise RuntimeError(msg)

        New_val = (conv_rate * float(val)) * Factor
        strUnit = str(New_val)
        ParamToModify.CurrentUnits = parts[1]

        if len(prefSuff) > 0:
            prefType = prefType + prefSuff  # set it back the way it was
        if len(currSuff) > 0:
            currType = currType + currSuff
        parts = strUnit.split(' ')
        strUnit = parts[0]
        return strUnit

    else:  # must be something other than boolean, string, or currency
        if isinstance(strUnit, pint.Quantity):
            val = ParamToModify.value
            currType = str(strUnit)
        else:
            parts = strUnit.split(' ')
            val = parts[0].strip()
            currType = parts[1].strip()
        # check to see if the units provided (CurrentUnits) are the same as the preferred units.
        # In that case, we don't need to do anything.
        try:
            # Make a Pint Quantity out of the old value: the amount of the unit doesn't matter,
            # just the units, so I set the amount to 0
            Old_valQ = _ureg.Quantity(0.000, str(ParamToModify.CurrentUnits.value))
            New_valQ = _ureg.Quantity(float(val), currType)  # Make a Pint Quantity out of the new value
        except BaseException as ex:
            print(str(ex))
            msg = (
                f'Error: GEOPHIRES failed to initialize your units for {ParamToModify.Name} '
                f'to something it understands. '
                f'You gave {strUnit} - Are the units defined for Pint library, or have you defined them in the '
                f'user-defined units file (GEOPHIRES3_newunits)?  Cannot continue. Exiting.'
            )
            print(msg)
            model.logger.critical(str(ex))
            model.logger.critical(msg)

            raise RuntimeError(msg)

        if Old_valQ.units != New_valQ.units:  # do the transformation only if the units don't match
            ParamToModify.CurrentUnits = LookupUnits(currType)[0]
            try:
                # update the quantity to the preferred units,
                # so we don't have to change the underlying calculations.  This assumes that Pint recognizes our unit.
                # If we have a new unit, we have to add it to the Pint configuration text file
                New_valQ.ito(Old_valQ)
            except BaseException as ex:
                print(str(ex))
                msg = (
                    f'Error: GEOPHIRES failed to convert your units for {ParamToModify.Name} '
                    f'to something it understands. You gave {strUnit} - Are the units defined for Pint library, '
                    f'or have you defined them in the user defined units file (GEOPHIRES3_newunits)? '
                    f'Cannot continue. Exiting.'
                )
                print(msg)
                model.logger.critical(str(ex))
                model.logger.critical(msg)

                raise RuntimeError(msg)

            # set sValue to the value based on the new units - don't add units to it - it should just be a raw number
            strUnit = str(New_valQ.magnitude)

            new_val_units_lookup = LookupUnits(str(New_valQ.units))
            if new_val_units_lookup is not None and new_val_units_lookup[0] is not None:
                ParamToModify.CurrentUnits = new_val_units_lookup[0]

        else:
            # if we come here, we must have a unit declared, but the unit must be the same as the preferred unit,
            # so we need to just get rid of the extra text after the space
            parts = strUnit.split(' ')
            strUnit = parts[0]

    model.logger.info(f'Complete {str(__name__)}: {sys._getframe().f_code.co_name}')
    return strUnit


def ConvertUnitsBack(ParamToModify: Parameter, model):
    """
    CovertUnitsBack: Converts units back to what the user specified they as.  It does this so that the user can see them
    in the report as the units they specified. We know that because CurrentUnits contains the desired units
    :param ParamToModify: The Parameter that will be modified (assuming it passes validation and conversion)
        - this is the object that will be modified by this method - see Parameter class for details on the fields in it
    :type ParamToModify: :class:`~geophires_x.Parameter.Parameter`
    :param model: The container class of the application, giving access to everything else, including the logger
    :type model: :class:`~geophires_x.Model.Model`
    :return: None
    """
    model.logger.info(f'Init {str(__name__)}: {sys._getframe().f_code.co_name} for {ParamToModify.Name}')

    try:
        ParamToModify.value = _ureg.Quantity(ParamToModify.value, convertible_unit(ParamToModify.CurrentUnits)).to(convertible_unit(ParamToModify.PreferredUnits)).magnitude
        ParamToModify.CurrentUnits = ParamToModify.PreferredUnits
    except Exception as conversion_error:
        # TODO refactor to check for/convert currency instead of relying on try/except once currency conversion is
        #  re-enabled - https://github.com/NREL/GEOPHIRES-X/issues/236?title=Currency+conversions+disabled
        if ParamToModify.UnitType not in [Units.CURRENCY, Units.CURRENCYFREQUENCY, Units.COSTPERMASS, Units.ENERGYCOST]:
            msg = (
                f'Error: GEOPHIRES failed to convert your units for {ParamToModify.Name} to something it understands. '
                f'You gave {ParamToModify.CurrentUnits}  - Are the units defined for Pint library, '
                f' or have you defined them in the user defined units file (GEOPHIRES3_newunits)? '
                f'Cannot continue. Exiting.'
            )
            model.logger.critical(f'Pint conversion failed ({conversion_error})')
            model.logger.critical(msg)
            raise RuntimeError(msg) from conversion_error

        model.logger.warning(f'Failed to convert units with pint, attempting currency conversion ({conversion_error})')

        try:
            param_modified: Parameter = _parameter_with_currency_units_converted_back_to_preferred_units(ParamToModify,
                                                                                                         model)
            ParamToModify.value = param_modified.value
            ParamToModify.CurrentUnits = param_modified.CurrentUnits
            ParamToModify.UnitType = param_modified.UnitType
        except AttributeError as currency_conversion_error:
            model.logger.error(f'Currency conversion failed ({currency_conversion_error})')

            msg = (
                f'Error: GEOPHIRES failed to convert your units for {ParamToModify.Name} to something it understands. '
                f'You gave {ParamToModify.CurrentUnits}  - Are the units defined for Pint library, '
                 f' or have you defined them in the user defined units file (GEOPHIRES3_newunits)? '
                f'Cannot continue. Exiting.'
            )
            model.logger.critical(msg)

            raise RuntimeError(msg)

    model.logger.info(f'Complete {str(__name__)}: {sys._getframe().f_code.co_name}')


def _ratio_suffix_conversion_factor(preferred_suffix: str, current_suffix: str) -> float:
    if not preferred_suffix or not current_suffix or preferred_suffix == current_suffix:
        return 1.0

    preferred_denominator = preferred_suffix.removeprefix("/")
    current_denominator = current_suffix.removeprefix("/")
    return _ureg.Quantity(1, convertible_unit(preferred_denominator)).to(convertible_unit(current_denominator)).magnitude


def _parameter_with_currency_units_converted_back_to_preferred_units(param: Parameter, model) -> Parameter:
    """
    TODO clean up and consolidate with pint-based conversion in ConvertUnitsBack
    """

    param_with_units_converted_back = copy.deepcopy(param)

    # deal with the currency case
    if param.UnitType in [Units.CURRENCY, Units.CURRENCYFREQUENCY, Units.COSTPERMASS, Units.ENERGYCOST]:
        prefType = param.PreferredUnits.value
        currType = param.CurrentUnits

        # First we need to deal the possibility that there is a suffix on the units (like /yr, kwh, or /tonne)
        # that will make it not be recognized by the currency conversion engine.
        # generally, we will just strip the suffix off of a copy of the string that represents the units, then allow
        # the conversion to happen. For now, we ignore the suffix.
        # this has the consequence that we don't do any conversion based on that suffix, so units like EUR/MMBTU
        # will trigger a conversion to USD/MMBTU, where MMBY+TU doesn't get converted to KW (or whatever)
        currSuff = prefSuff = ""
        elements = str(currType).split("/")
        if len(elements) > 1:
            currType = elements[0]  # strip off the suffix, but save it
            currSuff = "/" + elements[1]
        elements = prefType.split("/")
        if len(elements) > 1:
            prefType = elements[0]  # strip off the suffix, but save it
            prefSuff = "/" + elements[1]

        # Let's try to deal with first the simple conversion where the required units have a prefix like M (m) or K (k)
        # that means a "million" or a "thousand", like MUSD (or KUSD), and the user provided USD (or KUSD) or KEUR, MEUR
        # we have to deal with the case that the M, m, K, or k are NOT prefixes,
        # but rather are a part of the currency name.
        cc = CurrencyCodes()
        currFactor = prefFactor = 1.0
        currPrefix = prefPrefix = False
        Factor = 1.0
        prefShort = prefType
        currShort = currType
        symbol = cc.get_symbol(
            prefType[1:]
        )  # if either of these returns a symbol, then we must have prefixes we need to deal with
        symbol2 = cc.get_symbol(currType[1:])
        if symbol is not None:
            prefPrefix = True
        if symbol2 is not None:
            currPrefix = True
        if prefPrefix and prefType[0] in ['M', 'm']:
            prefFactor = prefFactor * 1_000_000.0
        elif prefPrefix and prefType[0] in ['K', 'k']:
            prefFactor = prefFactor * 1000.0
        if currPrefix and currType[0] in ['M', 'm']:
            currFactor = currFactor / 1_000_000.0
        elif currPrefix and currType[0] in ['K', 'k']:
            currFactor = currFactor / 1000.0
        Factor = currFactor * prefFactor
        Factor *= _ratio_suffix_conversion_factor(prefSuff, currSuff)
        if prefPrefix:
            prefShort = prefType[1:]
        if currPrefix:
            currShort = currType[1:]
        if prefShort == currShort:
            # this is true, then we just have a conversion between KUSD and USD, MUSD to KUSD, MUER to EUR, etc.,
            # so just do the simple factor conversion
            param_with_units_converted_back.value = param.value * Factor
            param_with_units_converted_back.CurrentUnits = currType
            return param_with_units_converted_back

        # Now lets deal with the case where the units still don't match, so we have a real currency conversion,
        # like USD to EUR
        # start the currency conversion process
        cc = CurrencyCodes()
        try:
            if _DISABLE_FOREX_API:
                raise RuntimeError('Forex API disabled')

            cr = CurrencyRates()
            conv_rate = cr.get_rate(currType, prefType)
        except BaseException as ex:
            print(str(ex))
            msg = (
                f'Error: GEOPHIRES failed to convert your currency for {param.Name} to something it understands.'
                f'You gave {currType} - conversion may be affected by https://github.com/NREL/GEOPHIRES-X/issues/236.  '
                f'Please change your units to {param.PreferredUnits.value} '
                f'to continue. Cannot continue unless you do.  Exiting.'
            )
            print(msg)
            model.logger.critical(str(ex))
            model.logger.critical(msg)

            raise RuntimeError(msg, ex)

        param_with_units_converted_back.value = (conv_rate * float(param.value)) / prefFactor
        return param_with_units_converted_back

    else:
        raise AttributeError(
            f'Unit/unit type ({param.CurrentUnits}/{param.UnitType}) for {param.Name} '
            f'is not a recognized currency unit'
        )


def LookupUnits(sUnitText: str):
    """
    LookupUnits Given a unit class and a text string, this will return the value from the Enumeration if it is there
    (or return nothing if it is not)
    :param sUnitText: The text string to look for in the Enumeration of units (like "m" or "feet") - this is the text
        that the user provides in the input file or the GUI to specify the units they want to use for a parameter or
        output value (like "m" or "feet")
    :type sUnitText: str (text)
    :return: The Enumerated value and the Unit class Enumeration
    :rtype: tuple
    """
    # look through all unit types and names for a match with my units
    for uType in Units:
        MyEnum = None
        if uType == Units.LENGTH:
            MyEnum = LengthUnit
        elif uType == Units.AREA:
            MyEnum = AreaUnit
        elif uType == Units.VOLUME:
            MyEnum = VolumeUnit
        elif uType == Units.MASS:
            MyEnum = MassUnit
        elif uType == Units.DENSITY:
            MyEnum = DensityUnit
        elif uType == Units.TEMPERATURE:
            MyEnum = TemperatureUnit
        elif uType == Units.PRESSURE:
            MyEnum = PressureUnit
        elif uType == Units.TIME:
            MyEnum = TimeUnit
        elif uType == Units.FLOWRATE:
            MyEnum = FlowRateUnit
        elif uType == Units.TEMP_GRADIENT:
            MyEnum = TemperatureGradientUnit
        elif uType == Units.DRAWDOWN:
            MyEnum = DrawdownUnit
        elif uType == Units.IMPEDANCE:
            MyEnum = ImpedanceUnit
        elif uType == Units.PRODUCTIVITY_INDEX:
            MyEnum = ProductivityIndexUnit
        elif uType == Units.INJECTIVITY_INDEX:
            MyEnum = InjectivityIndexUnit
        elif uType == Units.HEAT_CAPACITY:
            MyEnum = HeatCapacityUnit
        elif uType == Units.THERMAL_CONDUCTIVITY:
            MyEnum = ThermalConductivityUnit
        elif uType == Units.CURRENCY:
            MyEnum = CurrencyUnit
        elif uType == Units.CURRENCYFREQUENCY:
            MyEnum = CurrencyFrequencyUnit
        elif uType == Units.PERCENT:
            MyEnum = PercentUnit
        elif uType == Units.ENERGY:
            MyEnum = EnergyUnit
        elif uType == Units.ENERGYCOST:
            MyEnum = EnergyCostUnit
        elif uType == Units.ENERGYFREQUENCY:
            MyEnum = EnergyFrequencyUnit
        elif uType == Units.COSTPERMASS:
            MyEnum = CostPerMassUnit
        elif uType == Units.AVAILABILITY:
            MyEnum = AvailabilityUnit
        elif uType == Units.ENTROPY:
            MyEnum = EntropyUnit
        elif uType == Units.ENTHALPY:
            MyEnum = EnthalpyUnit
        elif uType == Units.POROSITY:
            MyEnum = PorosityUnit
        elif uType == Units.PERMEABILITY:
            MyEnum = PermeabilityUnit
        elif uType == Units.ENERGYDENSITY:
            MyEnum = EnergyDensityUnit
        elif uType == Units.MASSPERTIME:
            MyEnum = MassPerTimeUnit
        elif uType == Units.COSTPERDISTANCE:
            MyEnum = CostPerDistanceUnit
        elif uType == Units.POWER:
            MyEnum = PowerUnit
        elif uType == Units.CO2PRODUCTION:
            MyEnum = CO2ProductionUnit
        elif uType == Units.ENERGYPERCO2:
            MyEnum = EnergyPerCO2Unit

        if MyEnum is not None:
            for item in MyEnum:
                if item.value == sUnitText:
                    return item, uType

    try:
        canonical_unit_text = f'{_ureg.Quantity(1, sUnitText).units:~}'.replace(' ', '')
    except (UndefinedUnitError, ValueError):
        canonical_unit_text = None

    if canonical_unit_text is not None and canonical_unit_text != sUnitText.replace(' ', ''):
        return LookupUnits(canonical_unit_text)

    # No match was found with the unit text string, so try with the canonical symbol (if different).
    try:
        symbol = _ureg.get_symbol(sUnitText)
    except UndefinedUnitError:
        return None, None
    if symbol != sUnitText:
        return LookupUnits(symbol)
    return None, None


def ConvertOutputUnits(oparam: OutputParameter, newUnit: Units, model):
    """
    ConvertOutputUnits Given an output parameter, convert the value(s) from what they contain
    (as calculated by GEOPHIRES) to what the user specified as what they want for outputs.  Conversion happens inline.

    :param oparam: The parameter you want to be converted (value or list of values).  Because Parameters know the
        PreferredUnits and CurrentUnits, this routine knows what to do. It will convert the value(s) in the parameter
        to the new units, and then reset the CurrentUnits to the new units. This is done so that the user can see the units
        they specified in the output report. The value(s) in the parameter are converted back to the original units after
        the report is generated. This is done so that the calculations are done in the units that GEOPHIRES expects. If
        the user wants to see the output in different units, they can specify that in the input file or the GUI.
    :type oparam: :class:`~geophires_x.Parameter.Parameter`
    :param newUnit: The new units you want to convert value to (like "m" or "feet") - this is the text that the user
        provides in the input file or the GUI to specify the units they want to use for a parameter or output value
        (like "m" or "feet")
    :type newUnit: str (text)
    :param model: The container class of the application, giving access to everything else, including the logger
    :type model: :class:`~geophires_x.Model.Model`
    :return: None
    """

    try:
        oparam.value = _ureg.Quantity(oparam.value, oparam.CurrentUnits.value).to(convertible_unit(newUnit.value)).magnitude
        oparam.CurrentUnits = newUnit
        return
    except AttributeError as ae:
        # TODO refactor to check for/convert currency instead of relying on try/except once currency conversion is
        #  re-enabled - https://github.com/NREL/GEOPHIRES-X/issues/236?title=Currency+conversions+disabled
        model.logger.warning(f'Failed to convert units with pint, falling back to legacy conversion code ({ae})')

    if isinstance(oparam.value, str):
        return  # strings have no units
    elif isinstance(oparam.value, bool):
        return  # booleans have no units
    DefUnit, UnitSystem = LookupUnits(str(newUnit.value))

    if UnitSystem not in [Units.CURRENCY, Units.CURRENCYFREQUENCY, Units.COSTPERMASS, Units.ENERGYCOST]:
        msg = (
                    "Warning: GEOPHIRES failed to initialize your units for "
                    + oparam.Name
                    + " to something it understands. You gave "
                    + str(newUnit.value)
                    + " -Are the units defined for"
                    + " Pint library, or have you defined them in the user defined units file (GEOPHIRES3_newunits)?"
                    + " Continuing without output conversion."
                )
        print(msg)
        model.logger.warning(msg)
        return

    else:
        prefType = oparam.PreferredUnits.value
        currType = newUnit.value

        # First we need to deal the possibility that there is a suffix on the units (like /yr, kwh, or /tonne)
        # that will make it not be recognized by the currency conversion engine.
        # generally, we will just strip the suffix off of a copy of the string that represents the units, then
        # allow the conversion to happen. For now, we ignore the suffix.
        # this has the consequence that we don't do any conversion based on that suffix, so units like EUR/MMBTU
        # will trigger a conversion to USD/MMBTU, where MMBY+TU doesn't get converted to KW (or whatever)
        currSuff = prefSuff = ""
        elements = str(currType).split("/")
        if len(elements) > 1:
            currType = elements[0]  # strip off the suffix, but save it
            currSuff = "/" + elements[1]
        elements = prefType.split("/")
        if len(elements) > 1:
            prefType = elements[0]  # strip off the suffix, but save it
            prefSuff = "/" + elements[1]

        # Let's try to deal with first the simple conversion where the required units have a prefix like M (m) or K (k)
        # that means a "million" or a "thousand", like MUSD (or KUSD), and the user provided USD (or KUSD) or KEUR, MEUR
        # we have to deal with the case that the M, m, K, or k are NOT prefixes, but rather
        # are a part of the currency name.
        cc = CurrencyCodes()
        currFactor = prefFactor = 1.0
        currPrefix = prefPrefix = False
        Factor = 1.0
        prefShort = prefType
        currShort = currType
        symbol = cc.get_symbol(
            prefType[1:]
        )  # if either of these returns a symbol, then we must have prefixes we need to deal with
        symbol2 = cc.get_symbol(currType[1:])
        if symbol is not None:
            prefPrefix = True
        if symbol2 is not None:
            currPrefix = True
        if prefPrefix and prefType[0] in ['M', 'm']:
            prefFactor = prefFactor * 1_000_000.0
        elif prefPrefix and prefType[0] in ['K', 'k']:
            prefFactor = prefFactor * 1000.0
        if currPrefix and currType[0] in ['M', 'm']:
            currFactor = currFactor / 1_000_000.0
        elif currPrefix and currType[0] in ['K', 'k']:
            currFactor = currFactor / 1000.0
        Factor = currFactor * prefFactor
        if prefPrefix:
            prefShort = prefType[1:]
        if currPrefix:
            currShort = currType[1:]
        if prefShort == currShort:
            # this is true, then we just have a conversion between KUSD and USD, MUSD to KUSD, MUER to EUR, etc.,
            # so just do the simple factor conversion and exit
            oparam.value = oparam.value * Factor
            oparam.CurrentUnits = DefUnit
            return

        # start the currency conversion process
        # if we have a symbol for a currency type, then the type is known to the library.
        # If we don't try some tricks to make it into something it does do recognize
        symbol = cc.get_symbol(currShort)
        if symbol is None:
            msg = (
                f'Error: GEOPHIRES failed to convert your currency for {oparam.Name} to something it understands.'
                f'You gave {currType} - conversion may be affected by https://github.com/NREL/GEOPHIRES-X/issues/236.  '
                f'Please change your units to {oparam.PreferredUnits.value} '
                f'to continue. Cannot continue unless you do.  Exiting.'
            )
            print(msg)
            model.logger.critical(msg)

            raise RuntimeError(msg)

        symbol = cc.get_symbol(prefShort)
        # if we have a symbol for a currency type, then the type is known to the library.  If we don't
        # try some tricks to make it into something it does do recognize
        if symbol is None:
            msg = (
                f'Error: GEOPHIRES failed to convert your currency for {oparam.Name} to something it understands.'
                f'You gave {currType} - conversion may be affected by https://github.com/NREL/GEOPHIRES-X/issues/236.  '
                f'Please change your units to {oparam.PreferredUnits.value} '
                f'to continue. Cannot continue unless you do.  Exiting.'
            )

            print(msg)
            model.logger.critical(msg)

            raise RuntimeError(msg)
        try:
            if _DISABLE_FOREX_API:
                raise RuntimeError('Forex API disabled')

            cr = CurrencyRates()
            conv_rate = cr.get_rate(prefShort, currShort)
        except BaseException as ex:
            print(str(ex))

            msg = (
                f'Error: GEOPHIRES failed to convert your currency for {oparam.Name} to something it understands. '
                f'You gave {currType} - conversion may be affected by https://github.com/NREL/GEOPHIRES-X/issues/236. '
                f'Please change your units to {oparam.PreferredUnits.value} '
                f'to continue. Cannot continue unless you do. Exiting.'
            )

            print(msg)
            model.logger.critical(str(ex))
            model.logger.critical(msg)

            raise RuntimeError(msg)

        oparam.value = Factor * conv_rate * float(oparam.value)
        oparam.CurrentUnits = DefUnit
        model.logger.info(f'Complete {str(__name__)}: {sys._getframe().f_code.co_name}')



def coerce_int_params_to_enum_values(parameter_dict:dict[str,Parameter]) -> None:
    """
    Some modules have intParameters with an int default value whose working value gets set to an
    enum when the parameter is read. If the parameter is not provided as an input, the default int value needs to
    be coerced into the corresponding enum value.
    TODO: resolve these enum/int value discrepancies so this workaround can be removed
    """

    for param_name, param in parameter_dict.items():
        if isinstance(param, intParameter):
            parameter_dict[param_name].coerce_value_to_enum()



def default_parameter_value_message(new_val: Any, param_to_modify_name: str, default_value: Any) -> str:
    return (
        f'Parameter given ({str(new_val)}) for {param_to_modify_name} is the same as the default value. '
        f'Consider removing {param_to_modify_name} from the input file unless you wish '
        f'to change it from the default value of ({str(default_value)})'
    )


def process_int_or_float_parameter(parameter_read_in: ParameterEntry, param_to_modify: Parameter, model):
    # Process a floatParameter or intParameter in the same way, except the conversion type is different.
    # :param parameter_read_in: The ParameterEntry that was read in from the input file, which contains the value
    #       to be processed and put into the Parameter object
    # :type parameter_read_in: :class:`~geophires_x.Parameter.ParameterEntry`
    # :param param_to_modify: The Parameter object that we are trying to set the value for, which contains the default value,
    #     the allowable range, the preferred units, and other information that we need to know in order to process the value
    #     correctly and validate it
    # :type param_to_modify: :class:`~geophires_x.Parameter.Parameter`
    is_simple_conversion = True
    is_list = False
    try:
        # Very first thing - try to convert the value to a float or int, If it is a clean conversion, go on with the validation.
        # If it is not clean, then it means it could be a list, file, URL that needs processing, or it just needs unit conversion.
        if isinstance(param_to_modify, intParameter):
            param_to_modify.value = int(float(parameter_read_in.sValue))
        else:
            param_to_modify.value = float(parameter_read_in.sValue)
    except ValueError:
        #didn't make a clean conversion, so set a flag
        is_simple_conversion = False

    if not is_simple_conversion:  # I could do all this is the except clause, but that is not pythonic, so I set a flag and then do it outside the try/except block
        # test to see if we are working on a parameter that allows extended input. If we are,
        # and the user has supplied a normal parameter as a list (in the form of a  string), as a filename, or as a URL instead of a value
        # then process the list, file, or URL and replace the value of the parameter with the content of the list, file, or URL.
        # it does try to handle the units correctly, if it can.
        # This allows for more flexible input for parameters, but it also bypasses the normal validation and conversion of the value,
        # so it should be used with caution and only for parameters where this kind of input is appropriate.
        # also note that the code that uses this parameter MUST know how to handle the values that were in the list, file, or URL correctly!
        # we need to do this before the check for a space in the value because if there is a space in the value,
        # we will try to do unit conversion on it, and that will fail if it is a list, file, or URL instead of a simple value with units.
        # this also allows us to have the file or URL return a value with units in it,
        # and then we can do the unit conversion, which is more flexible for the user.
        if param_to_modify.AllowExtendedInput:
            if is_numeric_sequence_string(parameter_read_in.sValue):  # must be a simple list without units
                result = parse_container_string_simple(parameter_read_in.sValue, param_to_modify, model)
                if result is not None:
                    param_to_modify.value = result
                    is_list = True
                # if it is None, then it did not successfully parse out as a list, so it is likely just a simple value with units,
                # so we will try to process it as that below
            elif parameter_read_in.sValue.strip().startswith('http') or is_existing_file_path(parameter_read_in.sValue):
                param_to_modify.value = get_data_from_file_or_url(parameter_read_in.sValue, param_to_modify, model)

                # the file or URl may have returned a list, so set the flag
                is_list = is_numeric_sequence_string(str(param_to_modify.value))

        # deal with the case where the user has provided units. That will be indicated by a space in it
        # the strategy is to look for a space in the value, and if there is one, then we will assume that the value has a unit in it,
        # and we will try to convert it to the preferred units. If it can, it will replace the value with the converted value
        # (without the unit, which is now implicit because it is in the preferred units) and then later onn, process it as normal.
        # If it can't, then it will leave the value as is and we will try to process it as is,
        # which may lead to an error later on but at least we will have tried to convert it if we could.
        # lists doesn't have units
        if not is_list and ' ' in parameter_read_in.sValue:
            new_str = ConvertUnits(param_to_modify, parameter_read_in.sValue, model)
            if len(new_str) > 0:
                parameter_read_in.sValue = new_str
                if isinstance(param_to_modify, intParameter):
                    param_to_modify.value = int(float(parameter_read_in.sValue))
                else:
                    param_to_modify.value = float(parameter_read_in.sValue)
        # else:
        # The value came in without any units
        # TODO: determine the proper action in this case
        # (previously, it was assumed that the value must be
        # using the default PreferredUnits, which was not always
        # valid and led to incorrect units in the output)

    # Check the valid range for the parameter. If it is outside the valid range, set Valid to False and set the value back to the default value.
    # Note that now it could be a list of integers or float but that may be too complex to check ranges for.
    in_range = False
    if isinstance(param_to_modify, intParameter) and not isinstance(param_to_modify.value, list):
        # just check to see if it is in the AllowRange list
        in_range = param_to_modify.value in param_to_modify.AllowableRange

    elif isinstance(param_to_modify, floatParameter) and not isinstance(param_to_modify.value, list):
        in_range = param_to_modify.Min <= param_to_modify.value <= param_to_modify.Max

    elif isinstance(param_to_modify, intParameter) and isinstance(param_to_modify.value, list):
        # It is really hard to do a range check on a list because it may be long and complex list with many data types in it,
        # so we will just skip the range check in that case, and hope that the code that uses this parameter can handle it correctly.
        # We will log a warning about this.
        # TODO We want to have range checking for lists, but that is complicated for complex lists.
        #  We need to check each value in the list against that range but only for the columns we are intereste in.
        #  That is a more complex implementation, but it would be more robust and would allow us to have range checking for lists.
        msg = (
            f'Warning: GEOPHIRES failed to validate the range of the values provided for {param_to_modify.Name} because it could not convert all of the values to int. This may be because you provided a long and complex list with many data types in it. GEOPHIRES will skip the range check for this parameter, and hope that the code that uses this parameter can handle it correctly. If you want to have range checking for this parameter, please provide a simple list of integer values that can be converted to int. The current value provided is: {parameter_read_in.sValue}'
        )
        print(msg)
        model.logger.warning(msg)
        in_range = True  # set it to true so that we don't fail the validation just because we can't convert to int

    elif isinstance(param_to_modify, floatParameter) and isinstance(param_to_modify.value, list):
        # It is really hard to do a range check on a list because it may be long and complex list with many data types in it,
        # so we will just skip the range check in that case, and hope that the code that uses this parameter can handle it correctly.
        # We will log a warning about this.
        msg = (
            f'Warning: GEOPHIRES failed to validate the range of the values provided for {param_to_modify.Name} because it could not convert all of the values to float. This may be because you provided a long and complex list with many data types in it. GEOPHIRES will skip the range check for this parameter, and hope that the code that uses this parameter can handle it correctly. If you want to have range checking for this parameter, please provide a simple list of numeric values that can be converted to float. The current value provided is: {parameter_read_in.sValue}'
        )
        print(msg)
        model.logger.warning(msg)
        in_range = True  # set it to true so that we don't fail the validation just because we can't convert to float

    if not in_range:
        param_to_modify.Valid = False
        err_msg = f"Warning: Parameter given ({parameter_read_in.sValue}) for {param_to_modify.Name} is outside of valid range. Please use a value in the valid range."
        print(err_msg)
        model.logger.info(err_msg)
        invalid_value_for_exception = parameter_read_in.sValue
        if isinstance(param_to_modify, floatParameter):
            try:
                invalid_value_for_exception = float(parameter_read_in.sValue)
            except (TypeError, ValueError):
                pass
        elif isinstance(param_to_modify, intParameter):
            try:
                invalid_value_for_exception = int(float(parameter_read_in.sValue))
            except (TypeError, ValueError):
                pass
        param_to_modify.value = param_to_modify.DefaultValue #set it to the default value, but log a warning about it
        model.logger.info(f'Continuing with default value ({param_to_modify.DefaultValue}) for {param_to_modify.Name}')
        param_to_modify.Provided = True

        if param_to_modify.auto_raise_exception_on_invalid_read:
            raise RuntimeError(
                f'Error: Parameter given ({invalid_value_for_exception}) for {param_to_modify.Name} outside of valid range.'
            )

        model.logger.info(f'Complete {str(__name__)}: {sys._getframe().f_code.co_name}')
        return

    # All is good
    param_to_modify.Provided = True  # set provided to true because we are using a user provide value now
    param_to_modify.Valid = True  # set Valid to true because it passed the validation tests
    model.logger.info(f'Complete {str(__name__)}: {sys._getframe().f_code.co_name}')
