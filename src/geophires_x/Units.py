from __future__ import annotations
import os
import inspect
import re
import sys
from enum import IntEnum, Enum, auto
from typing import Any
from typing import Dict, Iterable, List, Union
import pint


_UREG = None


def get_unit_registry():
    global _UREG
    if _UREG is None:
        _UREG = pint.get_application_registry()
        _UREG.load_definitions(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'GEOPHIRES3_newunits.txt'))

    return _UREG


def convertible_unit(unit: Any) -> Any:
    """
    pint can't handle '%' as a unit in python 3.8, so use this method when constructing quantities

    :type unit: str|Enum
    """
    if unit == Units.PERCENT or unit == PercentUnit.PERCENT or unit == Units.PERCENT.value:
        return 'percent'

    if unit == PercentUnit.TENTH or unit == PercentUnit.TENTH.value:
        return 'dimensionless'

    return unit


class Units(IntEnum):
    """All possible systems of measure"""
    NONE = auto()
    CHOICE = auto()
    LENGTH = auto()
    ANGLE = auto()
    AREA = auto()
    VOLUME = auto()
    MASS = auto()
    DENSITY = auto()
    TEMPERATURE = auto()
    PRESSURE = auto()
    TIME = auto()
    FLOWRATE = auto()
    TEMP_GRADIENT = auto()
    DRAWDOWN = auto()
    IMPEDANCE = auto()
    PRODUCTIVITY_INDEX = auto()
    INJECTIVITY_INDEX = auto()
    HEAT = auto()
    HEAT_CAPACITY = auto()
    ENTROPY = auto()
    ENTHALPY = auto()
    THERMAL_CONDUCTIVITY = auto()
    POROSITY = auto()
    PERMEABILITY = auto()
    CURRENCY = auto()
    CURRENCYFREQUENCY = auto()
    ENERGYCOST = auto()
    ENERGYDENSITY = auto()
    COSTPERMASS = auto()
    MASSPERTIME = auto()
    COSTPERDISTANCE = auto()
    PERCENT = auto()
    ENERGY = auto()
    POWER = auto()
    ENERGYFREQUENCY = auto()
    AVAILABILITY = auto()
    CO2PRODUCTION = auto()
    ENERGYPERCO2 = auto()
    POPDENSITY = auto()
    HEATPERUNITAREA = auto()
    POWERPERUNITAREA = auto()
    HEATPERUNITVOLUME = auto()
    POWERPERUNITVOLUME = auto()
    DECAY_RATE = auto()
    INFLATION_RATE = auto()
    DYNAMIC_VISCOSITY = auto()


class AngleUnit(str, Enum):
    """Angle Units"""
    DEGREES = "degrees"
    RADIANS = "radians"


class TemperatureUnit(str, Enum):
    """Temperature Units"""
    CELSIUS = "degC"
    FAHRENHEIT = "degF"
    KELVIN = "degK"


class TemperatureGradientUnit(str, Enum):
    """Temperature Gradient Units"""
    DEGREESCPERKM = "degC/km"
    DEGREESFPERMILE = "degF/mi"
    DEGREESCPERM = "degC/m"
    DEGREESFPER100FT = "degF/100ft"
    DEGREESFPER1000FT = "degF/kft"
    DEGREESFPERFT = "degF/ft"


class PercentUnit(str, Enum):
    """Percent Units"""
    PERCENT = "%"
    TENTH = ""


class LengthUnit(str, Enum):
    """Length Units"""
    METERS = "meter"
    CENTIMETERS = "centimeter"
    KILOMETERS = "kilometer"
    FEET = "ft"
    INCHES = "in"
    MILES = "mile"


class AreaUnit(str, Enum):
    """Area Units"""
    METERS2 = "m**2"
    CENTIMETERS2 = "cm**2"
    KILOMETERS2 = "km**2"
    FEET2 = "ft**2"
    INCHES2 = "in**2"
    MILES2 = "mi**2"


class VolumeUnit(str, Enum):
    """Volume Units"""
    METERS3 = "m**3"
    CENTIMETERS3 = "cm**3"
    KILOMETERS3 = "km**3"
    FEET3 = "ft**3"
    INCHES3 = "in**3"
    MILES3 = "mi**3"


class DensityUnit(str, Enum):
    """Density Units"""
    KGPERMETERS3 = "kg/m**3"
    GRPERCENTIMETERS3 = "gr/cm**3"
    KGPERKILOMETERS3 = "kg/km**3"
    LBSPERFEET3 = "lbs/ft**3"
    OZPERINCHES3 = "oz/in**3"
    LBSPERMILES3 = "lbs/mi**3"


class EnergyUnit(str, Enum):
    """Energy (electricity or heat) Units"""
    WH = "Wh"
    KWH = "kWh"
    MWH = "MWh"
    GWH = "GWh"
    MMBTU = "MMBTU"


class PowerUnit(str, Enum):
    """Power (electricity or heat) Units"""
    W = "W"
    KW = "kW"
    MW = "MW"
    GW = "GW"


class EnergyFrequencyUnit(str, Enum):
    """Energy per interval Units"""
    WPERYEAR = "W/yr"
    KWPERYEAR = "kW/yr"
    MWPERYEAR = "MW/yr"
    GWPERYEAR = "GW/yr"
    KWhPERYEAR = "kWh/yr"
    MWhPERHOUR = "MWh/hr"
    MWhPERDAY = "MWh/day"
    MWhPERYEAR = "MWh/year"
    GWhPERYEAR = "GWh/year"


class CurrencyUnit(str, Enum):
    """Currency Units"""
    MDOLLARS = "MUSD"
    KDOLLARS = "KUSD"
    DOLLARS = "USD"
    MEUR = "MEUR"
    KEUR = "KEUR"
    EUR = "EUR"
    MMXN = "MMXN"
    KMXN = "KMXN"
    MXN = "MXN"


class CurrencyFrequencyUnit(str, Enum):
    MDOLLARSPERYEAR = "MUSD/yr"
    KDOLLARSPERYEAR = "KUSD/yr"
    DOLLARSPERYEAR = "USD/yr"
    MEURPERYEAR = "MEUR/yr"
    KEURPERYEAR = "KEUR/yr"
    EURPERYEAR = "EUR/yr"
    MMXNPERYEAR = "MXN/yr"
    KMXNPERYEAR = "KMXN/yr"
    MXNPERYEAR = "MXN/yr"

    def get_currency_unit_str(self) -> str:
        return self.value.split('/')[0]


class EnergyCostUnit(str, Enum):
    DOLLARSPERKWH = "USD/kWh"
    DOLLARSPERMWH = "USD/MWh"
    CENTSSPERKWH = "cents/kWh"
    DOLLARSPERKW = "USD/kW"
    CENTSSPERKW = "cents/kW"
    DOLLARSPERMMBTU = "USD/MMBTU"
    DOLLARSPERMCF = "USD/MCF"


class EnergyDensityUnit(str, Enum):
    KWHPERMCF = "kWh/MCF"


class MassPerTimeUnit(str, Enum):
    TONNEPERYEAR = "tonne/yr"


class CostPerMassUnit(str, Enum):
    CENTSSPERMT = "cents/mt"
    DOLLARSPERMT = "USD/mt"
    DOLLARSPERTONNE = "USD/tonne"
    CENTSSPERLB = "cents/lb"
    DOLLARSPERLB = "USD/lb"


class CostPerDistanceUnit(str, Enum):
    DOLLARSPERM = "USD/m"
    KDOLLARSPERKM = "KUSD/km"


class PressureUnit(str, Enum):
    """Pressure Units"""
    MPASCAL = "MPa"
    KPASCAL = "kPa"
    PASCAL = "Pa"
    BAR = "bar"
    KBAR = "kbar"
    PSI = "psi"


class AvailabilityUnit(str, Enum):
    """Availability Units"""
    MWPERKGPERSEC = "MW/(kg/s)"


class DrawdownUnit(str, Enum):
    """Drawdown Units"""
    KGPERSECPERSQMETER = "kg/s/m**2"
    PERYEAR = "1/year"


class HeatUnit(str, Enum):
    """Heat Units"""
    J = "J"
    KJ = "kJ"


class HeatCapacityUnit(str, Enum):
    """Heat Capacity Units"""
    JPERKGPERK = "J/kg/K"
    KJPERKM3C = "kJ/km**3C"
    kJPERKGC = "kJ/kgC"


class EntropyUnit(str, Enum):
    """Entropy Units"""
    KJPERKGK = "kJ/kgK"


class EnthalpyUnit(str, Enum):
    """Enthalpy Units"""
    KJPERKG = "kJ/kg"


class ThermalConductivityUnit(str, Enum):
    """Thermal Conductivity Units"""
    WPERMPERK = "W/m/K"


class TimeUnit(str, Enum):
    """Time Units"""
    MSECOND = "msec"
    SECOND = "sec"
    MINUTE = "min"
    HOUR = "hr"
    DAY = "day"
    WEEK = "week"
    YEAR = "yr"


class FlowRateUnit(str, Enum):
    """Flow Rate Units"""
    KGPERSEC = "kg/sec"


class ImpedanceUnit(str, Enum):
    """Impedance Units"""
    GPASPERM3 = "GPa.s/m**3"


class ProductivityIndexUnit(str, Enum):
    """Productivity IndexUnits"""
    KGPERSECPERBAR = "kg/sec/bar"


class InjectivityIndexUnit(str, Enum):
    """Injectivity IndexUnits"""
    KGPERSECPERBAR = "kg/sec/bar"


class PorosityUnit(str, Enum):
    """Porosity Units"""
    PERCENT = "%"


class PermeabilityUnit(str, Enum):
    """Permeability Units"""
    SQUAREMETERS = "m**2"


class CO2ProductionUnit(str, Enum):
    """CO2 Production Units"""
    LBSPERKWH = "lbs/kWh"
    KPERKWH = "k/kWh"
    TONNEPERMWH = "t/MWh"


class EnergyPerCO2Unit(str, Enum):
    """Energy cost per tonne of CO2 extracted Units"""
    KWHEPERTONNE = "kWh/t"
    KWTHPERTONNE = "kW/t"


class MassUnit(str, Enum):
    """Mass Units"""
    GRAM = "gram"
    KILOGRAM = "kilogram"
    TONNE = "tonne"
    TON = "ton"
    KILOTONNE = "kilotonne"
    LB = "pound"
    OZ = "ounce"


class PopDensityUnit(str, Enum):
    """Population Density Units"""
    perkm2 = "Population per square km"


class HeatPerUnitAreaUnit(str, Enum):
    """Population Density Units"""
    KJPERSQKM = "kJ/km**2"


class PowerPerUnitAreaUnit(str, Enum):
    """Population Density Units"""
    MWPERSQKM = "MW/km**2"


class HeatPerUnitVolumeUnit(str, Enum):
    """Population Density Units"""
    KJPERCUBICKM = "kJ/km**3"


class PowerPerUnitVolumeUnit(str, Enum):
    """Population Density Units"""
    MWPERCUBICKM = "MW/km**3"


class Decay_RateUnit(str, Enum):
    """Decay rate Units"""
    PERCENTPERYEAR = "%/yr"


class Inflation_RateUnit(str, Enum):
    """Decay rate Units"""
    KPASCALPERYEAR = "kPa/yr"


class Dynamic_ViscosityUnit(str, Enum):
    """Dynamic Viscosity Units"""
    PASCALSEC = "PaSec"


# ----------------------------
# Normalization helpers
# ----------------------------
_SUPERSCRIPTS = str.maketrans({"²": "2", "³": "3"})

def _norm(s: str) -> str:
    """Normalize a string for matching against unit names.
    Steps:
        - Lowercase and strip whitespace
        - Replace common words/symbols with normalized forms (e.g. "per" -> "/", "sq." -> "sq")
        - Remove all remaining whitespace
        - Replace unicode superscripts with regular digits
    :param s: The input string to normalize.
    :return: A normalized string suitable for matching against unit names.
    """

    if s is None:
        return ""
    s = str(s).strip().lower().translate(_SUPERSCRIPTS)
    s = s.replace(" per ", "/").replace("\\", "/").replace("−", "-").replace("·", "*")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\*\*(\d+)", r"\1", s)
    s = re.sub(r"\^(\d+)", r"\1", s)
    s = s.replace("sq.", "sq").replace("square", "sq").replace("cubic", "cu")
    return s


def _heuristic_candidates(s: str) -> List[str]:
    s0 = _norm(s)
    cands = [s0]

    if s0.endswith("s") and len(s0) > 2:
        cands.append(s0[:-1])

    if s0.startswith("sq") and len(s0) > 2:
        cands.append(s0[2:] + "2")

    if s0.startswith("cu") and len(s0) > 2:
        cands.append(s0[2:] + "3")

    # de-dupe preserving order
    return list(dict.fromkeys(cands))


def _add_alias(map_: Dict[str, Enum], key: str, target: Enum) -> None:
    k = _norm(key)
    if k:
        map_[k] = target


# ----------------------------
# Auto-discovery of unit enums
# ----------------------------
def discover_unit_enums(module=None) -> List[type[Enum]]:
    """
    Discover unit Enum classes in a module.

    Criteria:
      - subclass of Enum
      - NOT an IntEnum (excludes Units)
      - has at least one member
      - members have string values (typical for your unit enums)

    :param module: The module to inspect. If None, uses the current module.
    """
    if module is None:
        module = sys.modules[__name__]

    enums: List[type[Enum]] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if not issubclass(obj, Enum):
            continue
        if issubclass(obj, IntEnum):   # excludes Units(IntEnum)
            continue

        # Ensure it "looks like" a unit enum: string-valued members
        try:
            members = list(obj)  # Enum iteration gives members
        except TypeError:
            continue

        if not members:
            continue

        # Require that all (or nearly all) values are strings
        if all(isinstance(m.value, str) for m in members):
            enums.append(obj)

    # Sort for deterministic behavior (nice for reproducibility)
    enums.sort(key=lambda c: c.__name__)
    return enums


def build_unit_lookup(unit_enums: Iterable[type[Enum]]) -> Dict[str, Enum]:
    """
    Build a lookup dictionary mapping various string representations to Enum members.
    :param unit_enums: An iterable of Enum classes to include in the lookup.
    :return: A dictionary mapping normalized strings to Enum members.
    """
    lookup: Dict[str, Enum] = {}

    for enum_cls in unit_enums:
        for member in enum_cls:
            _add_alias(lookup, member.value, member)                    # "mile"
            _add_alias(lookup, member.name, member)                     # "MILES"
            _add_alias(lookup, f"{enum_cls.__name__}.{member.name}", member)  # "LengthUnit.MILES"

    return lookup


def add_common_aliases(lookup: Dict[str, Enum]) -> None:
    """
    Add cross-cutting aliases that don't exist in the enums themselves.
    Keep this small and obvious; it's easy to extend later.
    :param lookup: The unit lookup dictionary to add aliases to.
    :return: None
    """
    # Length shorthand
    _add_alias(lookup, "m", LengthUnit.METERS)
    _add_alias(lookup, "cm", LengthUnit.CENTIMETERS)
    _add_alias(lookup, "km", LengthUnit.KILOMETERS)
    _add_alias(lookup, "ft", LengthUnit.FEET)
    _add_alias(lookup, "in", LengthUnit.INCHES)
    _add_alias(lookup, "mi", LengthUnit.MILES)

    # Area shorthand
    _add_alias(lookup, "m2", AreaUnit.METERS2)
    _add_alias(lookup, "cm2", AreaUnit.CENTIMETERS2)
    _add_alias(lookup, "km2", AreaUnit.KILOMETERS2)
    _add_alias(lookup, "ft2", AreaUnit.FEET2)
    _add_alias(lookup, "in2", AreaUnit.INCHES2)
    _add_alias(lookup, "mi2", AreaUnit.MILES2)
    _add_alias(lookup, "sqm", AreaUnit.METERS2)
    _add_alias(lookup, "sqkm", AreaUnit.KILOMETERS2)
    _add_alias(lookup, "sqft", AreaUnit.FEET2)
    _add_alias(lookup, "sqin", AreaUnit.INCHES2)
    _add_alias(lookup, "sqmi", AreaUnit.MILES2)

    # Volume shorthand
    _add_alias(lookup, "m3", VolumeUnit.METERS3)
    _add_alias(lookup, "cm3", VolumeUnit.CENTIMETERS3)
    _add_alias(lookup, "km3", VolumeUnit.KILOMETERS3)
    _add_alias(lookup, "ft3", VolumeUnit.FEET3)
    _add_alias(lookup, "in3", VolumeUnit.INCHES3)
    _add_alias(lookup, "mi3", VolumeUnit.MILES3)
    _add_alias(lookup, "cc", VolumeUnit.CENTIMETERS3)
    _add_alias(lookup, "cuft", VolumeUnit.FEET3)

    # Temp
    _add_alias(lookup, "°c", TemperatureUnit.CELSIUS)
    _add_alias(lookup, "c", TemperatureUnit.CELSIUS)
    _add_alias(lookup, "°f", TemperatureUnit.FAHRENHEIT)
    _add_alias(lookup, "f", TemperatureUnit.FAHRENHEIT)
    _add_alias(lookup, "k", TemperatureUnit.KELVIN)

    # Time
    _add_alias(lookup, "s", TimeUnit.SECOND)
    _add_alias(lookup, "h", TimeUnit.HOUR)
    _add_alias(lookup, "d", TimeUnit.DAY)

    # Percent
    _add_alias(lookup, "pct", PercentUnit.PERCENT)
    _add_alias(lookup, "percent", PercentUnit.PERCENT)


# Build once at import time (fast runtime lookup)
UNIT_ENUMS = discover_unit_enums()
UNIT_LOOKUP = build_unit_lookup(UNIT_ENUMS)
add_common_aliases(UNIT_LOOKUP)


def get_unit_from_string(unit_str: str) -> Union[Enum, Units]:
    """
    Convert a string to a unit Enum member, if possible.

    :param unit_str: The input string to interpret as a unit.
    :return: The corresponding Enum member if a match is found; otherwise, Units.NONE.
    """
    if not unit_str or not str(unit_str).strip():
        return Units.NONE

    for cand in _heuristic_candidates(unit_str):
        hit = UNIT_LOOKUP.get(cand)
        if hit is not None:
            return hit

    return Units.NONE
