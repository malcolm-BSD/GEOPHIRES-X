import dataclasses
import math

from geophires_x.GeoPHIRESUtils import UpgradeSymbologyOfUnits


@dataclasses.dataclass
class OutputTableItem:
    parameter: str = ''
    value: str = ''
    units: str = ''

    def __init__(self, parameter: str, value: str = '', units: str = ''):
        self.parameter = parameter
        self.value = value
        self.units = units
        if self.units:
            self.units = UpgradeSymbologyOfUnits(self.units)


HEAT_PRICE_UNAVAILABLE_REASON = "Unavailable: net heat <= 0"


def format_heat_price(value: float) -> str:
    """Format a numeric heat price or its unavailable sentinel for reports."""
    return f'{"N/A":>10}' if math.isnan(value) else f'{value:10.2f}'
