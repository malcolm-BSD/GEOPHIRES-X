from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np


@dataclass
class HistoricalXYSeries:
    x_raw: np.ndarray
    y_raw: np.ndarray
    x_units_raw: Optional[str]
    y_units_raw: Optional[str]
    x_canonical: np.ndarray
    y_canonical: np.ndarray
    x_units_canonical: str
    y_units_canonical: str
    source_kind: str
    has_header: bool
    normalization_notes: List[str] = field(default_factory=list)


def detect_header_units(header_line: str) -> Tuple[Optional[str], Optional[str]]:
    row = next(csv.reader([header_line]), None)
    if not row or len(row) < 2:
        return None, None

    def _extract(token: str) -> Optional[str]:
        token = token.strip()
        if '(' in token and ')' in token:
            start = token.find('(')
            end = token.rfind(')')
            if end > start + 1:
                return token[start + 1:end].strip()
        return None

    return _extract(row[0]), _extract(row[1])


def _to_time_hours(values: np.ndarray, unit: str) -> np.ndarray:
    u = unit.strip().lower()
    factors = {
        's': 1 / 3600,
        'sec': 1 / 3600,
        'second': 1 / 3600,
        'seconds': 1 / 3600,
        'min': 1 / 60,
        'minute': 1 / 60,
        'minutes': 1 / 60,
        'h': 1,
        'hr': 1,
        'hour': 1,
        'hours': 1,
        'day': 24,
        'days': 24,
        'week': 24 * 7,
        'weeks': 24 * 7,
        'month': 24 * 30,
        'months': 24 * 30,
        'year': 24 * 365,
        'years': 24 * 365,
    }
    if u not in factors:
        raise ValueError(f'Unsupported time unit: {unit}')
    return values * factors[u]


def _to_distance_meters(values: np.ndarray, unit: str) -> np.ndarray:
    u = unit.strip().lower()
    factors = {
        'm': 1,
        'meter': 1,
        'meters': 1,
        'metre': 1,
        'metres': 1,
        'cm': 0.01,
        'centimeter': 0.01,
        'centimeters': 0.01,
        'mm': 0.001,
        'in': 0.0254,
        'inch': 0.0254,
        'inches': 0.0254,
        'ft': 0.3048,
        'foot': 0.3048,
        'feet': 0.3048,
        'km': 1000,
        'kilometer': 1000,
        'kilometers': 1000,
    }
    if u not in factors:
        raise ValueError(f'Unsupported distance unit: {unit}')
    return values * factors[u]


def _to_celsius(values: np.ndarray, unit: str) -> np.ndarray:
    u = unit.strip().lower()
    if u in ('c', 'degc', 'celsius', 'centigrade'):
        return values
    if u in ('f', 'degf', 'fahrenheit'):
        return (values - 32.0) * 5.0 / 9.0
    if u in ('k', 'kelvin'):
        return values - 273.15
    raise ValueError(f'Unsupported temperature unit: {unit}')


def _to_usd_per_kwh(values: np.ndarray, unit: str) -> np.ndarray:
    u = unit.strip().lower().replace(' ', '')
    if u in ('usd/kwh', '$/kwh', 'dollar/kwh', 'dollars/kwh'):
        return values
    if u in ('eur/kwh', '€/kwh', 'euro/kwh', 'euros/kwh'):
        return values * 1.1
    raise ValueError(f'Unsupported cost-rate unit: {unit}')


def convert_xy_units(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    x_dimension: str,
    y_dimension: str,
    x_units: str,
    y_units: str,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    if x_dimension == 'time':
        x = _to_time_hours(x, x_units)
        x_out = 'hour'
    elif x_dimension == 'distance':
        x = _to_distance_meters(x, x_units)
        x_out = 'meter'
    else:
        x_out = x_units

    if y_dimension == 'temperature':
        y = _to_celsius(y, y_units)
        y_out = 'degC'
    elif y_dimension == 'cost_rate':
        y = _to_usd_per_kwh(y, y_units)
        y_out = 'USD/kWh'
    else:
        y_out = y_units

    return x, y, x_out, y_out


def resample_to_hourly_year(x_values: np.ndarray, y_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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


def parse_xy_series(
    content: str,
    *,
    x_dimension: str,
    y_dimension: str,
    default_x_units: str,
    default_y_units: str,
    resample_to_hourly: bool,
    source_kind: str,
) -> HistoricalXYSeries:
    lines = [ln.strip() for ln in content.splitlines() if ln.strip() and not ln.strip().startswith(('#', '--', '*'))]
    if not lines:
        raise ValueError('No XY data found')

    x_units, y_units = detect_header_units(lines[0])
    has_header = x_units is not None or y_units is not None
    data_lines = lines[1:] if has_header else lines
    if not data_lines:
        raise ValueError('No XY rows found')

    xs: List[float] = []
    ys: List[float] = []
    for ln in data_lines:
        row = next(csv.reader([ln]), None)
        if row is None or len(row) != 2:
            raise ValueError(f'Invalid XY row: {ln}')
        xs.append(float(row[0].strip()))
        ys.append(float(row[1].strip()))

    x_raw = np.asarray(xs, dtype=float)
    y_raw = np.asarray(ys, dtype=float)

    if np.any(~np.isfinite(x_raw)) or np.any(~np.isfinite(y_raw)):
        raise ValueError('XY data contains non-finite values')

    x_conv, y_conv, x_canonical_units, y_canonical_units = convert_xy_units(
        x_raw,
        y_raw,
        x_dimension=x_dimension,
        y_dimension=y_dimension,
        x_units=x_units or default_x_units,
        y_units=y_units or default_y_units,
    )

    if np.any(np.diff(x_conv) < 0):
        raise ValueError('X values must be non-decreasing')

    notes: List[str] = []
    if resample_to_hourly:
        x_conv, y_conv, extra_notes = resample_to_hourly_year(x_conv, y_conv)
        notes.extend(extra_notes)

    return HistoricalXYSeries(
        x_raw=x_raw,
        y_raw=y_raw,
        x_units_raw=x_units,
        y_units_raw=y_units,
        x_canonical=x_conv,
        y_canonical=y_conv,
        x_units_canonical=x_canonical_units,
        y_units_canonical=y_canonical_units,
        source_kind=source_kind,
        has_header=has_header,
        normalization_notes=notes,
    )


def load_xy_series_from_source(
    value_or_source: str,
    *,
    x_dimension: str,
    y_dimension: str,
    default_x_units: str,
    default_y_units: str,
    resample_to_hourly: bool,
    raw_entry: Optional[str] = None,
) -> HistoricalXYSeries:
    candidate = (value_or_source or '').strip()
    source_kind = 'inline'
    content = None

    if candidate:
        p = Path(candidate)
        if p.exists() and p.is_file():
            source_kind = 'file'
            content = p.read_text(encoding='utf-8')
        else:
            parsed = urlparse(candidate)
            if parsed.scheme in ('http', 'https'):
                source_kind = 'url'
                with urlopen(candidate, timeout=10) as resp:
                    content = resp.read(1_000_000).decode('utf-8')

    if content is None:
        if raw_entry and ',' in raw_entry:
            content = raw_entry.split(',', 1)[1]
        else:
            content = candidate

    return parse_xy_series(
        content,
        x_dimension=x_dimension,
        y_dimension=y_dimension,
        default_x_units=default_x_units,
        default_y_units=default_y_units,
        resample_to_hourly=resample_to_hourly,
        source_kind=source_kind,
    )
