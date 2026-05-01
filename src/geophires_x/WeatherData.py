"""Fetch, normalize, and aggregate Open-Meteo weather data."""

from __future__ import annotations

import calendar
import time
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_WEATHER_DATA_YEAR = 2024
OPEN_METEO_HISTORICAL_START_YEAR = 1940
OPEN_METEO_ARCHIVE_DELAY_DAYS = 5
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1.0
EXPECTED_HOURLY_ROWS = 8760
LEAP_YEAR_HOURLY_ROWS = 8784
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

REQUIRED_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "surface_pressure",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
]

OPTIONAL_HOURLY_VARIABLES = [
    "wet_bulb_temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
]

DEFAULT_HOURLY_VARIABLES = REQUIRED_HOURLY_VARIABLES + OPTIONAL_HOURLY_VARIABLES


class OpenMeteoWeatherError(ValueError):
    """Raised when Open-Meteo weather data cannot be fetched or validated."""


class OpenMeteoNetworkError(OpenMeteoWeatherError):
    """Raised when Open-Meteo cannot be reached or returns only transient failures."""


@dataclass
class WeatherData:
    """Normalized hourly Open-Meteo weather data for one project location."""

    latitude: float
    longitude: float
    year: int
    hourly_data: pd.DataFrame
    hourly_units: dict[str, str]

    def hourly(self) -> pd.DataFrame:
        """Return a copy of the normalized 8760-row hourly weather data."""
        return self.hourly_data.copy()

    def daily_average(self) -> pd.DataFrame:
        """Return calendar-day mean values from normalized hourly weather data."""
        values = self._numeric_hourly_data()
        return values.groupby(self.hourly_data["time"].dt.date).mean()

    def weekly_average(self) -> pd.DataFrame:
        """Return ISO-week mean values from normalized hourly weather data."""
        values = self._numeric_hourly_data()
        iso_calendar = self.hourly_data["time"].dt.isocalendar()
        return values.groupby([iso_calendar["year"], iso_calendar["week"]]).mean()

    def monthly_average(self) -> pd.DataFrame:
        """Return calendar-month mean values from normalized hourly weather data."""
        values = self._numeric_hourly_data()
        return values.groupby(self.hourly_data["time"].dt.to_period("M")).mean()

    def annual_average(self) -> pd.Series:
        """Return annual mean values from normalized hourly weather data."""
        return self._numeric_hourly_data().mean()

    def _numeric_hourly_data(self) -> pd.DataFrame:
        return self.hourly_data.drop(columns=["time"]).apply(pd.to_numeric, errors="coerce")


def fetch_open_meteo_weather(
    latitude: float,
    longitude: float,
    year: int = DEFAULT_WEATHER_DATA_YEAR,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    session: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> WeatherData:
    """Fetch and normalize one year of hourly weather data from Open-Meteo."""
    _validate_request_inputs(latitude, longitude, year, timeout_seconds, max_retries, backoff_seconds)

    http_session = session or requests.Session()
    try:
        response_json = _request_open_meteo_json(
            http_session,
            latitude,
            longitude,
            year,
            DEFAULT_HOURLY_VARIABLES,
            timeout_seconds,
            max_retries,
            backoff_seconds,
            sleep,
        )
    except OpenMeteoWeatherError as error:
        if not _error_mentions_optional_variable(str(error)):
            raise
        warnings.warn(
            f"Open-Meteo did not accept one or more optional weather variables; retrying with required variables only. {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        response_json = _request_open_meteo_json(
            http_session,
            latitude,
            longitude,
            year,
            REQUIRED_HOURLY_VARIABLES,
            timeout_seconds,
            max_retries,
            backoff_seconds,
            sleep,
        )

    hourly_data, hourly_units = _weather_data_from_response(response_json, year)
    return WeatherData(
        latitude=float(latitude),
        longitude=float(longitude),
        year=int(year),
        hourly_data=hourly_data,
        hourly_units=hourly_units,
    )


def _validate_request_inputs(
    latitude: float,
    longitude: float,
    year: int,
    timeout_seconds: float,
    max_retries: int,
    backoff_seconds: float,
) -> None:
    latitude = float(latitude)
    longitude = float(longitude)
    year = int(year)

    if latitude < -90.0 or latitude > 90.0:
        raise OpenMeteoWeatherError("Project Latitude must be between -90 and 90 degrees.")
    if longitude < -180.0 or longitude > 180.0:
        raise OpenMeteoWeatherError("Project Longitude must be between -180 and 180 degrees.")
    latest_year = _latest_open_meteo_complete_historical_year()
    if year < OPEN_METEO_HISTORICAL_START_YEAR or year > latest_year:
        raise OpenMeteoWeatherError(
            "Weather Data Year must be within the Open-Meteo historical archive range, "
            f"{OPEN_METEO_HISTORICAL_START_YEAR} through {latest_year}."
        )
    if timeout_seconds <= 0.0:
        raise OpenMeteoWeatherError("Weather API timeout must be greater than zero.")
    if max_retries < 0:
        raise OpenMeteoWeatherError("Weather API retries must be greater than or equal to zero.")
    if backoff_seconds < 0.0:
        raise OpenMeteoWeatherError("Weather API backoff must be greater than or equal to zero.")


def _request_open_meteo_json(
    session: Any,
    latitude: float,
    longitude: float,
    year: int,
    hourly_variables: list[str],
    timeout_seconds: float,
    max_retries: int,
    backoff_seconds: float,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "start_date": f"{int(year)}-01-01",
        "end_date": f"{int(year)}-12-31",
        "hourly": ",".join(hourly_variables),
        "timezone": "auto",
    }

    attempt_count = max_retries + 1
    last_error: Exception | None = None
    for attempt_index in range(attempt_count):
        try:
            response = session.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=timeout_seconds)
            status_code = int(getattr(response, "status_code", 200))
            response_json = _response_json(response)
            if 200 <= status_code < 300:
                return response_json

            message = _open_meteo_error_message(response_json, status_code)
            if status_code not in TRANSIENT_STATUS_CODES:
                raise OpenMeteoWeatherError(message)

            last_error = OpenMeteoNetworkError(message)
        except requests.exceptions.RequestException as error:
            last_error = error

        if attempt_index < attempt_count - 1:
            sleep(backoff_seconds * (2 ** attempt_index))

    if isinstance(last_error, (requests.exceptions.RequestException, OpenMeteoNetworkError)):
        raise OpenMeteoNetworkError(
            f"Open-Meteo request failed after {attempt_count} attempts: {last_error}"
        ) from last_error

    raise OpenMeteoWeatherError(f"Open-Meteo request failed after {attempt_count} attempts: {last_error}") from last_error


def _latest_open_meteo_complete_historical_year(today: date | None = None) -> int:
    latest_available_date = (today or date.today()) - timedelta(days=OPEN_METEO_ARCHIVE_DELAY_DAYS)
    if latest_available_date < date(latest_available_date.year, 12, 31):
        return latest_available_date.year - 1
    return latest_available_date.year


def _response_json(response: Any) -> dict[str, Any]:
    try:
        response_json = response.json()
    except ValueError as error:
        raise OpenMeteoWeatherError("Open-Meteo response was not valid JSON.") from error

    if not isinstance(response_json, dict):
        raise OpenMeteoWeatherError("Open-Meteo response JSON must be an object.")

    return response_json


def _open_meteo_error_message(response_json: dict[str, Any], status_code: int) -> str:
    reason = response_json.get("reason") or response_json.get("error") or "Unknown Open-Meteo error"
    return f"Open-Meteo request failed with HTTP {status_code}: {reason}"


def _error_mentions_optional_variable(message: str) -> bool:
    return any(variable in message for variable in OPTIONAL_HOURLY_VARIABLES)


def _weather_data_from_response(response_json: dict[str, Any], year: int) -> tuple[pd.DataFrame, dict[str, str]]:
    hourly = response_json.get("hourly")
    if not isinstance(hourly, dict):
        raise OpenMeteoWeatherError("Open-Meteo response did not include hourly weather data.")

    hourly_units = response_json.get("hourly_units", {})
    if not isinstance(hourly_units, dict):
        hourly_units = {}

    _validate_required_hourly_variables(hourly)
    _warn_for_missing_optional_hourly_variables(hourly)

    row_count = _validate_hourly_lengths(hourly)
    hourly_data = _normalized_hourly_dataframe(hourly, year, row_count)
    return hourly_data, dict(hourly_units)


def _validate_required_hourly_variables(hourly: dict[str, Any]) -> None:
    missing_required = [variable for variable in REQUIRED_HOURLY_VARIABLES if variable not in hourly]
    if missing_required:
        raise OpenMeteoWeatherError(
            f"Open-Meteo response is missing required hourly weather variables: {', '.join(missing_required)}."
        )


def _warn_for_missing_optional_hourly_variables(hourly: dict[str, Any]) -> None:
    missing_optional = [variable for variable in OPTIONAL_HOURLY_VARIABLES if variable not in hourly]
    if missing_optional:
        warnings.warn(
            f"Open-Meteo response is missing optional hourly weather variables: {', '.join(missing_optional)}.",
            RuntimeWarning,
            stacklevel=2,
        )


def _validate_hourly_lengths(hourly: dict[str, Any]) -> int:
    if "time" not in hourly:
        raise OpenMeteoWeatherError("Open-Meteo response is missing hourly time values.")

    lengths = {variable: len(values) for variable, values in hourly.items() if isinstance(values, list)}
    if "time" not in lengths:
        raise OpenMeteoWeatherError("Open-Meteo hourly time values must be a list.")

    inconsistent_lengths = {variable: length for variable, length in lengths.items() if length != lengths["time"]}
    if inconsistent_lengths:
        raise OpenMeteoWeatherError("Open-Meteo hourly weather variables must all have the same length.")

    row_count = lengths["time"]
    if row_count not in [EXPECTED_HOURLY_ROWS, LEAP_YEAR_HOURLY_ROWS]:
        raise OpenMeteoWeatherError(
            f"Open-Meteo hourly weather data must contain 8760 or 8784 rows; received {row_count}."
        )

    return row_count


def _normalized_hourly_dataframe(hourly: dict[str, Any], year: int, row_count: int) -> pd.DataFrame:
    normalized_data: dict[str, Any] = {"time": _normalized_hourly_time_index(year)}
    for variable, values in hourly.items():
        if variable == "time":
            continue

        values_array = np.asarray(values, dtype=float)
        if row_count == EXPECTED_HOURLY_ROWS:
            normalized_data[variable] = values_array
        else:
            normalized_data[variable] = _resample_series_to_8760(values_array)

    return pd.DataFrame(normalized_data)


def _normalized_hourly_time_index(year: int) -> pd.Series:
    reference_year = int(year)
    if calendar.isleap(reference_year):
        reference_year = reference_year - 1
    return pd.Series(pd.date_range(f"{reference_year}-01-01", periods=EXPECTED_HOURLY_ROWS, freq="h"))


def _resample_series_to_8760(values: np.ndarray) -> np.ndarray:
    source_positions = np.linspace(0.0, 1.0, num=len(values), endpoint=True)
    target_positions = np.linspace(0.0, 1.0, num=EXPECTED_HOURLY_ROWS, endpoint=True)
    return np.interp(target_positions, source_positions, values)
