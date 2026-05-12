from __future__ import annotations

import os
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import requests

from geophires_x.WeatherData import DEFAULT_HOURLY_VARIABLES
from geophires_x.WeatherData import EXPECTED_HOURLY_ROWS
from geophires_x.WeatherData import LEAP_YEAR_HOURLY_ROWS
from geophires_x.WeatherData import OPEN_METEO_HISTORICAL_START_YEAR
from geophires_x.WeatherData import OPTIONAL_HOURLY_VARIABLES
from geophires_x.WeatherData import REQUIRED_HOURLY_VARIABLES
from geophires_x.WeatherData import OpenMeteoNetworkError
from geophires_x.WeatherData import OpenMeteoWeatherError
from geophires_x.WeatherData import _latest_open_meteo_complete_historical_year
from geophires_x.WeatherData import fetch_open_meteo_weather


class FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _weather_payload(row_count: int = EXPECTED_HOURLY_ROWS, include_optional: bool = True) -> dict:
    hourly = {
        "time": [f"2023-01-01T{i % 24:02d}:00" for i in range(row_count)],
    }
    for variable in REQUIRED_HOURLY_VARIABLES:
        if variable == "temperature_2m":
            hourly[variable] = np.arange(row_count, dtype=float).tolist()
        else:
            hourly[variable] = np.full(row_count, 10.0, dtype=float).tolist()
    if include_optional:
        for variable in OPTIONAL_HOURLY_VARIABLES:
            hourly[variable] = np.full(row_count, 5.0, dtype=float).tolist()

    return {
        "hourly": hourly,
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": "degC",
            "relative_humidity_2m": "%",
            "dew_point_2m": "degC",
            "apparent_temperature": "degC",
            "surface_pressure": "hPa",
            "et0_fao_evapotranspiration": "mm",
            "vapour_pressure_deficit": "kPa",
            "wet_bulb_temperature_2m": "degC",
            "wind_speed_10m": "km/h",
            "shortwave_radiation": "W/m2",
        },
    }


class WeatherDataTestCase(unittest.TestCase):
    def test_fetch_builds_open_meteo_request_with_expected_variables_and_timeout(self):
        session = FakeSession([FakeResponse(200, _weather_payload())])

        weather_data = fetch_open_meteo_weather(
            39.7392,
            -104.9903,
            year=2024,
            timeout_seconds=12.5,
            session=session,
            sleep=lambda _: None,
        )

        self.assertEqual(1, len(session.calls))
        call = session.calls[0]
        self.assertEqual(12.5, call["timeout"])
        self.assertEqual(39.7392, call["params"]["latitude"])
        self.assertEqual(-104.9903, call["params"]["longitude"])
        self.assertEqual("2024-01-01", call["params"]["start_date"])
        self.assertEqual("2024-12-31", call["params"]["end_date"])
        self.assertEqual(",".join(DEFAULT_HOURLY_VARIABLES), call["params"]["hourly"])
        self.assertEqual("auto", call["params"]["timezone"])
        self.assertEqual(EXPECTED_HOURLY_ROWS, len(weather_data.hourly()))
        self.assertEqual("degC", weather_data.hourly_units["temperature_2m"])

    def test_cache_hit_skips_open_meteo_request(self):
        with TemporaryDirectory() as cache_dir:
            first_session = FakeSession([FakeResponse(200, _weather_payload())])
            first_weather_data = fetch_open_meteo_weather(
                39.7392,
                -104.9903,
                year=2024,
                session=first_session,
                sleep=lambda _: None,
                cache_dir=cache_dir,
            )
            second_session = FakeSession([])

            second_weather_data = fetch_open_meteo_weather(
                39.7392,
                -104.9903,
                year=2024,
                session=second_session,
                sleep=lambda _: None,
                cache_dir=cache_dir,
            )

        self.assertEqual(1, len(first_session.calls))
        self.assertEqual(0, len(second_session.calls))
        self.assertEqual(first_weather_data.hourly_units, second_weather_data.hourly_units)
        self.assertEqual(EXPECTED_HOURLY_ROWS, len(second_weather_data.hourly()))
        self.assertEqual(0.0, second_weather_data.hourly()["temperature_2m"].iloc[0])

    def test_cache_key_rounds_location_to_six_decimal_places(self):
        with TemporaryDirectory() as cache_dir:
            first_session = FakeSession([FakeResponse(200, _weather_payload())])
            fetch_open_meteo_weather(
                39.7392001,
                -104.9903001,
                year=2024,
                session=first_session,
                sleep=lambda _: None,
                cache_dir=cache_dir,
            )
            second_session = FakeSession([])

            fetch_open_meteo_weather(
                39.7392002,
                -104.9903002,
                year=2024,
                session=second_session,
                sleep=lambda _: None,
                cache_dir=cache_dir,
            )

        self.assertEqual(1, len(first_session.calls))
        self.assertEqual(0, len(second_session.calls))

    def test_unreadable_cache_file_warns_and_refetches(self):
        with TemporaryDirectory() as cache_dir:
            first_session = FakeSession([FakeResponse(200, _weather_payload())])
            fetch_open_meteo_weather(
                39.7392,
                -104.9903,
                year=2024,
                session=first_session,
                sleep=lambda _: None,
                cache_dir=cache_dir,
            )
            for cache_file in Path(cache_dir).glob("*.json"):
                cache_file.write_text("not json", encoding="utf-8")

            second_session = FakeSession([FakeResponse(200, _weather_payload())])
            with self.assertWarnsRegex(RuntimeWarning, "Ignoring unreadable weather cache file"):
                fetch_open_meteo_weather(
                    39.7392,
                    -104.9903,
                    year=2024,
                    session=second_session,
                    sleep=lambda _: None,
                    cache_dir=cache_dir,
                )

        self.assertEqual(1, len(second_session.calls))

    def test_default_cache_directory_is_weather_data_cache(self):
        original_cwd = Path.cwd()
        with TemporaryDirectory() as run_dir:
            os.chdir(run_dir)
            try:
                session = FakeSession([FakeResponse(200, _weather_payload())])
                fetch_open_meteo_weather(
                    39.7392,
                    -104.9903,
                    year=2024,
                    session=session,
                    sleep=lambda _: None,
                    use_cache=True,
                )
            finally:
                os.chdir(original_cwd)

            cache_files = list((Path(run_dir) / "weather_data_cache").glob("*.json"))

        self.assertEqual(1, len(session.calls))
        self.assertEqual(1, len(cache_files))

    def test_missing_required_variable_raises(self):
        payload = _weather_payload()
        del payload["hourly"]["temperature_2m"]
        session = FakeSession([FakeResponse(200, payload)])

        with self.assertRaisesRegex(OpenMeteoWeatherError, "temperature_2m"):
            fetch_open_meteo_weather(39.0, -105.0, year=2024, session=session, sleep=lambda _: None)

    def test_missing_optional_variable_warns_and_continues(self):
        session = FakeSession([FakeResponse(200, _weather_payload(include_optional=False))])

        with self.assertWarnsRegex(RuntimeWarning, "optional hourly weather variables"):
            weather_data = fetch_open_meteo_weather(39.0, -105.0, year=2024, session=session, sleep=lambda _: None)

        self.assertEqual(EXPECTED_HOURLY_ROWS, len(weather_data.hourly()))
        self.assertNotIn("wet_bulb_temperature_2m", weather_data.hourly().columns)

    def test_optional_variable_rejection_retries_with_required_variables_only(self):
        session = FakeSession(
            [
                FakeResponse(
                    400,
                    {
                        "reason": "Variable wet_bulb_temperature_2m is not available",
                    },
                ),
                FakeResponse(200, _weather_payload(include_optional=False)),
            ]
        )

        with self.assertWarnsRegex(RuntimeWarning, "optional weather variables"):
            fetch_open_meteo_weather(39.0, -105.0, year=2024, session=session, sleep=lambda _: None)

        self.assertEqual(2, len(session.calls))
        self.assertEqual(",".join(DEFAULT_HOURLY_VARIABLES), session.calls[0]["params"]["hourly"])
        self.assertEqual(",".join(REQUIRED_HOURLY_VARIABLES), session.calls[1]["params"]["hourly"])

    def test_transient_errors_are_retried(self):
        session = FakeSession(
            [
                requests.exceptions.Timeout("timed out"),
                FakeResponse(503, {"reason": "temporarily unavailable"}),
                FakeResponse(200, _weather_payload()),
            ]
        )
        sleeps = []

        fetch_open_meteo_weather(
            39.0,
            -105.0,
            year=2024,
            max_retries=2,
            backoff_seconds=0.5,
            session=session,
            sleep=sleeps.append,
        )

        self.assertEqual(3, len(session.calls))
        self.assertEqual([0.5, 1.0], sleeps)

    def test_request_failures_raise_network_error_after_retries(self):
        session = FakeSession([requests.exceptions.ConnectionError("offline")])

        with self.assertRaises(OpenMeteoNetworkError):
            fetch_open_meteo_weather(39.0, -105.0, year=2024, max_retries=0, session=session, sleep=lambda _: None)

    def test_transient_failures_raise_network_error_after_retries(self):
        session = FakeSession([FakeResponse(503, {"reason": "temporarily unavailable"})])

        with self.assertRaises(OpenMeteoNetworkError):
            fetch_open_meteo_weather(39.0, -105.0, year=2024, max_retries=0, session=session, sleep=lambda _: None)

    def test_permanent_errors_are_not_retried(self):
        session = FakeSession([FakeResponse(400, {"reason": "Latitude must be in range"})])

        with self.assertRaisesRegex(OpenMeteoWeatherError, "HTTP 400"):
            fetch_open_meteo_weather(39.0, -105.0, year=2024, session=session, sleep=lambda _: None)

        self.assertEqual(1, len(session.calls))

    def test_invalid_inputs_raise_before_request(self):
        session = FakeSession([FakeResponse(200, _weather_payload())])

        with self.assertRaisesRegex(OpenMeteoWeatherError, "Project Latitude"):
            fetch_open_meteo_weather(91.0, -105.0, year=2024, session=session, sleep=lambda _: None)
        with self.assertRaisesRegex(OpenMeteoWeatherError, "Project Longitude"):
            fetch_open_meteo_weather(39.0, -181.0, year=2024, session=session, sleep=lambda _: None)
        with self.assertRaisesRegex(OpenMeteoWeatherError, "Open-Meteo historical archive range"):
            fetch_open_meteo_weather(39.0, -105.0, year=9999, session=session, sleep=lambda _: None)
        with self.assertRaisesRegex(OpenMeteoWeatherError, "Open-Meteo historical archive range"):
            fetch_open_meteo_weather(
                39.0,
                -105.0,
                year=OPEN_METEO_HISTORICAL_START_YEAR - 1,
                session=session,
                sleep=lambda _: None,
            )

        self.assertEqual(0, len(session.calls))

    def test_latest_complete_historical_year_accounts_for_archive_delay(self):
        self.assertEqual(2024, _latest_open_meteo_complete_historical_year(today=date(2025, 12, 31)))
        self.assertEqual(2025, _latest_open_meteo_complete_historical_year(today=date(2026, 1, 6)))

    def test_leap_year_response_is_normalized_to_8760_rows(self):
        session = FakeSession([FakeResponse(200, _weather_payload(row_count=LEAP_YEAR_HOURLY_ROWS))])

        weather_data = fetch_open_meteo_weather(39.0, -105.0, year=2024, session=session, sleep=lambda _: None)

        hourly = weather_data.hourly()
        self.assertEqual(EXPECTED_HOURLY_ROWS, len(hourly))
        self.assertEqual(0.0, hourly["temperature_2m"].iloc[0])
        self.assertEqual(float(LEAP_YEAR_HOURLY_ROWS - 1), hourly["temperature_2m"].iloc[-1])

    def test_aggregation_helpers_return_expected_means(self):
        session = FakeSession([FakeResponse(200, _weather_payload())])
        weather_data = fetch_open_meteo_weather(39.0, -105.0, year=2023, session=session, sleep=lambda _: None)

        daily = weather_data.daily_average()
        weekly = weather_data.weekly_average()
        monthly = weather_data.monthly_average()
        annual = weather_data.annual_average()

        self.assertEqual(365, len(daily))
        self.assertEqual(53, len(weekly))
        self.assertEqual(12, len(monthly))
        self.assertEqual(11.5, daily["temperature_2m"].iloc[0])
        self.assertEqual((EXPECTED_HOURLY_ROWS - 1) / 2.0, annual["temperature_2m"])


if __name__ == "__main__":
    unittest.main()
