"""The weather regression must neither fetch live data nor accept another location."""

from contextlib import ExitStack
from unittest.mock import patch

import pytest

from geophires_x import Model
from tests.example_weather import frozen_example_weather

WEATHER_EXAMPLE = "example1_dispatchable_tess_weather.txt"


def test_frozen_weather_is_offline_and_normalizes_leap_year():
    with ExitStack() as stack:
        request = stack.enter_context(patch("requests.sessions.Session.request", side_effect=AssertionError("network")))
        stack.enter_context(frozen_example_weather(WEATHER_EXAMPLE))
        weather = Model.fetch_open_meteo_weather(39.7392, -104.9903, year=2024)
        assert len(weather.hourly()) == 8760
        assert weather.annual_average()["temperature_2m"] == pytest.approx(10.566158100396148)
        assert weather.hourly()["temperature_2m"].min() != weather.hourly()["temperature_2m"].max()
        request.assert_not_called()


def test_frozen_weather_rejects_different_location():
    with pytest.raises(AssertionError, match="does not match"):
        with frozen_example_weather(WEATHER_EXAMPLE):
            Model.fetch_open_meteo_weather(0.0, 0.0, year=2024)


def test_missing_weather_fixture_fails_without_network(tmp_path):
    with ExitStack() as stack:
        request = stack.enter_context(patch("requests.sessions.Session.request", side_effect=AssertionError("network")))
        stack.enter_context(patch("tests.example_weather.WEATHER_FIXTURE_DIR", tmp_path))
        with pytest.raises(FileNotFoundError):
            with frozen_example_weather(WEATHER_EXAMPLE):
                pytest.fail("Missing weather fixture was accepted")
        request.assert_not_called()
