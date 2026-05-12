import os
import sys
from pathlib import Path
from unittest.mock import patch

from geophires_x.Model import Model
from geophires_x.Parameter import ParameterEntry
from geophires_x.WeatherData import OpenMeteoNetworkError
from tests.base_test_case import BaseTestCase


class FakeWeatherData:
    def __init__(self, annual_average_temperature: float):
        self.annual_average_temperature = annual_average_temperature

    def annual_average(self):
        return {"temperature_2m": self.annual_average_temperature}


class WeatherParametersTestCase(BaseTestCase):
    def _new_model(self) -> Model:
        stash_cwd = Path.cwd()
        stash_sys_argv = sys.argv
        sys.argv = [""]
        model = Model(enable_geophires_logging_config=False)
        sys.argv = stash_sys_argv
        os.chdir(stash_cwd)
        return model

    def test_weather_parameters_default_to_inactive_coordinates_and_2024_year(self) -> None:
        model = self._new_model()

        model.surfaceplant.read_parameters(model)

        self.assertEqual(0.0, model.surfaceplant.project_latitude.value)
        self.assertEqual(0.0, model.surfaceplant.project_longitude.value)
        self.assertEqual(2024, model.surfaceplant.weather_data_year.value)
        self.assertFalse(model.surfaceplant.project_latitude.Provided)
        self.assertFalse(model.surfaceplant.project_longitude.Provided)
        self.assertFalse(model.surfaceplant.weather_data_year.Provided)

    def test_weather_parameters_are_read_from_input(self) -> None:
        model = self._new_model()
        model.InputParameters = {
            "Project Latitude": ParameterEntry(Name="Project Latitude", sValue="39.7392"),
            "Project Longitude": ParameterEntry(Name="Project Longitude", sValue="-104.9903"),
            "Weather Data Year": ParameterEntry(Name="Weather Data Year", sValue="2025"),
        }

        model.surfaceplant.read_parameters(model)

        self.assertEqual(39.7392, model.surfaceplant.project_latitude.value)
        self.assertEqual(-104.9903, model.surfaceplant.project_longitude.value)
        self.assertEqual(2025, model.surfaceplant.weather_data_year.value)
        self.assertTrue(model.surfaceplant.project_latitude.Provided)
        self.assertTrue(model.surfaceplant.project_longitude.Provided)
        self.assertTrue(model.surfaceplant.weather_data_year.Provided)

    def test_project_coordinates_fetch_weather_with_default_year_and_fill_missing_temperatures(self) -> None:
        model = self._new_model()
        weather_data = FakeWeatherData(annual_average_temperature=12.5)
        model.InputParameters = {
            "Project Latitude": ParameterEntry(Name="Project Latitude", sValue="39.7392"),
            "Project Longitude": ParameterEntry(Name="Project Longitude", sValue="-104.9903"),
        }

        with patch("geophires_x.Model.fetch_open_meteo_weather", return_value=weather_data) as fetch_weather:
            model.read_parameters()

        fetch_weather.assert_called_once_with(39.7392, -104.9903, year=2024)
        self.assertIs(weather_data, model.weather_data)
        self.assertEqual(12.5, model.surfaceplant.ambient_temperature.value)
        self.assertEqual(12.5, model.reserv.Tsurf.value)

    def test_project_coordinates_fetch_weather_with_user_year(self) -> None:
        model = self._new_model()
        model.InputParameters = {
            "Project Latitude": ParameterEntry(Name="Project Latitude", sValue="39.7392"),
            "Project Longitude": ParameterEntry(Name="Project Longitude", sValue="-104.9903"),
            "Weather Data Year": ParameterEntry(Name="Weather Data Year", sValue="2023"),
        }

        with patch(
            "geophires_x.Model.fetch_open_meteo_weather",
            return_value=FakeWeatherData(annual_average_temperature=12.5),
        ) as fetch_weather:
            model.read_parameters()

        fetch_weather.assert_called_once_with(39.7392, -104.9903, year=2023)

    def test_weather_does_not_overwrite_user_provided_ambient_or_surface_temperature(self) -> None:
        model = self._new_model()
        model.InputParameters = {
            "Project Latitude": ParameterEntry(Name="Project Latitude", sValue="39.7392"),
            "Project Longitude": ParameterEntry(Name="Project Longitude", sValue="-104.9903"),
            "Ambient Temperature": ParameterEntry(Name="Ambient Temperature", sValue="20"),
            "Surface Temperature": ParameterEntry(Name="Surface Temperature", sValue="18"),
        }

        with patch(
            "geophires_x.Model.fetch_open_meteo_weather",
            return_value=FakeWeatherData(annual_average_temperature=12.5),
        ):
            model.read_parameters()

        self.assertEqual(20.0, model.surfaceplant.ambient_temperature.value)
        self.assertEqual(18.0, model.reserv.Tsurf.value)

    def test_weather_fetch_requires_both_project_coordinates(self) -> None:
        model = self._new_model()
        model.InputParameters = {
            "Project Latitude": ParameterEntry(Name="Project Latitude", sValue="39.7392"),
        }

        with patch("geophires_x.Model.fetch_open_meteo_weather") as fetch_weather:
            with self.assertRaisesRegex(ValueError, "Project Latitude and Project Longitude"):
                model.read_parameters()

        fetch_weather.assert_not_called()

    def test_weather_is_not_fetched_without_project_coordinates(self) -> None:
        model = self._new_model()

        with patch("geophires_x.Model.fetch_open_meteo_weather") as fetch_weather:
            model.read_parameters()

        fetch_weather.assert_not_called()
        self.assertIsNone(model.weather_data)

    def test_weather_fetch_network_failure_continues_without_weather_data(self) -> None:
        model = self._new_model()
        model.InputParameters = {
            "Project Latitude": ParameterEntry(Name="Project Latitude", sValue="39.7392"),
            "Project Longitude": ParameterEntry(Name="Project Longitude", sValue="-104.9903"),
        }
        default_ambient_temperature = model.surfaceplant.ambient_temperature.value
        default_surface_temperature = model.reserv.Tsurf.value

        with patch(
            "geophires_x.Model.fetch_open_meteo_weather",
            side_effect=OpenMeteoNetworkError("Open-Meteo request failed after 4 attempts"),
        ):
            model.read_parameters()

        self.assertIsNone(model.weather_data)
        self.assertEqual(default_ambient_temperature, model.surfaceplant.ambient_temperature.value)
        self.assertEqual(default_surface_temperature, model.reserv.Tsurf.value)
