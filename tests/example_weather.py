"""Frozen weather input for the weather/TESS example regression and regeneration."""

import json
import shutil
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from geophires_x.WeatherData import WeatherData
from geophires_x.WeatherData import _weather_data_from_response

WEATHER_EXAMPLES = {
    "example1_dispatchable_tess_weather.txt": ("denver", 39.7392, -104.9903),
    "example8.txt": ("ithaca", 42.441977082209235, -76.43242140337269),
    "example9.txt": ("ithaca", 42.441977082209235, -76.43242140337269),
    "example_SHR-3.txt": ("newberry", 43.812139, -121.281671),
}
WEATHER_FIXTURE_DIR = Path(__file__).parent / "fixtures/weather"


@contextmanager
def frozen_example_weather(example_name):
    """Keep this regression independent of live archive revisions and local caches."""
    if example_name not in WEATHER_EXAMPLES:
        yield
        return

    # Read errors must fail the test; never silently fall back to the network.
    location, expected_latitude, expected_longitude = WEATHER_EXAMPLES[example_name]
    fixture_path = WEATHER_FIXTURE_DIR / f"{location}-2024-open-meteo.json"
    response = json.loads(fixture_path.read_text(encoding="utf-8"))
    # The snapshot intentionally contains only the API's required variables.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Open-Meteo response is missing optional hourly weather variables:",
            category=RuntimeWarning,
        )
        hourly, units = _weather_data_from_response(response, 2024)

    def load_weather(latitude, longitude, year):
        if (latitude, longitude, year) != (expected_latitude, expected_longitude, 2024):
            raise AssertionError("The frozen weather does not match the example location/year.")
        return WeatherData(latitude, longitude, year, hourly.copy(), units.copy())

    with patch("geophires_x.Model.fetch_open_meteo_weather", side_effect=load_weather) as fetch:
        yield
        fetch.assert_called_once_with(expected_latitude, expected_longitude, year=2024)


if __name__ == "__main__":
    from geophires_x_client import GeophiresInputParameters
    from geophires_x_client import GeophiresXClient

    example_name = sys.argv[1] if len(sys.argv) == 2 else ""
    if example_name not in WEATHER_EXAMPLES:
        raise SystemExit("Usage: python -m tests.example_weather <example filename.txt>")
    sys.argv = [sys.argv[0]]
    examples_dir = Path(__file__).resolve().parent / "examples"
    with frozen_example_weather(example_name):
        result = GeophiresXClient().get_geophires_result(
            GeophiresInputParameters(from_file_path=examples_dir / example_name)
        )
    shutil.copyfile(result.output_file_path, examples_dir / Path(example_name).with_suffix(".out"))
