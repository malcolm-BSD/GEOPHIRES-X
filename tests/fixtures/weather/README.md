# Weather regression snapshots

These files contain frozen Open-Meteo Historical Weather API
responses retrieved on October 5, 2026. Weather data by
[Open-Meteo.com](https://open-meteo.com/), distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
See the [historical API documentation](https://open-meteo.com/en/docs/historical-weather-api)
for the underlying reanalysis datasets and attribution.

Request endpoint: `https://archive-api.open-meteo.com/v1/archive`

| Snapshot | Examples | Requested latitude | Requested longitude | Normalized annual mean (C) |
| --- | --- | --- | --- | --- |
| `denver-2024-open-meteo.json` | weather/TESS | 39.7392 | -104.9903 | 10.566158100396148 |
| `ithaca-2024-open-meteo.json` | 8 and 9 | 42.441977082209235 | -76.43242140337269 | 9.789750198230546 |
| `newberry-2024-open-meteo.json` | SHR-3 | 43.812139 | -121.281671 | 5.735162799281214 |

- Date range: `2024-01-01` through `2024-12-31`; timezone: `auto`.
- Hourly variables: `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`,
  `apparent_temperature`, `surface_pressure`, `et0_fao_evapotranspiration`,
  `vapour_pressure_deficit` (the application's required variables).
- API-selected grid coordinates are retained in each response.
- The response values are unchanged; JSON was serialized with compact separators
  and a final newline.
SHA-256 checksums:

- Denver: `041ebff708e2a29df6f7b0deaa706af5a95b2cab772521ed2bdfdb63998aa4bf`
- Ithaca: `0b5002e49cc0318319b51cf7e38e4e13e80e61ddf35850a2b9e930d157edf9c5`
- Newberry: `3c32d12c5582e23589d9c18fdf53d471df7bb334188b8ecd375de5a0ec19c3aa`

Each snapshot contains 8,784 leap-year hours. The regressions use the production
normalizer to obtain 8,760 hours.
Missing or invalid fixture data must fail the regression, never fetch replacement
weather. The examples retain their weather inputs; production API and cache behavior
are unchanged.

The original weather/TESS output used a 10.55 C annual mean and predates current location
reporting and annual-profile behavior. The pre-October-5 combined baseline
(tree `ec3923dff2511a28878989751e14333863680023`) and the reconciled candidate
(tree `19bf5a2deed3c541e849acc361dd1e85e7c01e7e`) produced exactly equal complete
parsed outputs for all four examples with these snapshots, after stripping simulation metadata and
normalizing NaN as the regression harness already does. The four references were updated
only after these comparisons. No numerical tolerance was changed.

From the repository root, regenerate a weather example's reference with the frozen
input using:

```shell
python -m tests.example_weather example1_dispatchable_tess_weather.txt
```

Review both the weather input and resulting output if replacing this snapshot.
The generic example regeneration scripts use live weather and must not be used
to update these regression references.
