# Weather and Climate Data Operating Guide

## Purpose

GEOPHIRES-X can optionally use project-location historical weather data from
the Open-Meteo Historical Weather API. When project coordinates are provided,
GEOPHIRES-X downloads one year of hourly weather data, normalizes it to the
standard 8760-hour annual profile, and uses it where ambient temperature affects
model calculations.

The feature is inactive by default. Existing input files and calculations do
not change unless both project coordinates are provided.

## Required Inputs

Weather data is activated only when both coordinate parameters are supplied:

```text
Project Latitude
Project Longitude
```

`Project Latitude` is an optional float from `-90` to `90`.
`Project Longitude` is an optional float from `-180` to `180`.

If only one coordinate is supplied, GEOPHIRES-X raises a validation error.

## Optional Inputs

The weather year is controlled by:

```text
Weather Data Year
```

The default is:

```text
2024
```

`Weather Data Year` must be within Open-Meteo's known historical archive range:

```text
1940 through the latest complete Open-Meteo historical archive year
```

GEOPHIRES-X accounts for the Open-Meteo archive delay by accepting only complete
historical years whose December 31 data is expected to be available.

## Activation Rules

```text
If Project Latitude and Project Longitude are provided:
    download weather data for Weather Data Year, defaulting to 2024
else:
    do not download weather data
```

No network request is made unless both coordinates are present.

## Open-Meteo Data Source

GEOPHIRES-X uses the Open-Meteo Historical Weather API:

```text
https://archive-api.open-meteo.com/v1/archive
```

Open-Meteo documentation:

```text
https://open-meteo.com/en/docs/historical-weather-api
```

Requested hourly variables:

```text
temperature_2m
relative_humidity_2m
dew_point_2m
apparent_temperature
surface_pressure
et0_fao_evapotranspiration
vapour_pressure_deficit
wet_bulb_temperature_2m
wind_speed_10m
shortwave_radiation
```

The first seven variables are required. The last three are optional; if
Open-Meteo does not return them, GEOPHIRES-X continues with the required data
and emits a warning.

## Units

GEOPHIRES-X stores Open-Meteo's returned `hourly_units` metadata with the
downloaded weather data. Expected units are:

```text
temperature_2m: degC
relative_humidity_2m: %
dew_point_2m: degC
apparent_temperature: degC
surface_pressure: hPa
et0_fao_evapotranspiration: mm
vapour_pressure_deficit: kPa
wet_bulb_temperature_2m: degC
wind_speed_10m: km/h or m/s depending API unit selection
shortwave_radiation: W/m2
```

## Network Behavior

The weather client uses robust default Open-Meteo access behavior:

```text
Timeout: 30 seconds
Retries: 3
Backoff: exponential, approximately 1s, 2s, 4s
```

GEOPHIRES-X retries transient failures:

```text
HTTP 429
HTTP 500
HTTP 502
HTTP 503
HTTP 504
connection errors
timeouts
```

GEOPHIRES-X does not retry permanent request errors such as invalid
coordinates, invalid years, malformed requests, or unsupported required data.

## Offline Fallback

If both project coordinates are provided but Open-Meteo cannot be reached
because the machine has no network access, the request times out, or the
service returns only transient failures, GEOPHIRES-X continues as if project
coordinates had not been provided.

In that fallback case:

- `model.weather_data` remains `None`;
- missing `Ambient Temperature` and `Surface Temperature` keep their normal
  GEOPHIRES-X defaults;
- Dispatchable, TESS, and Baseload calculations use existing scalar
  temperature behavior;
- the run logs a warning explaining that weather data was requested but was
  unavailable.

The offline fallback applies only to connectivity and transient service
failures. Invalid coordinates, invalid years, missing required variables, and
malformed API responses still raise errors.

## Hourly Normalization

GEOPHIRES-X uses a normalized 8760-hour annual convention. Weather data follows
that same convention.

Rules:

```text
Non-leap year response with 8760 rows:
    use directly

Leap year response with 8784 rows:
    resample to 8760 rows

Any other row count:
    raise a validation error
```

Leap years are not modeled explicitly. If a leap year is selected, the
downloaded data is numerically resampled to 8760 rows before it is used.

## Weather Data Access

When weather data is active and successfully downloaded, it is stored as:

```python
model.weather_data
```

When weather data is inactive or unavailable through the offline fallback:

```python
model.weather_data = None
```

The weather data object exposes:

```python
hourly() -> pandas.DataFrame
daily_average() -> pandas.DataFrame
weekly_average() -> pandas.DataFrame
monthly_average() -> pandas.DataFrame
annual_average() -> pandas.Series
```

Aggregation behavior:

- `hourly`: normalized 8760-row hourly data.
- `daily_average`: calendar-day mean after hourly normalization.
- `weekly_average`: ISO week mean.
- `monthly_average`: calendar-month mean.
- `annual_average`: mean across all 8760 hourly records.

`et0_fao_evapotranspiration` is averaged in these functions. ET0 sum outputs
are not provided.

## Ambient and Surface Temperature Defaults

When weather data is active, GEOPHIRES-X uses annual average `temperature_2m`
as the default for missing ambient and surface temperatures:

```text
If Ambient Temperature was not supplied:
    Ambient Temperature = annual average temperature_2m

If Surface Temperature was not supplied:
    Surface Temperature = annual average temperature_2m
```

User-provided `Ambient Temperature` and `Surface Temperature` values are never
overwritten.

## Dispatchable Mode Behavior

Dispatchable mode uses hourly `temperature_2m` where ambient temperature affects
plant physics:

```text
temperature_2m[timestep_index % 8760]
```

Weather-aware dispatch affects ORC and flash electricity and CHP calculations,
including:

- gross electricity production;
- net electricity production;
- reinjection temperature;
- geofluid availability;
- first-law efficiency;
- CHP heat and electricity split behavior;
- dispatch capacity factor;
- peak geothermal contribution;
- peak unmet load.

Dispatch control calculations also use the timestep-specific weather-aware
plant output where applicable.

Dispatch results store the hourly ambient profile in:

```python
hourly_ambient_temperature
```

That column is included in dispatch profile CSV output when dispatch profile
output is requested.

If no weather data is available, Dispatchable behavior remains unchanged.

## TESS Behavior

TESS operates only in Dispatchable mode, so TESS weather behavior follows the
Dispatchable weather behavior.

TESS does not currently use weather data to calculate storage heat losses.
`TESS Daily Heat Loss Fraction` remains the active loss model.

Weather affects TESS indirectly where weather affects the plant output used to
charge storage:

- TESS electric cases inherit hourly ambient effects through weather-aware
  electric plant output.
- TESS CHP cases inherit hourly ambient effects through weather-aware CHP plant
  output.
- TESS direct-heat, heat-pump, absorption-chiller, and district-heating cases
  remain unchanged unless their dispatch plant pathway is explicitly
  weather-sensitive.

TESS demand-to-storage conversion and moving-average charge control use
timestep-specific weather-aware plant output when applicable.

## Baseload Mode Behavior

Baseload mode uses weather data selectively.

Baseload support includes:

- annual average `temperature_2m` as the default for missing `Ambient
  Temperature`;
- annual average `temperature_2m` as the default for missing `Surface
  Temperature`;
- hourly `temperature_2m` for baseload electricity and CHP surface plant
  calculations when the plant type is ORC or flash.

For baseload time series longer than one year, GEOPHIRES-X tiles the normalized
weather year:

```text
ambient_temperature_for_timestep = temperature_2m[timestep_index % 8760]
```

Pure direct-use heat, heat pump, absorption chiller, and district heating
baseload modes do not change physical calculations beyond missing
ambient/surface-temperature defaults.

If no weather data is available, Baseload behavior remains unchanged.

## Output Reports

When weather data is active and successfully downloaded, output reports include:

```text
***WEATHER DATA RESULTS***
Weather data source
Weather data year
Project latitude
Project longitude
Annual average temperature (from Open-Meteo)
Minimum hourly temperature (from Open-Meteo)
Maximum hourly temperature (from Open-Meteo)
```

This section is omitted when weather data is inactive or when the offline
fallback continues without downloaded data.

The weather profile is not written as a CSV artifact and is not included in
JSON output. The text report summary is the intended audit surface.

## Backward Compatibility

Backward compatibility requirements:

- no network request unless both project coordinates are provided;
- existing input files without project coordinates produce identical results;
- user-provided `Ambient Temperature` and `Surface Temperature` are never
  overwritten;
- weather data is normalized to the existing 8760-hour convention;
- unavailable optional weather variables do not block a run;
- required weather data validation errors fail early with clear errors.

## Future Weather-Sensitive Models

The additional downloaded variables support future improvements:

- `wet_bulb_temperature_2m`: cooling tower, evaporative cooling, and condenser
  performance.
- `relative_humidity_2m`, `dew_point_2m`, `surface_pressure`, and
  `vapour_pressure_deficit`: psychrometric calculations.
- `wind_speed_10m`: convection-sensitive tank losses and air-cooled equipment.
- `shortwave_radiation`: solar gains on above-ground storage or equipment.
- `et0_fao_evapotranspiration`: water-use and cooling makeup studies.

Potential future model enhancements include:

- optional weather-derived district heating demand source, without replacing
  the existing temperature-file workflow;
- weather-derived cooling demand;
- temperature-dependent heat pump COP;
- wet-bulb-dependent absorption chiller or cooling tower performance;
- ambient-dependent TESS standby loss;
- water consumption or makeup-water reporting.

## Testing Expectations

Weather tests use mocked HTTP responses and do not depend on live Open-Meteo
service availability.

Maintained test coverage includes:

- weather input validation and default year behavior;
- Open-Meteo request construction, timeouts, retries, and transient failure
  fallback;
- required and optional weather variable handling;
- leap-year normalization to 8760 hours;
- hourly, daily, ISO-weekly, monthly, and annual aggregation;
- ambient and surface temperature default integration;
- Dispatchable weather-aware electricity behavior;
- Dispatchable weather-aware control-state behavior;
- TESS weather-aware electric dispatch behavior;
- Baseload ORC/flash weather-aware electricity behavior;
- report parsing for `***WEATHER DATA RESULTS***`.
