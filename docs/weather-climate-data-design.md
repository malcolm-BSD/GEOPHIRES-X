# Weather and Climate Data Design

## Purpose

Add optional project-location weather data to GEOPHIRES-X. When a user provides
project coordinates, GEOPHIRES-X will download one year of hourly historical
weather data from the Open-Meteo Historical Weather API, expose that data at
hourly and aggregated resolutions, and use it to improve ambient-temperature
dependent calculations.

The feature is inactive by default. Existing input files and calculations do not
change unless the user provides project coordinates.

## User Inputs

The weather data feature is activated by:

```text
Project Latitude
Project Longitude
```

Both coordinates are required to activate weather download. If only one is
provided, GEOPHIRES-X should raise a validation error.

The weather year is optional:

```text
Weather Data Year
```

Its default is:

```text
2024
```

`Weather Data Year` is validated against Open-Meteo's known historical archive
range instead of waiting for the API response to determine validity. Open-Meteo
documents historical access from 1940 onward. Because some archive products are
published with a delay, GEOPHIRES-X only accepts complete historical years whose
December 31 data should already be available after the archive delay. At
runtime, the accepted range is:

```text
1940 through the latest complete Open-Meteo historical archive year
```

The full activation rule is:

```text
If Project Latitude and Project Longitude are provided:
    download weather data for Weather Data Year, defaulting to 2024
else:
    do not download weather data
```

## Open-Meteo Data Source

Use the Open-Meteo Historical Weather API:

```text
https://archive-api.open-meteo.com/v1/archive
```

Open-Meteo Historical Weather API documentation:

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

The first seven variables are required by the feature request. The last three
are included because they are useful for future cooling, condenser,
evaporative-cooling, and ambient-dependent storage-loss models.

Example request shape:

```text
https://archive-api.open-meteo.com/v1/archive
    ?latitude=<Project Latitude>
    &longitude=<Project Longitude>
    &start_date=<Weather Data Year>-01-01
    &end_date=<Weather Data Year>-12-31
    &hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,surface_pressure,et0_fao_evapotranspiration,vapour_pressure_deficit,wet_bulb_temperature_2m,wind_speed_10m,shortwave_radiation
    &timezone=auto
```

## Units

Use Open-Meteo's returned `hourly_units` metadata as the source of truth and
store it with the downloaded data.

Expected units are:

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

The implementation should request metric defaults and avoid changing units in
the API request unless there is a clear GEOPHIRES-X unit convention to enforce.
If units are converted internally, conversion should happen through the existing
unit utilities rather than ad hoc conversion math.

## API Robustness

The Open-Meteo access layer should be isolated in a new module so network
behavior is easy to test and maintain:

```text
src/geophires_x/WeatherData.py
```

Recommended request defaults:

```text
Connect/read timeout: 30 seconds
Retries: 3
Backoff: exponential, approximately 1s, 2s, 4s
```

Retry only transient failures:

```text
HTTP 429
HTTP 500
HTTP 502
HTTP 503
HTTP 504
connection errors
timeouts
```

Do not retry permanent request errors, including invalid coordinates, invalid
dates, malformed requests, or unsupported variables.

If the optional variables `wet_bulb_temperature_2m`, `wind_speed_10m`, or
`shortwave_radiation` are unavailable from the selected Open-Meteo response,
continue with the available required variables and emit a warning. If any of
the required seven variables are missing, raise a validation error.

## Data Model

Add a small weather data object:

```python
@dataclass
class WeatherData:
    latitude: float
    longitude: float
    year: int
    hourly_data: pandas.DataFrame
    hourly_units: dict[str, str]
```

Canonical hourly columns:

```text
time
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

Optional variables may be absent if Open-Meteo does not return them. Required
variables must always be present after validation.

The model should store the fetched data as:

```python
model.weather_data
```

When no project coordinates are provided:

```python
model.weather_data = None
```

## Hourly Normalization

GEOPHIRES-X dispatch and annual profile machinery assumes one normalized
8760-hour year. Weather data must follow the same convention.

The weather data object must always expose 8760 hourly rows.

Rules:

```text
Non-leap year response with 8760 rows:
    use directly

Leap year response with 8784 rows:
    resample to 8760 rows

Any other row count:
    raise a validation error unless it can be safely resampled to 8760
```

Leap years are not modeled explicitly. If the user selects a leap year, the
downloaded data is normalized to 8760 rows before it is used anywhere else.

The design preference is to resample numerically over normalized year fraction
rather than simply dropping February 29. That avoids a discontinuity and keeps
the full climate signal represented in the 8760-hour profile.

## Aggregation Functions

The weather data object should provide:

```python
hourly() -> pandas.DataFrame
daily_average() -> pandas.DataFrame
weekly_average() -> pandas.DataFrame
monthly_average() -> pandas.DataFrame
annual_average() -> pandas.Series
```

Aggregation rules:

- `hourly`: normalized 8760-row hourly data.
- `daily_average`: calendar-day mean after hourly normalization.
- `weekly_average`: ISO week mean.
- `monthly_average`: calendar-month mean.
- `annual_average`: mean across all 8760 hourly records.

The requested behavior is average aggregation. Therefore
`et0_fao_evapotranspiration` should be averaged in these functions even though
Open-Meteo describes it as a preceding-hour quantity. ET0 sum functions are not
part of this effort; the weather aggregation API exposes averages only.

## Parameter Default Integration

When weather data is active, use annual average `temperature_2m` as the default
for missing ambient and surface temperatures.

Rules:

```text
If weather data is active and Ambient Temperature was not supplied:
    Ambient Temperature = annual average temperature_2m

If weather data is active and Surface Temperature was not supplied:
    Surface Temperature = annual average temperature_2m
```

Do not overwrite user-provided values. Use existing parameter `Provided` flags
or direct membership in `model.InputParameters` to distinguish explicit user
inputs from defaults.

If `Project Latitude` and `Project Longitude` are provided but Open-Meteo
cannot be reached because the machine has no network access, the request times
out, or the service returns only transient failures, the model should continue
as if project coordinates had not been provided. In that fallback case:

- `model.weather_data` remains `None`;
- missing `Ambient Temperature` and `Surface Temperature` keep their normal
  GEOPHIRES defaults instead of being filled from weather data;
- Dispatchable, TESS, and Baseload calculations use their existing scalar
  temperature behavior;
- the run should log a warning explaining that weather data was requested but
  unavailable.

This offline fallback applies only to connectivity and transient service
failures. Invalid coordinates, invalid years, or malformed weather responses
should still raise errors because those indicate bad input or incompatible API
data rather than an offline run.

## Dispatchable Mode Integration

Dispatchable mode should use hourly weather where it affects plant physics.

Current dispatch electricity and CHP calculations use scalar
`Ambient Temperature` for ORC and flash performance. The weather feature should
allow the dispatch timestep to use:

```text
temperature_2m[timestep_index % 8760]
```

instead of scalar `Ambient Temperature` when weather data is available.

This improves:

- ORC and flash gross electricity production.
- Net electricity production.
- Reinjection temperature.
- Geofluid availability.
- First-law efficiency.
- CHP heat and electricity split behavior.
- Dispatch capacity factor.
- Peak geothermal contribution.
- Peak unmet load.
- Electric and CHP TESS charging behavior, because storage charge value depends
  on plant output.

Add dispatch result storage for auditability:

```python
hourly_ambient_temperature
```

Include that column in dispatch profile CSV output when dispatch results are
written.

If no weather data is available, dispatch behavior remains unchanged.

## TESS Integration

The first implementation should not make TESS losses weather-dependent. It
should only allow TESS to benefit indirectly from weather-aware dispatch plant
performance.

That means:

- TESS electric and CHP cases inherit hourly ambient impacts through the plant
  output used to charge storage.
- TESS direct-heat cases are unchanged unless a future storage loss model uses
  weather.
- Existing `TESS Daily Heat Loss Fraction` behavior remains the default.

Future TESS enhancement:

```text
TESS Heat Loss Model = Fixed Fraction | Ambient Dependent
```

An ambient-dependent model would use a UA-style heat-loss calculation:

```text
hourly_loss_mwh = UA * (TESS temperature - ambient temperature) * dt
```

That requires new TESS parameters such as `TESS UA` or an equivalent heat loss
coefficient. It should be a later effort because the current TESS model uses a
simple fractional loss parameter and does not include tank geometry or
insulation.

## Baseload Mode Integration

Weather data should be available to baseload mode, but the first implementation
should use it selectively.

Baseload support includes:

- annual average `temperature_2m` as the default for missing `Ambient
  Temperature`;
- annual average `temperature_2m` as the default for missing `Surface
  Temperature`;
- hourly `temperature_2m` for baseload electricity and CHP surface plant
  calculations when the plant type is ORC or flash.

Baseload electricity and CHP currently use scalar `Ambient Temperature` in the
same performance calculations used by dispatchable mode. Replacing that scalar
with an hourly ambient profile can improve annual electric output, reinjection
temperature, heat extraction, and CHP results.

For pure direct-use heat, heat pump, absorption chiller, and district heating
baseload modes, do not change the first implementation's physical calculations
beyond the missing-parameter defaults. Weather can improve those modes only
after additional weather-sensitive demand or equipment models are added.

For baseload time series longer than one year, tile the normalized weather year:

```text
ambient_temperature_for_timestep = temperature_2m[timestep_index % 8760]
```

If no weather data is available, baseload behavior remains unchanged.

## Future Weather-Sensitive Models

The additional downloaded variables support later improvements:

- `wet_bulb_temperature_2m`: cooling tower, evaporative cooling, and condenser
  performance.
- `relative_humidity_2m`, `dew_point_2m`, `surface_pressure`, and
  `vapour_pressure_deficit`: psychrometric calculations.
- `wind_speed_10m`: convection-sensitive tank losses and air-cooled equipment.
- `shortwave_radiation`: solar gains on above-ground storage or equipment.
- `et0_fao_evapotranspiration`: water-use and cooling makeup studies.

Potential future model enhancements:

- optional weather-derived district heating demand source, without replacing
  the existing temperature-file workflow;
- weather-derived cooling demand;
- temperature-dependent heat pump COP;
- wet-bulb-dependent absorption chiller or cooling tower performance;
- ambient-dependent TESS standby loss;
- water consumption or makeup-water reporting.

## Development Steps

### Step 1: Parameters and Schema

Add input parameters:

```text
Project Latitude
Project Longitude
Weather Data Year
```

Parameter details:

- `Project Latitude`: optional float, valid range `-90` to `90`.
- `Project Longitude`: optional float, valid range `-180` to `180`.
- `Weather Data Year`: optional integer, default `2024`, valid only within the
  known Open-Meteo historical archive range from `1940` through the latest
  complete archive year.

Update generated request schema and parameter documentation.

### Step 2: WeatherData Module

Add `src/geophires_x/WeatherData.py`.

Implement:

- Open-Meteo request construction.
- timeout and retry behavior.
- response validation.
- optional-variable warning behavior.
- hourly normalization to 8760 rows.
- aggregation functions.

Keep the module independent from `Model` where practical so it can be tested
with simple mocked API responses.

### Step 3: Model Integration

After parameters are read and before calculations:

- determine whether weather data is active;
- fetch and validate weather data;
- assign `model.weather_data`;
- fill missing `Ambient Temperature`;
- fill missing `Surface Temperature`.

The fetch should occur only once per model run.

If the fetch fails because Open-Meteo is unreachable or returns only transient
failures after retry exhaustion, log a warning, set `model.weather_data = None`,
and continue as if project coordinates were absent.

### Step 4: Dispatchable Integration

Add a helper for ambient temperature at timestep:

```python
def dispatch_ambient_temperature(model, timestep_index: int) -> float:
    ...
```

Use it in electricity and CHP dispatch plant calculations.

Add `hourly_ambient_temperature` to `DispatchResults` and dispatch profile CSV
output.

### Step 5: Baseload Electricity and CHP Integration

Add shared surface-plant helper behavior so ORC and flash baseload calculations
can use an ambient temperature array when weather data is available.

Preserve scalar behavior for all existing non-weather cases.

### Step 6: Output and Documentation

Document the new input parameters and behavior:

- activation rules;
- default `Weather Data Year`;
- Open-Meteo variables;
- 8760-hour normalization;
- how ambient and surface defaults are set;
- which operating modes use hourly weather immediately.

Output reports include a small weather summary section whenever weather data is
active:

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

The section is omitted when weather data is inactive or when the offline
fallback continues without downloaded data. The client text-output parser should
also expose this section so automated workflows can audit which weather year and
temperature profile were used.

The weather profile is not written as a CSV artifact and is not included in
JSON output. The text report summary is the intended audit surface for this
effort.

## Unit Tests

Use mocked HTTP responses. Normal unit tests must not depend on live
Open-Meteo service availability.

The implementation plan must include comprehensive testing documents and
coverage checklists for both affected operating paths:

- Dispatchable mode, including ordinary dispatch without TESS.
- TESS mode, including every supported TESS dispatch interaction case.

These testing documents should be kept with the weather feature documentation
or linked from it so future changes can verify that weather integration did not
regress either Dispatchable or TESS behavior.

Weather module tests:

- no coordinates means no fetch;
- latitude without longitude raises;
- longitude without latitude raises;
- default `Weather Data Year` is `2024`;
- explicit `Weather Data Year` is honored;
- request contains expected endpoint and variables;
- timeout is passed to the HTTP client;
- transient HTTP errors are retried;
- permanent HTTP errors are not retried;
- missing required variable raises;
- missing optional variable warns and continues;
- 8760-row response is accepted;
- 8784-row leap-year response is normalized to 8760;
- unexpected row count raises or resamples only through the documented path;
- hourly, daily, ISO-weekly, monthly, and annual averages return expected
  values.

Model integration tests:

- project coordinates activate weather download;
- no project coordinates leave `model.weather_data` as `None`;
- weather annual average fills missing `Ambient Temperature`;
- provided `Ambient Temperature` is preserved;
- weather annual average fills missing `Surface Temperature`;
- provided `Surface Temperature` is preserved.

Dispatchable tests:

- dispatch electricity run records `hourly_ambient_temperature`;
- dispatch electricity output changes when weather hourly temperature differs
  from scalar ambient temperature;
- dispatch behavior is unchanged when weather data is absent;
- dispatch profile CSV includes hourly ambient temperature when dispatch
  profile output is requested.

TESS tests:

- TESS electric or CHP dispatch uses weather-aware plant output indirectly;
- TESS direct-heat behavior remains unchanged except for report/profile columns;
- TESS runs without weather data preserve current behavior.
- TESS with weather data produces valid TESS report output, dispatch report
  output, profile CSV output, JSON output, HTML output, and generated PNG
  references.
- TESS weather-aware dispatch should be tested for each supported demand mode:
  direct heat, electricity, cooling, CHP, heat pump, and district heating with
  peaking-boiler interaction.
- TESS tests should verify that weather data is tiled or aligned correctly over
  the dispatch analysis window.
- TESS electric and CHP tests should verify that changing hourly weather
  temperature changes weather-sensitive plant output and therefore changes the
  storage charge profile.
- TESS thermal direct-use tests should verify that weather data does not change
  storage behavior except through explicitly supported weather-dependent
  pathways.
- TESS tests should cover missing optional weather variables, ensuring optional
  weather columns do not block TESS execution.
- TESS tests should cover absent weather data, ensuring current non-weather
  TESS behavior remains the regression baseline.
- TESS tests should cover leap-year weather input normalized to 8760 hours and
  confirm all TESS hourly arrays still have the expected dispatch length.

Baseload tests:

- baseload ORC/flash electricity can consume hourly weather ambient
  temperature;
- baseload ORC/flash behavior is unchanged when weather data is absent;
- baseload direct-use heat output is unchanged except missing
  ambient/surface-temperature defaults.
- full baseload electricity test cases should be added for all supported
  electricity plant types:
  - subcritical ORC;
  - supercritical ORC;
  - single flash;
  - double flash.
- baseload electricity tests should compare a scalar-ambient run to a
  weather-aware hourly ambient run and verify that annual gross electricity,
  annual net electricity, first-law efficiency, geofluid availability, and
  reinjection temperature are populated and respond to weather data.
- baseload electricity tests should verify that hourly weather temperature is
  tiled across multi-year plant lifetimes.
- baseload electricity tests should verify that user-provided `Ambient
  Temperature` is preserved for the scalar parameter while the hourly weather
  profile is used only by the weather-aware calculation path.
- baseload electricity tests should verify that missing `Ambient Temperature`
  and missing `Surface Temperature` are filled from annual average
  `temperature_2m`.
- baseload electricity tests should cover leap-year weather input normalized to
  8760 hours and confirm the baseload output arrays still match the expected
  model timestep convention.

Optional integration test:

- one live Open-Meteo request behind an environment variable such as
  `GEOPHIRES_ENABLE_LIVE_WEATHER_TESTS=1`.

## Backward Compatibility

Backward compatibility requirements:

- no new network request unless both project coordinates are provided;
- existing input files without project coordinates produce identical results;
- user-provided `Ambient Temperature` and `Surface Temperature` are never
  overwritten;
- weather data is normalized to the existing 8760-hour convention;
- failures in optional weather variables do not block a run;
- failures in required weather variables fail early with clear errors.

## Open Questions

Resolved decisions:

- Weather data should not be written to a CSV artifact.
- The weather profile should not be included in JSON output.
- `Weather Data Year` should be constrained to the known Open-Meteo historical
  data range before making the API request.
- ET0 aggregation should expose averages only, not sums.
- Weather-driven district heating demand should be a future optional demand
  source, not a replacement for the existing temperature-file workflow.
