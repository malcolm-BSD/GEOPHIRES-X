# Proposal: Historical array inputs for GEOPHIRES operating and resource data

## Objective

Extend GEOPHIRES to allow input of historically-oriented X/Y arrays with explicit units, normalize those units specified defaults, and (for time-series data) produce an hourly one-year representation (8760 points).

This proposal is implementation-oriented and staged to minimize risk and preserve backward compatibility.

## Scope

### New parameters to add

1. `Annual Heating Demand` (`SurfacePlant.HeatingDemand`)
2. `Annual Cooling Demand` (`SurfacePlant.CoolingDemand`)
3. `Annual Electricity Demand` (`SurfacePlant.ElectricityDemand`)

These will initially be added as fully supported historical array parameters (parsing, normalization, validation, storage, reporting hooks). Their downstream use in plant calculations can be enabled later. Cooling/Heating/Electricity demand series can optionally be resampled to 8760 points when not already that length.

### Existing parameters to upgrade for array support

1. `Ambient Temperature` (`SurfacePlant.ambient_temperature`)
2. `Surface Temperature` (`Reservoir.Tsurf`)

## Input model: canonical X/Y historical array

To do this, we need to introduce one canonical data model used by all time-series array-enabled parameters:

- `PairVectorAsNumpyArray (bool, True)`: Whether to store the pair vector as a numpy array (if False, will store as list of lists)
- `PreferredXUnits (Optional[str], None)`: Specifies the name of the preferred x dimension units, default = "hours"
- `PreferredYUnits (Optional[str], None)`: Specifies the name of the preferred y dimension units
- `CurrentXUnits (Optional[str], None)`: Holds the current units of the x dimension (determined from file)
- `CurrentYUnits (Optional[str], None)`: Holds the current units of the y dimension (determined from file)
- `ResampleToHourlyYear (bool, False)`: specify whether to resample the historical array to an hourly year (8760 hours)

## Accepted sources and syntax

Use existing source precedence and extend consistently:

1. URL (CSV-like response)
2. Local file path (CSV)
3. Inline array content in input file (rare)

### Comments

There can be comments and blank lines in the source, but they are ignored after optional header processing. Comments can begin with "#", "//", ";", "--", "/*", "*", or "`".

### CSV/header convention

The first line may provide axis names and units, for example:
- `Time (minutes), Temperature (degF)`
- `Distance (feet), Temperature (K)`
- `Time (days), Cost (EUR/kWh)`

If header is absent:
- Assume default X units for parameter family.
- Assume default Y units for parameter family.

### Parsing rules

- Ignore blank and comment lines (e.g., `#`, `--`, `*`) after optional header processing.
- Require a minimum of two numeric columns per data row (but more columns are possible, as long as the first two rows are the rows we are interested in).
- Reject NaN/Inf and non-finite post-conversion values.
- Require monotonic non-decreasing X values.

## Unit normalization policy by parameter family

### 1) Ambient Temperature and Surface Temperature

- X axis semantic: time
- X default unit: hours
- Y axis semantic: temperature
- Y default unit: Celsius (`degC`)

### 2) Annual Heating Demand, Annual Cooling Demand, and Annual Electricity Demand

- X axis semantic: time
- X default unit: hours
- Y axis semantic: temperature
- Y default unit: demand (`kWh`)

#### Required behavior for parsing historical data

1. Parse user-specified time units from header (seconds/minutes/hours/days/weekly/monthly etc.).
2. Convert all X values to hours, if needed.
3. Parse user-specified Y-value units from header (Fahrenheit/Celsius/Kelvin etc.).
4. Convert all Y values to preferred units.
5. Optionally, Resample/interpolate to hourly cadence for one year (8760 points).


## Resampling strategy for time series (8760-hour target)

Apply to:

- Ambient Temperature
- Surface Temperature
- Annual Cooling Demand
- Annual Heating Demand
- Annual Electricity Demand

### Recommended algorithm

1. Convert X to canonical hours.
2. Normalize X origin:
   - If first X is not 0, shift series so first sample aligns to 0 hour (warn).
3. Build target grid: `x_target = np.arange(0, 8760, 1.0)`.
4. Interpolate Y onto target:
   - Default interpolation: linear.
   - Optional parameterized interpolation mode later (`linear`, `previous`, `nearest`, `cubic`).
5. Extrapolation policy:
   - If source span is shorter than 8760 h, extend using edge-hold by default (warn).
   - If source span exceeds 8760 h, truncate to first 8760 h (warn).

This guarantees stable model input shape while preserving user trends.


## Parameter-by-parameter proposed defaults

| Parameter                              | X semantic | X default | Y semantic | Y default | Resample to 8760h |
|----------------------------------------|---|---|----------|-----------|---|
| `SurfacePlant.ambient_temperature`     | time | hour | temperature | degC      | yes |
| `Reservoir.Tsurf`                      | time | hour | temperature | degC      | yes |
| `SurfacePlant.HeatingDemand`           | time | hour | demand   | kWh*      | yes |
| `SurfacePlant.CoolingDemand`           | time | hour | demand   | kWh*      | yes |
| `SurfacePlant.ElectricityDemand`       | time | hour | demand   | kWh*      | yes |

## Validation and warning matrix

Warnings (non-fatal unless specified):

- Header missing → defaults assumed.
- Unrecognized unit token → error (fatal for that parameter).
- Non-monotonic X → error.
- Duplicate X → warning + deterministic consolidation.
- Time series shorter/longer than 8760h → warning + edge-hold/truncate behavior.

## Testing plan and sample data

### Test assets to include

Add/maintain the following sample files under:

- `tests/assets/params/ambient_temperature.csv`
- `tests/assets/params/surface_temperature.csv`
- `tests/assets/params/annual_heat_demand.csv`
- `tests/assets/params/annual_cooling_demand_csv`
- `tests/assets/params/annual_electricity_demand.csv`

Public URL variants should be supported using the tagged asset repo pattern, for example:

- `https://raw.githubusercontent.com/malcolm-BSD/GEOPHIRES-X/test-assets-v1.0/tests/assets/params/ambient_temperature.csv`

The same URL shape applies to the other assets by swapping the filename (including `annual_heat_demand.csv`, `annual_cooling_demand_csv`, and `annual_electricity_demand.csv`).

### Tests

test_load_time_temperature_from_file: Test loading of ambient temperature from a local CSV file, including unit parsing and normalization.
test_load_time_cooling_demand_from_file: Test loading of annual cooling demand from a local CSV file, including unit parsing and normalization.
test_load_time_temperature_from_downsampled_file: Test loading of ambient temperature from a file with fewer than 8760 points, ensuring proper resampling and edge-hold behavior.
test_load_time_temperature_from_daily_file: Test loading of ambient temperature from a file with daily data (24 points), ensuring proper resampling to 8760 points.
test_load_time_temperature_from_weekly_file: Test loading of ambient temperature from a file with weekly data (52 points), ensuring proper resampling to 8760 points.
test_load_time_temperature_from_monthly_file: Test loading of ambient temperature from a file with monthly data (12 points), ensuring proper resampling to 8760 points.
test_read_parameter_historical_array_sets_series_and_scalar: Test that reading a historical array parameter correctly sets both the series and scalar values, with appropriate unit normalization.
test_read_parameter_historical_array_allows_scalar_with_units: Test that reading a historical array parameter with a single scalar value and units correctly sets the series to a constant array and the scalar to the provided value.
test_read_parameter_allows_pair_vector_csv_file_for_float_parameter: Test that reading a historical array parameter from a CSV file with appropriate headers correctly parses and normalizes the data.
test_read_parameter_allows_pair_vector_csv_url_for_float_parameter: Test that reading a historical array parameter from a CSV URL with appropriate headers correctly parses and normalizes the data.
