# Proposal: Historical array inputs for GEOPHIRES operating and resource data

## Objective

Extend GEOPHIRES array input support to (a) add new historical-demand parameters and (b) upgrade selected existing parameters so they can ingest X/Y arrays with explicit units, normalize those units to GEOPHIRES defaults, and (for time-series data) produce an hourly one-year representation (8760 points).

This proposal is implementation-oriented and staged to minimize risk and preserve backward compatibility.

## Scope

### New parameters to add

1. `Annual Heating Demand` (`SurfacePlant.HeatingDemand`)
2. `Annual Cooling Demand` (`SurfacePlant.CoolingDemand`)
3. `Annual Electricity Demand` (`SurfacePlant.ElectricityDemand`)

These will initially be added as fully supported historical array parameters (parsing, normalization, validation, storage, reporting hooks). Their downstream use in plant calculations can be enabled later. Cooling/Heating/Electricity demand series should be resampled to 8760 points when not already that length.

### Existing parameters to upgrade for array support

1. `Ambient Temperature` (`SurfacePlant.ambient_temperature`)
2. `Surface Temperature` (`Reservoir.Tsurf`)
3. `Gradients` (`SurfacePlant.gradient`)
4. `Electricity Rate` (`SurfacePlant.electricity_cost_to_buy`)
5. `Heat Rate` (`SurfacePlant.heat_price`)

### Related parameter behavior change

- `Number Of Segments` (`SurfacePlant.numseg`) should be auto-calculated from gradient/thickness array lengths, with warning behavior when user-provided `numseg` does not match.

## Input model: canonical X/Y historical array

Introduce one canonical data model used by all seven array-enabled parameters:

- `x_raw: np.ndarray`
- `y_raw: np.ndarray`
- `x_units_raw: str | None`
- `y_units_raw: str | None`
- `x_canonical: np.ndarray`
- `y_canonical: np.ndarray`
- `x_units_canonical: str` (e.g., `hour`, `meter`)
- `y_units_canonical: str` (e.g., `degC`, `USD/kWh`)
- `source_kind: inline | file | url`
- `has_header: bool`
- `normalization_notes: list[str]`

This separates parsing concerns from simulation logic and gives an auditable normalization trail.

## Accepted sources and syntax

Use existing source precedence and extend consistently:

1. Inline array content in input file
2. Local file path (CSV)
3. URL (CSV-like response)

### CSV/header convention

The first line may provide axis names and units, for example:

- `Time (minutes), Temperature (degF)`
- `Distance (feet), Temperature (K)`
- `Time (days), Cost (EUR/kWh)`

If header is absent:

- Assume default X units for parameter family.
- Assume default Y units for parameter family.

### Parsing rules

- Ignore blank and comment lines (`#`, `--`, `*`) after optional header processing.
- Require exactly two numeric columns per data row.
- Reject NaN/Inf and non-finite post-conversion values.
- Require monotonic non-decreasing X values.
- For duplicate X values, apply deterministic resolution (default: keep last, warn).

## Unit normalization policy by parameter family

### 1) Ambient Temperature and Surface Temperature

- X axis semantic: time
- X default unit: hours
- Y axis semantic: temperature
- Y default unit: Celsius (`degC`)

#### Required behavior

1. Parse user-specified time units from header (seconds/minutes/hours/days/weekly/monthly etc.).
2. Convert all X values to hours.
3. Parse user-specified temperature units from header (Fahrenheit/Celsius/Kelvin etc.).
4. Convert all Y values to Celsius.
5. Resample/interpolate to hourly cadence for one year (8760 points).

### 2) Gradients

- X axis semantic: distance
- X default unit: meters
- Y axis semantic: Gradient temperature change per distance
- Y default unit: Celsius/m (`degC/m`)

#### Required behavior

1. Parse distance units from header (feet/meters/centimeters/inches/etc.).
2. Convert X values to meters.
3. Parse gradient units from header and convert Y values to Celsius/m.
4. Do **not** resample/downsample/interpolate array length.

### 3) Electricity Rate and Heat Rate

- X axis semantic: time
- X default unit: hours
- Y axis semantic: cost per energy
- Y default unit: `USD/kWh`

#### Required behavior

1. Parse time units from header and convert X to hours.
2. Parse cost-rate units from header (e.g., `USD/kWh`, `EUR/kWh`) and convert Y to `USD/kWh`.
3. Resample/interpolate to hourly cadence for one year (8760 points).

## Resampling strategy for time series (8760-hour target)

Apply to:

- Ambient Temperature
- Surface Temperature
- Electricity Rate
- Heat Rate
- Annual CoolingDemand
- Annual HeatingDemand
- Annual ElectricityDemand

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

## `numseg` auto-calculation behavior

When both `Gradients` and `Thicknesses` arrays are provided:

1. Validate both arrays are present and lengths match expected segmentation semantics.
2. Compute `derived_numseg` from validated array length.
3. If `SurfacePlant.numseg` is absent: set automatically and log warning/info.
4. If user-provided `numseg` differs from `derived_numseg`: override with `derived_numseg` and emit warning that includes both values.
5. If user-provided `numseg` matches: keep as-is, no warning.

This prevents silent inconsistencies in segmented reservoir inputs.

## Proposed code architecture

## 1) New reusable parser/normalizer module

Add a dedicated utility (example path):

- `src/geophires_x/historical_arrays.py`

Core functions:

- `parse_xy_series(source_text_or_tokens, defaults, header_policy) -> ParsedXYSeries`
- `detect_header_units(header_line) -> AxisUnitMetadata`
- `convert_xy_units(series, x_dim, y_dim, x_default, y_default) -> ConvertedXYSeries`
- `resample_to_hourly_year(series, mode="linear") -> ConvertedXYSeries`
- `validate_xy_series(series, constraints) -> None`

## 2) Parameter metadata extensions

Extend parameter descriptors with optional historical-array metadata:

- `AllowHistoricalArrayInput: bool`
- `HistoricalXDimension: Literal["time", "distance", "none"]`
- `HistoricalYDimension: Literal["temperature", "cost_rate", "energy_rate", "generic"]`
- `HistoricalDefaultXUnits: str`
- `HistoricalDefaultYUnits: str`
- `HistoricalResampleToHourlyYear: bool`

This avoids one-off logic in each component and keeps parsing declarative.

## 3) Integration in input reading path

In `ReadParameter` flow:

1. Detect historical-array-capable parameters.
2. Route value/source to shared X/Y parser.
3. Apply unit normalization.
4. Apply optional resampling.
5. Store canonical arrays in parameter value container.
6. Emit warnings through existing logging infrastructure.

## 4) Backward compatibility behavior

- Existing scalar/list syntax remains valid for parameters not opted into historical arrays.
- For upgraded parameters, maintain existing scalar behavior when scalar input is supplied (if feasible), but prefer explicit migration warning when scalar-to-array coercion occurs.
- No behavior change for unrelated parameters.

## Parameter-by-parameter proposed defaults

| Parameter                              | X semantic | X default | Y semantic  | Y default | Resample to 8760h |
|----------------------------------------|---|---|-------------|-----------|---|
| `SurfacePlant.ambient_temperature`     | time | hour | temperature | degC      | yes |
| `Reservoir.Tsurf`                      | time | hour | temperature | degC      | yes |
| `SurfacePlant.gradient`                | distance | meter | gradient    | degC/m  | no |
| `SurfacePlant.electricity_cost_to_buy` | time | hour | cost        | USD/kWh   | yes |
| `SurfacePlant.heat_price`              | time | hour | cost        | USD/kWh   | yes |
| `SurfacePlant.HeatingDemand`           | time* | hour* | demand*     | MWh*      | yes |
| `SurfacePlant.CoolingDemand`           | time* | hour* | demand*     | MWh*      | yes |
| `SurfacePlant.ElectricityDemand`       | time* | hour* | demand*     | MWh*      | yes |

`*` Heating/Cooling/Electricity demand arrays are added now; exact computational usage and potentially final Y-axis units will be finalized in follow-on implementation.

## Validation and warning matrix

Warnings (non-fatal unless specified):

- Header missing → defaults assumed.
- Unrecognized unit token → error (fatal for that parameter).
- Non-monotonic X → error.
- Duplicate X → warning + deterministic consolidation.
- Time series shorter/longer than 8760h → warning + edge-hold/truncate behavior.
- `numseg` mismatch with gradient/thickness arrays → warning + auto-correct.

## Suggested implementation phases

### Phase 1: Plumbing and parser foundation

- Add shared XY parsing/units module.
- Add parameter metadata knobs.
- Add tests for header parsing and unit conversion.

### Phase 2: Upgrade existing parameters

- Enable historical arrays for Ambient Temperature, Surface Temperature, Gradient, Electricity Rate, Heat Rate.
- Add resampling for time-series group only.
- Add `numseg` auto-calc behavior.

### Phase 3: Add new demand parameters

- Add `SurfacePlant.HeatingDemand`, `SurfacePlant.CoolingDemand`, and `SurfacePlant.ElectricityDemand` parameter definitions.
- Enable ingestion, normalization, storage, and output reporting.
- Leave computational coupling guarded/feature-flagged until demand usage rules are specified.

### Phase 4: Hardening and documentation

- Add full integration tests with inline/file/URL data.
- Add full documentation of new input formats, header conventions, and unit policies.
- Add docstrings, code inlines comments and usage examples in code and docs.
- Document CSV header/unit grammar and defaults.
- Provide migration notes and examples.

## Testing plan and sample data

### Test assets to include

Add/maintain the following sample files under:

- `tests/assets/params/ambient_temperature.csv`
- `tests/assets/params/surface_temperature.csv`
- `tests/assets/params/gradients.csv`
- `tests/assets/params/electricity_rate.csv`
- `tests/assets/params/heat_rate.csv`
- `tests/assets/params/annual_heat_demand.csv`
- `tests/assets/params/annual_cooling_demand_csv`
- `tests/assets/params/annual_electricity_demand.csv`

Public URL variants should be supported using the tagged asset repo pattern, for example:

- `https://raw.githubusercontent.com/malcolm-BSD/GEOPHIRES-X/test-assets-v1.0/tests/assets/params/ambient_temperature.csv`

The same URL shape applies to the other assets by swapping the filename (including `annual_heat_demand.csv`, `annual_cooling_demand_csv`, and `annual_electricity_demand.csv`).

### Test categories

1. Header parsing and defaults:
   - `Time (minutes), Temperature (degF)` recognized and converted.
   - Missing header uses defaults (hours, Celsius or family defaults).
2. Unit conversion:
   - time (`s`, `min`, `day`, `month`) to hour.
   - distance (`ft`, `cm`, `in`) to meter.
   - temperature (`degF`, `K`) to Celsius.
   - cost rate (`EUR/kWh`) to `USD/kWh` (using configured currency conversion policy).
3. Resampling/interpolation:
   - downsample, upsample, irregular intervals, short-span, long-span.
   - output length always 8760 for time-series family.
4. Gradient-specific behavior:
   - no resampling; converted lengths unchanged.
5. `numseg` behavior:
   - absent → auto-set.
   - mismatch → warning + correction.
   - match → no warning.
6. Source loading modes:
   - inline input parsing.
   - local file loading from `tests/assets/params/*.csv`.
   - URL loading from `raw.githubusercontent.com/.../test-assets-v1.0/...`.
   - comment and blank-line handling.

### Minimum integration matrix (recommended)

- Ambient temperature from local file + URL, verify hour/°C normalization and 8760 output.
- Surface temperature from local file + URL, verify hour/°C normalization and 8760 output.
- Gradients from local file + URL, verify meter/°C normalization and unchanged length.
- Electricity rate from local file + URL, verify hour/USD-kWh normalization and 8760 output.
- Heat rate from local file + URL, verify hour/USD-kWh normalization and 8760 output.
- Annual heating demand from local file + URL, verify expected ingestion/normalization behavior and resampling to 8760 samples.
- Annual cooling demand from local file + URL, verify expected ingestion/normalization behavior and resampling to 8760 samples.
- Annual electricity demand from local file + URL, verify expected ingestion/normalization behavior and resampling to 8760 samples.

## Open design decisions to finalize

1. Currency conversion source for `EUR/kWh` → `USD/kWh` (fixed user-supplied factor vs. dynamic rates).
2. Interpolation mode default for price series (linear vs. step/previous).
3. Whether scalar legacy inputs for upgraded parameters should auto-expand to flat 8760 arrays or remain scalar until first array use.
4. Exact Y-axis unit semantics for Heating/Cooling/Electricity demand (e.g., `kWh`, `MWh`, `kWh per interval`).

## Expected user-facing benefits

- Enables direct use of historical/forecast operating data across thermal and economic inputs.
- Reduces pre-processing burden by handling unit normalization and cadence conversion internally.
- Improves model consistency with auto-validated segment counts for gradient/thickness inputs.
- Preserves compatibility via declarative opt-in parameter upgrades.
