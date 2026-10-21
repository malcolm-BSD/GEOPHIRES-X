# Proposal: Extend GEOPHIRES parameters so they can optionally read their data from Files or URLs. Also allow for arrays of values instead of just scalars, with associated unit parsing and normalization. Also add a new new parameter type - Filename.

## Objective

Extend GEOPHIRES parameters so they can read their data from files or URLs. Also allow for standardized inputs from canonical sources instead of just scalars from files. Data from those sources retains the usual type, unit, range, and normalization checks. `AllowExtendedInput` is the single switch controlling all extended-input forms. It defaults to `True` for user-input parameters; output-destination strings explicitly opt out. Scalars remain scalars, while arrays are accepted only by list/time-series parameters and scalar parameters with a dedicated collection mode. The `Filename` parameter type provides validity checking while preserving filename/URL references for the consuming code.

This proposal is implementation-oriented and staged to minimize risk and preserve backward compatibility. It includes full unit testing and sample data.

## Scope

### New parameter type to add

`Filename`

This additional will allow us to upgrade online GEOPHIRES - we can now offer functionality that reads file-based data where previously, that was only possible for desktop GEOPHIRES. In particular, we can now offer access to the TOUGH2 example in web-based GEOPHIRES.

These will initially be added and tested but not implemented in any existing code paths until a future phase. This allows for a clean separation of concerns and incremental rollout.

### Existing Parameter types to allow for reading from files and URLs

`All parameters that currently only accept scalar or list inputs can now optionally accept file/URL input.`

This can strengthen the model's ability to use real-world data and reduce manual pre-processing, but requires careful handling of parsing, unit normalization, and error handling. It is also easy to refer to a single canonical source for a value via URL but access it from various configuration files. If that canonical source is upgraded, then all parameters that reference that canonical source are also upgraded without remembering to update several input parameter lists/files.

Extended input is controlled on a per-parameter basis by the `AllowExtendedInput` metadata flag. It is enabled by default for all user inputs. The three output-destination strings (`Improved Text Output File`, `HTML Output File`, and `Dispatch Profile Output File`) explicitly set it to `False`, so a destination path is never mistaken for a source to read.

### Selecting a value from a JSON file or URL

An extended-input parameter can select one value from a JSON document by appending an
[RFC 6901 JSON Pointer](https://www.rfc-editor.org/rfc/rfc6901) after `#`:

```text
Reservoir Depth, D:\Work\ProjectFalco\inputs\geophires\profiles\edwards.json#/target_depth_km
Project Latitude, D:\Work\ProjectFalco\inputs\geophires\profiles\edwards.json#/latitude
Project Longitude, D:\Work\ProjectFalco\inputs\geophires\profiles\edwards.json#/longitude
Surface Temperature, D:\Work\ProjectFalco\inputs\geophires\profiles\edwards.json#/surface_temperature_c
Ambient Temperature, D:\Work\ProjectFalco\inputs\geophires\profiles\edwards.json#/ambient_temperature_c
```

The source can be an absolute path, a path relative to the GEOPHIRES input file, or an HTTP(S) URL. Nested values use
paths such as `profile.json#/site/location/latitude`; array entries use paths such as `profile.json#/temperatures/0`.
JSON Pointer escaping is supported (`~1` for `/` and `~0` for `~`). The selected JSON value is processed exactly like
a literal parameter value, including the parameter's normal type, range, and unit handling. JSON objects and `null`
cannot be assigned directly; select a scalar or array value within the document. Scalar parameters reject JSON arrays,
and list/time-series parameters require arrays. Invalid boolean spellings are rejected rather than replaced by a default.
A source without a `#/...` selector retains the existing whole-file extended-input behavior, except for a `Filename`
parameter: its bare filename or URL remains the value to be consumed later. A JSON selector may select a string filename
for a `Filename` parameter, after which the normal file/URL validation is applied.

### Existing parameters to upgrade for array support

1. `Gradients` (`SurfacePlant.gradient`) - this was already an array, but now we will allow it to be read from a file or URL and parsed with unit normalization.
2. `Thicknesses` (`SurfacePlant.thickness`) - this was already an array, but now we will allow it to be read from a file or URL and parsed with unit normalization.
3. `Electricity Rate` (`SurfacePlant.electricity_cost_to_buy`) - this was previously a scalar, but now we will allow it to be read from a file or URL and parsed with unit normalization.
4. `Heat Rate` (`SurfacePlant.heat_price`) - this was previously a scalar, but now we will allow it to be read from a file or URL and parsed with unit normalization.

These examples of how upgrading a scalar to an array improves the real-world applicability of the model, but also requires careful handling of parsing, unit normalization, and error handling. The addition of array support for these parameters allows for more detailed and accurate modeling based on real-world data.

### Related parameter behavior change

- `Number Of Segments` (`SurfacePlant.numseg`) should be auto-calculated from gradient/thickness array lengths, with warning behavior when user-provided `numseg` does not match.

### 1) Gradients and Thicknesses as arrays

- Gradients default unit: Celsius/km (`degC/km`)
- Thicknesses default unit: kilometers

#### Required behavior

1. Remove all comments
2. Parse distance units from header (feet/meters/centimeters/inches/etc.). The header must be in the form of "Gradient (degF/ft)". If not there, assume it is in the default units.
3. Convert values to Preferred Units if required.
4. Do **not** resample/downsample/interpolate array length.

## `numseg` auto-calculation behavior

When both `Gradients` and `Thicknesses` arrays are provided:

1. Validate both arrays are present and lengths match expected segmentation semantics.
2. Compute `derived_numseg` from validated array length.
3. If `SurfacePlant.numseg` is absent: set automatically and log warning/info.
4. If user-provided `numseg` differs from `derived_numseg`: override with `derived_numseg` and emit warning that includes both values.
5. If user-provided `numseg` matches: keep as-is, no warning.

This prevents silent inconsistencies in segmented reservoir inputs.

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

- Electricity Rate
- Heat Rate

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


## Testing plan and sample data

### Test assets to include

Add/maintain the following sample files under:

- `tests/assets/params/gradients.csv`
- `tests/assets/params/electricity_rate.csv`
- `tests/assets/params/heat_rate.csv`

Public URL variants should be supported using the tagged asset repo pattern, for example:

- `https://raw.githubusercontent.com/malcolm-BSD/GEOPHIRES-X/test-assets-v1.0/tests/assets/params/gradients.csv`

The same URL shape applies to the other assets by swapping the filename.
Note the use of tagged asset repo pattern (`test-assets-v1.0`) allows for stable test data references while still enabling updates to the canonical source as needed.

### Unit Tests
test_read_bool_from_file - This test will verify that boolean parameters can be correctly read from a file. It will check for correct parsing of true/false values, handling of different cases (e.g., "True", "true", "False", "false"), and proper error handling for invalid inputs.
test_read_int_from_file - This test will ensure that integer parameters can be accurately read from a file. It will validate that the values are correctly parsed as integers, check for proper handling of negative numbers, and verify that non-integer inputs are appropriately flagged as errors.
test_read_float_from_file - This test will confirm that floating-point parameters can be read from a file with correct parsing. It will check for proper handling of decimal points, scientific notation, and ensure that non-numeric inputs are correctly identified as errors.
test_read_str_from_file - This test will validate that string parameters can be read from a file without issues. It will check for correct parsing of various string formats, including those with spaces, special characters, and ensure that empty strings are handled appropriately.
test_read_list_from_file - This test will verify that list parameters can be read from a file correctly. It will check for proper parsing of list formats (e.g., comma-separated values), handling of different data types within the list, and ensure that malformed lists are flagged as errors.
test_read_bool_from_URL - This test will verify that boolean parameters can be correctly read from a URL. It will check for correct parsing of true/false values, handling of different cases (e.g., "True", "true", "False", "false"), and proper error handling for invalid inputs when reading from a URL.
test_read_int_from_URL - This test will ensure that integer parameters can be accurately read from a URL. It will validate that the values are correctly parsed as integers, check for proper handling of negative numbers, and verify that non-integer inputs are appropriately flagged as errors when reading from a URL.
test_read_float_from_URL - This test will confirm that floating-point parameters can be read from a URL with correct parsing. It will check for proper handling of decimal points, scientific notation, and ensure that non-numeric inputs are correctly identified as errors when reading from a URL.
test_read_str_from_URL - This test will validate that string parameters can be read from a URL without issues. It will check for correct parsing of various string formats, including those with spaces, special characters, and ensure that empty strings are handled appropriately when reading from a URL.
test_read_list_from_URL - This test will verify that list parameters can be read from a URL correctly. It will check for proper parsing of list formats (e.g., comma-separated values), handling of different data types within the list, and ensure that malformed lists are flagged as errors when reading from a URL.
test_read_gradient_from_file - This test will verify that gradients parameter can be read from a file correctly. It will check for proper parsing of gradient values, handling of different units (e.g., Celsius/km, Fahrenheit/ft), and ensure that unit conversion is performed accurately.
test_read_thicknesses_from_URL - This test will verify that thicknesses parameter can be read from a URL correctly. It will check for proper parsing of thickness values, handling of different units (e.g., ft/m, km/ft), and ensure that unit conversion is performed accurately.
