# Proposal: Extend GEOPHIRES parameters so they can optionally read their data from Files or URLs. Also allow for arrays of values instead of just scalars, with associated unit parsing and normalization. Also add a new new parameter type - Filename.

## Objective

Extend GEOPHIRES parameters so they can optionally read their data from Files or URLs. Also allow for standardization inputs from canonical sources instead of just scalars from files. The data from those new sources must come with the standard associated unit parsing and normalization. The extension is done on a case-by-case basis with a boolean flag. It is assumed that it will generally be OK to allow a scalar to be read from a file or URL instead of directly from the parameter file/list, since that is a like-for-like substitution. The extension to allow for arrays is trickier, and must be accounted for in the code. The addition of the new parameter type, Filename, allows for smarter use of filenames/URLs. Previously, we used strings to hold file names, but there was no validity checking.

This proposal is implementation-oriented and staged to minimize risk and preserve backward compatibility. It includes full unit testing and sample data.

## Scope

### New parameter type to add

`Filename`

This additional will allow us to upgrade online GEOPHIRES - we can now offer functionality that reads file-based data where previously, that was only possible for desktop GEOPHIRES. In particular, we can now offer access to the TOUGH2 example in web-based GEOPHIRES.

These will initially be added and tested but not implemented in any existing code paths until a future phase. This allows for a clean separation of concerns and incremental rollout.

### Existing Parameter types to allow for reading from files and URLs

`All parameters that currently only accept scalar or list inputs can now optionally accept file/URL input.`

This can strengthen the model's ability to use real-world data and reduce manual pre-processing, but requires careful handling of parsing, unit normalization, and error handling. It is also easy to refer to a single canonical source for a value via URL but access it from various configuration files. If that canonical source is upgraded, then all parameters that reference that canonical source are also upgraded without remembering to update several input parameter lists/files.

The upgrade is opt-in on a per-parameter basis, with a boolean flag in the parameter metadata to enable Extended Input like reading from files or URLS. By default, Extended Input is disabled for all parameters, preserving existing behavior. This allows for a gradual migration path and minimizes risk of unintended consequences.

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
