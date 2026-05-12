# Proposal: Optional structured input for scalar and list numeric parameters

## Goal

Support additional structured user input formats while preserving existing GEOPHIRES behavior:

1. Allow an input parameter that is normally a scalar (`intParameter` or `floatParameter`) to be provided as a two-element
   pair vector and internally converted to an ndarray.
2. Allow numeric list parameters (`listParameter`) to load values from a local file path or URL (CSV/line-delimited),
   with per-item unit conversion.

## Pair-vector format for scalar parameters

The two-element pair-vector format is:

1. independent coordinate (distance, time, depth, etc.)
2. dependent value for the parameter at that coordinate

Examples:

- inline: `Reservoir Depth, [3500, 180]`
- local CSV file path: `Reservoir Depth, /path/to/pair.csv`
- URL: `Reservoir Depth, https://example.org/pair.csv`

CSV contents (single line or first data line):

```csv
3500,180
```

## List-parameter source format

For list parameters such as `Gradients` and `Thicknesses`, user input may be:

- inline list values (existing behavior), e.g. `Gradients, 50, 40, 30, 20`
- file path, e.g. `Gradients, /path/to/gradients.csv`
- URL, e.g. `Thicknesses, https://example.org/thicknesses.csv`

File/URL content may be comma-separated or line-delimited numeric values. Comment lines beginning with `#`, `--`, or `*`
are ignored.

## Why this should be optional / backward compatible

Most scalar parameters are currently used directly in arithmetic expressions, so automatic conversion to arrays must
remain opt-in for scalar parameters. List parameters already behave as arrays; extending source formats preserves their
core behavior.

## Implemented architecture

### 1) Scalar metadata extension

In `src/geophires_x/Parameter.py`, `Parameter` includes metadata for pair vectors:

- `AllowPairVectorInput: bool = False`
- `PairVectorAsNumpyArray: bool = True`
- `PairVectorValue: Optional[np.ndarray] = None`
- `PairVectorAxisLabel: Optional[str] = None`

### 2) Scalar pair-vector parsing helpers

Implemented helpers include:

- `_try_read_pair_vector(...)`
- `_try_parse_pair_vector_inline(...)`
- `_try_parse_pair_vector_csv_file(...)`
- `_try_parse_pair_vector_csv_url(...)`
- `_pair_vector_from_csv_text(...)`
- `_parse_csv_pair_line(...)`

Validation rules:

- exactly two numeric values
- finite values only
- reject NaN/Inf

### 3) List source parsing helpers

Implemented helpers include:

- `_try_read_numeric_list_from_source(...)`
- `_parse_numeric_list_text(...)`
- `_parse_numeric_list_tokens(...)`

Behavior:

- file/URL source detection for list parameters
- max size checks for file/URL content
- finite numeric validation
- preserve inline list fallback if file/URL parsing does not apply

### 4) `ReadParameter` integration

- Scalar branch (`intParameter` / `floatParameter`): pair-vector parsing runs when `AllowPairVectorInput=True`.
- List branch (`listParameter`): attempts file/URL source parsing before inline token parsing.
- Scalar pre-conversion using `ConvertUnits()` is skipped for list parameters and for detected pair-vector candidates,
  so conversion can be applied at the appropriate token/component level.

## Unit conversion requirements

Structured inputs must use GEOPHIRES unit conversion semantics consistently.

### Pair vectors

1. Detect pair-vector candidates before scalar pre-conversion (avoid sending whole strings such as `[10, 212 degF]`
   into scalar-only conversion logic).
2. Parse into two components (`x`, `y`).
3. Apply `ConvertUnits()` component-wise when a component includes unit text.
4. Validate converted values are finite floats.

### List parameters

1. For file/URL/inline tokenized list values, parse each token independently.
2. If a token includes units (e.g., `1 km`), apply `ConvertUnits()` to that token.
3. Convert token to float and validate finiteness.
4. Continue using existing list min/max validation after list construction.

Unit semantics guidance:

- For pair vectors, `y` uses the parameter's normal unit semantics.
- `x` may be unitless by default unless axis-unit metadata is introduced.

## Input source policy

For scalar pair vectors and list sources:

1. inline values (where applicable)
2. local file path
3. `http://` or `https://` URL

If ambiguous or invalid, preserve compatibility by falling back to existing parsing paths where possible.

## Error handling

- If parsing is malformed, fail parsing for that structured path and fall back to existing path when possible.
- If URL/file retrieval fails, continue to fallback behavior.
- Do not silently coerce malformed numeric tokens.

## Security and robustness notes

For URL support:

- only allow `http`/`https`
- set request timeout
- cap maximum response bytes

For file support:

- cap file size
- parse with comment-line filtering

## Test coverage (implemented)

`tests/test_parameter.py` includes tests for:

- scalar pair-vector parsing (inline, file, URL)
- pair-vector component-wise unit conversion
- list-parameter file source parsing with units
- list-parameter URL source parsing with units

## Backward compatibility

This remains backward compatible because:

- scalar pair vectors are opt-in (`AllowPairVectorInput=False` by default)
- existing scalar parsing remains unchanged for non-opt-in parameters
- list parameters still accept existing inline list syntax
- file/URL list support is additive
