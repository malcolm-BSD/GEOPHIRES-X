# Parameter Formula Support Design

## Goal

Add formula-based input support to the main GEOPHIRES parameter parser so selected numeric input parameters can be defined as expressions instead of only literal values.

This effort should reuse the useful parts of the Monte Carlo formula approach without importing Monte Carlo-specific behavior into standard model input parsing.

## Existing State

- Main parameter parsing is centered on `ReadParameter` in `src/geophires_x/Parameter.py`.
- Monte Carlo formula evaluation exists in `src/geophires_monte_carlo/MC_GeoPHIRES3.py`.
- Monte Carlo formulas currently:
  - evaluate a symbolic expression with `sympy`
  - substitute a single variable `x`
  - optionally expand random expressions such as `random.uniform(...)`

The main parser does not currently have a concept equivalent to Monte Carlo's linked master variable, so formula semantics must be defined explicitly.

## V1 Scope

Version 1 should be intentionally narrow:

- Support formulas only for `floatParameter` and `intParameter`
- Require formula support to be explicitly enabled per-parameter
- Support deterministic formulas only
- Support cross-parameter references by normalized parameter identifier
- Evaluate formulas after all input parameters have been read
- Interpret formula results in the parameter's preferred units

## Out Of Scope For V1

- Random expressions such as `random.uniform(...)`
- String, boolean, list, filename, or time-series formulas
- Unit-bearing literals inside formulas
- Arbitrary Python execution
- Mixing formula evaluation with extended input parsing for files/URLs/lists

## Proposed Input Syntax

Use an explicit formula marker:

```text
Reservoir Depth, = 2 * well_spacing + 0.5
```

Reasons:

- avoids ambiguity with existing scalar parsing
- preserves current behavior for plain numeric input
- keeps formula detection cheap and obvious

## Parameter Model Changes

Add metadata to `Parameter` in `src/geophires_x/Parameter.py`:

- `AllowFormulaInput: bool = False`
- `FormulaExpression: Optional[str] = None`
- `EvaluatedFromFormula: bool = False`

These fields should allow the parser to distinguish between:

- a literal input value
- a deferred formula
- a final resolved numeric value

## Evaluation Model

### Detection

During `ReadParameter`:

- if the parameter is not numeric, keep current behavior
- if the parameter does not allow formulas, keep current behavior
- if the value does not start with `=`, keep current behavior
- otherwise:
  - store the raw expression without the leading `=`
  - mark the parameter as provided
  - defer numeric conversion and validation

### Resolution

Add a second pass after `Model.read_parameters()` has completed the normal per-component reads.

The formula resolution pass should:

1. collect all input parameters from all component `ParameterDict`s
2. build a symbol table for formula-enabled numeric parameters
3. normalize parameter names to stable identifiers
4. evaluate formulas in dependency order
5. detect undefined symbols
6. detect cycles
7. assign the final numeric value
8. run normal range and enum validation on the resolved value

This avoids order-dependence during the normal parser read pass.

## Symbol Naming

Use normalized parameter identifiers derived from parameter names:

- `Reservoir Depth` -> `reservoir_depth`
- `Number of Production Wells` -> `number_of_production_wells`

Normalization should be strict and deterministic:

- lowercase
- replace non-alphanumeric runs with `_`
- collapse repeated `_`
- strip leading/trailing `_`

This should be implemented in one helper and used consistently in tests and error messages.

## Safe Expression Evaluation

Do not import Monte Carlo logic directly into the parser.

Instead, add a shared helper module, for example:

- `src/geophires_x/formula_evaluator.py`

Responsibilities:

- parse and validate expressions
- allow only safe mathematical constructs
- accept a symbol table of numeric values
- evaluate to `float`

Suggested allowed surface:

- numeric literals
- arithmetic operators
- parentheses
- a small set of math functions if needed later
- named variables from the supplied symbol table

Avoid:

- attribute access
- function imports
- arbitrary Python names
- Monte Carlo random behavior

## Validation Strategy

Resolved formulas should go through the same post-evaluation validation path as literal numeric values:

- `intParameter` should coerce to integer in the same way as ordinary input
- `floatParameter` should remain float
- min/max checks should still apply
- enum conversion for integer-backed enums should still apply

The formula system should not bypass existing validation rules.

## Failure Modes

Expected user-visible errors:

- parameter does not allow formula input
- unknown formula symbol
- circular formula dependency
- expression evaluates to non-finite value
- resolved value fails min/max validation

Each failure should include:

- parameter name
- raw formula text
- specific reason

## Test Plan

Add focused tests in `tests/test_parameter.py`:

- formula accepted for formula-enabled `floatParameter`
- formula accepted for formula-enabled `intParameter`
- formula references another numeric parameter
- formula resolution order is not dependent on input order
- unknown symbol raises a clear error
- circular dependency raises a clear error
- min/max validation still applies after evaluation
- non-formula-enabled parameter rejects `= ...` input
- plain numeric input remains unchanged

If formula resolution moves into `Model`, add integration-style tests for multi-parameter evaluation across component dictionaries.

## Recommended Rollout

### Phase 1

- add metadata and helper module
- add deferred formula detection in `ReadParameter`
- add second-pass formula resolution in `Model`
- enable the feature on a very small set of parameters for testing

### Phase 2

- widen parameter coverage deliberately
- add documentation and user examples
- consider optional math helpers if real use cases need them

### Phase 3

- decide whether any Monte Carlo random-expression syntax belongs in the main parser
- only do this if there is a clear reproducibility story

## Open Decisions

These should be settled before implementation expands beyond the first slice:

1. Which initial parameters should allow formulas?
2. Should formulas be allowed to reference only user inputs, or also derived/calculated values?
3. Should integer-valued formulas round, floor, or require exact integer results?
4. Should formulas be preserved in output/reporting metadata for traceability?

## Recommended First Slice

Implement the smallest useful vertical slice:

1. one helper for name normalization
2. one helper for safe expression evaluation
3. `AllowFormulaInput` on `Parameter`
4. deferred formula capture in `ReadParameter`
5. one model-level resolution pass
6. tests for two simple dependent numeric parameters

That is enough to prove the architecture without forcing a large parser rewrite.
