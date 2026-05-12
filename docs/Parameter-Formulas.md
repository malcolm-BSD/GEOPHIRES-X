# Parameter Formulas

GEOPHIRES supports formulas for a small, explicit set of numeric input parameters.

## Input Syntax

Provide a formula by starting the value with `=`.

Example:

```text
Number of Production Wells, = number_of_injection_wells * 1.5
```

Formulas are evaluated after the normal parameter read pass, so references are not order-dependent.

## Currently Supported Parameter Coverage

Formula input is currently enabled for these well-count parameters in the standard `WellBores` model:

- `Number of Production Wells`
- `Number of Injection Wells`
- `Number of Doublets`
- `Number of Injection Wells per Production Well`

This scope is intentionally narrow so the behavior can expand carefully.

## Supported Expression Surface

Formulas currently support:

- numeric literals
- arithmetic operators such as `+`, `-`, `*`, `/`, `%`, and `**`
- parentheses
- named variables derived from parameter names
- a small set of math functions: `abs`, `max`, `min`, `round`, `sqrt`

Examples:

```text
Number of Production Wells, = number_of_injection_wells * 1.5
Number of Injection Wells, = number_of_production_wells - 1
Number of Doublets, = max(2, number_of_injection_wells)
Number of Injection Wells per Production Well, = number_of_production_wells / 8
```

## Parameter Name References

Formulas reference parameters by normalized identifier:

- lowercase
- non-alphanumeric characters replaced with `_`
- repeated `_` collapsed

Examples:

- `Number of Production Wells` -> `number_of_production_wells`
- `Number of Injection Wells per Production Well` -> `number_of_injection_wells_per_production_well`

## Validation and Errors

Resolved formulas still go through normal parameter validation.

This means GEOPHIRES will raise an error if:

- a parameter does not allow formula input
- a formula references an unknown symbol
- formulas create a circular dependency
- an expression evaluates to a non-finite value
- the resolved value is outside the parameter's valid range

## Example Input File

See [tests/examples/example_parameter_formulas.txt](../tests/examples/example_parameter_formulas.txt) for a minimal example that uses formulas for linked well-count parameters.

## Notes

- Formula support is deterministic only.
- Unit-bearing literals inside formulas are not currently supported.
- Additional math helpers should only be added when a concrete use case needs them.
