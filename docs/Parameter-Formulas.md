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

Formula input is also enabled for these geometry parameters in the `SBTWellbores` model:

- `Lateral Endpoint Depth`
- `Junction Depth`
- `Vertical Section Length`
- `Lateral Spacing`
- `Vertical Wellbore Spacing`
- `Discretization Length`

Formula input is enabled for these dimensionless performance parameters:

- `Water Loss Fraction`
- `Reservoir Porosity`
- `Drawdown Parameter`
- `Maximum Drawdown`
- `Overpressure Percentage`
- `Overpressure Depletion Rate`
- `Circulation Pump Efficiency`
- `Utilization Factor`
- `End-Use Efficiency Factor`
- `CHP Fraction`

Formula input is enabled for these thermophysical properties:

- `Reservoir Heat Capacity`
- `Reservoir Density`
- `Reservoir Thermal Conductivity`
- `Reservoir Permeability`
- `Water Thermal Conductivity`
- `Working Fluid Heat Capacity`
- `Working Fluid Density`
- `Working Fluid Thermal Conductivity`
- `Working Fluid Dynamic Viscosity`

Formula input is enabled for these passive wellbore dimensions and construction properties:

- `Production Well Diameter`
- `Injection Well Diameter`
- `Production Flow Rate per Well`
- `Well Separation`
- `Nonvertical Length per Multilateral Section`
- `Nonvertical Wellbore Diameter`
- `Lateral Casing Thermal Conductivity`
- `Lateral Cement Thermal Conductivity`
- `Lateral Casing Thickness`
- `Lateral Cement Thickness`

Formula input is enabled for these independent economic percentages:

- `Contingency Percentage`
- `Indirect Capital Cost Percentage`
- `Well Drilling and Completion Indirect Capital Cost Percentage`
- `Reservoir Stimulation Indirect Capital Cost Percentage`

This scope remains explicit so parameters with read-time conversions, model-selection effects, or cross-parameter
validation are not inadvertently enabled.

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
Number of Doublets, "= max(2, number_of_injection_wells)"
Number of Injection Wells per Production Well, = number_of_production_wells / 8
Junction Depth, = lateral_endpoint_depth - 2000
Vertical Section Length, = junction_depth - 1000
Discretization Length, "= min(250, lateral_spacing)"
Reservoir Porosity, = water_loss_fraction * 2
Injection Well Diameter, = production_well_diameter + 0.5
Lateral Cement Thickness, = lateral_casing_thickness * 2
Contingency Percentage, = indirect_capital_cost_percentage - 5
```

Input files are comma-delimited, so quote a formula when the expression itself contains a comma.

## Parameter Name References

Formulas reference parameters by normalized identifier:

- lowercase
- non-alphanumeric characters replaced with `_`
- repeated `_` collapsed

Examples:

- `Number of Production Wells` -> `number_of_production_wells`
- `Number of Injection Wells per Production Well` -> `number_of_injection_wells_per_production_well`
- `Lateral Endpoint Depth` -> `lateral_endpoint_depth`

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
- Formula results for the supported `SBTWellbores` geometry parameters are interpreted in meters.
- More generally, formula results and referenced values use each parameter's current internal units, just like a
  unitless literal input. Formulas do not perform dimensional analysis or automatic unit conversion between symbols.
- Additional math helpers should only be added when a concrete use case needs them.
