# GEOPHIRES Examples

## Examples List

View the list of all GEOPHIRES examples in the [README](https://github.com/NREL/GEOPHIRES-X?tab=readme-ov-file#examples)
or in the [web interface](https://gtp.scientificwebservices.com/geophires) under the Examples tab.

## Parameter Formulas Example

Formula-enabled well-count examples are documented in [Parameter Formulas](Parameter-Formulas.html).
A minimal example input file is available at
[`tests/examples/example_parameter_formulas.txt`](../tests/examples/example_parameter_formulas.txt).

## Case Study: 500 MW EGS Project Modeled on Fervo Cape Station

See documentation: [Case Study: 500 MWe EGS Project Modeled on Fervo Cape Station](Fervo_Project_Cape-5.html).

## Extended Levelized Cost Examples

`XLC*` examples are available in `tests/examples/`:

- heat-focused example:
  [`tests/examples/example_XLCOH.txt`](../tests/examples/example_XLCOH.txt)
- cooling-focused example:
  [`tests/examples/example_XLCOC.txt`](../tests/examples/example_XLCOC.txt)
- electricity paper-validation fixtures:
  [`tests/examples/example_XLCOE_paper_low.tst`](../tests/examples/example_XLCOE_paper_low.tst)
  and
  [`tests/examples/example_XLCOE_paper_high.tst`](../tests/examples/example_XLCOE_paper_high.tst)

Important validation boundary:

- `XLCOE` is directly regression-locked to the published paper table through explicit validation fixtures
- `XLCOH` and `XLCOC` are GEOPHIRES-native extensions of the same market/social methodology, validated through
  direct-use, cooling, and mixed-output regression tests rather than a published external paper target
