# VALCO Design Plan

## Basis

This design is based on:

- [VALCOEinGlobalEnergyandClimateModelDocumentation2025.pdf](D:/Work/malcolm-BSD/GEOPHIRES-X-parser-formulas/docs/reference/VALCOEinGlobalEnergyandClimateModelDocumentation2025.pdf)
- [XLCO-design.md](D:/Work/malcolm-BSD/GEOPHIRES-X-parser-formulas/docs/XLCO-design.md)

The source document is the IEA 2025 Global Energy and Climate Model documentation, Section 4.2,
`Value-adjusted Levelised Cost of Electricity`.

The source feature is `VALCOE`, not `VALCOH` or `VALCOC`.

That means this design intentionally separates two things:

- source-grounded `VALCOE`
- GEOPHIRES extensions `VALCOH` and `VALCOC` built by analogy to the same value-adjusted framework

Any implementation should state this boundary clearly in code comments, docs, and tests.

## Purpose

The goal is to add an optional `VALCO(E|H|C)` feature family to GEOPHIRES that complements baseline
`LCOE`, `LCOH`, and `LCOC` in the same way that `XLCO(E|H|C)` complements them, while remaining a distinct metric.

`VALCO` is not another ESG benefit calculation.

It is a competitiveness metric that adjusts baseline levelized cost by the relative system value of a technology.
Per the IEA formulation, that value has three components:

- energy value
- capacity value
- flexibility value

## Status

Current status:

- Phase 1 complete
- Phase 2 complete
- Phase 3 complete
- Phase 4 complete
- Phase 5 complete
- Phase 6 complete
- Phase 7 complete
- Phase 8 complete
- Phase 9 complete
- Phase 10 not started

Phase 1 completion date:

- 2026-04-17

Phase 2 completion date:

- 2026-04-17

Phase 3 completion date:

- 2026-04-17

Phase 4 completion date:

- 2026-04-17

Phase 5 completion date:

- 2026-04-17

Phase 6 completion date:

- 2026-04-17

Phase 7 completion date:

- 2026-04-17

Phase 8 completion date:

- 2026-04-17

Phase 9 completion date:

- 2026-04-17

## Relationship To XLCO

`VALCO` should follow the `XLCO` implementation pattern operationally, but not algebraically.

Shared implementation expectations:

- optional calculation flag
- parallel electricity, heat, and cooling outputs
- commodity-aware internal engine
- reuse of baseline levelized-cost bases
- support across all economic models that support baseline `LCO*`
- transparent tests and example inputs

Key differences from `XLCO`:

- `XLCO` modifies the discounted numerator with market and social benefit streams
- `VALCO` adjusts baseline `LCO*` by relative value terms
- `XLCO` has `Market` and `MarketSocial` outputs
- `VALCO` should have a single primary output per commodity plus optional component outputs

`VALCO` and `XLCO` should remain distinct calculations, but they should compose in a defined order.

Locked requirement:

- if `XLCO*` is not active for a commodity, `VALCO*` adjusts the baseline `LCO*`
- if `XLCO*` is active for a commodity, `VALCO*` adjusts the already-extended `XLCO*`

So the effective base for `VALCO` becomes:

- `LCO*` when `XLCO*` is disabled or unavailable
- `XLCO*_Market` when `XLCO*` is enabled and available

`VALCO` should not stack on `XLCO*_MarketSocial`.

Reason:

- `VALCO` is a competitiveness adjustment to project cost in market terms
- `XLCO*_Market` is the market-adjusted breakeven price
- `XLCO*_MarketSocial` already includes monetized social-value benefits, which would mix two different adjustment
  frameworks in a way that is harder to interpret

## Source Formula

From the IEA documentation, `VALCOE` is conceptually:

```text
VALCOE_m = LCOE_m
         + (E_bar - E_m)
         + (C_bar - C_m)
         + (F_bar - F_m)
```

where:

- `m` is a generation technology
- `E_m` is the technology-specific energy value
- `C_m` is the technology-specific capacity value
- `F_m` is the technology-specific flexibility value
- bars denote system-average values

Interpretation:

- if a technology provides above-average value, its `VALCO` is lower than its baseline `LCO`
- if it provides below-average value, its `VALCO` is higher than its baseline `LCO`

This is a competitiveness metric, not a revenue forecast and not a project cash-flow replacement.

## GEOPHIRES Adaptation

GEOPHIRES is not the IEA GEC Model and does not contain a built-in regional hourly market model.

So the design should preserve the IEA structure while adapting the inputs pragmatically.

Recommended rule:

- GEOPHIRES should implement the `VALCO` accounting identity directly
- but should not attempt in v1 to reproduce the IEA hourly market simulation internally
- GEOPHIRES should compute `VALCO` from the active commodity cost basis:
  - `LCO*` if `XLCO` is inactive
  - `XLCO*_Market` if `XLCO` is active

Instead, GEOPHIRES should support two calculation modes.

### Mode 1: Direct Value Inputs

This is the recommended v1 implementation.

Users provide component values directly in public cost units:

- system-average energy value
- technology energy value
- system-average capacity value
- technology capacity value
- system-average flexibility value
- technology flexibility value

Then GEOPHIRES computes:

```text
VALCO = ActiveBaseCost + (SystemEnergyValue - TechnologyEnergyValue)
                      + (SystemCapacityValue - TechnologyCapacityValue)
                      + (SystemFlexibilityValue - TechnologyFlexibilityValue)
```

where:

- `ActiveBaseCost = LCO` if `XLCO` is inactive
- `ActiveBaseCost = XLCO_Market` if `XLCO` is active

This is the cleanest way to stay faithful to the source structure without inventing a pseudo-hourly market model.

### Mode 2: Derived Component Inputs

This should be a later phase.

Users provide underlying terms from which GEOPHIRES derives component values.

For electricity this follows the IEA concepts:

- energy value from hourly or segmented prices and output
- capacity value from capacity credit, basis capacity value, and annual utilization
- flexibility value from flexibility multiplier, base flexibility value, and annual utilization

This mode should be added only after the direct-input mode is stable.

## Proposed Outputs

Primary outputs:

- `VALCOE`
- `VALCOH`
- `VALCOC`

Recommended transparency outputs:

- `VALCOE_EnergyAdjustment`
- `VALCOE_CapacityAdjustment`
- `VALCOE_FlexibilityAdjustment`
- `VALCOH_EnergyAdjustment`
- `VALCOH_CapacityAdjustment`
- `VALCOH_FlexibilityAdjustment`
- `VALCOC_EnergyAdjustment`
- `VALCOC_CapacityAdjustment`
- `VALCOC_FlexibilityAdjustment`

Optional diagnostic outputs for later phases:

- technology energy/capacity/flexibility values by commodity
- system-average energy/capacity/flexibility values by commodity

Display names:

- `Value-Adjusted Electricity Breakeven Price (VALCOE)`
- `Value-Adjusted Heat Breakeven Price (VALCOH)`
- `Value-Adjusted Cooling Breakeven Price (VALCOC)`

Units:

- match the baseline public units of `LCOE`, `LCOH`, and `LCOC`

## Proposed Inputs

General:

- `Do VALCO(E|H|C) Calculations`
- `VALCO Calculation Mode`
  - `Direct`
  - `Derived`

Electricity direct-value inputs:

- `VALCOE System Average Energy Value`
- `VALCOE Technology Energy Value`
- `VALCOE System Average Capacity Value`
- `VALCOE Technology Capacity Value`
- `VALCOE System Average Flexibility Value`
- `VALCOE Technology Flexibility Value`

Heat direct-value inputs:

- `VALCOH System Average Energy Value`
- `VALCOH Technology Energy Value`
- `VALCOH System Average Capacity Value`
- `VALCOH Technology Capacity Value`
- `VALCOH System Average Flexibility Value`
- `VALCOH Technology Flexibility Value`

Cooling direct-value inputs:

- `VALCOC System Average Energy Value`
- `VALCOC Technology Energy Value`
- `VALCOC System Average Capacity Value`
- `VALCOC Technology Capacity Value`
- `VALCOC System Average Flexibility Value`
- `VALCOC Technology Flexibility Value`

Recommended units:

- electricity: `cents/kWh`
- heat: same public unit as `LCOH`
- cooling: same public unit as `LCOC`

Later derived-input electricity parameters:

- `VALCOE Basis Capacity Value`
- `VALCOE Capacity Credit`
- `VALCOE Base Flexibility Value`
- `VALCOE Flexibility Multiplier`
- optional explicit price/output series inputs

Later derived-input heat and cooling parameters should mirror this shape, but remain clearly marked as
GEOPHIRES extensions rather than direct IEA terminology.

## Internal Architecture

Recommended new module:

- `src/geophires_x/valco.py`

Recommended internal types:

- `Commodity`
  - `electricity`
  - `heat`
  - `cooling`
- `ValueAdjustmentInputs`
  - system energy value
  - technology energy value
  - system capacity value
  - technology capacity value
  - system flexibility value
  - technology flexibility value
- `ValueAdjustmentResult`
  - active base cost
  - final `valco`
  - energy adjustment
  - capacity adjustment
  - flexibility adjustment

Recommended main entrypoints:

- `calculate_value_adjusted_levelized_costs(econ, model) -> dict[str, ValueAdjustmentResult]`
- `assign_value_adjusted_levelized_cost_outputs(econ, model) -> dict[str, ValueAdjustmentResult]`

Compatibility helpers:

- `calculate_valcoe_output(...)`
- `calculate_valcoh_output(...)`
- `calculate_valcoc_output(...)`

## Reuse Of Existing XLCO Infrastructure

The `VALCO` implementation should reuse:

- `build_levelized_cost_bases(...)` in [levelized_costs.py](D:/Work/malcolm-BSD/GEOPHIRES-X-parser-formulas/src/geophires_x/levelized_costs.py)
- the existing commodity names used by `xlco.py`
- the same economics-subclass write-back pattern used after the recent `XLCO` refactor
- the existing `XLCO` outputs as the preferred base when `XLCO` is active

The existing `XLCO` pattern already solved two important problems:

- centralized assignment into `Economics`, `SBTEconomics`, `AGSEconomics`, and `EconomicsAddOns`
- support across `FCR`, `STANDARDIZED_LEVELIZED_COST`, `BICYCLE`, `CLGS`, and `SAM_SINGLE_OWNER_PPA`

`VALCO` should build on that rather than re-solving it.

## Commodity Generalization

### Electricity

This is source-grounded.

`VALCOE` should follow the IEA structure directly.

### Heat

This is a GEOPHIRES extension.

Recommended interpretation:

- energy value = thermal service value relative to system-average heat value
- capacity value = contribution to peak heat adequacy
- flexibility value = ability to shift, ramp, or support system balancing in thermal service

### Cooling

This is a GEOPHIRES extension.

Recommended interpretation:

- energy value = cooling service value relative to system-average cooling value
- capacity value = contribution during peak cooling demand
- flexibility value = value of dispatchability or shiftability in meeting cooling load

For both heat and cooling, v1 should rely on user-specified component values rather than inferred market models.

## Mathematical Formulation For GEOPHIRES

For each active commodity `c`:

```text
ActiveBaseCost_c = LCO_c                              if XLCO_c is inactive
                 = XLCO_c_market                      if XLCO_c is active

EnergyAdjustment_c      = SystemEnergyValue_c      - TechnologyEnergyValue_c
CapacityAdjustment_c    = SystemCapacityValue_c    - TechnologyCapacityValue_c
FlexibilityAdjustment_c = SystemFlexibilityValue_c - TechnologyFlexibilityValue_c

VALCO_c = ActiveBaseCost_c
        + EnergyAdjustment_c
        + CapacityAdjustment_c
        + FlexibilityAdjustment_c
```

This design intentionally keeps the adjustments in the same public units as the corresponding baseline `LCO_c`.

That avoids extra unit-conversion complexity and makes the component outputs directly legible.

## Derived-Mode Formulas

These should be implemented after direct mode.

Electricity energy value:

```text
TechnologyEnergyValue = sum_h(price_h * output_h) / sum_h(output_h)
```

Electricity capacity value, adapted from the IEA formulation:

```text
TechnologyCapacityValue
  = CapacityCredit
  * BasisCapacityValue($/kW-yr)
  / AnnualMWhPerkWYear

AnnualMWhPerkWYear = CapacityFactor * 8760 / 1000
```

Electricity flexibility value:

```text
TechnologyFlexibilityValue
  = FlexibilityMultiplier
  * BaseFlexibilityValue($/kW-yr)
  / AnnualMWhPerkWYear
```

Heat and cooling derived-mode formulas should follow the same structure, replacing electric adequacy and
flexibility interpretations with thermal analogues.

## Design Decisions

These decisions are now locked for implementation:

1. `VALCO` uses `XLCO*_Market` as its cost base whenever `XLCO` is active for the same commodity
2. `VALCO` has one primary output per commodity, not market/social variants
3. v1 uses direct component-value inputs
4. v1 reuses existing baseline `LCO*` outputs rather than replacing them
5. `VALCOH` and `VALCOC` are explicit GEOPHIRES extensions modeled after `VALCOE`
6. the first implementation should expose component adjustments as outputs for transparency

## Scope Boundaries

### In Scope

- optional `VALCOE`, `VALCOH`, `VALCOC`
- support across all current economic models
- direct component-value inputs
- component adjustment outputs
- examples and tests

### Out Of Scope For Initial Delivery

- reproducing the full IEA hourly model inside GEOPHIRES
- network integration costs
- ancillary-service market simulation
- automatic regional market data lookup
- combined `XLCO + VALCO` hybrid metrics

## Validation Strategy

Validation should be split into three layers.

### Layer 1: Algebra Tests

- zero adjustments gives `VALCO == LCO`
- zero adjustments gives `VALCO == XLCO_Market` when `XLCO` is active
- higher-than-average technology value decreases `VALCO`
- lower-than-average technology value increases `VALCO`
- each component affects only its own adjustment output

### Layer 2: Commodity Tests

- electricity-only project computes `VALCOE`
- heat-only project computes `VALCOH`
- cooling-only project computes `VALCOC`
- cogeneration computes both `VALCOE` and `VALCOH`

### Layer 3: Model Coverage Tests

- `FCR`
- `STANDARDIZED_LEVELIZED_COST`
- `BICYCLE`
- `CLGS`
- `SAM_SINGLE_OWNER_PPA`

This mirrors the upgraded `XLCO` test approach.

## Examples

Recommended initial example files:

- `example_VALCOE.txt`
- `example_VALCOH.txt`
- `example_VALCOC.txt`

Recommended v1 example philosophy:

- do not claim these reproduce IEA system values directly
- instead show transparent, controlled component adjustments with easy-to-check arithmetic

## Phased Implementation Plan

### Phase 1: Design Lock

- finalize terminology
- lock direct-input v1 scope
- document that `VALCOH/C` are GEOPHIRES extensions

### Phase 2: Core Module

- create `valco.py`
- add internal dataclasses and calculation helpers
- add shared output-assignment helper
- add helper to select `LCO` vs `XLCO_Market` as the active cost base per commodity

### Phase 3: Economics Parameters

- add `Do VALCO(E|H|C) Calculations`
- add `VALCO Calculation Mode`
- add direct component-value input parameters for `E`, `H`, and `C`

### Phase 4: Outputs

- add `VALCOE`, `VALCOH`, `VALCOC`
- add component adjustment outputs
- wire outputs into `Outputs.py`, `OutputsRich.py`, and any AGS output surface

### Phase 5: Model Integration

- call `assign_value_adjusted_levelized_cost_outputs(...)` from:
  - `Economics`
  - `SBTEconomics`
  - `AGSEconomics`
  - `EconomicsAddOns`

### Phase 6: Tests

- add pure algebra tests
- add composition tests for `LCO -> VALCO`
- add composition tests for `LCO -> XLCO_Market -> VALCO`
- add electricity, heat, cooling, and cogeneration tests
- add model-coverage tests across all economic models

### Phase 7: Client And Schema Surfaces

- update client parsing
- update generated request/result schemas
- add surface tests

### Phase 8: Examples And Docs

- add example input files
- add output reference files if needed
- document interpretation and limitations

### Phase 9: Derived Mode

- add electricity derived-mode inputs
- support capacity-credit and flexibility derivations
- optionally add explicit series inputs for energy value

### Phase 10: Advanced Extensions

- evaluate whether heat and cooling should also get derived-mode support
- evaluate future interaction with hourly or segment-based dispatch models

## Recommended Implementation Order

1. Implement direct-input `VALCOE/H/C` with shared commodity-aware internals and explicit `XLCO_Market` composition.
2. Reuse the existing `levelized_costs.py` basis helper instead of touching baseline `LCO*` math.
3. Expose component adjustment outputs from day one to keep the metric auditable.
4. Wire all economics subclasses immediately so the feature works across every supported economic model.
5. Add schema/client/output surfaces only after core calculation behavior is stable.
6. Add derived-mode formulas only after direct mode is fully tested and documented.

## Open Questions

These should be answered explicitly before coding starts:

1. Should `VALCO Calculation Mode` be an enum parameter or implicit from which inputs are provided?
2. Should direct-value inputs be accepted only in public units, or also in internal normalized units?
3. Do we want component outputs only, or also the raw technology/system values as diagnostics?
4. Should `VALCO` be shown in the same output block as `LCO*` and `XLCO*`, or in a separate competitiveness block?
5. Do we want any v1 combined examples where both `XLCO` and `VALCO` are active simultaneously?
6. Should the active `VALCO` base cost be surfaced as an output for transparency?

## Recommended Answers

My recommendation:

1. use an explicit `VALCO Calculation Mode` enum
2. accept public units only in v1
3. expose component adjustments now; raw value diagnostics later
4. show `VALCO` near `LCO*` and `XLCO*`, but label it clearly as a competitiveness metric
5. keep `XLCO` and `VALCO` simultaneously computable, with `VALCO` adjusting `XLCO_Market` when available
6. add an output such as `VALCOE_BaseCostUsed` later only if debugging or auditability needs it

## Phase 1 Checklist

Phase 1 is complete. The following design-lock items are accepted:

- `[x]` naming:
  - `Do VALCO(E|H|C) Calculations`
  - `VALCO Calculation Mode`
  - `VALCOE`, `VALCOH`, `VALCOC`
- `[x]` composition rule:
  - `VALCO` uses `LCO*` when `XLCO` is off
  - `VALCO` uses `XLCO*_Market` when `XLCO` is on
  - `VALCO` never uses `XLCO*_MarketSocial` as its base
- `[x]` v1 scope:
  - direct-value input mode only
  - no hourly market simulation in v1
  - no network integration costs in v1
- `[x]` transparency scope:
  - component adjustments are exposed as outputs
  - final `VALCO*` outputs share baseline public units
- `[x]` commodity scope:
  - `VALCOE` is source-grounded
  - `VALCOH` and `VALCOC` are GEOPHIRES extensions by analogy
- `[x]` model scope:
  - `FCR`
  - `STANDARDIZED_LEVELIZED_COST`
  - `BICYCLE`
  - `CLGS`
  - `SAM_SINGLE_OWNER_PPA`
- `[x]` testing scope:
  - algebra tests
  - composition tests with and without `XLCO`
  - commodity tests
  - multi-economic-model tests
- `[x]` output-surface scope:
  - text outputs
  - rich outputs
  - client parsing
  - schema generation

## Immediate Next Step

Phase 10 should evaluate the next extension boundary:

- whether `VALCOH` and `VALCOC` should gain derived-mode analogs
- whether electricity energy-value derivation should support explicit time-series or segmented inputs
- how far GEOPHIRES should go toward dispatch-like value modeling without duplicating the external market model
