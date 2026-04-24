# XLCO Design Document

This document reflects the current implemented state of the `XLCO(E|H|C)` feature family on this branch.

## Basis

This design is based on:

- [FINAL-WGC2026-Reframing-ESG-as-Tangible-Economic-Inputs-to-Techno-Economic-Analysis-in-Geothermal-Projects.docx](D:/Work/malcolm-BSD/GEOPHIRES-X-parser-formulas/docs/reference/FINAL-WGC2026-Reframing-ESG-as-Tangible-Economic-Inputs-to-Techno-Economic-Analysis-in-Geothermal-Projects.docx)

This feature should be understood as a portability effort from Eavor-Suite into GEOPHIRES, not as a brand-new
economic concept invented independently inside GEOPHIRES.

Eavor-Suite and GEOPHIRES perform the same overall techno-economic role and are expected to produce similar results
when given the same project assumptions. For this effort, the working assumption is stronger than "similar":

- if the same closed-loop project inputs used for the paper's Eavor-Suite analysis are represented faithfully in
  GEOPHIRES
- then GEOPHIRES should reproduce the same baseline `LCOE`
- and the `XLCOE` extensions should build from that same baseline successfully

This framing materially affects validation:

- the primary target is not merely reproducing the paper's published `XLCOE` table algebraically
- the primary target is reproducing the Eavor-Suite-derived baseline and extended values in GEOPHIRES from equivalent
  inputs

If GEOPHIRES fails to match the Eavor-Suite baseline under equivalent assumptions, that mismatch must be treated as a
calibration or model-mapping problem before `XLCOE` implementation is considered validated.

The paper defines two extended levelized-cost outcomes:

- `XLCOEm`
  - baseline LCOE with market-priced ESG modifiers included in the numerator
- `XLCOEm,s`
  - `XLCOEm` plus monetized social-value modifiers, discounted separately

The paper's governing expressions are:

```text
XLCOEm   = sum_t((CapEx_t + OpEx_t - Bmarket_t) / (1 + r)^t) / sum_t(E_t / (1 + r)^t)
XLCOEm,s = [sum_t((CapEx_t + OpEx_t - Bmarket_t) / (1 + r)^t) - sum_t(Bsocial_t / (1 + rS)^t)] / sum_t(E_t / (1 + r)^t)
```

where:

- `Bmarket_t` aggregates market-priced ESG benefits
- `Bsocial_t` aggregates social-value ESG benefits
- `r` is the market discount rate
- `rS` is the social discount rate

## Purpose

The implemented feature adds `XLCOE` support to GEOPHIRES without disturbing existing `LCOE`, `LCOH`, and `LCOC`
behavior.

The implementation:

- preserve the existing investor-grade `LCOE`
- add explicit, traceable ESG cash-flow modifiers
- produce separate outputs for market-only and market-plus-social extended breakeven cost
- remain transparent enough for policy and investor discussion

This is not a request to replace `LCOE`.

## Locked Design Decisions

These decisions were fixed for the original electricity-only v1:

1. Expose both `XLCOE_Market` and `XLCOE_MarketSocial`
2. Implement all five ESG categories in v1
3. Keep `XLCOE` electricity-only in v1
4. Make the first example mirror the paper's closed-loop California framing
5. Add tests and examples that reproduce the paper's published results-table values exactly

These decisions are implemented in the generalized extension:

1. Preserve all existing `XLCOE_*` inputs and outputs for backward compatibility
2. Add parallel extended outputs for heat and cooling rather than overloading the electricity outputs
3. Generalize the implementation around commodity-specific extended levelized-cost calculations
4. Reuse the same market-versus-social split for electricity, heat, and cooling
5. Allocate shared project-level ESG benefits across active commodities using baseline discounted-cost share by default

## Implemented Terminology

To stay faithful to the paper and avoid ambiguity, GEOPHIRES treats extended levelized cost as a family of outputs
rather than a single scalar with shifting meaning.

Recommended output names:

- `XLCOE_Market`
  - display name: `Extended Electricity Breakeven Price (XLCOE Market)`
- `XLCOE_MarketSocial`
  - display name: `Extended Electricity Breakeven Price (XLCOE Market + Social)`
- `XLCOH_Market`
  - display name: `Extended Heat Breakeven Price (XLCOH Market)`
- `XLCOH_MarketSocial`
  - display name: `Extended Heat Breakeven Price (XLCOH Market + Social)`
- `XLCOC_Market`
  - display name: `Extended Cooling Breakeven Price (XLCOC Market)`
- `XLCOC_MarketSocial`
  - display name: `Extended Cooling Breakeven Price (XLCOC Market + Social)`

Optional aliasing for presentation can still use `XLCOE`, `XLCOH`, and `XLCOC`, but internally the code should keep the market and market-plus-social variants distinct.

When discussing the generalized implementation, this document will use `XLC*` as shorthand for:

- `XLCOE`
- `XLCOH`
- `XLCOC`

## Scope Boundaries

### In Scope

- electricity extension of `LCOE`
- heat extension of `LCOH`
- cooling extension of `LCOC`
- explicit period-by-period market and social ESG benefit streams
- configurable monetization inputs for the same five ESG benefit categories
- discounting of social benefits using a distinct social discount rate
- outputs in units consistent with the corresponding baseline `LCOE`, `LCOH`, or `LCOC`
- projects that produce any active combination of electricity, heat, and cooling

### Out Of Scope For Initial Delivery

- automatic regional data lookup for carbon, RECs, water, labor, or rig markets
- new policy optimization solvers
- probabilistic ESG distributions

## Implemented XLC Star Generalization

The original electricity-only `XLCOE` logic has been promoted into a generalized `XLC*` framework that supports
electricity, heat, cooling, or any active combination of them.

The governing pattern is the same for each active commodity:

```text
XLC_market(c)        = [PV(BaselineCost_c) - PV(MarketBenefits_c)] / PV(Output_c)
XLC_market_social(c) = [PV(BaselineCost_c) - PV(MarketBenefits_c) - PV(SocialBenefits_c)] / PV(Output_c)
```

where:

- `c` is one commodity in `{electricity, heat, cooling}`
- market benefits are discounted at the standard market rate `r`
- social benefits are discounted at the social rate `rS`
- the denominator uses the same discounted-output basis as the corresponding baseline levelized-cost calculation

This is an extension of the existing `XLCOE` structure, not a replacement of the baseline `LCOE`, `LCOH`, or `LCOC`
calculations.

## Implemented Internal Architecture

The implementation does not add separate one-off code paths for `XLCOE`, `XLCOH`, and `XLCOC`. Instead, it uses a
generalized commodity-aware internal model.

Implemented internal concepts include:

- `Commodity`
  - `ELECTRICITY`
  - `HEAT`
  - `COOLING`
- `LevelizedCostBasis`
  - commodity id
  - baseline public value
  - baseline preferred/current units
  - discounted output denominator
  - reconstructed baseline discounted cost numerator
  - active/inactive flag
- `ExtendedCostResult`
  - `market`
  - `market_social`

Implemented main internal entrypoints live in the shared levelized-cost support modules and commodity-specific XLCO
helpers.

Compatibility wrappers may still exist:

- `calculate_xlcoe_outputs(...)`
- `calculate_xlcoh_outputs(...)`
- `calculate_xlcoc_outputs(...)`

## Implemented Baseline Cost Basis Refactor

The original electricity-only implementation reconstructed the baseline discounted numerator from the already-computed
`LCOE` and the discounted electricity denominator. That approach was too narrow for a clean `XLCOH` / `XLCOC`
extension because:

- `LCOH` and `LCOC` use different denominators
- `LCOH` and `LCOC` are publicly reported in `$ / MMBTU`, not `cents / kWh`
- multiple end-use branches in `CalculateLCOELCOHLCOC(...)` already perform commodity-specific cost allocation

The generalized implementation now uses shared helpers that mirror the existing baseline logic for:

- electricity-only projects
- direct-use heat projects
- absorption-chiller cooling projects
- heat-pump heat projects
- district-heating heat projects
- cogeneration splits
- supported economic-model branches

That keeps the baseline and extended calculations aligned and avoids a second drifting implementation of cost
allocation logic.

## Unit Handling

The generalized implementation uses a single internal computational basis:

- internal basis: `cents / kWh-equivalent`

The public outputs are reported in the same units as the corresponding baseline commodity:

- electricity: `cents / kWh`
- heat: the same public units as `LCOH`
- cooling: the same public units as `LCOC`

This means:

- internal discounted numerators and denominators should be computed on a commodity-specific energy basis
- public unit conversion should happen only at the output boundary
- `XLCOH` and `XLCOC` should match the baseline `LCOH` / `LCOC` units exactly when all ESG modifiers are zero

## Benefit Mapping By Commodity

The five ESG categories remain the same, but they split into commodity-direct benefits and shared project-level
benefits.

### Commodity-Direct Market Benefits

- avoided-emissions benefit
- commodity credit benefit
  - electricity: REC-like credit
  - heat: renewable-thermal or heat-credit analogue
  - cooling: clean-cooling or cooling-credit analogue

### Commodity-Direct Social Benefits

- displaced-water benefit
- operations-jobs benefit tied to the active output scale of that commodity

### Shared Project-Level Benefits

- idle-rig discount
- construction-jobs benefit

The first generalized implementation should treat idle-rig discount and construction jobs as project-level benefits,
not duplicate them independently for electricity, heat, and cooling.

## Shared-Benefit Allocation Rule

Projects with more than one active commodity need a deterministic allocation rule for project-level benefits.

The recommended default is:

- allocate shared project-level market and social benefits by baseline discounted-cost share

For active commodity `c`:

```text
share_c = BaselineDiscountedCost_c / sum(BaselineDiscountedCost_active_commodities)
```

Then:

- `SharedMarketBenefit_c = share_c * SharedProjectMarketBenefit`
- `SharedSocialBenefit_c = share_c * SharedProjectSocialBenefit`

Rationale:

- the baseline levelized-cost logic already performs commodity-specific capital and O&M allocation
- discounted-cost share aligns better with investor-style levelized-cost framing than raw energy share
- energy-share allocation would mix commodities with different unit conventions and economic meaning

Future versions could expose the allocation basis as a user option, but v1 of `XLCOH` / `XLCOC` should keep the
allocation rule fixed.

## Input Parameter Extension

Backward compatibility requirements:

- retain all existing electricity `XLCOE_*` parameters unchanged
- preserve the current meaning of `Do XLCO(E|H|C) Calculations`
- allow a future alias such as `Do Extended Levelized Cost Calculations`, but do not require it initially

Recommended parameter structure:

### Shared Inputs

- `Social Discount Rate`
- `Idle Rig Discount Rate`
- `XLCO Construction Jobs Per Rig`
- `XLCO Indirect Jobs Multiplier`
- `XLCO Average Monthly Wage`

### Electricity Inputs

- retain existing:
  - `XLCOE Avoided Emissions Intensity`
  - `XLCO(E|H|C) Carbon Price`
  - `XLCOE REC Price`
  - `XLCOE Displaced Water Use Intensity`
  - `XLCO(E|H|C) Water Shadow Price`
  - `XLCO(E|H|C) Operations Jobs Per MW`

### Heat Inputs

- `XLCOH Avoided Emissions Intensity`
- `XLCO(E|H|C) Carbon Price`
- `XLCOH Thermal REC`
- `XLCOH Displaced Water Use Intensity`
- `XLCO(E|H|C) Water Shadow Price`
- `XLCO(E|H|C) Operations Jobs Per MW`

### Cooling Inputs

- `XLCOC Avoided Emissions Intensity`
- `XLCO(E|H|C) Carbon Price`
- `XLCOC Thermal REC`
- `XLCOC Displaced Water Use Intensity`
- `XLCO(E|H|C) Water Shadow Price`
- `XLCO(E|H|C) Operations Jobs Per MW`

Construction jobs and idle-rig discount should remain shared project-level inputs unless a later design revision
proves a commodity-specific split is needed.

## Output Surface Extension

The economics layer should expose the following additional scalar outputs:

- `XLCOH_Market`
- `XLCOH_MarketSocial`
- `XLCOC_Market`
- `XLCOC_MarketSocial`

Output behavior rules:

- electricity-only projects compute only `XLCOE_*`
- heat-only projects compute only `XLCOH_*`
- cooling-only projects compute only `XLCOC_*`
- cogeneration projects compute both `XLCOE_*` and `XLCOH_*`
- projects with inactive commodities should leave the corresponding extended outputs unset or zero, consistent with
  current `XLCOE` disabled behavior

The output surfaces that already carry `XLCOE` should be extended in parallel:

- `Outputs.py`
- `OutputsRich.py`
- `AGSOutputs.py` if applicable
- client result parsing
- schema generation

## Testing Requirements For XLCOH And XLCOC

Minimum required automated coverage:

- zero-modifier heat project: `XLCOH_Market == LCOH` and `XLCOH_MarketSocial == LCOH`
- zero-modifier cooling project: `XLCOC_Market == LCOC` and `XLCOC_MarketSocial == LCOC`
- positive heat market modifiers reduce `XLCOH_Market`
- positive cooling social modifiers reduce `XLCOC_MarketSocial`
- social discount rate changes only the `*_MarketSocial` variant
- mixed-output project computes extended outputs for each active commodity
- shared project-level benefits are allocated across active commodities by the documented cost-share rule
- schema and client surfaces include the new outputs and inputs

## ESG Category Mapping

The paper groups benefits into market and social categories.

### Market-Priced Benefits

- Carbon Offset
  - `Bcarbon = Iavoided * E * Pcarbon`
- Renewable Energy Credits
  - `BREC = E * PREC`
- Idle Rig Discount
  - modeled as a market-facing cost reduction / capital benefit

### Social-Value Benefits

- Water Offset
  - `Bwater = E * ΔVwater * Pwater`
- Jobs Benefit
  - `Bjobs = M * Njobs * W`

## GEOPHIRES Scenario Mapping

The paper's validation scenario is a closed-loop electricity project with California-facing ESG assumptions and a baseline result table of:

- low-value case
  - `LCOE = $80/MWh`
  - `XLCOEm = $69/MWh`
  - `XLCOEm,s = $51/MWh`
- high-value case
  - `LCOE = $80/MWh`
  - `XLCOEm = $34/MWh`
  - `XLCOEm,s = $11/MWh`

### Recommended Modeling Basis In GEOPHIRES

For v1 validation, the closest native GEOPHIRES mapping is:

- closed-loop modeling basis
  - `Reservoir Model = 8` (`SBT`)
  - `Is AGS = True`
- electricity-only project
  - `End-Use Option = 1`
  - `Power Plant Type = 2` unless paper-specific power block assumptions require another plant type
- investor-style levelized cost basis
  - `Economic Model = 2` (`Standard Levelized Cost`)

Rationale:

- the paper's equations are direct discounted cash-flow levelized-cost equations, which align more naturally with the existing standard levelized-cost path than with SAM-specific finance outputs
- the paper explicitly references a closed-loop project and Eavor-Suite exports
- GEOPHIRES's best native closed-loop analog is the `SBT` / `AGS` pathway

### Paper Assumption Mapping To GEOPHIRES Inputs

The paper does not provide a full GEOPHIRES-ready input deck, so the first implementation step must produce one.

#### Core Project Framing

Paper assumption:

- approximately `250 MW`
- developed over `5 years`
- connected to California grid
- simplified closed-loop geothermal project

GEOPHIRES mapping target:

- `Construction Years = 5`
- closed-loop reservoir/wellbore configuration via `SBT`
- plant sizing and well count calibrated so baseline `LCOE` reproduces `$80/MWh`

#### Economic Basis

Paper assumption:

- baseline and extended values are compared in `$ / MWh`

GEOPHIRES mapping target:

- internal GEOPHIRES `LCOE` output is in `cents/kWh`
- validation rules must explicitly convert:
  - `$80/MWh = 8.0 cents/kWh`
  - `$69/MWh = 6.9 cents/kWh`
  - `$51/MWh = 5.1 cents/kWh`
  - `$34/MWh = 3.4 cents/kWh`
  - `$11/MWh = 1.1 cents/kWh`

#### Market ESG Inputs

Paper assumptions:

- `Pcarbon = $35/tCO2` for both low and high cases
- `Iavoided = 0.44 tCO2/MWh`
- `PREC = $7/MWh` low, `$38/MWh` high
- idle rig discount `5%` low, `10%` high

GEOPHIRES mapping target:

- explicit XLCOE input parameters for these values
- carbon and REC benefits computed against annual electricity production
- idle rig discount applied to a well-defined drilling-capex basis

#### Social ESG Inputs

Paper assumptions:

- `Pwater = $0.25/m3` low, `$0.50/m3` high
- displaced water use `1.0 m3/MWh` low, `7.5 m3/MWh` high
- average wage `$4000/worker/month` low, `$7000/worker/month` high
- construction jobs `37/rig`
- operations jobs `1.7/MW`
- social discount rate `8%` low, `4%` high

GEOPHIRES mapping target:

- explicit XLCOE input parameters for these values
- water benefit computed against annual electricity generation
- jobs benefit split into:
  - construction-period stream
  - operations-period stream
- social benefit discounting performed with `rS`

## Validation Strategy

### Validation Objective

The implementation is not complete unless GEOPHIRES can reproduce the paper's published result table exactly, using
automated regression tests.

More specifically:

- GEOPHIRES should reproduce the same baseline `LCOE` as the Eavor-Suite scenario when fed equivalent inputs
- GEOPHIRES should then reproduce the same `XLCOE_Market` and `XLCOE_MarketSocial` values from those same scenario
  inputs plus the paper's ESG parameter sets

### Required Validation Artifacts

V1 should add all of the following:

- one paper-aligned example input for the low-value scenario
- one paper-aligned example input for the high-value scenario
- one test fixture that asserts:
  - baseline `LCOE`
  - `XLCOE_Market`
  - `XLCOE_MarketSocial`
- one documentation section that states the exact assumptions used to match the paper table

### Proposed File Layout

Recommended initial validation assets:

- `tests/examples/example_XLCOE_paper_low.tst`
- `tests/examples/example_XLCOE_paper_high.tst`
- `tests/test_xlco.py`
- `docs/XLCOE.md` or equivalent user-facing document later in Phase 5

### Exact Assertion Rules

The paper table values should be treated as exact published targets.

Recommended assertion targets:

- low-value case
  - `LCOE = 8.0 cents/kWh`
  - `XLCOE_Market = 6.9 cents/kWh`
  - `XLCOE_MarketSocial = 5.1 cents/kWh`
- high-value case
  - `LCOE = 8.0 cents/kWh`
  - `XLCOE_Market = 3.4 cents/kWh`
  - `XLCOE_MarketSocial = 1.1 cents/kWh`

Recommended test tolerance:

- start with exact values rounded to one decimal place in `cents/kWh`
- if GEOPHIRES's internal floating-point path requires more precision, keep the internal assertions tighter and only allow display-layer rounding at one decimal place

### Reproduction Path Decision

There are two implementation-valid approaches:

#### Path A: Native GEOPHIRES Reproduction

Use a normal GEOPHIRES closed-loop scenario whose baseline cost and production profile yield the published table once XLCOE modifiers are applied.

Advantages:

- strongest product integrity
- no special-case validation mechanism

Risk:

- the paper's baseline was generated from Eavor-Suite exports, so GEOPHIRES may not exactly match without substantial calibration

#### Path B: Explicit Paper Validation Fixture

Represent the paper scenario using explicit annualized series for:

- energy
- capex
- opex
- construction years
- ESG benefit streams

Advantages:

- exact reproduction is straightforward
- paper validation is decoupled from any single reservoir-model calibration

Risk:

- this is less "native GEOPHIRES" and more "GEOPHIRES XLCOE reproduces the paper's algebra"

### Recommended Validation Approach

Use a two-layer validation strategy:

1. native GEOPHIRES paper-aligned closed-loop examples using the same effective inputs as the Eavor-Suite case
   - proves GEOPHIRES reproduces the Eavor-Suite baseline and extended outputs
2. explicit paper algebra regression fixture if needed
   - guarantees exact reproduction of published table values even if there is a remaining model-mapping gap

If Path A alone cannot achieve exact reproduction, Path B should still be implemented so the published paper results are locked down in tests.

## GEOPHIRES Integration Points

The existing levelized-cost path is centered in:

- [Economics.py](D:/Work/malcolm-BSD/GEOPHIRES-X-parser-formulas/src/geophires_x/Economics.py)
  - `CalculateLCOELCOHLCOC(...)`
  - `Economics.Calculate(...)`
  - output parameter declarations for `LCOE`, `LCOH`, `LCOC`
- [Outputs.py](D:/Work/malcolm-BSD/GEOPHIRES-X-parser-formulas/src/geophires_x/Outputs.py)
- [OutputsRich.py](D:/Work/malcolm-BSD/GEOPHIRES-X-parser-formulas/src/geophires_x/OutputsRich.py)
- [AGSOutputs.py](D:/Work/malcolm-BSD/GEOPHIRES-X-parser-formulas/src/geophires_x/AGSOutputs.py)
- [SUTRAOutputs.py](D:/Work/malcolm-BSD/GEOPHIRES-X-parser-formulas/src/geophires_x/SUTRAOutputs.py)

Likely downstream exposure points:

- `src/geophires_x_client/`
- `src/geophires_x_schema_generator/`
- `tests/`

## Design Principles

### Preserve Existing Outputs

`LCOE` must remain unchanged unless users explicitly opt into `XLCOE` parameters or outputs.

### Keep Benefits First-Class and Traceable

Each ESG driver should be represented as its own parameter set and output stream, not buried in one opaque adjustment factor.

### Time-Resolved Before Aggregated

Benefits should be computed per project period where possible, then aggregated into discounted results.

### Separate Market and Social Streams

The paper's central distinction between investor-grade and policy-oriented value must remain visible in the implementation.

## Proposed Data Model

### New Input Parameters

The initial implementation should add explicit inputs for the five paper categories.

Recommended first-pass parameters:

#### Enablement and Discounting

- `Do XLCO(E|H|C) Calculations`
  - boolean
- `Social Discount Rate`
  - percent

#### Carbon Offset Inputs

- `XLCOE Avoided Emissions Intensity`
  - units: `tCO2/MWh`
- `Carbon Price`
  - units: currency per `tCO2`

#### REC Inputs

- `REC Price`
  - units: currency per `MWh`

#### Water Offset Inputs

- `Displaced Water Use Intensity`
  - units: `m3/MWh`
- `Water Shadow Price`
  - units: currency per `m3`

#### Idle Rig Inputs

- `Idle Rig Discount Rate`
  - percent
- possibly one of:
  - `Apply Idle Rig Discount To Drilling CAPEX`
  - boolean
  - or a more explicit drilling-cost basis parameter if the existing cost model requires it

#### Jobs Inputs

- `Construction Jobs Per Rig`
- `Operations Jobs Per MW`
- `Indirect Jobs Multiplier`
- `Average Monthly Wage`

### New Output Parameters

Recommended output families:

- scalar summary outputs
  - `XLCOE_Market`
  - `XLCOE_MarketSocial`
- annual benefit profiles
  - `Annual Carbon Offset Benefit`
  - `Annual REC Benefit`
  - `Annual Water Offset Benefit`
  - `Annual Jobs Benefit`
- aggregate benefit outputs
  - `Total Market ESG Benefit`
  - `Total Social ESG Benefit`

### Internal Helpers

Implementation should likely introduce one new helper module, for example:

- `src/geophires_x/xlco.py`

Responsibilities:

- compute annual market and social ESG benefit streams
- aggregate discounted benefit totals
- compute `XLCOE_Market` and `XLCOE_MarketSocial`

This avoids bloating `Economics.py` further.

## Proposed Rollout Plan

### Phase 0: Definition Lock

Goal:

- freeze the meaning of `XLCOE` before code changes

Tasks:

- confirm and document the locked v1 decisions
- confirm the exact output names to expose
- confirm that the paper's published result table is the reference validation target
- confirm whether social benefits should use `rS` by default, with optional override to market rate later

Deliverable:

- approved design document with frozen v1 scope

### Phase 1: Core Calculation Skeleton

Goal:

- add internal `XLCOE` scaffolding with no policy-heavy complexity yet

Tasks:

- create `xlco.py` helper module
- add new input parameters for enablement and social discount rate
- add output parameters for `XLCOE_Market` and `XLCOE_MarketSocial`
- wire calculation invocation into `Economics.Calculate(...)`
- keep outputs zero or unset when disabled

Tests:

- disabled path does not change `LCOE`
- outputs exist and serialize cleanly

Deliverable:

- non-breaking XLCOE calculation skeleton

### Phase 2: Market ESG Modifiers

Goal:

- implement the investor-grade portion from the paper

Tasks:

- add carbon offset calculation
- add REC benefit calculation
- add idle rig discount treatment
- compute `Bmarket_t`
- compute `XLCOE_Market`

Design notes:

- carbon and REC benefits are straightforward annual cash-flow offsets
- idle rig discount likely applies to drilling-related capital costs; the exact insertion point must be chosen carefully so it does not silently double-count existing cost adjustments
- all five categories are required in v1, so idle rig discount cannot be deferred out of the first release

Tests:

- each market component increases benefit monotonically with its driving input
- `XLCOE_Market <= LCOE` when market benefits are positive
- turning one component on affects only the expected result path

Deliverable:

- investor-grade XLCOE variant

### Phase 3: Social ESG Modifiers

Goal:

- implement the policy-oriented extension from the paper

Tasks:

- add water offset calculation
- add jobs benefit calculation
- compute `Bsocial_t`
- discount social benefits at `rS`
- compute `XLCOE_MarketSocial`

Design notes:

- jobs should likely be split into construction-period and operations-period streams
- water benefit should be driven by electric output and displaced water intensity

Tests:

- distinct social discount rate changes only the social-extended output
- `XLCOE_MarketSocial <= XLCOE_Market` when social benefits are positive
- annual profiles aggregate to the summary values

Deliverable:

- full paper-aligned XLCOE pair of outputs

### Phase 4: Output Surfaces and Schema

Goal:

- expose XLCOE cleanly everywhere users already see breakeven economics

Tasks:

- add XLCOE summary lines to `Outputs.py`
- add rich-output table entries to `OutputsRich.py`
- add AGS/SUTRA output handling if relevant
- update client/result parsers
- update schema generation

Tests:

- text output contains XLCOE when enabled
- client result parsing includes new fields
- schemas remain valid

Deliverable:

- user-visible XLCOE output support

### Phase 5: Documentation and Examples

Goal:

- make the feature understandable and reproducible

Tasks:

- add `XLCOE` user documentation
- document the paper equations as implemented in GEOPHIRES terms
- add one paper-aligned closed-loop California example
- add one minimal developer-focused example if needed for debugging
- clearly separate market and social interpretations
- document the exact assumptions used to reproduce the paper table

Deliverable:

- user-facing documentation and examples

### Phase 6: Calibration and Validation

Goal:

- ensure the numbers are defensible and understandable

Tasks:

- reproduce the paper's published results table exactly
- compare baseline `LCOE`, `XLCOE_Market`, and `XLCOE_MarketSocial`
- sensitivity-check each ESG parameter
- verify sign conventions and units carefully
- add automated tests asserting the exact low-value and high-value table results
- add example input(s) and expected output artifact(s) for the paper-aligned closed-loop California framing

Deliverable:

- validated reference scenario with exact-number regression coverage

## XLCOH And XLCOC Extension Rollout Plan

The following phased plan extends the completed `XLCOE` work into generalized `XLC*` support for heat and cooling
without rewriting the original electricity-only rollout history.

### Phase 7: Baseline Cost Basis Refactor

Goal:

- extract a commodity-aware baseline levelized-cost basis from the existing `LCOE` / `LCOH` / `LCOC` path

Tasks:

- refactor `CalculateLCOELCOHLCOC(...)` so the commodity-specific discounted numerator and denominator logic can be
  reused by extended-cost calculations
- introduce a helper such as `build_levelized_cost_bases(econ, model)`
- cover electricity, direct-use heat, cooling, district-heating, heat-pump, and cogeneration branches
- preserve existing public `LCOE`, `LCOH`, and `LCOC` values exactly

Tests:

- existing `LCOE`, `LCOH`, and `LCOC` regression suites remain unchanged
- zero-modifier electricity path still reproduces current `XLCOE`
- no output-unit regressions for heat and cooling branches

Deliverable:

- reusable commodity-level baseline-cost basis for all extended-cost variants

### Phase 8: Generalized Internal XLC Star Engine

Goal:

- replace the electricity-only internal helper with a commodity-aware `XLC*` engine

Tasks:

- generalize `xlco.py` internals around active commodities
- introduce internal concepts for:
  - commodity id
  - baseline cost basis
  - market benefit stream
  - social benefit stream
- keep compatibility wrappers for:
  - `calculate_xlcoe_outputs(...)`
  - future `calculate_xlcoh_outputs(...)`
  - future `calculate_xlcoc_outputs(...)`
- keep current electricity behavior byte-for-byte compatible at the public output layer

Tests:

- electricity-only tests continue to pass unchanged
- generalized helper rejects mismatched stream lengths cleanly
- inactive commodities return zero or unset outputs consistently

Deliverable:

- one internal engine capable of computing extended levelized cost for any supported commodity

### Phase 9: Heat And Cooling Input Model

Goal:

- add commodity-specific `XLCOH` and `XLCOC` monetization inputs while preserving existing electricity parameters

Tasks:

- keep all current `XLCOE_*` and legacy electricity input names unchanged
- add heat-specific inputs:
  - `XLCOH Avoided Emissions Intensity`
  - `XLCO(E|H|C) Carbon Price`
  - `XLCOH Thermal REC`
  - `XLCOH Displaced Water Use Intensity`
  - `XLCO(E|H|C) Water Shadow Price`
  - `XLCO(E|H|C) Operations Jobs Per MW`
- add cooling-specific inputs:
  - `XLCOC Avoided Emissions Intensity`
  - `XLCO(E|H|C) Carbon Price`
  - `XLCOC Thermal REC`
  - `XLCOC Displaced Water Use Intensity`
  - `XLCO(E|H|C) Water Shadow Price`
  - `XLCO(E|H|C) Operations Jobs Per MW`
- keep idle-rig discount and construction-jobs inputs shared at the project level

Tests:

- new parameters exist with correct defaults and units
- old electricity input names remain accepted
- schemas include the new heat and cooling inputs

Deliverable:

- complete input surface for electricity, heat, and cooling extended-cost monetization

### Phase 10: Heat And Cooling Output Computation

Goal:

- compute `XLCOH` and `XLCOC` outputs from the generalized engine

Tasks:

- add outputs:
  - `XLCOH_Market`
  - `XLCOH_MarketSocial`
  - `XLCOC_Market`
  - `XLCOC_MarketSocial`
- map direct market benefits by commodity
- map direct social benefits by commodity
- allocate shared project-level benefits using baseline discounted-cost share
- support active combinations of:
  - electricity only
  - heat only
  - cooling only
  - electricity plus heat
  - any other supported mixed-output combination

Tests:

- zero-modifier heat project yields `XLCOH == LCOH`
- zero-modifier cooling project yields `XLCOC == LCOC`
- positive heat market inputs reduce `XLCOH_Market`
- positive cooling social inputs reduce `XLCOC_MarketSocial`
- social discount rate changes only the `*_MarketSocial` outputs

Deliverable:

- first working `XLCOH` and `XLCOC` calculations

### Phase 11: Mixed-Commodity Allocation Validation

Goal:

- validate and lock down the shared-benefit allocation behavior for projects with more than one active output

Tasks:

- implement shared project-level benefit allocation using baseline discounted-cost share
- validate allocation in cogeneration cases
- validate allocation in heat-plus-cooling style branches if present
- document the fixed default allocation rule and why it was chosen over raw energy share

Tests:

- cogeneration computes both `XLCOE_*` and `XLCOH_*`
- shared idle-rig benefit affects each active commodity according to the documented allocation rule
- shared construction-jobs benefit affects each active commodity according to the documented allocation rule
- direct commodity benefits do not leak into unrelated commodity outputs

Deliverable:

- locked and tested mixed-output allocation behavior

### Phase 12: Output Surfaces, Client, And Schema For XLCOH And XLCOC

Goal:

- expose the new heat and cooling extended-cost outputs everywhere users already see breakeven economics

Tasks:

- add summary lines to `Outputs.py`
- add rich-output table entries to `OutputsRich.py`
- update AGS/SUTRA output handling if relevant
- update client/result parsing
- regenerate request/result schemas
- ensure unit presentation matches baseline `LCOH` / `LCOC` formatting

Tests:

- text outputs contain `XLCOH` and `XLCOC` when relevant
- client parsing includes new heat and cooling extended outputs
- schemas remain valid and up to date

Deliverable:

- fully surfaced `XLCOH` and `XLCOC` outputs

### Phase 13: Documentation, Examples, And Validation

Goal:

- document and validate the generalized `XLC*` feature family

Tasks:

- extend user-facing documentation from `XLCOE` to `XLC*`
- add at least one heat-oriented example and one cooling-oriented example
- add a mixed-output example if a realistic existing branch supports it
- document the shared-benefit allocation rule
- add regression tests for:
  - heat-only
  - cooling-only
  - mixed-output
- explicitly document that the original paper validates electricity directly, while `XLCOH` and `XLCOC` are
  GEOPHIRES-native generalizations of the same methodology

Deliverable:

- documented and regression-tested `XLCOE` / `XLCOH` / `XLCOC` family support

### Phase 13 Completion Notes

Phase 13 is now implemented in the repository.

Completed artifacts:

- user-facing example references in [GEOPHIRES-Examples.md](D:/Work/malcolm-BSD/GEOPHIRES-X-parser-formulas/docs/GEOPHIRES-Examples.md)
- heat-oriented example input:
  [example_XLCOH.txt](D:/Work/malcolm-BSD/GEOPHIRES-X-parser-formulas/tests/examples/example_XLCOH.txt)
- cooling-oriented example input:
  [example_XLCOC.txt](D:/Work/malcolm-BSD/GEOPHIRES-X-parser-formulas/tests/examples/example_XLCOC.txt)
- regression coverage for the documented heat and cooling example files
- output/client/schema coverage for surfaced `XLCOH` and `XLCOC` summary fields

Validation boundary:

- `XLCOE` is directly locked to the paper's published electricity table via explicit low/high fixtures
- `XLCOH` and `XLCOC` are GEOPHIRES-native generalizations of the same method
- there is no external published paper table for `XLCOH` or `XLCOC` in this design basis, so validation for those
  commodities is based on:
  - zero-modifier equivalence to baseline `LCOH` / `LCOC`
  - sign and isolation checks for commodity-specific market/social inputs
  - mixed-output allocation tests for shared project-level benefits
  - surfaced-output parsing and schema regression tests

## Implementation Status

The implementation represented by this document is now complete through Phase 13.

Completed sequence:

1. Phase 0: design freeze for electricity-only `XLCOE`
2. Phases 1-6: electricity implementation, surfacing, and paper validation
3. Phase 7: baseline cost-basis refactor
4. Phase 8: generalized internal `XLC*` engine
5. Phase 9: heat and cooling monetization inputs
6. Phase 10: `XLCOH` and `XLCOC` calculations
7. Phase 11: mixed-output allocation validation
8. Phase 12: text output, rich output, client, and schema surfacing
9. Phase 13: documentation, example inputs, and final validation pass

## Key Technical Risks

### Unit Consistency

The paper mixes:

- `$ / MWh`
- `cents / kWh`
- `$ / tCO2`
- `$ / m3`
- wages per worker-month

GEOPHIRES currently reports `LCOE` in `cents/kWh`. The implementation must convert all ESG-derived numerator benefits into a currency basis consistent with the existing discounted cash-flow calculations before dividing by discounted energy.

### CAPEX Interaction for Idle Rig Discount

This is the least direct category. If applied carelessly, it could:

- double-count drilling cost reductions
- bypass existing well-cost correlations
- produce unclear behavior when users also override drilling costs

This category needs the most care in implementation.

### Social Benefit Interpretation

The paper explicitly leaves room for interpretation about whether monetized social benefits should still be discounted socially once converted into incentive-like values. The first implementation should use the paper's baseline framing:

- market benefits discounted at `r`
- social benefits discounted at `rS`

but the design should leave room for a future option to use `r` for socialized cash flows if needed.

### Exact-Reproduction Risk

The paper describes:

- Eavor-Suite cash-flow and production exports
- a closed-loop California framing
- simplified but project-specific assumptions

GEOPHIRES must therefore explicitly address how the published results table will be reproduced exactly.

There are two plausible paths:

- encode a GEOPHIRES scenario whose internal calculations reproduce the published table directly
- create a paper-validation fixture with explicit annualized series derived from the paper scenario

If standard GEOPHIRES inputs alone cannot reproduce the exact published table, that derivation gap must be documented before implementation is considered complete.

## Recommended Implementation Order

The recommended order for this feature family is now:

1. freeze the electricity-only scope and paper-validation target
2. implement baseline `XLCOE` outputs before heat/cooling generalization
3. validate the electricity paper target independently of any heat/cooling work
4. extract the commodity-aware baseline cost basis before extending `XLCOE` internals
5. generalize the internal engine before adding new public heat/cooling calculations
6. add heat/cooling inputs before heat/cooling outputs
7. validate mixed-output allocation before surfacing new summary fields broadly
8. surface outputs only after the internal allocation rules are locked
9. finish with user-facing examples and explicit documentation of the electricity-versus-heat/cooling validation boundary

Actual completed order in this repository:

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. idle-rig completion before Phase 4 surfacing
6. Phase 4
7. Phase 5
8. Phase 6
9. Phase 7
10. Phase 8
11. Phase 9
12. Phase 10
13. Phase 11
14. Phase 12
15. Phase 13

This sequencing kept the paper-locked electricity implementation stable while the generalized `XLC*` support was
added in layers.

## Next Step

Implementation work described by this document is complete.

Reasonable follow-on work, if needed later:

1. add more user-facing narrative documentation outside this design document if `XLC*` becomes a headline feature
2. consider exposing the shared-benefit allocation basis as a configurable option if users need alternatives to
   discounted-cost share
3. add additional example variants for district-heating and heat-pump `XLCOH` scenarios if those branches become
   common support questions
