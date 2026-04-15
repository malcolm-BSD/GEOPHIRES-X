# XLCOE Design Plan

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

The goal is to add `XLCOE` support to GEOPHIRES without disturbing existing `LCOE`, `LCOH`, and `LCOC` behavior.

The implementation should:

- preserve the existing investor-grade `LCOE`
- add explicit, traceable ESG cash-flow modifiers
- produce separate outputs for market-only and market-plus-social extended breakeven cost
- remain transparent enough for policy and investor discussion

This is not a request to replace `LCOE`.

## Locked Design Decisions

These decisions are fixed for v1:

1. Expose both `XLCOE_Market` and `XLCOE_MarketSocial`
2. Implement all five ESG categories in v1
3. Keep `XLCOE` electricity-only in v1
4. Make the first example mirror the paper's closed-loop California framing
5. Add tests and examples that reproduce the paper's published results-table values exactly

## Proposed Terminology

To stay faithful to the paper and avoid ambiguity, GEOPHIRES should treat `XLCOE` as a family of outputs rather than a single scalar with shifting meaning.

Recommended output names:

- `XLCOE_Market`
  - display name: `Extended Electricity Breakeven Price (XLCOE Market)`
- `XLCOE_MarketSocial`
  - display name: `Extended Electricity Breakeven Price (XLCOE Market + Social)`

Optional aliasing for presentation can still use `XLCOE`, but internally the code should keep the two variants distinct.

## Scope Boundaries

### In Scope

- electricity-focused extension of `LCOE`
- explicit period-by-period market and social ESG benefit streams
- configurable monetization inputs for the paper's five benefit categories
- discounting of social benefits using a distinct social discount rate
- outputs in electricity breakeven-price units consistent with `LCOE`

### Out Of Scope For Initial Delivery

- heat/cooling analogues such as `XLCOH` or `XLCOC`
- automatic regional data lookup for carbon, RECs, water, labor, or rig markets
- new policy optimization solvers
- probabilistic ESG distributions

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

- `tests/examples/example_XLCOE_paper_low.txt`
- `tests/examples/example_XLCOE_paper_high.txt`
- `tests/test_xlcoe.py`
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

- `Do XLCOE Calculations`
  - boolean
- `Social Discount Rate`
  - percent

#### Carbon Offset Inputs

- `Avoided Emissions Intensity`
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

- `src/geophires_x/xlcoe.py`

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

- create `xlcoe.py` helper module
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

The safest order is:

1. Phase 0
2. Phase 1
3. Phase 2 with carbon and REC first
4. Phase 3
5. complete idle rig discount before Phase 4
6. Phase 4
7. Phase 5
8. Phase 6

This sequencing avoids having the most ambiguous category block the rest of the feature.

## Immediate Next Step

Before implementation starts, the following should now be done from the locked decisions:

1. map the paper's low-value and high-value scenario assumptions into explicit GEOPHIRES-compatible inputs
2. determine whether standard GEOPHIRES closed-loop modeling alone can reproduce the paper's baseline `LCOE = $80/MWh`
3. define the exact expected output values and rounding rules for regression tests
4. decide whether the paper-validation example should live only in `tests/examples/` or also in user-facing docs/examples
