# Variable Demand Operating Mode Design

## Objective

Extend GEOPHIRES-X to support a new `dispatchable` operating mode alongside the existing default `baseload` mode.

In `baseload` mode, the current behavior remains unchanged: the model assumes constant production flow and continuous operation, with reservoir decline driven by sustained extraction.

In `dispatchable` mode, the user supplies an hourly thermal demand profile at the point of use. GEOPHIRES-X then simulates plant operation against that demand using variable flow, variable runtime, or both, subject to geothermal system limits. The design must maximize code reuse, minimize disruption to the existing codebase, and preserve backward compatibility.

## Design Principles

1. Preserve the current baseload path unchanged.
2. Introduce dispatchable behavior as an additive operating mode, not as a rewrite of existing calculations.
3. Reuse existing demand-profile infrastructure by building on the historical array framework already present in `Parameter.py` and `SurfacePlant.py`.
4. Separate operating control from reservoir and plant physics.
5. Make transient dispatch simulation opt-in and model-specific.
6. Provide clean extension points for future shut-in regeneration and additional reservoir-model support.

## Scope

### In Scope for v1

- New operating mode selection: `Baseload` or `Dispatchable`
- Hourly thermal demand input using historical arrays
- Dispatch simulation using hourly timesteps
- Operation driven by both variable flow and variable runtime through a dispatch strategy
- Separate design-capacity and operating-profile concepts for economics
- Dispatch-aware outputs:
  - timestep produced temperature
  - demand served and unmet demand
  - timestep flow profile
  - dispatch-aware LCOH, LCOE, and LCOC where applicable
- Initial dispatchable reservoir-model support priority:
  - first: cylindrical
  - second: MPF / LHS / SF
  - third: UPP
  - fourth: SBT

### Explicitly Out of Scope for v1

- High-fidelity shut-in regeneration physics
- New solver code for every reservoir model
- Daily/weekly/monthly/seasonal dispatch input as a primary internal format
- Changes to default baseload sizing logic for existing runs
- Retrofitting unsupported models to silently behave as dispatchable

## Problem Statement

The current codebase is structured around a mostly steady production rate per well and precomputed decline-style profiles. That works for baseload operation but is not sufficient for demand-following operation because dispatch requires:

- a time-varying operating command
- a timestep-based operating loop
- delivered heat comparison against demand
- partial-load operation
- shut-in behavior
- capacity planning that distinguishes installed design from actual utilization

The core design challenge is not demand ingestion. It is introducing a new operating loop without destabilizing the current baseload calculation path.

## Proposed High-Level Solution

Add a new operating-mode abstraction above the existing reservoir, wellbore, surface plant, and economics objects.

### Operating Modes

- `Baseload`
  - current GEOPHIRES-X behavior
  - uses the existing calculation path unchanged

- `Dispatchable`
  - new hourly simulation path
  - uses historical-array hourly demand input
  - uses a dispatch strategy to choose flow and runtime each timestep
  - updates geothermal outputs over time

The dispatchable path should be implemented as a separate simulation workflow, not as conditional logic woven deeply into every existing calculation.

## Key Architectural Decision

The design introduces a new dispatch simulation layer, rather than modifying the current baseload methods to become dual-purpose.

That layer should:

1. read the hourly demand profile
2. compute an operating command for each timestep
3. evaluate geothermal capability for that operating command
4. calculate served and unmet demand
5. accumulate dispatch-aware profiles and annual aggregates

This keeps the current model objects reusable while minimizing invasive changes.

## Integration with Historical Arrays

GEOPHIRES-X already supports historical arrays and already defines demand-oriented parameters in `SurfacePlant`:

- `Annual Heat Demand`
- `Annual Cooling Demand`
- `Annual Electricity Demand`

The dispatchable-mode design will reuse that framework instead of adding a new parallel demand-input mechanism.

### Canonical v1 Input

For dispatchable geothermal heat operation, the canonical input is:

- an hourly historical array
- 8760 timesteps
- representing thermal demand at the point of use

### Historical Array Requirements for Dispatchable Mode

- The dispatch demand series must be stored as an hourly one-year array.
- Existing historical-array parsing, normalization, and optional resampling should be reused.
- For dispatchable heat mode, `Annual Heat Demand` becomes the required demand input.
- The dispatch loop should treat the array as the target thermal load delivered to the user, not as geothermal extraction demand.

### Backward Compatibility

- Existing historical-array support remains unchanged.
- Existing baseload cases that provide these demand arrays but do not opt into dispatchable mode retain their current behavior.
- Dispatchable mode introduces explicit new parameters; no existing parameter changes meaning.

## Proposed Object-Oriented Design

### New Concepts

The design introduces the following new responsibilities.

#### 1. Operating Mode Strategy

Purpose:
- select the correct simulation path without changing the current default behavior

Proposed interface:

- `OperatingModeStrategy`
  - `run(model) -> None`

Implementations:

- `BaseloadOperatingModeStrategy`
  - delegates to existing `Model.Calculate()` behavior

- `DispatchableOperatingModeStrategy`
  - runs the hourly dispatch simulation

This is the primary Strategy pattern in the design.

#### 2. Demand Profile Adapter

Purpose:
- provide a canonical hourly demand series to the dispatch loop
- isolate historical-array parsing/storage details from dispatch logic

Proposed interface:

- `DemandProfile`
  - `series`
  - `units`
  - `num_timesteps`
  - `time_step_hours`

- `DemandProfileFactory`
  - builds `DemandProfile` from existing `SurfacePlant.HeatingDemand`

This allows the dispatch loop to consume demand cleanly without depending on raw parameter internals.

#### 3. Dispatch Strategy

Purpose:
- decide how to meet demand using both flow and runtime

Proposed interface:

- `DispatchStrategy`
  - `dispatch(timestep_state, demand) -> DispatchCommand`

Proposed command object:

- `DispatchCommand`
  - `target_flow_fraction`
  - `runtime_fraction`
  - `is_shut_in`

For v1, one default strategy is sufficient:

- `DemandFollowingDispatchStrategy`
  - attempts to meet thermal demand
  - can vary both flow and runtime
  - respects min/max operating limits
  - caps output at geothermal capability
  - records unmet demand if demand exceeds capability

This keeps dispatch policy separate from the geothermal physics implementation.

#### 4. Dispatch-Capable Plant Adapter

Purpose:
- bridge the current model objects into a timestep-based interface without rewriting them all at once

Proposed interface:

- `DispatchPlantAdapter`
  - `initialize(model, design_state)`
  - `evaluate_timestep(dispatch_command, timestep_index) -> DispatchTimestepResult`
  - `finalize()`

`DispatchTimestepResult` should include:

- produced temperature
- plant outlet thermal power
- pumping power
- electrical output if applicable
- served thermal demand
- unmet thermal demand
- actual flow
- runtime fraction

This adapter layer is the key to minimizing codebase disruption.

#### 5. Dispatch State Store

Purpose:
- accumulate timestep outputs and annual summaries without changing existing baseload output arrays

Proposed object:

- `DispatchResults`
  - hourly produced temperature
  - hourly flow
  - hourly runtime fraction
  - hourly demand served
  - hourly unmet demand
  - hourly pumping power
  - hourly geothermal thermal output
  - annual aggregates
  - dispatch-specific summary metrics

## Reuse of Existing Components

### Reservoir and Wellbore Models

The current reservoir and wellbore classes should not be fully rewritten for v1.

Instead, v1 should introduce a dispatch-capability adapter per supported model family.

Examples:

- `CylindricalDispatchAdapter`
- `MPFDispatchAdapter`
- `LHSDispatchAdapter`
- `SFDispatchAdapter`
- `UPPDispatchAdapter`
- `SBTDispatchAdapter`

These adapters are responsible for mapping an hourly operating command onto the existing underlying model behavior.

### Surface Plant

Existing `SurfacePlant` and subclass logic should continue to be the source of plant-side heat and power calculations where feasible.

However, dispatchable mode should not assume the current annual-profile logic is sufficient. The dispatch path should expose an instantaneous or per-timestep plant-evaluation interface, even if that interface is initially implemented through reduced-order approximations derived from existing behavior.

### Economics

Existing economics classes remain the source of:

- CAPEX calculations
- standard OPEX logic
- LCOH/LCOE/LCOC framework

Dispatchable mode should extend economics with:

- separate design capacity metrics
- operating-profile-derived utilization metrics
- dispatch-aware annual delivered energy
- dispatch-aware unserved demand reporting

The economics layer should not own dispatch logic. It should consume dispatch results.

### Outputs

Existing output generation can be reused for baseload mode without change.

Dispatchable mode should add:

- dispatch summary section
- dispatch profile section
- optional hourly-profile export hooks

The output classes should read from `DispatchResults`, not rerun dispatch calculations.

## Proposed Minimal Changes to Existing Code

To minimize disruption, the following changes should be made at controlled extension points.

### 1. Model-Level Mode Selection

Add a new explicit input parameter:

- `Operating Mode`
  - default: `Baseload`
  - allowed values: `Baseload`, `Dispatchable`

`Model` should select the operating-mode strategy after input parsing and before calculation.

### 2. SurfacePlant Input Extension

Reuse the existing historical-array demand parameters instead of creating new raw input types.

Add dispatch-specific parameters near demand and plant-operation settings:

- `Dispatch Demand Source`
  - default: `Annual Heat Demand`
- `Dispatch Flow Strategy`
  - default: `Demand Following`
- `Maximum Dispatch Flow Fraction`
  - user-specified multiple of nominal flow
- `Minimum Dispatch Flow Fraction`
  - optional lower operating bound
- `Minimum Dispatch Runtime Fraction`
  - optional lower runtime threshold

These new parameters are additive and do not change the meaning of current parameters.

### 3. Reservoir/Plant Dispatch Adapter Registry

Add a small registry or factory that maps model family to a dispatch adapter.

Example:

- `DispatchAdapterFactory.create(model) -> DispatchPlantAdapter`

Behavior:

- if model supports dispatchable mode, return the appropriate adapter
- if unsupported, raise a clear error in dispatchable mode

This is preferable to scattering `if dispatchable:` blocks across model code.

### 4. Dispatch Results Storage

Add a new results object rather than overloading existing annual-profile arrays with mixed semantics.

This preserves the meaning of current outputs and reduces regression risk.

## v1 Dispatch Simulation Semantics

### Internal Timestep

The internal timestep for v1 is fixed at:

- 1 hour

This aligns with the user requirement and existing historical-array resampling design.

### Demand Interpretation

The dispatch demand series represents:

- thermal demand at the point of use

It does not represent:

- geothermal extraction demand
- requested wellhead heat rate
- surface plant gross thermal output

This distinction is important because served demand depends on produced temperature, flow, runtime, and plant-side conversion losses.

### Control Variables

The dispatch controller may use:

- variable flow
- variable runtime
- or both

For v1, the default strategy should allow both because the user explicitly wants dispatch to choose between them.

### Peak Demand and Oversizing

The model should allow:

- flow up to a user-specified multiple of nominal flow
- unmet demand when geothermal capability is insufficient

This preserves a realistic distinction between:

- what the demand requests
- what the geothermal system can actually provide

### Zero-Demand Timesteps

The conceptual long-term target is:

- true shut-in with thermal recovery

However, v1 fidelity is explicitly limited to a placeholder framework. Therefore:

- v1 should support shut-in semantics and state transitions
- v1 may set the system to zero production during zero-demand hours
- v1 should include explicit hooks for future thermal recovery
- v1 should not claim validated regeneration physics unless that model is actually implemented

## Reservoir Model Support Strategy

Dispatchable mode should be phased by model family.

### Tier 1: Cylindrical

First supported model family because it is the best candidate for a reduced-order dispatch implementation with minimal complexity.

### Tier 2: MPF / LHS / SF

Second priority because these are still reduced-order reservoir models and conceptually fit the dispatch adapter pattern.

### Tier 3: UPP

Supported later because it depends on externally supplied temperature profiles and therefore needs a clear interpretation under variable operation.

### Tier 4: SBT

Supported after the reduced-order families because of added geometry and transient complexity.

### Unsupported in v1

The following should be explicitly rejected in dispatchable mode unless separately implemented:

- SUTRA
- TOUGH2
- AGS / CLGS unless a dedicated dispatch adapter is added

Failing explicitly is safer than silent approximation.

## Economics Design

Dispatchable mode requires separate concepts for design and operation.

### Design Capacity

Design capacity should represent the system capability or installed design basis used for capital sizing.

Examples:

- nominal production flow
- peak allowed dispatch flow
- installed surface-plant capacity

### Operating Profile

Operating profile should represent actual simulated use over time.

Examples:

- hourly flow
- runtime fraction
- delivered thermal energy
- average and peak served demand
- annual capacity factor

### v1 Economics Approach

The dispatchable economics design should:

- continue using existing CAPEX logic wherever possible
- size design-related CAPEX from dispatch design capacity inputs
- calculate revenues and utilization from actual dispatch results
- calculate LCOH/LCOE/LCOC from dispatch-served annual energy

This preserves the current economics framework while introducing the required design-versus-operation separation.

## Outputs Design

Dispatchable mode should add a dedicated output section rather than altering the meaning of current baseload summaries.

### Mandatory v1 Dispatch Outputs

Priority order from user requirements:

1. timestep produced temperature
2. demand served and unmet demand
3. timestep flow profile
4. dispatch-aware LCOH / LCOE / LCOC

### Recommended Additional Outputs

- annual geothermal energy delivered
- annual unmet thermal demand
- dispatch capacity factor
- average runtime fraction
- peak geothermal contribution
- peak unmet load
- design flow versus observed peak flow

## Proposed Class Responsibilities

### New Classes

- `OperatingModeStrategy`
- `BaseloadOperatingModeStrategy`
- `DispatchableOperatingModeStrategy`
- `DemandProfile`
- `DemandProfileFactory`
- `DispatchStrategy`
- `DemandFollowingDispatchStrategy`
- `DispatchCommand`
- `DispatchPlantAdapter`
- `DispatchAdapterFactory`
- `DispatchResults`
- model-specific dispatch adapters

### Existing Classes with Minimal Extensions

- `Model`
  - choose operating-mode strategy

- `SurfacePlant`
  - expose dispatch-related parameters
  - provide access to canonical demand input

- `Economics`
  - consume dispatch results for utilization and levelized metrics

- `Outputs` / `OutputsRich`
  - render dispatch-specific summaries and profiles

## Sequence of Control in Dispatchable Mode

### Initialization

1. Parse input file
2. Build canonical hourly demand profile from historical arrays
3. Select dispatchable operating-mode strategy
4. Create the appropriate dispatch adapter for the selected reservoir/plant family
5. Establish design state:
   - nominal flow
   - max dispatch flow fraction
   - design capacity metrics

### Hourly Simulation Loop

For each hour:

1. read demand
2. build timestep state
3. ask dispatch strategy for a command
4. evaluate geothermal system response through the dispatch adapter
5. compute served and unmet demand
6. store timestep outputs in `DispatchResults`
7. update internal state for next hour

### Finalization

1. aggregate annual dispatch metrics
2. pass results into economics
3. render outputs

## Shut-In Regeneration Design Hook

Although true regeneration is not in scope for v1 fidelity, the design must reserve a clean extension point for it.

Recommended interface:

- `ReservoirRecoveryModel`
  - `update(state, dt_hours, is_shut_in) -> state`

v1 implementation:

- `NoRecoveryModel`

future implementation:

- reduced-order recovery model tuned by reservoir family

This allows v1 to implement dispatchable flow behavior now without closing the door on later recovery physics.

## Error Handling and Compatibility Rules

### Backward Compatibility

- `Baseload` remains the default operating mode
- existing input files keep their current meaning
- dispatchable mode requires explicit new parameters
- unsupported reservoir models in dispatchable mode should fail explicitly with a clear message

### Validation Rules for Dispatchable Mode

- hourly demand profile required
- dispatch demand must be thermal demand at point of use
- max dispatch flow fraction must be greater than or equal to nominal
- unsupported reservoir model plus `Dispatchable` operating mode is a fatal configuration error

## Risks

### Primary Technical Risks

1. Existing reservoir/wellbore code assumes scalar or fixed-profile flow in many locations.
2. Some plant and economics logic assume annual aggregates derived from baseload behavior.
3. Users may over-interpret v1 dispatchable results as including recovery physics when they do not.
4. UPP and SBT may require more model-specific adaptation than reduced-order families.

### Mitigations

1. Isolate dispatch through adapters and strategies.
2. Keep baseload path untouched.
3. Explicitly label v1 as variable-demand dispatch without validated regeneration.
4. Roll out support by model family in priority order.

## Recommended Implementation Phases

### Phase 1: Framework

- add operating mode parameter
- add dispatch strategy and demand-profile objects
- add dispatch results container
- integrate with historical arrays

### Phase 2: Cylindrical v1

- implement cylindrical dispatch adapter
- support hourly dispatch loop
- produce timestep outputs
- connect dispatch-aware economics and reporting

### Phase 3: Reduced-Order Expansion

- add MPF / LHS / SF adapters
- validate common dispatch abstractions

### Phase 4: UPP and SBT

- add model-specific adapters
- resolve model-specific transient assumptions

### Phase 5: Regeneration

- implement recovery model interface
- add reduced-order shut-in recovery behavior
- validate annual energy-recovery improvements

## Recommendation

This functionality should be implemented in the existing GEOPHIRES-X codebase, not in a new codebase.

The correct approach is a major but controlled extension:

- preserve the current baseload path
- add dispatchable operation as a distinct operating-mode strategy
- reuse historical arrays for hourly demand input
- use adapters and strategies to isolate transient dispatch behavior
- phase in model support by priority
- defer validated regeneration physics to a later phase

This approach maximizes code reuse, minimizes invasive changes, and keeps the current model behavior stable for existing users.
