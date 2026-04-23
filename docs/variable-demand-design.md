# Variable Demand Operating Mode Design

## Objective

Extend GEOPHIRES-X with a `Dispatchable` operating mode alongside the existing default `Baseload` mode.

`Baseload` keeps the legacy behavior unchanged.

`Dispatchable` uses an hourly one-year demand profile, simulates geothermal response with variable flow and runtime, stores dispatch-specific hourly results, and feeds dispatch-aware annual energy into the existing economics framework.

This document reflects the current implementation state and the next planned extension for dispatchable electricity cases.

## Implemented Scope

### Operating Modes

- `Baseload`
  - existing GEOPHIRES-X calculation path
  - still the default

- `Dispatchable`
  - implemented as a separate operating-mode strategy
  - currently supports direct-use heat cases
  - currently supports the `Annual Heat Demand` demand source
  - currently supports the `Demand Following` dispatch flow strategy

### Planned Extension: Electricity Dispatch

The next dispatch extension will add support for pure electricity cases without changing the existing heat-dispatch behavior.

Planned initial scope:

- `EndUseOptions.ELECTRICITY`
- `Annual Electricity Demand` as a new hourly demand input
- `Demand Following` dispatch flow strategy
- pure power-generation plants first
- CHP remains out of scope for the first electricity-dispatch implementation

The current hard stop in `DispatchableOperatingModeStrategy` for non-heat cases remains correct until the items in the design section below are implemented.

### Supported Reservoir Families

Dispatchable mode is implemented for:

- `CylindricalReservoir`
- `MPFReservoir`
- `LHSReservoir`
- `SFReservoir`
- `UPPReservoir`
- `SBTReservoir`
  - `U-loop` and `EavorLoop` configurations only
  - coaxial SBT remains unsupported

Dispatchable mode still fails explicitly for unsupported reservoir families rather than silently approximating them.

### Demand Input

The canonical dispatch input is:

- `Annual Heat Demand`
- hourly historical array
- 8760 timesteps
- point-of-use thermal demand

The canonical regression asset is:

- `tests/assets/params/annual_heat_demand.csv`

The first column is hour, and the second column is thermal demand in `MMBtu`. The dispatch layer normalizes that series to thermal power in `MW`.

Historical-array file inputs now resolve relative paths against the input file location, so file-based examples can reference nearby demand CSV files directly.

### Planned Electricity Demand Input

Electricity dispatch will add a second canonical demand input:

- `Annual Electricity Demand`
- hourly historical array
- 8760 timesteps
- point-of-use electric demand

Canonical test asset for the initial implementation:

- `D:\temp\Annual_Electricity_Demand.csv`

Observed file structure:

- header row: `Time (hour),Demand (MW)`
- first column: hour index
- second column: electric demand in `MW`

The electricity-dispatch layer should treat that series as already being in delivered electric power units, without thermal unit conversion.

## Implemented Architecture

### Operating-Mode Strategy

Implemented in `Dispatch.py`:

- `OperatingModeStrategy`
- `BaseloadOperatingModeStrategy`
- `DispatchableOperatingModeStrategy`

`Model` selects the strategy after input parsing. This keeps the baseload path isolated from dispatch logic.

### Demand Profile Adapter

Implemented in `Dispatch.py`:

- `DemandProfile`
- `DemandProfileFactory`

This layer converts `Annual Heat Demand` into a canonical hourly `MW` series for dispatch use.

Planned extension:

- `DemandProfileFactory` should support both `Annual Heat Demand` and `Annual Electricity Demand`
- the returned profile should carry a demand kind, such as `thermal` or `electric`
- heat demand remains normalized from `MMBtu` to thermal `MW`
- electricity demand remains in electric `MW`

### Dispatch Strategy

Implemented in `Dispatch.py`:

- `DispatchStrategy`
- `DemandFollowingDispatchStrategy`
- `DispatchCommand`

The default strategy can modulate both:

- flow fraction
- runtime fraction

It respects:

- `Maximum Dispatch Flow Fraction`
- `Minimum Dispatch Flow Fraction`
- `Minimum Dispatch Runtime Fraction`

Planned extension:

- the strategy input should be generalized from thermal-only metrics to delivered-output metrics
- for heat cases, delivered output remains useful geothermal heat
- for electricity cases, delivered output becomes net electric power
- the demand-following algorithm itself can remain structurally the same once the adapter contract exposes the correct delivered-output quantity

### Dispatch Plant Adapters

Implemented in `Dispatch.py`:

- `DispatchPlantAdapter`
- `DispatchAdapterFactory`
- `CylindricalDispatchPlantAdapter`
- `AnalyticalReservoirDispatchPlantAdapter`
- `SBTDispatchPlantAdapter`

Current behavior by family:

- Cylindrical uses a reduced-order heat-content model.
- MPF/LHS/SF/UPP use a baseline-profile depletion mapping built from existing reservoir outputs.
- SBT uses the same baseline-profile approach with SBT-specific validation.

Planned extension for electricity:

- keep the reservoir-side dispatch adapters and recovery model structure
- extend the plant adapter contract so the same reservoir response can be evaluated against either:
  - useful thermal output for heat cases
  - net electric output for electricity cases

The smallest safe first scope is:

- `SurfacePlantSubcriticalOrc`
- `SurfacePlantSupercriticalOrc`

Flash and CHP configurations should remain unsupported until off-design power behavior is validated.

Planned adapter additions:

- a neutral timestep state object or dictionary that includes:
  - `useful_heat_mw`
  - `gross_electric_mw`
  - `net_electric_mw`
  - `pumping_power_mw`
  - `produced_temperature`
- a neutral delivered-output accessor used by the dispatch strategy
- preservation of the current thermal-only fields for backward compatibility during migration if needed

### Dispatch Results Store

Implemented in `Dispatch.py`:

- `DispatchResults`

Stored outputs include:

- hourly produced temperature
- hourly flow
- hourly runtime fraction
- hourly demand served
- hourly unmet demand
- hourly pumping power
- hourly geothermal thermal output
- hourly thermal demand
- annual dispatch aggregates
- dispatch summary metrics

Planned extension:

- add electricity-specific hourly outputs:
  - hourly electric demand
  - hourly net electric output
  - hourly unmet electric demand
  - hourly gross electric generation if useful for reporting
- either generalize thermal field names to neutral demand/output names or add parallel electric fields while preserving current heat outputs

## Dispatch Analysis Window

Dispatchable mode now supports an explicit operating-year analysis window.

Implemented surface-plant parameters:

- `Dispatch Analysis Start Year`
- `Dispatch Analysis End Year`

Semantics:

- `Start Year` is inclusive
- `End Year` is exclusive
- duration is implied by `End Year - Start Year`

Default behavior:

- `Dispatch Analysis Start Year = 1`
- `Dispatch Analysis End Year = 2`

That means the default dispatch analysis covers the first operating year after construction.

Example:

- `Dispatch Analysis Start Year = 3`
- `Dispatch Analysis End Year = 5`

means the stored dispatch analysis covers operating years 3 and 4.

Validation rules:

- start year must be `>= 1`
- end year must be `> start year`
- end year must be `<= Plant Lifetime + 1`

Implementation detail:

- reservoir and plant state are still advanced across the full operating life for consistency with annual economics
- the stored hourly dispatch results, dispatch CSV export, and dispatch summary metrics are sliced to the requested operating-year window

## Economics Integration

Dispatchable mode now feeds the existing economics framework instead of replacing it.

Implemented behavior:

- design-capacity metrics are computed from dispatch design conditions
- key heat/pumping CAPEX sizing uses dispatch design capacity instead of only realized utilization
- annual served heat from dispatch is passed into the standard heat-only economics path
- dispatch summary metrics report delivered and unmet heat separately

This preserves the existing economics structure while distinguishing:

- design capacity
- actual operating profile

Planned electricity-dispatch economics behavior:

- use dispatch design conditions to size power-plant CAPEX consistently with the heat-dispatch path
- feed annual delivered net electricity into the existing electricity economics path
- preserve pumping and parasitic loads in net generation accounting
- report unmet electric demand separately from generated electricity

## Recovery Model Hook

Implemented in `Dispatch.py`:

- `ReservoirRecoveryModel`
- `NoRecoveryModel`
- `ReducedOrderRecoveryModel`

Current behavior:

- shut-in hours can restore some reservoir state through a reduced-order recovery model
- cylindrical and analytical/baseline-profile adapters both use the recovery hook

This is still reduced-order behavior, not a validated high-fidelity regeneration model.

## Outputs

### Standard Output

Implemented in `Outputs.py` and `OutputsRich.py`:

- dedicated `DISPATCH RESULTS` section

Current dispatch summary output includes:

- dispatch analysis start year
- dispatch analysis end year
- dispatch analysis duration
- design heat produced
- annual geothermal heat delivered
- annual unmet thermal demand
- dispatch capacity factor
- average runtime fraction
- peak geothermal contribution
- peak unmet load
- peak hourly demand
- design flow rate
- observed peak flow rate

Dispatch summary field meanings:

- `Dispatch analysis start year`
  - first operating year included in the reported dispatch analysis window
  - inclusive

- `Dispatch analysis end year`
  - operating-year boundary at which the reported dispatch analysis window stops
  - exclusive

- `Dispatch analysis duration`
  - number of operating years in the reported dispatch analysis window
  - equal to `Dispatch analysis end year - Dispatch analysis start year`

- `Design heat produced`
  - geothermal useful thermal output used as the dispatch design basis
  - reported in `MW`
  - this is a design-capacity style metric, not an annual average

- `Annual geothermal heat delivered`
  - total geothermal heat actually served to the demand profile over the selected analysis window
  - normalized to `GWh/year` in the report

- `Annual unmet thermal demand`
  - total demand not served by the geothermal system over the selected analysis window
  - normalized to `GWh/year` in the report

- `Dispatch capacity factor`
  - average served thermal output divided by dispatch design heat output over the selected analysis window
  - reported as a percentage
  - this indicates how heavily the dispatch design capacity is used on average

- `Average runtime fraction`
  - average of the hourly runtime fraction over the selected analysis window
  - reported as a percentage
  - `100%` means the plant ran the full hour on average, while lower values indicate partial-hour runtime or shut-in behavior

- `Peak geothermal contribution`
  - maximum hourly geothermal heat actually delivered to the load during the selected analysis window
  - reported in `MW`

- `Peak unmet load`
  - maximum hourly demand shortfall during the selected analysis window
  - reported in `MW`

- `Peak hourly demand`
  - maximum hourly thermal demand requested by the dispatch profile during the selected analysis window
  - reported in `MW`

- `Design flow rate`
  - nominal production-well design flow used as the dispatch design basis
  - reported in `kg/s`
  - this is currently the nominal per-well production flow input used by the dispatch model

- `Observed peak flow rate`
  - highest actual hourly flow reached during the selected analysis window
  - reported in `kg/s`
  - this reflects dispatch behavior and may be lower than design flow if demand never requires the full design point

### Hourly Dispatch CSV Export

Implemented output parameter:

- `Dispatch Profile Output File`

Current CSV columns include:

- `Year`
- `Hour of Year`
- `Simulation Hour`
- `Thermal Demand (MW)`
- `Geothermal Thermal Output (MW)`
- `Demand Served (MW)`
- `Unmet Demand (MW)`
- `Produced Temperature (degC)`
- `Flow Rate (kg/s)`
- `Runtime Fraction`
- `Pumping Power (MW)`

The `Year` and `Simulation Hour` fields respect the selected dispatch analysis window.

Dispatch CSV column meanings:

- `Year`
  - operating year for that timestep within the selected dispatch analysis window

- `Hour of Year`
  - hour index within the operating year
  - ranges from `1` to `8760`

- `Simulation Hour`
  - absolute hour index within the full operating-life dispatch simulation
  - useful when the analysis window starts after operating year 1

- `Thermal Demand (MW)`
  - point-of-use hourly thermal demand presented to the dispatch solver

- `Geothermal Thermal Output (MW)`
  - geothermal thermal output produced during that timestep before unmet demand is accounted for

- `Demand Served (MW)`
  - portion of thermal demand actually met by geothermal supply during that timestep

- `Unmet Demand (MW)`
  - portion of thermal demand not met during that timestep

- `Produced Temperature (degC)`
  - geothermal produced-fluid temperature used for that timestep

- `Flow Rate (kg/s)`
  - actual production flow used during that timestep

- `Runtime Fraction`
  - fraction of the hour for which the plant operated
  - `1.0` means the plant ran for the full hour
  - `0.5` means it ran for half the hour
  - `0.0` means shut-in for the full hour

- `Pumping Power (MW)`
  - pumping power consumed during that timestep

### Optional HTML Dispatch Graphs

Implemented output parameter:

- `Generate Dispatch HTML Graphs`

When enabled with an HTML output file for a dispatchable run, the HTML output includes dispatch graphs for:

- demand, served, and unmet heat
- produced temperature and flow rate
- runtime fraction and pumping power

Planned electricity-dispatch outputs:

- demand, served, and unmet electricity
- gross and net electric generation
- produced temperature and flow rate
- runtime fraction and pumping power

## Implemented Input Parameters

Dispatchable mode currently uses these additive parameters:

- `Operating Mode`
- `Dispatch Demand Source`
- `Dispatch Flow Strategy`
- `Maximum Dispatch Flow Fraction`
- `Minimum Dispatch Flow Fraction`
- `Minimum Dispatch Runtime Fraction`
- `Dispatch Analysis Start Year`
- `Dispatch Analysis End Year`
- `Annual Heat Demand`

Planned additive input parameters for electricity dispatch:

- `Annual Electricity Demand`

Planned `Dispatch Demand Source` extension:

- `Annual Heat Demand`
- `Annual Electricity Demand`

Optional dispatch-related output parameters:

- `Dispatch Profile Output File`
- `Generate Dispatch HTML Graphs`

## Validation and Compatibility Rules

### Backward Compatibility

- `Baseload` remains the default
- existing non-dispatch runs keep their existing meaning
- dispatchable mode is opt-in

### Dispatch Validation

- `Annual Heat Demand` is required for dispatchable heat mode
- the demand profile must normalize to an hourly one-year series
- maximum dispatch flow fraction must be greater than or equal to minimum dispatch flow fraction
- dispatch analysis year window must lie within plant lifetime
- unsupported reservoir/configuration combinations fail explicitly

Planned electricity-dispatch validation:

- `Annual Electricity Demand` is required for dispatchable electricity mode
- the demand profile must normalize to an hourly one-year series
- dispatchable electricity initially supports only pure electricity cases
- dispatchable electricity initially supports only explicitly enabled plant families with validated off-design power behavior
- CHP remains unsupported until a combined objective and reporting model is defined

## Example Input

The canonical full-scale dispatch example is:

- `tests/geophires_x_tests/example1_dispatchable_full_scale.txt`

It demonstrates:

- dispatchable operating mode
- canonical hourly demand ingestion
- reduced-order MPF dispatch support
- dispatch analysis-year window selection
- dispatch summary reporting
- improved text output
- HTML output
- dispatch profile CSV export
- optional dispatch HTML graphs

## Remaining Limitations

- dispatchable mode currently supports direct-use heat only
- only `Annual Heat Demand` is implemented as a dispatch source
- only `Demand Following` is implemented as a dispatch strategy
- SBT dispatch excludes coaxial configuration
- shut-in recovery is reduced-order and not yet a validated reservoir-regeneration model
- unsupported reservoir families still fail explicitly in dispatchable mode

## Planned Electricity Dispatch Design

### Problem Statement

The current dispatch implementation is thermal-demand-centric. It assumes the dispatched commodity is geothermal heat delivered to the load.

That is why `DispatchableOperatingModeStrategy` currently rejects non-heat cases.

Electricity cases need a different dispatch target:

- thermal dispatch target: useful geothermal heat delivered
- electricity dispatch target: net electric power delivered

Removing the current `EndUseOptions.HEAT` guard without redesign would produce physically misleading results and incorrect unmet-demand accounting.

### Design Goals

- preserve current heat-dispatch behavior without regression
- add electricity dispatch as a second supported commodity path
- reuse the existing strategy and adapter architecture
- keep reservoir-side reduced-order behavior aligned with the current dispatch implementation
- avoid introducing fake thermal-demand proxies for electric dispatch

### Proposed Implementation Sequence

1. Add `Annual Electricity Demand` as a `TimeSeriesParameter`.
2. Extend `DispatchDemandSource` with `Annual Electricity Demand`.
3. Generalize `DemandProfile` so it carries both:
   - hourly series
   - demand kind: `thermal` or `electric`
4. Generalize the dispatch adapter contract to expose both thermal and electric performance metrics.
5. Change dispatch strategy inputs to use a neutral delivered-output metric.
6. Add dispatch results storage for electricity-specific hourly and annual metrics.
7. Extend reporting, CSV export, HTML graphs, and JSON output for electricity dispatch.
8. Relax the non-heat guard only for explicitly supported electricity plant types.

### Initial Supported Electricity Scope

First supported electricity-dispatch target:

- `EndUseOptions.ELECTRICITY`
- `SurfacePlantSubcriticalOrc`
- `SurfacePlantSupercriticalOrc`

Deferred:

- flash plants
- CHP
- hybrid objectives that co-optimize heat and electricity simultaneously

### Testing Asset

Use this file for initial electricity-dispatch testing and regression coverage:

- `D:\temp\Annual_Electricity_Demand.csv`

Expected characteristics:

- 8760 hourly entries
- hour index in column 1
- electric demand in `MW` in column 2

Recommended test coverage:

- parse and normalize `Annual Electricity Demand`
- dispatchable electricity run with ORC plant and valid annual electricity profile
- unmet electric demand accounting
- annual net electric generation handed to economics
- dispatch CSV and HTML outputs for electricity mode
- explicit failure for unsupported plant types and CHP

## Recommendation

The current implementation is the right shape for continued expansion:

- keep baseload untouched
- keep dispatch isolated through strategies and adapters
- continue using historical arrays as the canonical hourly demand path
- extend support reservoir-by-reservoir rather than introducing generic silent approximations
- treat recovery and additional demand/control modes as later extensions rather than overloading the current v1 implementation
