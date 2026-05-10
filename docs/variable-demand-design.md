# Variable Demand Operating Mode Design

## Objective

Extend GEOPHIRES-X with a `Dispatchable` operating mode alongside the existing `Baseload` mode.

`Baseload` preserves the legacy calculation path.

`Dispatchable` consumes a one-year hourly demand profile, simulates variable flow and runtime, stores dispatch-specific hourly and annual results, and feeds dispatch-aware annual production into the existing economics framework.

This document reflects the current implemented state.

## Implemented Scope

### Operating Modes

- `Baseload`
  - legacy GEOPHIRES-X behavior
  - remains the default

- `Dispatchable`
  - implemented as a separate operating-mode strategy in `Dispatch.py`
  - supports heat, cooling, electricity, and CHP dispatch targets through the same demand-following framework
  - currently supports only the `Demand Following` dispatch flow strategy

### Supported End-Use / Plant Combinations

Implemented dispatch support includes:

- `EndUseOptions.HEAT`
  - `PlantType.INDUSTRIAL`
  - `PlantType.HEAT_PUMP`
  - `PlantType.DISTRICT_HEATING`
  - `PlantType.ABSORPTION_CHILLER`

- `EndUseOptions.ELECTRICITY`
  - `PlantType.SUB_CRITICAL_ORC`
  - `PlantType.SUPER_CRITICAL_ORC`
  - `PlantType.SINGLE_FLASH`
  - `PlantType.DOUBLE_FLASH`

- CHP end-use options
  - topping, bottoming, and parallel CHP cases
  - same supported dispatchable electric plant types as pure electricity

### Supported Demand Sources

Implemented dispatch demand sources:

- `Annual Heat Demand`
  - canonical thermal dispatch input
  - used for industrial heat, heat pump, and district heating dispatch

- `Annual Cooling Demand`
  - cooling-demand dispatch input
  - used for absorption chiller dispatch

- `Annual Electricity Demand`
  - electric-demand dispatch input
  - used for pure electricity and CHP electric-following dispatch

Canonical regression assets:

- `tests/assets/params/annual_heat_demand.csv`
- `tests/assets/params/annual_cooling_demand.csv`
- `tests/assets/params/annual_electricity_demand.csv`

All dispatch demand profiles normalize to hourly one-year series with 8760 timesteps.

Historical-array file inputs resolve relative paths against the input file location, so example input decks can reference nearby CSV assets directly.

### Supported Reservoir Families

Dispatchable mode is implemented for:

- `CylindricalReservoir`
- `MPFReservoir`
- `LHSReservoir`
- `SFReservoir`
- `UPPReservoir`
- `SBTReservoir`
  - `U-loop` and `EavorLoop` only
  - coaxial SBT remains unsupported

Unsupported reservoir families still fail explicitly.

## Implemented Architecture

### Operating-Mode Strategy

Implemented in `Dispatch.py`:

- `OperatingModeStrategy`
- `BaseloadOperatingModeStrategy`
- `DispatchableOperatingModeStrategy`

`Model` selects the strategy after parameter parsing.

### Demand Profile Adapter

Implemented in `Dispatch.py`:

- `DemandProfile`
- `DemandProfileFactory`

This layer normalizes:

- heat demand to delivered thermal `MW`
- cooling demand to delivered cooling `MW`
- electricity demand to delivered electric `MW`

The returned profile carries a demand kind:

- `thermal`
- `cooling`
- `electric`

### Dispatch Strategy

Implemented in `Dispatch.py`:

- `DispatchStrategy`
- `DemandFollowingDispatchStrategy`
- `DispatchCommand`

The strategy modulates:

- flow fraction
- runtime fraction

It respects:

- `Maximum Dispatch Flow Fraction`
- `Minimum Dispatch Flow Fraction`
- `Minimum Dispatch Runtime Fraction`

The dispatch strategy operates on a neutral delivered-output quantity supplied by the plant adapter:

- useful heat for heat-following cases
- cooling output for absorption chiller cases
- net electric output for electricity-following cases

### Dispatch Plant Adapters

Implemented in `Dispatch.py`:

- `DispatchPlantAdapter`
- `DispatchAdapterFactory`
- `CylindricalDispatchPlantAdapter`
- `AnalyticalReservoirDispatchPlantAdapter`
- `SBTDispatchPlantAdapter`

Current behavior by family:

- `CylindricalReservoir` uses a reduced-order heat-content model.
- `MPF/LHS/SF/UPP` use a baseline-profile depletion mapping derived from existing reservoir outputs.
- `SBT` uses the analytical/baseline approach with SBT-specific validation.

Plant-side dispatch output state now carries:

- `useful_heat_mw`
- `cooling_produced_mw`
- `gross_electricity_mw`
- `net_electricity_mw`
- `heat_pump_electricity_mw`
- `pumping_power_mw`
- `produced_temperature_c`

### Dispatch Results Store

Implemented in `Dispatch.py`:

- `DispatchResults`

Stored hourly outputs now include:

- produced temperature
- flow
- runtime fraction
- demand served
- unmet demand
- pumping power
- geothermal thermal output
- geothermal electric output
- gross electric output
- cooling output
- heat-pump electricity use
- heat extracted
- dispatch demand

Stored annual outputs include:

- annual aggregates used by JSON/schema output
- dispatch summary metrics used by text/HTML output and economics sizing

## Direct-Use Heat Specialization in Dispatch

Dispatch now reconstructs plant-type-specific output arrays after the dispatch timestep simulation so the existing economics and reporting stack sees the correct state.

### Heat Pump

Dispatch computes:

- geothermal heat extracted
- useful heat delivered
- heat-pump electricity consumption

Dispatch summary and JSON outputs now include:

- `annual_heat_pump_electricity_kwh`
- `design_heat_pump_electricity_consumed_mw`

### Absorption Chiller

Dispatch follows `Annual Cooling Demand`.

Dispatch computes:

- geothermal heat delivered to the chiller
- cooling output
- unmet cooling demand

Dispatch summary and JSON outputs now include:

- `annual_served_cooling_kwh`
- `annual_unmet_cooling_kwh`
- `design_cooling_produced_mw`

### District Heating

Dispatch uses the generic `Annual Heat Demand` profile rather than the baseload district-heating demand synthesis path.

After dispatch, district-heating-specific outputs are reconstructed from the served and unmet hourly heat profile, including:

- daily heating demand
- annual heating demand
- geothermal district-heating contribution
- peaking-boiler contribution
- annual peaking-boiler energy
- peak peaking-boiler demand

Dispatch summary and JSON outputs now include:

- `annual_district_heating_boiler_kwh`
- `peak_district_heating_boiler_mw`

## Dispatch Analysis Window

Dispatchable mode supports an explicit operating-year analysis window.

Implemented parameters:

- `Dispatch Analysis Start Year`
- `Dispatch Analysis End Year`

Semantics:

- start year is inclusive
- end year is exclusive
- duration equals `End Year - Start Year`

Default behavior:

- `Dispatch Analysis Start Year = 1`
- `Dispatch Analysis End Year = 2`

Validation rules:

- start year must be `>= 1`
- end year must be `> start year`
- end year must be `<= Plant Lifetime + 1`

Implementation detail:

- reservoir and plant state still evolve across the full operating life
- stored hourly dispatch results, CSV export, and summary metrics are sliced to the requested analysis window

## Economics Integration

Dispatchable mode feeds the existing economics framework instead of replacing it.

Implemented behavior:

- dispatch design metrics size key plant and pumping CAPEX terms
- annual served heat, cooling, and electricity propagate into the existing economics paths through reconstructed surface-plant arrays
- specialized direct-use economics are supported for:
  - heat pumps
  - absorption chillers
  - district heating

This preserves the legacy economics structure while separating:

- design capacity
- actual dispatch utilization

## Outputs

### Standard Output

Implemented in `Outputs.py` and `OutputsRich.py`:

- dedicated `DISPATCH RESULTS` section

The dispatch report now includes the original heat and electricity metrics plus specialized plant-type metrics where applicable, including:

- `Design heat produced`
- `Design cooling produced`
- `Design net electricity produced`
- `Annual geothermal heat delivered`
- `Annual geothermal cooling delivered`
- `Annual geothermal electricity delivered`
- `Annual unmet thermal demand`
- `Annual unmet cooling demand`
- `Annual unmet electricity demand`
- `Annual heat pump electricity consumed`
- `Annual peaking boiler heat delivered`
- `Dispatch capacity factor`
- `Average runtime fraction`
- `Peak geothermal contribution`
- `Peak unmet load`
- `Peak hourly demand`
- `Peak peaking boiler demand`
- `Design flow rate`
- `Observed peak flow rate`

### JSON Output

Implemented in `GEOPHIRESv3.py` and schema generation:

- `Dispatch Summary`

The structured dispatch JSON now includes:

- `schema_version`
- `demand_type`
- `surfaceplant_mode`
- `dispatch_settings`
- `analysis_window`
- `summary_metrics`
- `annual_aggregates`

Supported `demand_type` values:

- `thermal`
- `cooling`
- `electric`

Specialized dispatch JSON metrics now include:

- cooling served / unmet
- heat-pump electricity use
- district-heating boiler energy and peak demand

### Hourly Dispatch CSV Export

Implemented parameter:

- `Dispatch Profile Output File`

The CSV export remains commodity-aware:

- heat-following runs emit thermal demand/output columns
- electricity-following runs emit electric demand/output columns
- cooling-following runs emit cooling demand/output columns through the same dispatch-results infrastructure

The `Year` and `Simulation Hour` fields respect the selected dispatch analysis window.

### Optional HTML Dispatch Graphs

Implemented parameter:

- `Generate Dispatch HTML Graphs`

When enabled with HTML output for a dispatchable run, the HTML output includes dispatch graphs for:

- demand, served, and unmet heat
- demand, served, and unmet electricity
- demand, served, and unmet cooling
- produced temperature and flow rate
- runtime fraction and pumping power or delivered commodity output, depending on mode

Graph file names are namespaced by HTML output filename stem to avoid collisions between multiple runs in the same directory.

## Implemented Input Parameters

Dispatchable mode uses these additive parameters:

- `Operating Mode`
- `Dispatch Demand Source`
- `Dispatch Flow Strategy`
- `Maximum Dispatch Flow Fraction`
- `Minimum Dispatch Flow Fraction`
- `Minimum Dispatch Runtime Fraction`
- `Dispatch Analysis Start Year`
- `Dispatch Analysis End Year`
- `Annual Heat Demand`
- `Annual Cooling Demand`
- `Annual Electricity Demand`

Optional dispatch-related output parameters:

- `Dispatch Profile Output File`
- `Generate Dispatch HTML Graphs`

Supported `Dispatch Demand Source` values:

- `Annual Heat Demand`
- `Annual Cooling Demand`
- `Annual Electricity Demand`

Validation by mode:

- heat pump and district heating dispatch require `Annual Heat Demand`
- absorption chiller dispatch requires `Annual Cooling Demand`
- pure electricity dispatch requires `Annual Electricity Demand`
- CHP dispatch accepts heat or electricity demand, depending on the dispatch objective

## Validation and Compatibility Rules

### Backward Compatibility

- `Baseload` remains the default
- non-dispatch runs keep their existing meaning
- dispatchable mode is opt-in

### Dispatch Validation

- demand profiles must normalize to hourly one-year series
- maximum dispatch flow fraction must be greater than or equal to minimum dispatch flow fraction
- dispatch analysis window must lie within plant lifetime
- unsupported reservoir/configuration combinations fail explicitly
- dispatchable electricity and CHP are restricted to:
  - `Subcritical ORC`
  - `Supercritical ORC`
  - `Single-Flash`
  - `Double-Flash`

### Path Handling

Implemented behavior now includes:

- relative demand-file inputs resolve against the input file location
- the Python client runs simulations from the source input file directory so relative output paths in example inputs resolve predictably
- explicit rich-text and HTML output paths are preserved rather than overwritten later in output generation

## Example Input

Canonical full-scale dispatch example:

- `tests/geophires_x_tests/example1_dispatchable_full_scale.txt`

It demonstrates:

- dispatchable operating mode
- canonical hourly demand ingestion
- dispatch analysis-year window selection
- dispatch summary reporting
- improved text output
- HTML output
- dispatch profile CSV export
- optional dispatch HTML graphs

## Remaining Limitations

- only `Demand Following` is implemented as a dispatch strategy
- SBT dispatch still excludes coaxial configuration
- shut-in recovery remains reduced-order rather than a validated high-fidelity regeneration model
- unsupported reservoir families still fail explicitly

## Recommendation

The current implementation shape remains correct:

- keep `Baseload` isolated and unchanged
- keep dispatch isolated through operating-mode strategies and plant adapters
- continue using file-backed hourly historical arrays as the canonical demand path
- extend support reservoir-by-reservoir rather than silently approximating unsupported cases
- treat additional control policies and higher-fidelity recovery/regeneration as later extensions rather than overloading the current dispatch framework
