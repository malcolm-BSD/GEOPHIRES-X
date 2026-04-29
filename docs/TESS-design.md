# Thermal Energy Storage System Dispatch Design

## Objective

Extend GEOPHIRES-X dispatchable operating mode with an optional thermal energy storage system, abbreviated `TESS`.

The first implementation models TESS as a large pressurized liquid tank placed after the underground geothermal system and before the demand center.

The intended system topology is:

```text
reservoir/wellbores -> surface heat delivery -> TESS -> demand center
```

With TESS disabled, GEOPHIRES-X must preserve the current dispatchable behavior:

```text
demand -> geothermal dispatch -> demand served/unmet
```

With TESS enabled, demand no longer drives geothermal flow directly. Demand is served from the tank, while the geothermal system charges the tank according to a user-selected control strategy:

```text
demand -> TESS discharge -> demand served/unmet
TESS state -> geothermal charge dispatch -> TESS charge
```

The first on/off parameter must be:

```text
TESS Enabled
```

Its default value must be `False`.

## Design Intent

TESS is meant to decouple short-timescale demand variability from geothermal production.

In practical terms, the tank behaves like a low-pass filter on the thermal demand seen by the geothermal system:

- second-by-second and hourly demand spikes are served by storage discharge;
- tank recharge is controlled over longer intervals;
- geothermal flow and runtime respond primarily to tank state, not instantaneous demand;
- total seasonal energy still has to come from geothermal production or another heat source.

This is a power-smoothing and energy-buffering model. It does not create energy. If winter heat demand is high for weeks or months, geothermal supply must still cover that energy over time. A realistic water tank can smooth intraday and possibly multi-day variability; true summer-to-winter seasonal shifting would require very large storage volumes.

## Initial Scope

### In Scope

- Dispatchable direct-use thermal demand.
- Hourly tank energy balance using the existing dispatch hourly timebase.
- Pressurized liquid water as the initial TESS fluid.
- User-controlled tank volume and cost per cubic meter.
- User-controlled operating temperature bounds and target temperature.
- User-controlled `TESS Deadband Range`.
- Tank charge from geothermal output.
- Tank discharge to thermal demand.
- Tank heat losses.
- Tank charge/discharge efficiency.
- Optional tank-side charge and discharge power limits.
- TESS CAPEX and optional fixed O&M integration.
- Hourly TESS profile output.
- Summary metrics showing demand smoothing and storage utilization.

### Out of Scope for the First Implementation

- Stratified multi-node tank physics.
- Detailed heat exchanger sizing.
- Two-phase or boiling tank operation.
- Non-water storage media.
- Explicit pressure-vessel code compliance.
- Subhourly numerical simulation.
- Multi-tank storage networks.
- Seasonal storage optimization.
- Co-optimization against electricity price or heat price.
- District-heating peaking boiler interaction logic.
- Electric, cooling, and CHP TESS integration.

These should remain future extensions after the one-node thermal model is stable and tested.

## Current Code Integration Points

The existing dispatch implementation is concentrated in `src/geophires_x/Dispatch.py`.

Relevant current components:

- `DemandProfileFactory`
  - reads annual heat/cooling/electricity demand profiles;
  - normalizes dispatch demand to hourly `MW`;
  - currently requires 8760 hourly timesteps.

- `DispatchableOperatingModeStrategy`
  - runs the hourly dispatch loop;
  - currently sends hourly demand directly to `DemandFollowingDispatchStrategy`;
  - writes hourly results into `DispatchResults`;
  - finalizes annual production and economics-compatible outputs.

- `DispatchPlantAdapter`
  - abstracts reservoir-specific geothermal production behavior;
  - evaluates a `DispatchCommand`;
  - returns `DispatchTimestepResult`.

- `DispatchResults`
  - stores hourly dispatch outputs and summary metrics.

TESS should be added as a layer in `DispatchableOperatingModeStrategy`, not as a replacement reservoir adapter. The reservoir adapter should continue to answer: "what does the geothermal system produce under this flow/runtime command?" The TESS layer should answer: "how much demand is served by storage, and how should geothermal charge the tank?"

## Physical Model

### One-Node Tank State

The first implementation should use a single well-mixed tank state:

```text
T_tank[t]
E_stored[t]
SOC[t]
```

where `SOC` is normalized between 0 and 1 over the usable storage range:

```text
SOC = (T_tank - TESS Minimum Useful Temperature)
      / (TESS Maximum Temperature - TESS Minimum Useful Temperature)
```

clipped to `[0, 1]`.

### Energy Capacity

For water:

```text
mass = density * volume
E_max = mass * cp * (TESS Maximum Temperature - TESS Minimum Useful Temperature)
```

In convenient approximate units:

```text
E_max_kWh ~= 1.16 * volume_m3 * delta_T_C
```

This approximation is useful for explaining the model, but implementation should use the existing CoolProp-backed water property helpers where possible.

### Hourly Energy Balance

At each dispatch hour:

```text
E[t+1] = E[t]
       + eta_charge * Q_charge_accepted[t] * dt
       - Q_discharge_to_load[t] / eta_discharge * dt
       - Q_loss[t] * dt
```

where:

- `E` is usable stored thermal energy;
- `Q_charge_accepted` is geothermal heat accepted by the tank;
- `Q_discharge_to_load` is heat delivered from TESS to the demand center;
- `Q_loss` is standby thermal loss;
- `dt` is one hour in the first implementation.

The tank cannot exceed `E_max`. Excess geothermal heat should be reported as curtailed or rejected heat.

The tank cannot fall below zero usable energy. Any remaining demand after discharge limits and stored energy constraints should be reported as unmet demand.

### Charge Temperature Acceptance

The tank can only be charged usefully when geothermal output is hotter than the tank or above a minimum useful charging temperature. The initial implementation may use a conservative energy-only approximation:

```text
charge_accepted_mw = min(geothermal_useful_heat_mw, remaining_storage_capacity_mwh / dt)
```

A better first-pass thermal check is:

```text
if produced_temperature <= T_tank:
    accepted charge = 0
```

This avoids creating heat at a higher tank temperature from a lower-temperature geothermal source. Later work can model heat exchanger approach temperature explicitly.

### Discharge Temperature Constraint

Demand can only be served if tank temperature is at or above `TESS Minimum Useful Temperature`.

The first implementation should treat all energy above the minimum useful temperature as deliverable. Direct-use heat demand should not require a customer supply-temperature profile. A future optional output or advanced mode can report supply-temperature adequacy, but a required demand-side temperature profile is intentionally out of scope.

### Heat Loss

The first implementation should use a fractional daily loss model:

```text
Q_loss_mwh = E_stored_mwh * TESS Daily Heat Loss Fraction / 24
```

This is simple, stable, and easy to calibrate. A future model can support UA-based losses:

```text
Q_loss = UA * (T_tank - ambient_temperature)
```

## Pressure Handling

The first TESS fluid should be water.

The tank must remain liquid at operating temperature. For water, the model should validate pressure against saturation pressure:

```text
TESS pressure >= Psat(TESS Maximum Temperature) * TESS Pressure Safety Factor
```

The repo already depends on CoolProp and has water property helpers in `GeoPHIRESUtils.py`. Saturation pressure can be computed through CoolProp without adding a dependency.

Pressure should initially be a feasibility/reporting feature, not a detailed pressure-vessel cost model.

## Proposed Parameters

All parameters are optional unless `TESS Enabled` is `True`. With `TESS Enabled = False`, these values must not affect model behavior or economics.

### Core Switch

| Parameter | Type | Default | Range / Options | Units | Notes |
|---|---:|---:|---|---|---|
| `TESS Enabled` | bool | `False` | `True`, `False` | none | Master switch. Must be false by default for backward compatibility. |

### Tank Size and Cost

| Parameter | Type | Default | Range / Options | Units | Notes |
|---|---:|---:|---|---|---|
| `TESS Volume` | float | `1000.0` | `1.0` to `10000000.0` | m3 | Usable tank fluid volume. Default gives meaningful intraday buffering without implying seasonal storage. |
| `TESS Cost per Cubic Meter` | float | `500.0` | `0.0` to `100000.0` | USD/m3 | Complete installed TESS cost, including tank, foundations, insulation, heat exchangers, controls, and balance-of-plant items included in the user's estimate. Wide range reflects atmospheric vs pressurized and site-specific costs. |
| `TESS Fixed O&M Fraction` | float | `0.01` | `0.0` to `0.20` | 1/year | Annual fixed O&M as a fraction of TESS CAPEX. |

### Fluid and Pressure

| Parameter | Type | Default | Range / Options | Units | Notes |
|---|---:|---:|---|---|---|
| `TESS Working Fluid` | string/enum | `Water` | `Water` initially | none | Future extension point for other fluids. |
| `TESS Pressure Mode` | string/enum | `Auto` | `Auto`, `User Specified` | none | Auto computes pressure from saturation pressure and safety factor. |
| `TESS Pressure` | float | `1.0` | `0.1` to `25.0` | MPa | Used only when pressure mode is user specified. Must exceed saturation requirement. |
| `TESS Pressure Safety Factor` | float | `1.10` | `1.0` to `3.0` | none | Multiplier on saturation pressure in auto mode and validation. |

### Temperature and State

| Parameter | Type | Default | Range / Options | Units | Notes |
|---|---:|---:|---|---|---|
| `TESS Minimum Useful Temperature` | float | `120.0` | `0.0` to `370.0` | degC | Lower bound for usable stored heat. |
| `TESS Maximum Temperature` | float | `160.0` | `1.0` to `370.0` | degC | Upper tank operating temperature. Must exceed minimum useful temperature. |
| `TESS Target Temperature` | float | `150.0` | `0.0` to `370.0` | degC | Preferred tank control temperature. Must be between minimum and maximum. |
| `TESS Initial Temperature` | float | `150.0` | `0.0` to `370.0` | degC | Initial tank temperature. Must be between minimum and maximum. |
| `TESS Deadband Range` | float | `10.0` | `0.0` to `100.0` | degC | Total thermostat hysteresis band around target. Example: target 150 C and range 10 C gives lower/upper thresholds of 145 C and 155 C. |

### Charge and Discharge

| Parameter | Type | Default | Range / Options | Units | Notes |
|---|---:|---:|---|---|---|
| `TESS Charge Efficiency` | float | `0.98` | `0.0` to `1.0` | none | Fraction of geothermal heat accepted into useful tank energy. |
| `TESS Discharge Efficiency` | float | `0.98` | `0.0` to `1.0` | none | Fractional efficiency from tank usable energy to delivered heat. |
| `TESS Daily Heat Loss Fraction` | float | `0.005` | `0.0` to `0.10` | 1/day | Fraction of stored usable energy lost per day. |
| `TESS Maximum Charge Power` | float | `-1.0` | `-1.0`, or `0.0` to `100000.0` | MWth | `-1.0` means no tank-side limit beyond geothermal production and available tank capacity. |
| `TESS Maximum Discharge Power` | float | `-1.0` | `-1.0`, or `0.0` to `100000.0` | MWth | `-1.0` means auto-size to `peak hourly thermal demand * TESS Subhourly Demand Peak Multiplier`. Explicit nonnegative values override the demand-derived default. |
| `TESS Subhourly Demand Peak Multiplier` | float | `1.0` | `1.0` to `10.0` | none | Multiplier used by the automatic discharge-power default and for checking discharge power against within-hour peaks while preserving hourly energy demand. |

### Control Strategy

| Parameter | Type | Default | Range / Options | Units | Notes |
|---|---:|---:|---|---|---|
| `TESS Charge Control Strategy` | string/enum | `Temperature Band` | `Temperature Band`, `Moving Average` | none | First implementation should support `Temperature Band`; `Moving Average` can follow in phase 2. |
| `TESS Charge Flow Fraction` | float | `1.0` | `0.0` to `100.0` | none | Geothermal flow fraction used while the tank is actively charging under temperature-band control. Existing max dispatch flow validation should also apply. |
| `TESS Moving Average Window` | int | `24` | `1` to `8760` | hours | Used only by moving-average control. |
| `TESS SOC Control Gain` | float | `0.25` | `0.0` to `10.0` | none | Used only by moving-average control to bias charging toward the target tank state. |

## Parameter Validation

Validation should run during surface plant parameter validation.

Required checks when `TESS Enabled = True`:

- `TESS Volume > 0`.
- `TESS Maximum Temperature > TESS Minimum Useful Temperature`.
- `TESS Target Temperature` is within `[TESS Minimum Useful Temperature, TESS Maximum Temperature]`.
- `TESS Initial Temperature` is within `[TESS Minimum Useful Temperature, TESS Maximum Temperature]`.
- `TESS Deadband Range >= 0`.
- deadband lower threshold is not below `TESS Minimum Useful Temperature` unless clipped with a warning.
- deadband upper threshold is not above `TESS Maximum Temperature` unless clipped with a warning.
- `TESS Charge Efficiency > 0` if charging is expected.
- `TESS Discharge Efficiency > 0` if demand is expected.
- `TESS Maximum Discharge Power = -1.0` is converted to the demand-derived automatic value before dispatch.
- `TESS Pressure >= Psat(TESS Maximum Temperature) * TESS Pressure Safety Factor` when pressure mode is user specified.
- TESS initially supports only thermal dispatch demand.

If TESS is enabled with unsupported demand mode, fail explicitly:

```text
TESS dispatch initially supports Annual Heat Demand only.
```

## Control Behavior

### Temperature Band Control

This is the first recommended implementation.

Definitions:

```text
lower_threshold = TESS Target Temperature - TESS Deadband Range / 2
upper_threshold = TESS Target Temperature + TESS Deadband Range / 2
```

Thresholds should be clipped to the minimum/maximum tank temperatures after validation warnings.

Control logic:

```text
if T_tank <= lower_threshold:
    charging_state = on

if T_tank >= upper_threshold:
    charging_state = off
```

When charging is on:

```text
target_flow_fraction = min(TESS Charge Flow Fraction, Maximum Dispatch Flow Fraction)
runtime_fraction = 1.0
```

When charging is off:

```text
target_flow_fraction = 0.0
runtime_fraction = 0.0
```

This deliberately turns geothermal into a tank-maintenance resource rather than a demand-following resource.

### Moving Average Control

This can be a second-phase strategy.

Purpose:

- compute a smoothed geothermal target from recent or forecast demand;
- add correction for tank SOC deviation from target;
- reduce cycling relative to simple thermostat control.

Example:

```text
smoothed_demand_mw = moving_average(demand, TESS Moving Average Window)
soc_error = target_soc - current_soc
charge_target_mw = smoothed_demand_mw + TESS SOC Control Gain * soc_error * E_capacity_mwh
```

The first implementation should not attempt optimization. It should stay deterministic and explainable.

## Dispatch Loop Design

Current dispatch loop:

```text
for each hour:
    nominal_state = geothermal_adapter.thermal_state_for_flow_fraction(1.0)
    dispatch_command = demand_following_strategy.dispatch(nominal_state, demand)
    timestep_result = geothermal_adapter.evaluate_timestep(dispatch_command)
    record served/unmet demand from geothermal output
```

TESS dispatch loop:

```text
for each hour:
    demand_mw = demand_profile[hour]

    # 1. Serve demand from tank.
    tess_discharge = tess.discharge(demand_mw, dt_hours=1.0)

    # 2. Decide whether/how to charge tank from geothermal.
    charge_command = tess_controller.dispatch(tess_state, demand_mw, geothermal_adapter)

    # 3. Evaluate geothermal production.
    geothermal_result = geothermal_adapter.evaluate_timestep(charge_command, hour)

    # 4. Accept geothermal heat into tank subject to capacity, temperature, and charge power.
    tess_charge = tess.charge(geothermal_result, dt_hours=1.0)

    # 5. Apply standby loss.
    tess.apply_losses(dt_hours=1.0)

    # 6. Record geothermal output, tank output, tank state, unmet demand, losses, and curtailment.
```

The order above serves demand before charging in each hour. That is conservative for short-term demand service because the model does not let simultaneous charging mask an empty tank at the start of the hour. A simultaneous formulation can be considered later, but the first implementation should be simple and reproducible.

## New Internal Data Structures

The implementation can either add these to `Dispatch.py` or create a new `ThermalStorage.py`. A new file is cleaner because `Dispatch.py` is already large.

Recommended classes:

```text
ThermalStorageModel
ThermalStorageState
ThermalStorageTimestepResult
ThermalStorageController
TemperatureBandThermalStorageController
MovingAverageThermalStorageController
```

### ThermalStorageState

Fields:

- `temperature_c`
- `stored_energy_mwh`
- `usable_capacity_mwh`
- `soc_fraction`
- `available_discharge_mwh`
- `remaining_charge_capacity_mwh`

### ThermalStorageTimestepResult

Fields:

- `starting_temperature_c`
- `ending_temperature_c`
- `starting_soc_fraction`
- `ending_soc_fraction`
- `stored_energy_mwh`
- `demand_mw`
- `discharged_to_load_mw`
- `unmet_demand_mw`
- `geothermal_charge_available_mw`
- `geothermal_charge_accepted_mw`
- `curtailed_charge_mw`
- `standby_loss_mw`
- `charge_efficiency_loss_mw`
- `discharge_efficiency_loss_mw`

## DispatchResults Extensions

Add hourly arrays to `DispatchResults`:

- `hourly_tess_temperature`
- `hourly_tess_soc`
- `hourly_tess_stored_energy`
- `hourly_tess_discharge_to_load`
- `hourly_tess_charge_from_geothermal`
- `hourly_tess_charge_curtailed`
- `hourly_tess_standby_loss`
- `hourly_tess_efficiency_loss`
- `hourly_geothermal_charge_command`

Existing fields should keep their current meaning as much as possible:

- `hourly_thermal_demand`: customer demand.
- `hourly_demand_served`: heat delivered to demand, now from TESS when TESS is enabled.
- `hourly_unmet_demand`: customer demand not served.
- `hourly_geothermal_thermal_output`: geothermal heat produced upstream of TESS.
- `hourly_heat_extracted`: reservoir heat extracted.
- `hourly_flow`: geothermal production flow, not tank discharge flow.

This distinction matters because with TESS enabled, geothermal output and served demand are no longer the same hourly profile.

## Summary Metrics

Add summary metrics when TESS is enabled:

- `tess_enabled`
- `tess_volume_m3`
- `tess_usable_capacity_mwh`
- `tess_initial_temperature_c`
- `tess_final_temperature_c`
- `tess_min_temperature_c`
- `tess_max_temperature_c`
- `tess_average_soc`
- `tess_min_soc`
- `tess_max_soc`
- `tess_annual_charge_kwh`
- `tess_annual_discharge_kwh`
- `tess_annual_standby_loss_kwh`
- `tess_annual_efficiency_loss_kwh`
- `tess_annual_curtailed_heat_kwh`
- `tess_equivalent_full_cycles`
- `peak_customer_demand_mw`
- `peak_geothermal_charge_mw`
- `geothermal_peak_reduction_fraction`
- `geothermal_output_variability_reduction_fraction`
- `annual_tess_served_heat_kwh`

The smoothing metrics should compare customer demand against upstream geothermal charge:

```text
geothermal_peak_reduction_fraction =
    1 - peak_geothermal_charge_mw / peak_customer_demand_mw
```

Variability reduction can use standard deviation:

```text
1 - std(geothermal_charge_mw) / std(customer_demand_mw)
```

Report zero or omit the metric when the denominator is zero.

## Economics Integration

TESS should be represented as a distinct storage cost, not only as a generic add-on.

Recommended calculations:

```text
TESS_CAPEX_MUSD = TESS Volume * TESS Cost per Cubic Meter / 1e6
TESS_OPEX_MUSD_per_year = TESS_CAPEX_MUSD * TESS Fixed O&M Fraction
```

Add new economics output parameters:

- `TESS Capital Cost`
- `TESS O&M Cost`

For non-SAM economics:

- add TESS CAPEX to total capital cost when `TESS Enabled = True`;
- add TESS O&M to computed total annual O&M when `TESS Enabled = True`;
- preserve `Total Capital Cost` override behavior if a user provides total capital cost;
- preserve `Total O&M Cost` override behavior if a user provides total O&M cost.

For SAM economics:

- include TESS CAPEX in overnight capital cost;
- include TESS O&M in fixed annual operating cost if supported by the existing SAM bridge.

The implementation should avoid treating TESS as "free flexibility." If it smooths demand and reduces wellfield cycling, that benefit should be visible only through changed operations and costs, not through an implicit credit.

## Output Integration

### Text Output

Add dispatch result rows when TESS is enabled:

- TESS enabled
- TESS volume
- TESS usable capacity
- TESS capital cost
- TESS fixed O&M
- TESS average SOC
- TESS annual discharge
- TESS annual losses
- TESS curtailed geothermal heat
- peak customer demand
- peak geothermal charge
- geothermal peak reduction

### Dispatch Profile CSV

Extend dispatch CSV with TESS columns when enabled:

- `TESS Temperature (degC)`
- `TESS State of Charge (-)`
- `TESS Stored Energy (MWh)`
- `TESS Discharge to Load (MW)`
- `TESS Charge from Geothermal (MW)`
- `TESS Curtailed Charge (MW)`
- `TESS Standby Loss (MW)`
- `TESS Efficiency Loss (MW)`

### JSON Summary

Include TESS settings and TESS summary metrics in `build_dispatch_summary_json`.

Recommended shape:

```json
{
  "tess_settings": {
    "enabled": true,
    "volume_m3": 1000.0,
    "target_temperature_c": 150.0,
    "deadband_range_c": 10.0,
    "charge_control_strategy": "Temperature Band"
  },
  "summary_metrics": {
    "tess_usable_capacity_mwh": 46.0
  }
}
```

## Backward Compatibility

The default behavior must be unchanged. This is a hard design criterion, not a convenience:

```text
TESS Enabled = False
```

When disabled:

- the dispatch model must use the same demand-following geothermal path as before TESS was added;
- customer load must be served directly from the geothermal system, subject to the same geothermal capacity, flow, runtime, and unmet-demand logic as the legacy dispatch model;
- `hourly_demand_served`, `hourly_unmet_demand`, `hourly_geothermal_thermal_output`, `hourly_flow`, pumping power, reservoir depletion, and economics must match the legacy non-TESS run for the same inputs;
- no TESS parameters affect dispatch;
- no TESS costs are added;
- no dispatch CSV columns are added unless the implementation chooses a fixed schema with zero-filled TESS columns;
- existing dispatch regression tests should continue to pass, and at least one regression test should compare `TESS Enabled = False` against an equivalent run with TESS omitted;
- existing input files should not require modification.

## Numerical Considerations

### Hourly Timebase

Current dispatch mode is hourly, regardless of the standard GEOPHIRES `Time steps per year` setting. The first TESS implementation should use the same hourly timebase.

Second-by-second variability should not be modeled directly in the first implementation. Instead, `TESS Subhourly Demand Peak Multiplier` should allow users to check whether the tank discharge power limit can meet subhourly peaks while preserving hourly energy demand.

### Energy Accounting

All hourly energy accounting should be internally consistent:

```text
demand = served + unmet
geothermal_output = tank_charge_accepted + curtailed_charge + charge_losses
tank_energy_delta = charge_accepted - discharge_draw - standby_losses
```

Tests should verify these balances within numerical tolerance.

### Initial and Terminal Storage State

The first implementation should report final tank state but should not force cyclic operation.

A later option can add:

```text
TESS Require Cyclic State
```

which would require final stored energy to equal initial stored energy over the analysis window. That is useful for fair annual economics but requires either iteration or an end-condition penalty.

## Development Steps

### Phase 1: Design and Parameter Plumbing

1. Add `TESS-design.md`.
2. Add TESS parameter definitions to `SurfacePlant`.
3. Add enums for:
   - `TESS Pressure Mode`
   - `TESS Charge Control Strategy`
   - `TESS Working Fluid`
4. Add parameter parsing and validation.
5. Add unit tests verifying:
   - `TESS Enabled` defaults to `False`;
   - TESS parameter parsing works;
   - `TESS Deadband Range` is parsed under the correct name;
   - default `TESS Maximum Discharge Power` resolves from peak demand and `TESS Subhourly Demand Peak Multiplier`;
   - invalid temperature and pressure combinations fail clearly.

### Phase 2: Storage Physics

1. Add `ThermalStorageModel`.
2. Add water property calculations for tank capacity.
3. Add pressure saturation validation using CoolProp.
4. Implement charge, discharge, loss, and curtailment behavior.
5. Add unit tests for:
   - capacity calculation;
   - SOC conversion;
   - discharge-limited demand service;
   - charge-limited storage fill;
   - standby losses;
   - energy balance.

### Phase 3: Dispatch Integration

1. Add TESS branch inside `DispatchableOperatingModeStrategy.run`.
2. Preserve the current dispatch path when `TESS Enabled = False`, with load served directly from geothermal output exactly as in the legacy dispatch model.
3. Implement temperature-band control.
4. Record TESS hourly arrays.
5. Ensure geothermal adapter still handles reservoir depletion/recovery.
6. Add dispatch integration tests for:
   - disabled TESS exact legacy behavior, including direct geothermal load service and matching served/unmet demand, flow, and geothermal output arrays;
   - enabled TESS serves demand from initial stored energy;
   - enabled TESS charges when below lower threshold;
   - enabled TESS shuts off geothermal charging above upper threshold;
   - unmet demand appears when storage is empty or discharge-limited.

### Phase 4: Economics

1. Add TESS CAPEX and O&M output parameters.
2. Add TESS CAPEX to total capital cost when enabled.
3. Add TESS O&M to computed annual O&M when enabled.
4. Preserve total-capital-cost and total-O&M override semantics.
5. Add tests for:
   - no cost impact when disabled;
   - correct CAPEX from volume and cost per cubic meter;
   - correct O&M from fixed O&M fraction;
   - LCOH changes when TESS cost is enabled.

### Phase 5: Outputs

1. Add TESS summary rows to text output.
2. Add TESS columns to dispatch profile CSV.
3. Add TESS fields to dispatch summary JSON.
4. Add optional HTML graph support:
   - tank temperature and SOC;
   - demand vs TESS discharge vs geothermal charge;
   - losses and curtailment.
5. Add tests for CSV/JSON schema stability.

Implemented Phase 5 output contract:

- disabled TESS keeps the legacy dispatch summary fields and dispatch profile CSV schema;
- enabled TESS adds TESS summary rows to plain text and rich output;
- enabled TESS adds TESS hourly columns to the dispatch profile CSV;
- the `geophires_x_result` parsed dispatch summary schema includes the TESS fields;
- optional dispatch HTML graphs include TESS temperature/SOC, demand/discharge/charge, and losses/curtailment when TESS is enabled.

### Phase 6: Example Case

1. Add a TESS dispatch example input file.
2. Use `Annual Heat Demand` with a variable hourly profile.
3. Set:
   - `TESS Enabled, True`
   - `TESS Volume`
   - `TESS Cost per Cubic Meter`
   - `TESS Target Temperature`
   - `TESS Deadband Range`
4. Include expected output files for regression testing.

Implemented Phase 6 example contract:

- added `tests/geophires_x_tests/example1_dispatchable_tess.txt`;
- the example uses the canonical variable `annual_heat_demand.csv` profile;
- the example enables `TESS Enabled` and sets tank volume, installed cost per cubic meter, target temperature, initial temperature, useful temperature bounds, deadband range, charge flow fraction, and heat loss;
- the regression test generates and verifies text, HTML, dispatch profile CSV, and all standard plus TESS dispatch graph artifacts;
- the regression test checks parsed TESS summary values and TESS CSV columns.

### Phase 7: Moving-Average Control

1. Implement `Moving Average` charge control.
2. Add tests comparing reduced geothermal variability against demand-following operation.
3. Add smoothing metrics to output.
4. Document tradeoffs between thermostat and moving-average control.

Implemented Phase 7 control contract:

- `TESS Charge Control Strategy = Moving Average` now charges the tank from a trailing moving average of the heat demand profile;
- `TESS Moving Average Window` controls the smoothing horizon in hours;
- `TESS SOC Control Gain` adds an optional state-of-charge correction around the configured target tank temperature;
- moving-average control still serves customer demand before charging during each timestep;
- smoothing output metrics include customer demand standard deviation, geothermal output standard deviation, geothermal output smoothing ratio, and geothermal variability reduction;
- the moving-average regression test confirms geothermal output is smoother than both the legacy demand-following geothermal output and the unsmoothed customer demand profile.

Moving-average control is better for moderating short-term demand swings and approximating the low-pass-filter behavior expected from a large tank. Temperature-band control is simpler and more thermostat-like, but it can create blocky geothermal charging cycles with higher charge-power peaks. Moving-average control can run geothermal more steadily, but poor choices of window length or SOC gain can overcharge, undercharge, or curtail heat when the tank reaches its bounds.

## Acceptance Criteria

The initial feature should be considered complete when:

- `TESS Enabled = False` preserves existing dispatch behavior and tests: load is served directly from the geothermal system, and served demand, unmet demand, geothermal output, flow, pumping, depletion, and economics match the legacy non-TESS path.
- `TESS Enabled = True` works for direct-use thermal dispatch.
- TESS volume creates finite, auditable storage capacity.
- `TESS Deadband Range` controls charge cycling.
- tank energy balance closes over every timestep.
- geothermal output is decoupled from hourly demand when TESS is active.
- unmet demand is reported when tank energy or discharge power is insufficient.
- default TESS discharge power is demand-derived, not unlimited.
- TESS CAPEX and O&M are included in economics.
- hourly TESS profile output is available.
- clear errors are raised for unsupported modes and physically invalid pressure/temperature inputs.

## Resolved Design Decisions

- The first implementation should not require final tank state to match initial tank state for annual analysis.
- `TESS Maximum Discharge Power` should default to a demand-derived automatic value, not unlimited discharge.
- `TESS Cost per Cubic Meter` should represent complete installed TESS cost, not tank-only cost.
- Temperature-band control should serve demand before charging in each hourly timestep.
- Direct-use heat demand should not require a customer supply-temperature profile.
- District-heating peaking boiler interactions should be handled in a follow-up release, not the first TESS release.
