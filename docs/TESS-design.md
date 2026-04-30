# Thermal Energy Storage System Dispatch Design

## Purpose

GEOPHIRES-X includes an optional thermal energy storage system, abbreviated
`TESS`, for dispatchable direct-use heat simulations. TESS is modeled as a
large pressurized liquid-water tank placed after the geothermal production
system and before the customer heat demand.

The implemented topology is:

```text
reservoir/wellbores -> surface heat delivery -> TESS -> demand center
```

With TESS disabled, GEOPHIRES-X uses the legacy dispatchable behavior:

```text
demand -> geothermal dispatch -> demand served/unmet
```

With TESS enabled, the customer demand is served from storage while geothermal
production charges the tank:

```text
demand -> TESS discharge -> demand served/unmet
TESS state -> geothermal charge dispatch -> TESS charge
```

The master switch is:

```text
TESS Enabled
```

Its default value is `False`, so existing models do not change unless storage
is explicitly enabled.

## Operating Mode

TESS is supported only in the dispatchable operating path: `Operating Mode =
Dispatchable`. In the TESS work this is the "Discharge" mode of operation: the
tank discharges to meet customer demand and geothermal output is dispatched to
recharge the tank.

TESS is not supported in `Operating Mode = Baseload`. Storage in the baseload
case is not physically or operationally meaningful for this implementation
because baseload operation already assumes steady plant operation rather than a
variable customer-demand profile that storage can buffer. If `TESS Enabled =
True` is used outside dispatchable mode, the model raises a validation error.

The current implementation also limits TESS to direct-use industrial heat with
`Annual Heat Demand`. Electric, cooling, CHP, and district-heating boiler
interaction cases are not part of the completed TESS feature.

## Design Intent

TESS decouples short-timescale customer demand variability from geothermal
production. In practical terms, the tank behaves like a thermal buffer:

- hourly demand spikes can be served by storage discharge;
- geothermal production can recharge storage using a selected charge-control
  strategy;
- geothermal flow and runtime respond to tank state and control settings, not
  only to instantaneous customer demand;
- seasonal energy still has to come from geothermal production or another heat
  source.

This is a power-smoothing and energy-buffering model. It does not create
energy. A realistic tank can smooth intraday and potentially multi-day
variability; true seasonal shifting requires very large storage volumes.

## Implemented Scope

TESS includes:

- dispatchable direct-use industrial heat demand;
- hourly tank energy balance on the existing dispatch hourly timebase;
- pressurized liquid water as the storage fluid;
- user-controlled tank volume and installed cost per cubic meter;
- user-controlled minimum useful, target, maximum, and initial temperatures;
- user-controlled `TESS Deadband Range`;
- geothermal charge into storage;
- storage discharge to customer demand;
- tank standby heat losses;
- charge and discharge efficiency losses;
- optional tank-side maximum charge and discharge power limits;
- TESS CAPEX and fixed O&M integration;
- hourly dispatch profile CSV columns for TESS state and heat flows;
- plain text, rich text, HTML, parsed-result, and schema output integration;
- optional dispatch HTML graphs for TESS state, charge/discharge, losses, and
  curtailment;
- temperature-band and moving-average charge control.

Out of scope:

- stratified or multi-node tank physics;
- detailed heat exchanger sizing;
- two-phase or boiling tank operation;
- non-water storage media;
- explicit pressure-vessel code compliance;
- subhourly numerical simulation;
- multi-tank storage networks;
- seasonal storage optimization;
- co-optimization against electricity price or heat price;
- district-heating peaking boiler interaction logic;
- electric, cooling, and CHP TESS integration.

## Code Integration

The feature is implemented primarily in:

- `src/geophires_x/SurfacePlant.py`: TESS parameters, validation, and default
  discharge-power resolution.
- `src/geophires_x/ThermalStorage.py`: one-node tank state, charge, discharge,
  loss, curtailment, and control helper classes.
- `src/geophires_x/Dispatch.py`: dispatchable operation loop integration,
  hourly arrays, and summary metrics.
- `src/geophires_x/Economics.py`: TESS capital and fixed O&M cost integration.
- `src/geophires_x/Outputs.py`: plain text report and dispatch profile CSV.
- `src/geophires_x/OutputsRich.py`: rich text, HTML, and graph output.
- `src/geophires_x_client/geophires_x_result.py`: parsed report categories.
- `src/geophires_x_schema_generator/geophires-result.json`: result schema.

The reservoir adapter remains responsible for answering what geothermal output
is available for a flow/runtime command. TESS sits above that layer and decides
how much customer demand storage can serve and how geothermal production should
charge the tank.

## Physical Model

### One-Node Tank

TESS is a single well-mixed water tank represented by:

```text
T_tank[t]
E_stored[t]
SOC[t]
```

`SOC` is normalized over the usable storage range:

```text
SOC = (T_tank - TESS Minimum Useful Temperature)
      / (TESS Maximum Temperature - TESS Minimum Useful Temperature)
```

The value is clipped to `[0, 1]`.

### Energy Capacity

For water:

```text
mass = density * volume
E_max = mass * cp * (TESS Maximum Temperature - TESS Minimum Useful Temperature)
```

The implementation uses water property calculations through the existing
CoolProp-backed utilities. For intuition, a useful approximation is:

```text
E_max_kWh ~= 1.16 * volume_m3 * delta_T_C
```

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
- `dt` is one hour.

The tank cannot exceed usable capacity. Excess geothermal heat is reported as
curtailed charge. The tank cannot discharge below zero usable energy. Any
remaining customer demand after stored-energy and discharge-power limits is
reported as unmet demand.

### Charge Acceptance

TESS accepts geothermal charge subject to tank capacity, maximum charge power,
charge efficiency, and source-temperature feasibility. Geothermal heat can
charge the tank only when the produced fluid is hotter than the current tank
temperature. If the produced fluid is cooler than or equal to the tank, the
model does not count that heat as useful charge because it cannot raise the
tank temperature without an external heat pump or other temperature-lifting
device.

### Discharge Constraint

Demand can be served only from usable stored energy above `TESS Minimum Useful
Temperature`. The model treats all energy above that temperature as deliverable
to the direct-use heat demand. A customer-side supply-temperature profile is not
required.

### Heat Loss

The implemented standby loss model is fractional daily loss:

```text
Q_loss_mwh = E_stored_mwh * TESS Daily Heat Loss Fraction / 24
```

## Pressure Handling

The current TESS working fluid is water. The tank must remain liquid at its
operating temperature. Water pressure is validated against saturation pressure:

```text
TESS pressure >= Psat(TESS Maximum Temperature) * TESS Pressure Safety Factor
```

`TESS Pressure Mode = Auto` computes the required pressure from saturation
pressure and the safety factor. `TESS Pressure Mode = User Specified` validates
the user-provided pressure against the same requirement. Pressure is a
feasibility/reporting feature, not a detailed pressure-vessel cost model.

## Parameters

All TESS parameters are optional unless `TESS Enabled = True`. With `TESS
Enabled = False`, TESS parameters do not affect model behavior or economics.

### Core Switch

| Parameter | Default | Units | Notes |
|---|---:|---|---|
| `TESS Enabled` | `False` | none | Master switch. |

### Tank Size and Cost

| Parameter | Default | Units | Notes |
|---|---:|---|---|
| `TESS Volume` | `1000.0` | m3 | Usable tank fluid volume. |
| `TESS Cost per Cubic Meter` | `500.0` | USD/m3 | Complete installed TESS cost estimate. |
| `TESS Fixed O&M Fraction` | `0.01` | 1/year | Annual fixed O&M as a fraction of TESS CAPEX. |

### Fluid and Pressure

| Parameter | Default | Units | Notes |
|---|---:|---|---|
| `TESS Working Fluid` | `Water` | none | Current implementation supports water. |
| `TESS Pressure Mode` | `Auto` | none | `Auto` or `User Specified`. |
| `TESS Pressure` | `1.0` | MPa | Used when pressure mode is user specified. |
| `TESS Pressure Safety Factor` | `1.10` | none | Multiplier on saturation pressure. |

### Temperature and State

| Parameter | Default | Units | Notes |
|---|---:|---|---|
| `TESS Minimum Useful Temperature` | `120.0` | degC | Lower bound for usable stored heat. |
| `TESS Maximum Temperature` | `160.0` | degC | Upper tank operating temperature. |
| `TESS Target Temperature` | `150.0` | degC | Preferred control temperature. |
| `TESS Initial Temperature` | `150.0` | degC | Initial tank temperature. |
| `TESS Deadband Range` | `10.0` | degC | Total thermostat hysteresis band around target. |

### Charge and Discharge

| Parameter | Default | Units | Notes |
|---|---:|---|---|
| `TESS Charge Efficiency` | `0.98` | none | Fraction of accepted geothermal heat stored as useful tank energy. |
| `TESS Discharge Efficiency` | `0.98` | none | Fractional efficiency from tank usable energy to delivered heat. |
| `TESS Daily Heat Loss Fraction` | `0.005` | 1/day | Fraction of stored usable energy lost per day. |
| `TESS Maximum Charge Power` | `-1.0` | MWth | `-1.0` means no tank-side limit beyond geothermal production and tank capacity. |
| `TESS Maximum Discharge Power` | `-1.0` | MWth | `-1.0` resolves to peak hourly thermal demand times `TESS Subhourly Demand Peak Multiplier`. |
| `TESS Subhourly Demand Peak Multiplier` | `1.0` | none | Used by automatic discharge-power sizing. |

### Control Strategy

| Parameter | Default | Units | Notes |
|---|---:|---|---|
| `TESS Charge Control Strategy` | `Temperature Band` | none | `Temperature Band` or `Moving Average`. |
| `TESS Charge Flow Fraction` | `1.0` | none | Geothermal flow fraction while charging under temperature-band control. |
| `TESS Moving Average Window` | `24` | hours | Moving-average smoothing horizon. |
| `TESS SOC Control Gain` | `0.25` | none | State-of-charge correction for moving-average control. |

## Validation

When `TESS Enabled = True`, the model validates:

- operating mode is `Dispatchable`;
- demand source is `Annual Heat Demand`;
- end-use is direct-use industrial heat;
- `TESS Volume > 0`;
- maximum temperature is greater than minimum useful temperature;
- target and initial temperatures are inside the usable temperature range;
- deadband range is nonnegative;
- charge and discharge efficiencies are positive and no greater than 1;
- maximum charge and discharge powers are either `-1.0` or nonnegative;
- user-specified pressure satisfies the saturation-pressure requirement.

## Control Behavior

### Temperature-Band Control

Temperature-band control uses thermostat-style hysteresis:

```text
lower_threshold = TESS Target Temperature - TESS Deadband Range / 2
upper_threshold = TESS Target Temperature + TESS Deadband Range / 2
```

When tank temperature is at or below the lower threshold, geothermal charging
turns on. When tank temperature is at or above the upper threshold, geothermal
charging turns off. While charging, geothermal flow uses `TESS Charge Flow
Fraction`, limited by the dispatch maximum flow settings.

### Moving-Average Control

Moving-average control computes a smoothed geothermal charge target from the
heat demand profile and applies an optional SOC correction:

```text
smoothed_demand_mw = moving_average(demand, TESS Moving Average Window)
soc_error = target_soc - current_soc
charge_target_mw = smoothed_demand_mw + TESS SOC Control Gain * soc_error * E_capacity_mwh
```

This control mode is better for moderating short-term demand swings and
approximating low-pass-filter behavior. Poor choices of window length or SOC
gain can overcharge, undercharge, or curtail heat when the tank reaches its
bounds.

## Dispatch Loop

The TESS dispatch loop runs once per hour:

```text
for each hour:
    demand_mw = demand_profile[hour]

    # 1. Serve demand from tank.
    tess_discharge = storage.discharge(demand_mw, dt_hours=1.0)

    # 2. Decide geothermal charge command from tank state and control strategy.
    charge_command = controller.dispatch(tess_state, demand_mw, geothermal_adapter)

    # 3. Evaluate geothermal production.
    geothermal_result = geothermal_adapter.evaluate_timestep(charge_command, hour)

    # 4. Accept geothermal heat into tank.
    tess_charge = storage.charge(geothermal_result, dt_hours=1.0)

    # 5. Apply standby loss.
    storage.apply_losses(dt_hours=1.0)

    # 6. Record geothermal output, tank output, tank state, unmet demand,
    #    losses, and curtailment.
```

Demand is served before charging in each hour. This is conservative because
same-hour charging cannot hide an empty tank at the start of the hour.

## Dispatch Results

When TESS is enabled, `DispatchResults` includes hourly arrays for:

- `hourly_tess_temperature`;
- `hourly_tess_soc`;
- `hourly_tess_stored_energy`;
- `hourly_tess_discharge_to_load`;
- `hourly_tess_charge_from_geothermal`;
- `hourly_tess_charge_curtailed`;
- `hourly_tess_standby_loss`;
- `hourly_tess_efficiency_loss`;
- `hourly_geothermal_charge_command`.

Existing fields keep their dispatch meaning:

- `hourly_thermal_demand`: customer demand.
- `hourly_demand_served`: heat delivered to demand, from TESS when TESS is
  enabled.
- `hourly_unmet_demand`: customer demand not served.
- `hourly_geothermal_thermal_output`: geothermal heat produced upstream of
  TESS.
- `hourly_heat_extracted`: reservoir heat extracted.
- `hourly_flow`: geothermal production flow, not tank discharge flow.

With TESS enabled, geothermal output and served demand are no longer expected
to have the same hourly profile.

## Summary Metrics

TESS summary metrics include:

- `tess_volume_m3`;
- `tess_usable_capacity_mwh`;
- `tess_initial_temperature_c`;
- `tess_final_temperature_c`;
- `tess_min_temperature_c`;
- `tess_max_temperature_c`;
- `tess_average_soc`;
- `tess_min_soc`;
- `tess_max_soc`;
- `tess_annual_charge_kwh`;
- `tess_annual_discharge_kwh`;
- `tess_annual_standby_loss_kwh`;
- `tess_annual_efficiency_loss_kwh`;
- `tess_annual_curtailed_heat_kwh`;
- `tess_equivalent_full_cycles`;
- `peak_customer_demand_mw`;
- `peak_geothermal_charge_mw`;
- `customer_demand_standard_deviation_mw`;
- `geothermal_output_standard_deviation_mw`;
- `geothermal_output_smoothing_ratio`;
- `geothermal_peak_reduction_fraction`;
- `geothermal_output_variability_reduction_fraction`;
- `annual_tess_served_heat_kwh`.

### Reported TESS Metrics

`Peak geothermal charge` is the maximum hourly geothermal thermal output during
the dispatch analysis window. With TESS enabled, this can include geothermal
production used to charge storage, not just heat served directly to the
customer.

`Customer demand standard deviation` is the standard deviation of hourly
customer demand in MW. Higher values mean the customer heat load is more
variable.

`Geothermal output standard deviation` is the standard deviation of hourly
geothermal thermal output in MW after dispatch and TESS controls. Higher values
mean geothermal production is more variable.

`Geothermal peak reduction ratio` is calculated as:

```text
1 - peak_geothermal_charge_mw / peak_customer_demand_mw
```

It can be negative. A negative value means geothermal output peaked above the
customer demand peak. That is expected when the plant produces extra heat to
charge TESS. For example, if peak customer demand is 25 MW and peak geothermal
charge is 50 MW, the metric is `1 - 50 / 25 = -1`.

`Geothermal variability reduction ratio` is calculated as:

```text
1 - geothermal_output_standard_deviation_mw
    / customer_demand_standard_deviation_mw
```

A positive value means geothermal output is smoother than customer demand. A
negative value means geothermal output is more variable than customer demand,
often because the selected charge-control strategy creates high-output charging
periods.

`Geothermal output smoothing ratio` is:

```text
geothermal_output_standard_deviation_mw
/ customer_demand_standard_deviation_mw
```

Values below 1 indicate geothermal output is smoother than customer demand.
Values above 1 indicate geothermal output is more variable than customer demand.

## Economics

TESS costs are represented as distinct storage costs:

```text
TESS_CAPEX_MUSD = TESS Volume * TESS Cost per Cubic Meter / 1e6
TESS_OPEX_MUSD_per_year = TESS_CAPEX_MUSD * TESS Fixed O&M Fraction
```

For non-SAM economics:

- TESS CAPEX is added to total capital cost when enabled;
- TESS fixed O&M is added to computed annual O&M when enabled;
- user-provided `Total Capital Cost` and `Total O&M Cost` overrides retain
  their override behavior.

TESS is not treated as free flexibility. If it changes cycling, flow, pumping,
or unmet demand, those effects appear through dispatch behavior and costs.

## Output Integration

### Text, Rich Text, HTML, and Parsed Results

When TESS is enabled, TESS values are reported in their own section:

```text
***THERMAL ENERGY STORAGE SYSTEM (TESS) RESULTS***
```

This section appears before:

```text
***DISPATCH RESULTS***
```

The TESS report section includes storage size, cost, SOC, annual charge and
discharge, losses, curtailment, cycles, peak customer demand, peak geothermal
charge, and geothermal smoothing metrics.

The report intentionally does not include `TESS enabled` or `TESS final
temperature` as display rows. The enabled state is implied by the presence of
the TESS section, and final temperature remains available internally through
summary metrics.

### Dispatch Profile CSV

The dispatch profile CSV adds these columns when TESS is enabled:

- `TESS Temperature (degC)`;
- `TESS State of Charge (-)`;
- `TESS Stored Energy (MWh)`;
- `TESS Discharge to Load (MW)`;
- `TESS Charge from Geothermal (MW)`;
- `TESS Curtailed Charge (MW)`;
- `TESS Standby Loss (MW)`;
- `TESS Efficiency Loss (MW)`;
- `Geothermal Charge Command (MW)`.

### JSON Dispatch Summary

`build_dispatch_summary_json` includes TESS settings, TESS hourly data, annual
aggregates, and summary metrics when TESS is enabled. Parsed text results place
displayed TESS rows under `THERMAL ENERGY STORAGE SYSTEM (TESS) RESULTS`.

### HTML Graphs

When dispatch HTML graph generation is enabled, TESS cases include graphs for:

- TESS temperature and SOC;
- demand, TESS discharge, and geothermal charge;
- TESS losses and curtailment.

## Backward Compatibility

`TESS Enabled = False` preserves the existing dispatchable behavior:

- customer load is served directly from geothermal output;
- no TESS costs are added;
- no TESS report section is emitted;
- no TESS dispatch CSV columns are added;
- TESS parameters do not affect dispatch;
- dispatch regression tests compare disabled TESS against equivalent legacy
  dispatch behavior.

Existing input files do not require modification.

## Numerical Considerations

The dispatchable path uses an hourly timebase. TESS uses the same hourly
timebase. Subhourly demand variability is not simulated directly; `TESS
Subhourly Demand Peak Multiplier` is used for automatic discharge-power sizing.

Energy accounting is maintained across the tank:

```text
demand = served + unmet
geothermal_output = tank_charge_accepted + curtailed_charge + charge_losses
tank_energy_delta = charge_accepted - discharge_draw - standby_losses
```

The implementation reports final tank state but does not enforce cyclic annual
operation. Final stored energy is not required to equal initial stored energy.

## Example Case

The tracked regression input is:

```text
tests/geophires_x_tests/example1_dispatchable_tess.txt
```

The runnable example input is:

```text
tests/examples/example1_dispatchable_tess.txt
```

The example uses the canonical variable `annual_heat_demand.csv` profile,
enables TESS, and sets tank volume, installed cost per cubic meter, target
temperature, initial temperature, useful temperature bounds, deadband range,
charge flow fraction, and heat loss. Regression coverage verifies text output,
HTML output, dispatch profile CSV output, TESS graph artifacts, parsed TESS
summary values, and TESS CSV columns.

## Resolved Design Decisions

- TESS is available only in dispatchable direct-use industrial heat mode.
- TESS is not available in baseload mode because storage does not add useful
  dispatch behavior to a steady baseload case.
- The model does not require final tank state to equal initial tank state.
- `TESS Maximum Discharge Power` defaults to a demand-derived automatic value.
- `TESS Cost per Cubic Meter` represents complete installed TESS cost.
- Temperature-band control serves demand before charging in each hourly
  timestep.
- Direct-use heat demand does not require a customer supply-temperature profile.
- District-heating peaking boiler interactions are outside the completed TESS
  feature.
