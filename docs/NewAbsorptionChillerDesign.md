# New Absorption Chiller Design - Detailed Specification

Status: Partially implemented. The absorption chiller package, canonical GEOPHIRES parameter integration, catalog-based bank sizing, and MILP/greedy hourly dispatch paths are present. Remaining work includes full-year performance validation, vendor-grade performance maps, richer baseload scheduling, and broader PEP 257/484 cleanup across the new modules.

Author: (generated design)
Date: 2026-05-04

Purpose
-------
This document describes the target design for a new, advanced Absorption Chiller subsystem for GEOPHIRES-X and notes the current implementation status. It extends the previous design and adds requirements requested by the user: baseload/dispatch modes, Pint/CoolProp enabled by default, embedded manufacturer catalog with CSV + remote fallback, PEP 8/257/484 compliance, a UnitsManager, effect multipliers with override, expanded supplier catalog seed entries, and a concrete method/attribute signature list for all classes.

Design Principles
-----------------
- Backwards-compatible: default behavior replicates legacy outputs when the advanced subsystem is disabled.
- Opt-out/opt-in tradeoff: Pint and CoolProp are enabled by default to produce physically consistent units but the code supports a lightweight fallback mode for minimal installs.
- Modular, object-oriented, testable, and documented to PEP8, PEP257, PEP484 standards.
- Minimal new parameters with sensible defaults that reproduce current results.
- Plant-level staging and dispatch are supported. Baseload operation currently uses the same hourly dispatch machinery with an average-load target; richer cycling/startup scheduling remains future work.

Top-level Checklist (Implementation tasks)
----------------------------------------
1. Add new package directory: `src/geophires_x/absorption/`.
2. Create modules (skeletons and docstrings):
   - `absorption/absorption_chiller.py` (AbsorptionChiller controller)
   - `absorption/chiller_unit.py` (ChillerUnit class)
   - `absorption/chiller_bank.py` (ChillerBank dispatcher)
   - `absorption/performance_map.py` (PerformanceMap)
   - `absorption/catalog.py` (Catalog and CatalogEntry)
   - `absorption/fluid_props_adapter.py` (FluidPropsAdapter)
   - `absorption/units_manager.py` (UnitsManager using Pint)
   - `absorption/utils.py` (common helpers)
3. Add `docs/NewAbsorptionChillerDesign.md` (this document).
4. Add embedded catalog file `data/absorption_chiller_catalog_default.csv` with seed entries.
5. Implement unit tests under `tests/test_absorption_*` as described.
6. Modify `src/geophires_x/SurfacePlantAbsorptionChiller.py` to call into new subsystem with opt-in flag `Use Advanced Absorption Chiller` (default True for new design; legacy compatibility preserved if false).
7. Add README and usage examples in `docs/` and update `pyproject.toml`/`requirements-dev` to include Pint and CoolProp by default for development.
8. Run test suite and ensure regression tests pass.

New global default: Enable advanced Absorption Chiller subsystem by setting `Use Advanced Absorption Chiller = True` in the new configuration. This advanced subsystem will enable Pint and CoolProp by default; tests and CI should include these packages.

PEP compliance
--------------
- All source files must conform to PEP 8 style (flake8/black recommended). Use 79/88 columns per project conventions.
- Docstrings: module/class/function docstrings required per PEP 257; include parameter and return descriptions.
- Type hints: all public methods and class attributes should include PEP 484 type hints (use typing module types like Optional, List, Dict, Tuple, Union).
- Unit tests should check type contracts where relevant (mypy for CI recommended).

High-level class design (concrete signatures)
--------------------------------------------
The following section lists classes with full method and attribute signatures (type hints) and short docstrings. These are ready for direct implementation.

Note: use `numpy.ndarray` for hourly arrays. If `UnitsManager.enabled` is True, arrays may be `pint.Quantity` wrapping numpy arrays. Import typing aliases as needed: Optional, List, Dict, Tuple, Any.

1. UnitsManager
----------------
class UnitsManager:
    """Manage unit handling with Pint.

    If Pint is available and enabled, UnitsManager provides helpers to create and
    convert quantities. If Pint is disabled, UnitsManager methods return plain
    floats and arrays but keep units documented.
    """

    enabled: bool

    def __init__(self, enabled: bool = True) -> None:
        """Initialize the UnitsManager.

        Parameters:
            enabled: Whether to enable Pint unit handling. Default True.
        """

    def quantity(self, value: Any, units: str):
        """Return a quantity object or raw value.

        Parameters:
            value: Scalar or array-like value.
            units: Unit string (e.g. 'kW', 'kg/s', 'degC').

        Returns:
            A pint.Quantity if enabled, otherwise the raw value.
        """

    def to(self, q: Any, units: str):
        """Convert quantity to target units; if disabled returns numeric value.

        Parameters:
            q: Quantity or numeric.
            units: Target unit string.

        Returns:
            Converted quantity or numeric.
        """

    def strip(self, q: Any) -> Any:
        """Return the raw numeric value (ndarray or scalar) without units.

        This is useful for internal numpy computations.
        """

2. FluidPropsAdapter
--------------------
class FluidPropsAdapter:
    """Adapter for thermophysical property queries.

    Tries to use CoolProp when `use_coolprop` is True; otherwise falls back to
    safe constant approximations for water and ammonia and simple mixture
    heuristics for LiBr-water.
    """

    has_coolprop: bool

    def __init__(self, use_coolprop: bool = True) -> None:
        """Initialize the adapter and attempt to import CoolProp.

        Parameters:
            use_coolprop: Request use of CoolProp by default.
        """

    def cp(self, fluid: str, t_c: float, x: Optional[float] = None) -> float:
        """Return specific heat capacity [J/kg/K] at temperature t_c [degC].

        Parameters:
            fluid: Fluid identifier (e.g., 'water', 'NH3').
            t_c: Temperature in degC.
            x: Optional concentration/quality.

        Returns:
            cp in J/kg/K.
        """

    def density(self, fluid: str, t_c: float, x: Optional[float] = None) -> float:
        """Return density [kg/m3]."""

    def enthalpy(self, fluid: str, t_c: float, x: Optional[float] = None) -> float:
        """Return specific enthalpy [J/kg]."""

3. PerformanceMap
-----------------
class PerformanceMap:
    """Hold PLR/COP and auxiliary power curves and perform interpolation.

    The map can be constructed from a simple parametric fallback or from a
    manufacturer-supplied lookup table (PLR × temperature grid).
    """

    def __init__(self, rated_cop: float, plr_curve_params: Optional[Dict[str, float]] = None) -> None:
        """Construct a simple parametric performance map.

        Parameters:
            rated_cop: Rated COP at PLR=1.0.
            plr_curve_params: Optional params for PLR and temperature correction,
            including 'alpha', 'beta', '*_ref_c', and '*_slope_per_c'.
        """

    def evaluate(self, plr: float, t_gen_c: float, t_evap_c: float, t_cond_c: float) -> Dict[str, float]:
        """Evaluate map and return COP and auxiliary multipliers at the given conditions.

        The fallback implementation applies a bounded temperature correction
        around nominal generator, evaporator, and condenser conditions so hourly
        temperature profiles affect COP even without a manufacturer lookup table.

        Returns dict with keys: 'cop', 'aux_power_factor'.
        """

    @classmethod
    def from_lookup(cls, lookup_table: Any) -> "PerformanceMap":
        """Create a PerformanceMap from a lookup table (manufacturers' data).
        """

4. ChillerUnit
--------------
class ChillerUnit:
    """Represent a commercial absorption chiller model.

    Attributes are typed, and default values provided for optional fields.
    """

    model_id: str
    manufacturer: str
    nominal_cooling_kW: float
    nominal_COP: float
    effect_type: str  # 'single'|'double'|'triple'
    refrigerant_family: str  # 'LiBr-water'|'NH3-water'
    min_PLR: float
    turndown_ratio: float
    nominal_chilled_flow_kg_s: Optional[float]
    nominal_hot_flow_kg_s: Optional[float]
    pump_head_m: Optional[float]
    electrical_aux_kW: float
    installed_cost_USD: Optional[float]
    performance_map: Optional[PerformanceMap]

    def __init__(
        self,
        model_id: str,
        manufacturer: str,
        nominal_cooling_kW: float,
        nominal_COP: float = 0.7,
        effect_type: str = "single",
        refrigerant_family: str = "LiBr-water",
        min_PLR: float = 0.2,
        turndown_ratio: float = 3.0,
        nominal_chilled_flow_kg_s: Optional[float] = None,
        nominal_hot_flow_kg_s: Optional[float] = None,
        pump_head_m: Optional[float] = 30.0,
        electrical_aux_kW: float = 5.0,
        installed_cost_USD: Optional[float] = None,
        performance_map: Optional[PerformanceMap] = None,
    ) -> None:
        """Initialize the ChillerUnit dataclass.
        """

    def performance_at_plr(self, plr: float, t_gen_c: float, t_evap_c: float, t_cond_c: float) -> Dict[str, float]:
        """Return performance results for a given PLR and temperatures.

        Returns dict with keys: 'cop', 'cooling_kW', 'fuel_input_kW', 'electrical_aux_kW'.
        """

    def feasible_for_conditions(self, t_gen_c: float, t_evap_c: float, t_cond_c: float) -> bool:
        """Return whether this unit can operate under the provided conditions."""

5. Catalog & CatalogEntry
-------------------------
class CatalogEntry:
    """Container for a single catalog record. Fields mirror CSV columns."""

    def __init__(self, **kwargs) -> None:
        pass

class Catalog:
    """Maintain an embedded default dataset, allow CSV import, and remote query.

    The Catalog provides query() and select_min_cost_set() functions.
    """

    embedded_path: str

    def __init__(self, embedded_csv_path: Optional[str] = None) -> None:
        """Load embedded dataset.

        Parameters:
            embedded_csv_path: Path inside package data. If None, use packaged default.
        """

    def load_user_csv(self, csv_path: str) -> None:
        """Load user-provided CSV and merge/override embedded entries."""

    def query(self, capacity_kW: float, refrigerant_family: Optional[str] = None, effect_type: Optional[str] = None) -> List[CatalogEntry]:
        """Return candidate entries matching constraints."""

    def select_min_cost_set(self, required_capacity_kW: float, allow_staging: bool = True) -> Dict[str, Any]:
        """Return a selection of units and counts to meet capacity optimized by cost.

        Returns a dict containing keys: 'selected', 'total_capacity_kW', 'estimated_cost_USD'.
        """

    def query_remote_catalog(self, query_params: Dict[str, Any], timeout_s: int = 10) -> List[CatalogEntry]:
        """Attempt to query a remote catalog (fallback if embedded + user CSV not present).

        Notes:
            Remote queries may require internet access and are best-effort. If
            not available, the method should return an empty list.
        """

6. ChillerBank
--------------
class ChillerBank:
    """Aggregate multiple ChillerUnit instances and perform dispatch.

    Supports dispatch strategies 'min_cost', 'min_units', 'follow_heat', and
    both baseload and dispatch modes.
    """

    units: List[Tuple[ChillerUnit, int]]  # (unit, count)
    dispatch_strategy: str

    def __init__(self, units: Optional[List[Tuple[ChillerUnit, int]]] = None, dispatch_strategy: str = "min_cost") -> None:
        """Initialize the ChillerBank."""

    def add_unit(self, unit: ChillerUnit, count: int = 1) -> None:
        """Add `count` copies of a ChillerUnit to the bank."""

    def dispatch_hourly(
        self,
        cooling_load_kW_hourly: "numpy.ndarray",
        generator_heat_available_kW_hourly: Optional["numpy.ndarray"] = None,
        temps: Optional[Dict[str, "numpy.ndarray"]] = None,
        mode: str = "dispatch",
    ) -> Dict[str, Any]:
        """Dispatch units hourly.

        Parameters:
            cooling_load_kW_hourly: hourly cooling demand in kW.
            generator_heat_available_kW_hourly: if provided, the thermal power (kW) available to the chiller generator each hour.
            temps: dict of temperature arrays { 'geo_in', 'ambient', 'chilled_supply' } as needed.
            mode: 'dispatch' or 'baseload' — in 'baseload' operate units to provide a roughly constant output per unit schedule (e.g., minimize startups).

        Returns:
            dict with aggregated hourly outputs (same as AbsorptionChiller.evaluate_hourly output keys) and per-unit statuses.
        """

    def _dispatch_hour(self, hour_idx: int, load_kW: float, available_heat_kW: Optional[float]) -> Dict[str, Any]:
        """Internal: decide units and PLRs for a single hour."""

7. AbsorptionChiller (public controller)
----------------------------------------
class AbsorptionChiller:
    """High-level API class that orchestrates chiller calculations.

    This is the class called by `SurfacePlantAbsorptionChiller.Calculate`.
    """

    catalog: Catalog
    units_manager: UnitsManager
    fluid_adapter: FluidPropsAdapter

    def __init__(
        self,
        catalog: Optional[Catalog] = None,
        units_manager: Optional[UnitsManager] = None,
        fluid_adapter: Optional[FluidPropsAdapter] = None,
        refrigerant_family: str = "LiBr-water",
        effect_type: str = "single",
        rated_COP: float = 0.7,
        min_part_load_ratio: float = 0.2,
        turndown_ratio: float = 3.0,
        chilled_deltaT_K: float = 5.0,
        pump_head_m: float = 30.0,
        pump_efficiency: float = 0.70,
        use_pint: bool = True,
        use_coolprop: bool = True,
        effect_multiplier_override: Optional[float] = None,
    ) -> None:
        """Initialize AbsorptionChiller with sensible defaults.

        By default Pint and CoolProp are enabled for precise unit and property handling.

        New parameters added to configure MILP segmentation and temperature handling:
            - n_segments: number of PLR segments used to build piecewise-linear COP approximations (default 5)
            - use_hourly_temps: if True, segment COPs will be evaluated using the configured per-hour temperature profiles

        Note on parameter labels: the human-facing parameter label for enabling per-hour temperature evaluation is
        "Absorption Chiller Use Hourly Temperatures". Hourly temperature inputs use the canonical list-parameter labels
        "Absorption Chiller Generator Temperature Profile", "Absorption Chiller Evaporator Temperature Profile", and
        "Absorption Chiller Condenser Temperature Profile". The older dotted-key style has been removed from the
        canonical input parser. Update input files to use the canonical human-friendly parameter names documented in
        the project's Parameters Reference and examples.
        """

    def evaluate_hourly(
        self,
        cooling_demand_hourly: "numpy.ndarray",
        geo_inlet_temp_hourly: "numpy.ndarray",
        chilled_supply_setpoint_c: float = 7.0,
        ambient_temp_hourly: Optional["numpy.ndarray"] = None,
        mode: str = "dispatch",
    ) -> Dict[str, Any]:
        """Evaluate plant performance over the hourly timestep arrays.

        Parameters:
            cooling_demand_hourly: Hourly cooling load [kW].
            geo_inlet_temp_hourly: Hourly geothermal inlet temperature [degC].
            chilled_supply_setpoint_c: Chilled water supply setpoint [degC].
            ambient_temp_hourly: Hourly ambient temperature [degC].
            mode: 'dispatch' or 'baseload'.

        Returns:
            Dict containing hourly arrays for cooling, q_gen, q_rej, COP, mdots, pump_power, unit dispatch, temperatures, warnings.
        """

    def size_for_capacity(self, peak_cooling_kW: float) -> Dict[str, Any]:
        """Return a CatalogSelection for the requested peak capacity.

        Uses the Catalog and selection heuristics to return a recommended set of units and estimated installed cost and pump specs.
        """

Concrete algorithms and numerical details
---------------------------------------
- Units and types: all arrays are `numpy.ndarray` (float64). When `UnitsManager.enabled` is True, API accepts and returns `pint.Quantity` wrappers.
- COP calculation: compute rated COP adjusted by effect multiplier and part-load curve.

  COP_hourly = COP_rated * effect_multiplier * PLR_correction(PLR_hourly, alpha, beta)

  Default effect_multiplier = 1.0 for single, 1.65 for double, 2.0 for triple; user can override via `effect_multiplier_override`.

- Q_gen, Q_rej relationship:

  Q_gen = Q_cooling / COP
  Q_rej = Q_gen - Q_cooling + pump_heat_contribution

  Pump heat contribution is small and computed from pump_power_hourly (W) divided by 1000 to kW if considered in thermal balance.

- Mass flows (chilled loop):

  m_dot_chilled = Q_cooling (W) / (cp_water (J/kg/K) * ΔT_chilled (K))

  Units: convert kW → W where required.

- Geothermal mass flow:

  m_dot_geo = Q_gen (W) / (cp_geo (J/kg/K) * ΔT_geo (K))

  ΔT_geo default = max(2.0, provided_deltaT) K.

- Pump power per loop:

  P_hydrau_W = ρ * g * H * V_dot  (W) where V_dot = m_dot / ρ
  P_elec_W = P_hydrau_W / η_pump

- Part-load curve param defaults: alpha=0.15, beta=1.2 (gentle PLR correction). Values configurable through PerformanceMap.

Baseload vs Dispatch modes
--------------------------
- Dispatch mode (default): the system meets hourly load and staging algorithm minimizes operating cost subject to min_PLR, startup/shutdown penalties and available thermal input.
- Baseload mode: the chiller(s) run a schedule to provide steady output (or follow a profile chosen to minimize cycling). The controller computes a baseload schedule (e.g., constant output during daytime) and either curtails or stores excess heat if allowed. Modes are implemented inside `ChillerBank.dispatch_hourly` with `mode` parameter.

Pint and CoolProp by default
----------------------------
- Use `UnitsManager(enabled=True)` and `FluidPropsAdapter(use_coolprop=True)` by default for new designs.
- The implementation must gracefully handle missing optional dependencies in runtime: raise informative error at initialization time if policy disallows missing deps, or fallback with warnings and approximate numerical constants.

Catalog population strategy
---------------------------
We will ship an embedded default dataset `data/absorption_chiller_catalog_default.csv` containing seed entries for popular suppliers and representative models. The embedded dataset is intended to be a conservative, public-domain-like seed with estimated values that are safe for initial simulations. Users can override by supplying a CSV with the same schema or by enabling remote queries which will attempt to fetch manufacturer data.

CSV schema (columns):
- model_id, manufacturer, nominal_cooling_kW, nominal_COP, effect_type, refrigerant_family, recommended_generator_in_C, chilled_supply_C, chilled_deltaT_K, nominal_chilled_flow_kg_s, nominal_hot_flow_kg_s, min_PLR, turndown_ratio, pump_head_m, electrical_aux_kW, installed_cost_USD, footprint_m2

Embedded dataset (seed entries — estimated values)

Note: values are indicative and should be verified against vendor datasheets before use in procurement calculations.

Sample rows (CSV-style):
- YZK-YNH-1000,Yazaki,1000,0.75,single,LiBr-water,90,7,5,12.0,10.0,0.25,3.0,30,6,900000,28
- THX-TF-500,Thermax,500,0.72,single,LiBr-water,95,7,5,6.0,6.0,0.2,3.0,30,4,520000,22
- TRN-AC-1200,Trane,1200,0.70,single,LiBr-water,90,7,5,14.4,12.0,0.2,3.0,30,7,1100000,35
- CRR-ABS-1500,Carrier,1500,0.70,double,LiBr-water,150,7,5,18.0,16.0,0.2,3.0,30,9,1800000,50
- BRD-BDA-2000,Broad,2000,1.1,double,LiBr-water,150,7,5,24.0,20.0,0.2,3.0,30,12,2200000,70
- YZK-YNH-3000,Yazaki,3000,0.68,triple,LiBr-water,180,7,5,36.0,30.0,0.2,3.0,30,18,4200000,90
- MWK-MY-600,Mayekawa,600,0.7,single,NH3-water,95,7,5,7.2,6.0,0.2,3.0,30,5,650000,24
- JCI-JC-900,JohnsonControls,900,0.7,single,LiBr-water,90,7,5,10.8,9.0,0.2,3.0,30,6,920000,30
- MHI-MH-800,MitsubishiHeavy,800,0.72,double,LiBr-water,150,7,5,9.6,8.0,0.2,3.0,30,5,760000,32
- YZK-YNH-250,Yazaki,250,0.75,single,LiBr-water,90,7,5,3.0,2.5,0.25,3.0,30,2,260000,12

This seed list covers a range of capacities, single/double/triple effect types, and manufacturers. Implementers should expand the dataset with verified numbers from vendors if procurement-level accuracy is needed.

Provider list (non-exhaustive, to populate catalog):
- Yazaki (Japan)
- Thermax (India)
- Broad (China)
- Carrier (USA)
- Trane (USA)
- Johnson Controls (USA)
- Mitsubishi Heavy Industries (Japan)
- Mayekawa (Japan)
- York (Johnson Controls / York brand)
- Hitec, Hitachi (if applicable)

Remote catalog query
--------------------
- Provide a best-effort `Catalog.query_remote_catalog` implementation that attempts to access maintained JSON/CSV endpoints (if available) or scrapes publicly available manufacturer datasheets when permitted.
- The remote query is optional and should be throttled and cache results locally.
- If remote query fails, fall back to embedded + user CSV.

Effect multipliers and overrides
--------------------------------
- Default multipliers: single=1.0, double=1.65, triple=2.0.
- `AbsorptionChiller` exposes `effect_multiplier_override: Optional[float]` so users can specify exact multiplier; this overrides default.
- Catalog entries may carry a `multiplier` field to indicate manufacturer-claimed effect improvement; if present, the code will use the catalog value unless override provided.

Baseload & Dispatch operation specifics
--------------------------------------
- Mode switchable via call to `AbsorptionChiller.evaluate_hourly(..., mode='baseload'|'dispatch')`.
- `baseload` mode options:
  - Provide schedule type: constant output, daily profile, or custom array.
  - Minimize cycling by keeping units at steady PLR where possible.
- `dispatch` mode:
  - Hourly optimization implemented in `ChillerBank.dispatch_hourly`.
  - Dispatch objective depends on `dispatch_strategy`:
    - 'min_cost' minimizes cost = fuel_cost + electricity_aux + hourlyized_capex
    - 'min_units' minimizes number of units online
    - 'follow_heat' prioritize usage of available geothermal heat (when constrained)
- Thermal constraints: if geothermal heat is limited, dispatch must respect generator_heat_available_kW_hourly.

Part-load & turndown handling
-----------------------------
- Respect unit.min_PLR; if plant aggregate PLR would fall below useful limits, prefer adding a small unit or duty-cycling with startup penalty.
- Implement minimum on/off dwell time (configurable per-bank; default 1 hour) to avoid excessive cycling.

Testing & validation
--------------------
- Unit tests listed earlier are mandatory. Additional integration tests should run full-year examples with both baseload and dispatch modes.
- Add mypy checks and black/flake8 in CI.

API examples (interface-level, no code)
--------------------------------------
1) Simple: sizing for a peak demand
- Call `AbsorptionChiller.size_for_capacity(peak_kW=2500)` → returns recommended set of units from Catalog with counts, estimated installed cost, pump specs.

2) Dispatch: hourly simulation
- Prepare arrays: cooling_demand_hourly (8760 long), geo_inlet_temp_hourly, ambient_temp_hourly.
- Call `AbsorptionChiller.evaluate_hourly(..., mode='dispatch')` → returns hourly outputs described in the data model.

3) Baseload: steady schedule
- Call `evaluate_hourly(..., mode='baseload')` with baseload schedule parameters; chiller bank will deliver steady output unless constrained.

Implementation roadmap (step-wise)
---------------------------------
1. Create skeleton modules and docstrings for all classes listed above.
2. Implement `UnitsManager` and `FluidPropsAdapter` (including CoolProp integration) and add unit tests for fallback behavior.
3. Implement `PerformanceMap` parametric model and lookup support.
4. Implement `ChillerUnit` with `performance_at_plr` and feasibility checks.
5. Implement `Catalog` with embedded CSV loader, user CSV loader, and remote query stub + cache.
6. Implement `ChillerBank.dispatch_hourly` with simple greedy staging heuristic; expand to more advanced optimization later.
7. Implement `AbsorptionChiller.evaluate_hourly` to call into `ChillerBank` and aggregate results.
8. Add regression tests to verify legacy parity when `Use Advanced Absorption Chiller=False`.
9. Populate `data/absorption_chiller_catalog_default.csv` with seed data and document sources and verifications.
10. Ensure code passes flake8/black/mypy; add CI steps.

Security & Licensing considerations
----------------------------------
- Ensure embedded seed dataset contains only non-copyrighted or user-permissible data; mark values as "estimates" and require user verification for procurement.
- Remote scraping of manufacturer sites may violate terms; prefer manufacturer-provided APIs or documented CSV downloads.

Deliverables
------------
- `docs/NewAbsorptionChillerDesign.md` (this file)
- File skeletons under `src/geophires_x/absorption/` (to be implemented following this spec)
- `data/absorption_chiller_catalog_default.csv` seed dataset (CSV with schema above)
- Unit tests under `tests/` as described

Appendix: Example CSV header and first few rows (repeat of seed)
----------------------------------------------------------------
model_id,manufacturer,nominal_cooling_kW,nominal_COP,effect_type,refrigerant_family,recommended_generator_in_C,chilled_supply_C,chilled_deltaT_K,nominal_chilled_flow_kg_s,nominal_hot_flow_kg_s,min_PLR,turndown_ratio,pump_head_m,electrical_aux_kW,installed_cost_USD,footprint_m2
YZK-YNH-1000,Yazaki,1000,0.75,single,LiBr-water,90,7,5,12.0,10.0,0.25,3.0,30,6,900000,28
THX-TF-500,Thermax,500,0.72,single,LiBr-water,95,7,5,6.0,6.0,0.2,3.0,30,4,520000,22
CRR-ABS-1500,Carrier,1500,0.70,double,LiBr-water,150,7,5,18.0,16.0,0.2,3.0,30,9,1800000,50
BRD-BDA-2000,Broad,2000,1.1,double,LiBr-water,150,7,5,24.0,20.0,0.2,3.0,30,12,2200000,70

Notes on catalog expansion: keep fields normalized (units, effect types uniform). Add a 'source' column and 'last_verified' date to track provenance.

Closing notes
-------------
This design is intentionally comprehensive and conservative: defaults preserve legacy behavior while offering significantly greater fidelity when users enable the advanced features. The class signatures provided are ready for direct implementation, with docstrings and type hints suitable for PEP 257/484 compliance. The embedded catalog seed is lightweight and marked as estimated — procurement-grade accuracy requires populating with verified vendor data.

If you want, I can now:
- Generate the Python skeleton files with docstrings and type hints under `src/geophires_x/absorption/`, and create the embedded CSV file in `data/`.
- Or, start implementing the numerical core functions and tests.

Which should I do next?

ILP, MILP, PuLP, and PLR vs COP (notes for implementers and users)
---------------------------------------------------------------

Integer Linear Programming (ILP)
--------------------------------
- Definition: ILP is a mathematical optimization where the objective and
  constraints are linear, and some or all decision variables are constrained
  to take integer values. A common ILP use in this project is selecting how
  many whole commercial chiller units of each model to install (counts must
  be integers).
- What it does here: given a list of candidate units (nominal capacities and
  installed costs), ILP selects integer counts of each model to meet a
  required capacity while minimizing installed cost.
- User-provided inputs required to support ILP selection:
  - `installed_cost_USD` (per unit) in the catalog entries
  - `nominal_cooling_kW` per model
  - reasonable upper bounds (implicitly provided by catalog counts or
    by solver bounds) to keep the IP small and solve quickly

Mixed-Integer Linear Programming (MILP)
--------------------------------------

- Definition: MILP generalizes ILP by allowing both integer (often binary)
  variables and continuous variables; objectives and constraints remain
  linear. MILP lets us model both discrete decisions (which units are on)
  and continuous decisions (part-load ratios, PLR) in the same optimization.
- What it does here: the per-hour dispatch MILP contains binary on/off
  variables for each unit instance and a piecewise-linear representation of
  part-load behaviour. Instead of a single continuous PLR variable per unit,
  the implementation splits each unit's PLR domain into `n_segments` uniform
  segments (configurable per `ChillerBank`, default `n_segments=5`). The MILP
  creates continuous cooling variables `q_k_s` for each unit k and segment s
  so that total cooling from a unit is sum_s q_k_s and the fraction of each
  segment is implicitly represented by q_k_s / cap_k.

- Piecewise-linear COP(PLR) handling: COP is nonlinear in PLR and would make
  the model non-linear. To keep the problem linear the MILP evaluates COP at
  the midpoint of each segment (plr_mid) and uses that constant COP value for
  the segment. Generator heat (fuel) is then linearized as:

  fuel ≈ sum_k sum_s q_k_s / COP_seg(k,s)

  which is linear in the q_k_s variables and therefore compatible with MILP
  solvers.

- Constraints implemented per unit
  - sum_s q_k_s <= cap_k * y_k  (no cooling when unit off)
  - sum_s q_k_s >= cap_k * min_PLR * y_k  (enforce minimum part-load when on)
  - q_k_s <= cap_k * (plr_high - plr_low)  (segment capacity upper bound)

- Generator-heat constraint: when `generator_heat_available_kW_hourly` is
  provided, the MILP enforces the linear fuel constraint using the segment COPs
  for all dispatch strategies:

  sum_k sum_s q_k_s / COP_seg(k,s) <= available_generator_heat_kW

  This piecewise-linear approximation is tighter than a single conservative
  COP_min bound and typically yields better utilization of available heat
  while remaining a linear MILP.

What users must provide to enable MILP dispatch
-----------------------------------------------
- For accurate MILP dispatch, catalog/units should provide (per-model):
  - `nominal_cooling_kW` (kW)
  - `nominal_COP` (COP at rated conditions)
  - `min_PLR` (minimum part-load ratio)
  - `installed_cost_USD` (cost proxy used in objective)
  - ideally a `PerformanceMap` or part-load curve so the solver's
    continuous PLR variables can be mapped to an expected COP. If a
    `PerformanceMap` is not available, the MILP uses conservative values
    (COP at min_PLR) to bound generator heat.

PuLP: solver library used by the implementation
-----------------------------------------------
- What is PuLP: PuLP is a Python library that provides an interface to
  formulate linear and integer linear programs and to call external and
  bundled solvers (for example, the CBC solver bundled with PuLP).
- How we use PuLP here: the implementation builds small MILPs per hour
  (binary on/off variables for units and continuous PLR variables) and one
  ILP for initial catalog selection. The code calls `pulp.PULP_CBC_CMD(msg=False)`
  to solve quietly using the CBC solver if available.
- Developer / CI notes: PuLP is an optional dependency in the runtime
  sense (the code falls back to greedy heuristics if PuLP is unavailable),
  but the project test environment includes `pulp` for CI/dev so ILP/MILP
  tests run deterministically. Dev/test configuration updated to include
  `pulp>=2.6.0` in the `tox` test environment and in CI requirement manifests
  so the MILP paths are exercised during continuous integration.

PLR vs COP — brief explanation
-------------------------------
- PLR (Part-Load Ratio) is defined as the instantaneous output of a unit
  divided by its nominal (rated) capacity. PLR ∈ [0,1]. Example: a 1000 kW
  chiller producing 500 kW is at PLR = 0.5.
- COP (Coefficient of Performance) is the ratio of useful cooling provided
  (Q_cooling) to the thermal input required by the chiller generator
  (Q_generator) for absorption chillers. COP typically varies with PLR and
  operating temperatures:

  COP = Q_cooling / Q_generator

- Relationship and modeling:
  - At part-load, COP often degrades versus rated conditions; we model this
    with a PLR-correction factor (e.g., COP_part = COP_rated * (1 - alpha * (1-PLR)^beta)).
  - MILP uses PLR continuous variables to represent fractional loading; to
    remain linear we cannot represent nonlinear COP(PLR) inside the MILP.
    Therefore we either:
    1. Pre-linearize COP(PLR) into piecewise-linear segments and add those
       linear constraints to the MILP (more complex, but more accurate), or
    2. Use conservative constant COP estimates (e.g., COP at min_PLR) to
       form linear generator-heat constraints (implemented here). The
       conservative approach ensures feasibility (won't overcommit limited
       generator heat) but may under-utilize available heat compared to a
       nonlinear model.

Recommendations for users and developers
---------------------------------------
- Provide per-unit part-load performance maps when possible; these allow
  more accurate post-solve performance calculations. If you supply a
  detailed `PerformanceMap` (lookup table or piecewise-linear fit), we can
  optionally construct a MILP with piecewise-linear COP approximations.
- If you need the MILP to consider operating cost beyond installed cost
  (fuel cost, electrical parity, startup/shutdown penalties), extend the
  objective with those terms and, where necessary, include multi-hour
  coupling variables (e.g., startup binaries) — note this increases MILP
  complexity and solve time.
