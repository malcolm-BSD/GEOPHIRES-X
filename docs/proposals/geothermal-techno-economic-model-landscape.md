# Public/Open Geothermal Techno-Economic Modeling Landscape

**Assessment date:** 2026-07-11  
**Purpose:** Inform the proposed GEOPHIRES-X external-extension framework and the scope of PR #32.

## Executive conclusion

GEOPHIRES-X should remain the host platform, canonical project schema, default calculation engine, orchestration layer, and common reporting environment. It should not attempt to absorb every useful geothermal calculation into one monolithic codebase.

The strongest architecture is a **federated extension framework** that allows GEOPHIRES-X to:

1. replace a selected internal calculation with an alternative engine;
2. chain a specialist calculation before or after a GEOPHIRES-X run;
3. run an independent model in parallel for benchmarking;
4. preserve explicit units, model boundaries, provenance, licensing, and version information.

The recommended implementation priorities are:

| Priority | Tool | Recommended role |
|---|---|---|
| 1 | SAM through PySAM/SSC | Detailed electricity-generation, GETEM-style cost, and project-finance extension |
| 1 | GeoCLUSTER | Closed-loop/AGS alternative engine and rapid surrogate-model extension |
| 2 | GenGEO | Higher-fidelity coupled thermodynamic/reservoir engine after modernization |
| 2 | FGEM | Flexible-operation, storage, dispatch, and electricity-market extension |
| Defer | PyThermoNomics | External reservoir-simulator and well-trajectory economics adapter |
| Defer | REopt | Campus, district-energy, storage, and hybrid-system optimization connector |
| Defer | reV | Geospatial scaling, regional potential, and supply-curve workflow |
| Reference | GETEM spreadsheet | Historical validation benchmark; current functionality is represented in SAM |
| Ecosystem | dGeo | Market-adoption and deployment analysis rather than project simulation |

Only the framework foundation and the first two implementation tiers are proposed for PR #32. REopt, reV, PyThermoNomics, dGeo, and other possible extensions remain documented candidates rather than committed implementation work.

---

## Evaluation criteria

Each tool was assessed for:

- scientific and software maturity;
- ongoing development and institutional support;
- breadth and distinctiveness of functionality;
- packaging, testability, and programmatic access;
- license compatibility and distribution constraints;
- usefulness as a GEOPHIRES-X replacement engine, chained component, or benchmark;
- expected integration and maintenance cost.

A terminology note: most tools reviewed here are not literally in the public domain. They are publicly available, copyrighted open-source software under licenses such as MIT, BSD, Apache, LGPL, or GPL. The license distinction affects whether code should be linked, containerized, run out of process, or copied into GEOPHIRES-X.

---

## 1. GEOPHIRES-X

**Repository:** https://github.com/NatLabRockies/GEOPHIRES-X  
**Status:** Mature and actively developed.  
**Recommended role:** Host platform and default engine.

GEOPHIRES-X has the broadest project-level coverage among the reviewed open tools. It combines reservoir behavior, wellbores, surface plants, capital and operating costs, project economics, direct heat, electricity, cogeneration, closed-loop systems, storage, Monte Carlo analysis, and several specialized end-use models.

Repository activity remained substantial in July 2026, including releases, tests, cost-model refinements, and new project features.

### Strengths

- Broad geothermal application coverage.
- Python implementation and established batch use.
- Existing public calculator and community familiarity.
- Multiple reservoir, wellbore, surface-plant, and economic options.
- MIT license.
- Natural location for common inputs, outputs, validation, and provenance.

### Limitations

- Some calculations are deliberately screening-level or correlation-based.
- Continually adding specialized models directly to core will increase coupling and maintenance burden.
- Similar terms can have different physical boundaries across reservoir, plant, and economic models.

### Framework implication

GEOPHIRES-X should own:

- the canonical case definition;
- unit normalization and validation;
- capability selection;
- execution orchestration;
- normalized outputs;
- provenance and warnings;
- comparison reporting;
- local, hosted, and batch execution contracts.

External tools should not be required to subclass internal GEOPHIRES-X objects. A stable process-neutral protocol is more durable.

---

## 2. SAM, SSC, PySAM, and GETEM-derived geothermal functionality

**Repositories and documentation:**

- https://github.com/NatLabRockies/ssc
- https://github.com/NatLabRockies/pysam
- https://nrel-pysam.readthedocs.io/en/main/modules/Geothermal.html
- https://www.energy.gov/hgeo/geothermal/geothermal-electricity-technology-evaluation-model

**Status:** Production-grade and actively supported.  
**Recommended role:** First-priority detailed plant, cost, and finance extension.

DOE's historical Geothermal Electricity Technology Evaluation Model, GETEM, estimates geothermal electricity cost for hydrothermal and EGS projects using flash or binary plants. DOE identifies SAM as the current implementation path for this functionality.

The preferred programmatic integration is not the SAM desktop application. It is:

- **SSC**, the compiled simulation core;
- **PySAM**, the supported Python binding to SSC;
- the geothermal SSC module and selected SAM financial modules.

SSC and PySAM both had active releases and repository work in July 2026.

### Distinctive value

- Strong DOE/National Laboratory provenance.
- Detailed electricity-project treatment.
- Binary and flash conversion options.
- GETEM-derived geothermal cost structure.
- Multiple ownership, debt, tax, depreciation, incentive, and revenue models.
- Independent benchmark against native GEOPHIRES-X plant and economic calculations.
- BSD-style licensing and supported Python packaging.

### Limitations

- Electricity-oriented; not a complete direct-use geothermal model.
- Many interdependent inputs make naive field-to-field mapping unsafe.
- Some financial assumptions are US-centered.
- The geothermal module is not decomposed exactly along the same boundaries as GEOPHIRES-X.

### Recommended integration modes

1. **SAM power block:** GEOPHIRES-X supplies produced-fluid conditions and flow; SAM calculates plant performance.
2. **SAM geothermal project:** SAM runs its coupled geothermal electricity calculation for a supported case.
3. **SAM financial postprocessor:** GEOPHIRES-X supplies annual production and costs; a selected SAM financial model calculates project cash flow and financial metrics.
4. **Parallel benchmark:** native GEOPHIRES-X and SAM run independently from a mapped common case and produce a delta report.

Algorithms should not be copied from SAM into GEOPHIRES-X. The extension should pin supported PySAM versions and isolate translation logic in an adapter.

---

## 3. GeoCLUSTER

**Repository:** https://github.com/pnnl/GeoCLUSTER  
**Status:** Strong national-laboratory research platform with 2026 maintenance.  
**Recommended role:** First-priority closed-loop alternative engine and surrogate provider.

PNNL's GeoCLUSTER is a closed-loop geothermal techno-economic simulator covering multiple well geometries, working fluids, electricity and heat applications, and conduction-dominated reservoirs. It combines a large precomputed simulation space with on-demand semi-analytical calculations, including SBT-based workflows.

The repository remained public and unarchived and received license/documentation maintenance in February 2026.

### Distinctive value

- Purpose-built closed-loop/AGS focus.
- Very large precomputed design space for rapid screening.
- Multiple geometries, fluids, operating conditions, and end uses.
- Particularly relevant to conduction-dominated systems.
- Independent comparison family for GEOPHIRES-X SBT and U-loop calculations.
- Permissive BSD-style license.

### Limitations

- Application, datasets, surrogate logic, and computational models are more intertwined than ideal for library use.
- Large data assets create versioning and container-size concerns.
- Partial overlap now exists with GEOPHIRES-X closed-loop capabilities.
- Interpolated/surrogate results must be clearly distinguished from dynamically simulated results.

### Recommended integration

A GeoCLUSTER adapter should:

- query supported precomputed cases;
- invoke on-demand semi-analytical calculations where available;
- return thermal-production profiles, pumping requirements, plant outputs, and costs through the common schema;
- report whether each output was simulated, interpolated, or retrieved;
- support side-by-side comparison with native GEOPHIRES-X closed-loop models.

Dataset redistribution terms, checksums, and version identity should be managed separately from source-code licensing.

---

## 4. GenGEO

**Repository:** https://github.com/GEG-ETHZ/genGEO  
**Status:** Scientifically useful but operationally stale.  
**Recommended role:** Second-priority optional external engine after bounded modernization.

GenGEO is an object-oriented Python model coupling geothermal reservoir behavior, electricity conversion, and project costs. It includes conventional geothermal and CO2-based or CPG research configurations and was designed with extensibility in mind.

The latest repository commit identified during this review was July 2021. Its published environment is based on Python 3.7-era dependencies and older scientific libraries. It is LGPL-licensed.

### Distinctive value

- Explicitly coupled reservoir-thermodynamic-economic design.
- Detailed fluid-property and power-cycle calculations.
- Research value for sedimentary geothermal and CO2-based systems.
- Object-oriented organization that exposes useful component boundaries.

### Limitations

- No evidence of sustained upstream maintenance since 2021.
- Old runtime and dependency stack.
- Limited release, packaging, and support infrastructure.
- Existing tests are not sufficient as a production validation suite.
- LGPL obligations make direct source incorporation into an MIT core unattractive.
- Some models are academically important but not broadly validated against operating projects.

### Recommended integration

GenGEO should remain a separate engine, not be merged into the GEOPHIRES-X repository.

The initial implementation should:

1. capture legacy golden cases;
2. modernize or containerize GenGEO in a dedicated repository or compatibility branch;
3. support one narrowly defined water-based binary/ORC case;
4. implement explicit input/output translation;
5. compare against original GenGEO results and native GEOPHIRES-X;
6. add CO2/CPG cases only after the water-based interface is stable.

A subprocess or container boundary is preferred for dependency isolation, license separation, and reproducibility.

**Overall judgment:** high scientific value, medium-to-high integration cost, and significant maintenance risk.

---

## 5. FGEM — Flexible Geothermal Economics Model

**Repository and documentation:**

- https://github.com/aljubrmj/FGEM
- https://fgem.readthedocs.io/

**Status:** Credible and relatively recent research software.  
**Recommended role:** Second-priority operational, storage, dispatch, and market-value extension.

FGEM is a Python lifecycle techno-economic model for baseload and flexible geothermal operation. It represents hourly operation, wholesale-market prices, capacity and environmental-credit revenues, flash or binary plants, weather effects, and thermal or battery storage. Repository maintenance was observed in July 2025.

### Distinctive value

- Hourly rather than annual-average operation.
- Flexible generation, curtailment, dispatch, and storage.
- Wholesale energy, capacity, REC, and PPA revenue streams.
- Direct relevance to geothermal grid value.
- MIT license.

### Limitations

- Smaller development and user community than SAM or GEOPHIRES-X.
- Research-grade interfaces may continue to evolve.
- Its full model overlaps with GEOPHIRES-X reservoir and plant calculations.
- Market assumptions may be more reusable than its entire physical model.

### Recommended integration

FGEM should initially be a chained downstream layer:

```text
GEOPHIRES-X physical production and plant profile
                    -> FGEM dispatch, storage, and market calculation
                    -> hourly operation, revenue, and economic outputs
```

GEOPHIRES-X should ordinarily remain responsible for physical project definition. A full-FGEM alternative-engine mode may be retained for research comparison but should not be the default boundary.

---

## 6. PyThermoNomics

**Repository:** https://github.com/TNO/pythermonomics  
**Status:** Well-structured emerging research package.  
**Recommended role:** Deferred candidate for external simulator and trajectory-aware economics.

TNO's PyThermoNomics calculates geothermal project economics from reservoir-simulation and well information. It supports NPV and LCOE, well trajectories, YAML cases, CSV and OPM/Eclipse-style results, command-line execution, Python APIs, examples, tests, and documentation. Repository development continued into August 2025. It is GPL-3.0 licensed.

### Value

- Strong bridge from numerical reservoir simulation to economics.
- Well-trajectory-aware capital calculations.
- Useful OPM/Eclipse ingestion path.
- Modern packaging relative to several older academic tools.

### Constraints

- Primarily an economic postprocessor rather than a complete competing simulator.
- GPL licensing argues for out-of-process or separately distributed integration.
- Small contributor and user base.
- Limited evidence of broad project validation.

### Recommendation

Document as a future adapter candidate, especially for OPM/Eclipse result ingestion and independent NPV/LCOE benchmarking. It is outside the implementation scope of PR #32.

---

## 7. REopt

**Repository:** https://github.com/NatLabRockies/REopt.jl  
**Status:** Mature and actively maintained.  
**Recommended role:** Deferred whole-energy-system optimization connector.

REopt optimizes sizing and dispatch of energy technologies for buildings, campuses, districts, and microgrids. It supports electrical and thermal systems, storage, tariffs, resilience, and geothermal heat-pump applications.

### Best use with GEOPHIRES-X

GEOPHIRES-X can supply:

- capacity limits;
- hourly or representative thermal/electric production profiles;
- marginal operating costs;
- capital-cost curves;
- temperature grades;
- minimum-load and ramp constraints.

REopt can then determine optimal sizing, dispatch, storage, and whole-system economics across geothermal, boilers, chillers, heat pumps, solar, and batteries.

### Recommendation

Do not place REopt inside the GEOPHIRES-X physical model. Retain it as a future system-optimization connector outside PR #32.

---

## 8. reV

**Repository:** https://github.com/NatLabRockies/reV  
**Status:** Production-grade and actively maintained.  
**Recommended role:** Deferred geospatial orchestration and supply-curve workflow.

reV is a geospatial techno-economic platform for technical potential, costs, transmission, exclusions, and renewable supply curves from site to continental scale. It had active repository work in July 2026.

### Best use with GEOPHIRES-X

```text
Resource and spatial data
        -> reV site selection and orchestration
        -> GEOPHIRES-X project calculations
        -> reV aggregation and supply curves
```

reV's configuration, chunking, provenance, and high-throughput patterns are also useful architectural references for the extension framework.

### Recommendation

Document as a later workflow integration. It is not an internal calculation plugin and is outside PR #32 implementation scope.

---

## 9. dGeo

**Documentation:** https://research-hub.nlr.gov/en/publications/the-distributed-geothermal-market-demand-model-dgeo-documentation/  
**Status:** Important research lineage; current reusable software posture requires separate confirmation.  
**Recommended role:** Ecosystem reference for market adoption and deployment.

NREL's dGeo is an agent-based, geospatial market-deployment model for geothermal heat pumps and direct-use district heating. It analyzes adoption and market potential rather than detailed subsurface project performance.

### Recommendation

Use concepts and published results for downstream market analysis, but do not prioritize direct integration until the currently supported repository, reproducible data package, license, maintained API, and geographic applicability are confirmed.

---

## 10. Legacy GETEM spreadsheets

**DOE page:** https://www.energy.gov/hgeo/geothermal/geothermal-electricity-technology-evaluation-model  
**Status:** Historical benchmark, effectively superseded for current use by SAM.

### Recommendation

- preserve representative GETEM cases in the validation library;
- do not create a runtime Excel dependency;
- use SAM/PySAM for current calculations;
- document differences among GETEM vintages, SAM versions, and GEOPHIRES-X assumptions.

---

## Recommended framework architecture

### 1. Define extensions by capability

Extensions should declare the calculations they can perform rather than relying on inheritance from GEOPHIRES-X internals.

Example capabilities:

```text
reservoir.temperature_decline
reservoir.pressure_drop
wellbore.heat_loss
wellbore.hydraulics
plant.binary
plant.flash
plant.direct_heat
plant.absorption_chiller
economics.capex
economics.project_finance
operations.dispatch
operations.storage
geospatial.supply_curve
```

A capability declaration prevents a tool such as SAM from being offered for a direct-use case it does not support.

### 2. Use a stable external protocol

A minimal adapter interface should resemble:

```python
capabilities()
validate_case(case)
translate_inputs(case)
run(translated_case)
normalize_outputs(raw_results)
provenance()
```

The transport should use a versioned JSON-compatible case schema with explicit SI units. Large time series may be passed as versioned Parquet or HDF5 artifacts referenced by the request.

### 3. Support multiple execution patterns

The framework should support:

- native Python plugin;
- subprocess;
- local container;
- remote HTTP service;
- AWS Batch job;
- compiled-library wrapper.

This accommodates PySAM/SSC, legacy GenGEO environments, Julia-based tools, large GeoCLUSTER datasets, and GPL/LGPL-separated packages without forcing every dependency on every user.

### 4. Separate replacement, chained, and benchmark modes

**Replacement calculation**

```text
Native GEOPHIRES-X component -> external component result
```

**Chained calculation**

```text
GEOPHIRES-X production profile -> FGEM dispatch calculation
```

**Parallel benchmark**

```text
GEOPHIRES-X | SAM | GenGEO -> normalized comparison report
```

The mode must be explicit to prevent double counting or accidental use of two plant or cost models on the same boundary.

### 5. Make provenance a first-class output

Every result should identify:

- model and adapter name;
- exact version and commit;
- container image digest where applicable;
- data-package version and checksum;
- capability selected;
- defaults applied;
- schema version;
- warnings and extrapolations;
- unit conversions;
- whether the result was simulated, interpolated, or retrieved.

### 6. Build a common validation corpus

Representative cases should include:

- hydrothermal binary;
- hydrothermal flash;
- EGS doublet;
- direct-use heating;
- cogeneration;
- closed-loop water;
- closed-loop CO2 where supported;
- flexible generation with hourly prices;
- externally supplied reservoir-production profile;
- campus heating/cooling case.

For each case, preserve original-tool results, translated GEOPHIRES-X inputs, variable-specific tolerances, and an explanation of model-boundary differences. The objective is explainable divergence, not forced numerical identity.

---

## Scoped implementation recommendation for PR #32

### Phase 0 — Extension framework foundation

Build the reusable foundation before committing GEOPHIRES-X core to any single external tool:

- canonical input and output schemas;
- capability registry;
- adapter discovery and versioning;
- explicit replacement/chained/benchmark execution modes;
- local subprocess and container runners;
- batch-execution contract compatible with AWS Batch;
- provenance and compatibility metadata;
- golden-case validation harness;
- normalized comparative-results report;
- licensing and distribution review checklist.

### Phase 1 — Highest-value supported engines

#### PySAM/SSC

- binary and flash electricity cases;
- GETEM-style geothermal project calculations;
- selected SAM financial models;
- GEOPHIRES-X/SAM comparison reports;
- pinned PySAM compatibility matrix.

#### GeoCLUSTER

- closed-loop water cases;
- surrogate/database queries;
- on-demand semi-analytical calculations where exposed;
- comparison with GEOPHIRES-X SBT and U-loop results;
- explicit data-version and interpolation provenance.

### Phase 2 — Higher-risk specialized engines

#### GenGEO modernization pilot

- reproducible legacy environment and golden cases;
- bounded modernization or compatibility container;
- one water-based binary/ORC case;
- normalized comparison with GEOPHIRES-X and, where appropriate, SAM;
- no direct source merge into GEOPHIRES-X;
- CO2/CPG deferred until the water-based adapter is validated.

#### FGEM operational extension

- hourly production handoff;
- flexible dispatch and curtailment;
- thermal and battery storage;
- wholesale, capacity, credit, and PPA revenues;
- chained operation as the default integration boundary;
- full-engine comparison retained only as an optional research mode.

No additional implementation phases are proposed in this PR. The remaining tools are retained in this survey as candidates for later community decisions.

---

## Final recommendation

The key decision is not whether GEOPHIRES-X should choose one external model. It is whether GEOPHIRES-X should become a disciplined **federated modeling platform**.

PySAM/SSC, GeoCLUSTER, GenGEO, and FGEM solve meaningfully different problems. A capability-aware, versioned, container-friendly extension protocol would allow GEOPHIRES-X to use their strengths while preserving its existing simplicity, licensing, public calculator, local execution, and Monte Carlo workflows.

For PR #32, the recommended commitment is deliberately bounded:

1. build the extension foundation;
2. integrate PySAM/SSC and GeoCLUSTER;
3. pilot GenGEO and FGEM under stricter isolation and validation requirements.

Everything beyond those three phases remains informative context rather than approved implementation scope.
