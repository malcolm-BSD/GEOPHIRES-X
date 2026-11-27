# Proposal: GEOPHIRES-X External Extension Framework

> **PR continuity note:** This file retains its original GenGEO-oriented filename because it was introduced in PR #32. The proposal has been broadened following a review of the public/open geothermal techno-economic modeling landscape. GenGEO is now one candidate engine within a general GEOPHIRES-X extension framework rather than the sole architectural target.

## Purpose

This proposal describes a community-led framework that allows **GEOPHIRES-X** to interoperate with selected public/open geothermal techno-economic models while preserving GEOPHIRES-X as:

- the primary project model;
- the canonical input and output environment;
- the default calculation engine;
- the public user interface;
- the orchestration, validation, and comparison layer.

The objective is not to merge external codebases into GEOPHIRES-X. The objective is to establish a stable, versioned, capability-aware extension contract through which specialized tools can be used as:

1. replacement engines for selected calculations;
2. chained upstream or downstream models;
3. parallel benchmark engines.

The supporting landscape survey is:

- `docs/proposals/geothermal-techno-economic-model-landscape.md`

## Scope decision

This PR proposes only three implementation phases:

- **Phase 0:** extension framework foundation;
- **Phase 1:** PySAM/SSC and GeoCLUSTER integrations;
- **Phase 2:** bounded GenGEO and FGEM pilots.

No Phase 3 or later implementation commitment is included. PyThermoNomics, REopt, reV, dGeo, legacy GETEM spreadsheets, and other possible tools remain documented candidates or references for later community decisions.

## Summary recommendation

GEOPHIRES-X should become a modest **federated modeling platform**, not an ever-growing monolithic simulator.

The recommended architecture consists of:

- a canonical, versioned case schema;
- explicit SI-unit normalization;
- a capability registry;
- external adapters that own model-specific mappings;
- multiple execution modes, including in-process Python, subprocesses, containers, compiled-library wrappers, HTTP services, and batch jobs;
- normalized outputs with complete provenance;
- a common validation and comparison harness.

The first implementation priority should be the most mature, complementary, and maintainable engines:

1. **PySAM/SSC** for geothermal electricity, GETEM-derived cost treatment, and project finance;
2. **GeoCLUSTER** for closed-loop/AGS simulation and rapid surrogate-based screening.

The second implementation priority should address higher-value but higher-risk specialist tools:

3. **GenGEO** as an isolated and modernized higher-fidelity thermodynamic/reservoir engine;
4. **FGEM** as a chained hourly dispatch, storage, and market-value engine.

## Why this is worth considering

GEOPHIRES-X has unusually broad project-level coverage. It handles reservoirs, wellbores, surface plants, direct use, electricity, cogeneration, closed-loop systems, costs, economics, and uncertainty analysis. That breadth makes it the natural public-facing platform and common case-definition environment.

However, specialized tools provide important functionality that should not necessarily be recreated in GEOPHIRES-X core:

- SAM/PySAM provides nationally supported geothermal electricity and finance calculations;
- GeoCLUSTER provides a large closed-loop design space and specialized conduction-dominated models;
- GenGEO provides detailed coupled thermodynamic and research configurations;
- FGEM provides hourly dispatch, storage, and market-revenue calculations.

The opportunity is therefore:

> Can GEOPHIRES-X preserve its breadth and usability while delegating selected calculations to independently maintained specialist engines through explicit, testable boundaries?

A successful framework would provide:

- broader modeling capability without forcing all dependencies on every user;
- independent benchmarking of native GEOPHIRES-X calculations;
- reproducible local, hosted, and cloud execution;
- explicit model boundaries and prevention of double counting;
- consistent Monte Carlo and batch behavior;
- preservation of external licenses and governance;
- easier adoption of future specialist models without redesigning GEOPHIRES-X core.

## Candidate tools and proposed roles

| Tool | Maturity/support | Proposed role in this PR |
|---|---|---|
| GEOPHIRES-X | Mature; actively developed | Host, default engine, canonical schema, orchestration, reporting |
| PySAM/SSC | Production-grade; actively supported | Phase 1 plant, geothermal project, finance, and benchmark extension |
| GeoCLUSTER | Strong national-laboratory research platform; 2026 maintenance | Phase 1 closed-loop engine and surrogate extension |
| GenGEO | Scientifically valuable; upstream code stale since 2021 | Phase 2 isolated modernization and water-based ORC pilot |
| FGEM | Credible recent research software | Phase 2 hourly dispatch, storage, and market-value extension |
| PyThermoNomics | Emerging, structured research package | Documented only; no implementation commitment |
| REopt | Mature and active | Documented future whole-system connector |
| reV | Mature and active | Documented future geospatial workflow |
| dGeo | Important deployment-model lineage | Ecosystem reference only |
| GETEM spreadsheet | Historical | Validation reference; current path is SAM |

## Core architectural principles

### 1. Capability-based extensions

Extensions should declare what they can calculate rather than inheriting directly from internal GEOPHIRES-X model classes.

Example capabilities:

```text
reservoir.temperature_decline
reservoir.pressure_drop
wellbore.heat_loss
wellbore.hydraulics
plant.binary
plant.flash
plant.direct_heat
plant.closed_loop
plant.absorption_chiller
economics.capex
economics.project_finance
operations.dispatch
operations.storage
```

Capability declarations allow the framework to reject unsupported combinations before execution.

### 2. Canonical case and result schemas

The framework should define a versioned, JSON-compatible request/result schema with:

- explicit SI units;
- physical boundary definitions;
- time basis and project lifetime;
- financial convention identifiers;
- optional references to large Parquet or HDF5 time-series artifacts;
- schema migration and compatibility rules.

The canonical schema is not required to expose every native input of every tool. It should represent the interoperable subset and allow adapter-specific extension fields where justified.

### 3. Adapter-owned translations

Each adapter should conceptually implement:

```python
class ExternalModelAdapter:
    def capabilities(self) -> CapabilityManifest:
        ...

    def validate_case(self, case: CanonicalCase) -> ValidationResult:
        ...

    def translate_inputs(self, case: CanonicalCase) -> ExternalRequest:
        ...

    def run(self, request: ExternalRequest) -> ExternalRawResult:
        ...

    def normalize_outputs(self, result: ExternalRawResult) -> CanonicalResult:
        ...

    def provenance(self) -> ProvenanceRecord:
        ...
```

GEOPHIRES-X core and the public calculator should not import external-model internals directly.

### 4. Explicit execution modes

The framework must distinguish among:

#### Replacement

A selected GEOPHIRES-X calculation is replaced by an external result.

```text
GEOPHIRES-X project model
    -> external plant or finance calculation
    -> normalized result returned to GEOPHIRES-X
```

#### Chained

An external model consumes a completed GEOPHIRES-X profile.

```text
GEOPHIRES-X physical production profile
    -> FGEM dispatch/storage/market model
    -> hourly operation and revenue results
```

#### Parallel benchmark

Multiple engines run independently and produce a comparison report.

```text
GEOPHIRES-X | PySAM | GeoCLUSTER | GenGEO
                    -> normalized comparison
```

The mode must be explicit. The framework must not silently apply two plant, cost, or finance models to the same project boundary.

### 5. Deployment-neutral execution

The same schema and adapter contract should support:

- in-process Python execution where dependencies are compatible;
- subprocess execution;
- local containers;
- compiled SSC/PySAM wrappers;
- remote HTTP services;
- local multicore workers;
- AWS Batch or equivalent institutional/cloud batch systems.

The public hosted calculator should be the easiest entry point, but hosted execution should not be the only way to use an extension.

### 6. Monte Carlo and batch compatibility

Monte Carlo is a primary GEOPHIRES-X use case. The framework should support it from the foundation phase rather than retrofitting it later.

The batch contract should include:

- deterministic random seeds;
- realization and chunk identifiers;
- configurable chunk size;
- persistent initialization within a worker where useful;
- partial-failure reporting;
- retry and resume behavior;
- raw realization results;
- aggregate statistics;
- chunk-size invariance testing;
- provenance for every result bundle.

Threads should not be assumed safe for external engines. Independent processes or containers are preferred unless an engine is explicitly demonstrated to be thread-safe.

### 7. Provenance as a first-class result

Every external result should identify:

- model and adapter name;
- exact version and commit;
- container image digest where applicable;
- data-package version and checksum;
- schema version;
- capability and execution mode;
- defaults applied;
- unit conversions;
- warnings and extrapolations;
- whether results were simulated, interpolated, or retrieved;
- random seed and batch metadata;
- external-engine runtime and dependency versions.

### 8. Native GEOPHIRES-X remains authoritative by default

The first release of each adapter should operate in comparison or chained mode. Native GEOPHIRES-X should remain authoritative unless a user explicitly selects a validated replacement mode.

Unsupported cases should:

- fail validation before external execution;
- clearly identify unsupported fields or modes;
- fall back to native GEOPHIRES-X only when the user's policy permits it;
- never silently change project boundaries.

## Common validation corpus

A shared golden-case library should include, as relevant to supported adapters:

- hydrothermal binary plant;
- hydrothermal flash plant;
- EGS doublet;
- direct-use heating;
- cogeneration;
- closed-loop water system;
- closed-loop CO2 system when supported;
- flexible generation with hourly market prices;
- externally supplied reservoir-production profile;
- campus heating/cooling case.

For each case, retain:

- original-tool inputs and results;
- translated canonical inputs;
- GEOPHIRES-X results;
- output-specific tolerances;
- mapping assumptions;
- explanation of model-boundary differences;
- expected numerical divergence where models are not physically equivalent.

The objective is not forced agreement. It is transparent and explainable disagreement.

## Phase 0 — Extension framework foundation

### Goal

Create the reusable interoperability foundation before coupling GEOPHIRES-X to any one external model.

### Deliverables

- canonical case and result schemas;
- explicit SI-unit convention;
- capability vocabulary and registry;
- adapter discovery and version negotiation;
- replacement, chained, and benchmark execution semantics;
- standardized validation and unsupported-mode reporting;
- local subprocess runner;
- container runner;
- batch request/result contract compatible with AWS Batch;
- provenance record and compatibility manifest;
- golden-case validation harness;
- normalized comparative-results report;
- extension documentation template;
- licensing and distribution checklist;
- reference adapter or mock engine for framework testing.

### Acceptance criteria

- A mock adapter can be discovered, validated, run, and compared with a native result.
- The same canonical deterministic request can run locally and in a container.
- A representative Monte Carlo request can be chunked and reconstructed reproducibly.
- Unsupported capabilities fail before model execution with actionable diagnostics.
- Provenance is sufficient to reproduce the calculation environment.
- Native-only GEOPHIRES-X users do not install external dependencies.

## Phase 1 — PySAM/SSC and GeoCLUSTER

### PySAM/SSC workstream

#### Initial supported scope

- electricity generation;
- hydrothermal and EGS cases supported by the selected SAM geothermal module;
- binary and flash plant pathways where the input mapping is defensible;
- selected SAM financial models;
- advisory/benchmark mode first.

#### Integration modes

1. **Power-block mode:** GEOPHIRES-X provides produced-fluid state and flow.
2. **Full SAM geothermal-project mode:** SAM executes its supported geothermal calculation.
3. **Financial-postprocessor mode:** GEOPHIRES-X supplies production and cost time series to a selected SAM financial model.
4. **Parallel benchmark mode:** native and SAM results are normalized and compared.

#### Deliverables

- PySAM/SSC adapter;
- supported-version matrix and dependency pinning;
- explicit input-boundary mapping document;
- binary and flash golden cases;
- selected financial-model mappings;
- deterministic and Monte Carlo execution tests;
- delta report for native versus SAM results;
- documented treatment of GETEM-derived assumptions.

#### Acceptance criteria

- Supported PySAM cases reproduce direct PySAM reference runs within approved tolerances.
- Financial outputs identify the selected SAM financial structure and assumptions.
- Unsupported direct-use and other non-SAM cases are rejected clearly.
- Version changes cannot be adopted without rerunning the compatibility suite.

### GeoCLUSTER workstream

#### Initial supported scope

- closed-loop water cases;
- selected well geometries present in both systems;
- precomputed/surrogate queries;
- on-demand semi-analytical calculations where programmatically available;
- advisory comparison with GEOPHIRES-X SBT and U-loop calculations.

#### Deliverables

- GeoCLUSTER adapter;
- data-package manifest with checksums;
- surrogate/interpolation provenance fields;
- geometry and operating-condition mapping document;
- thermal-production and pumping output normalization;
- closed-loop golden cases;
- comparison report against native GEOPHIRES-X closed-loop models;
- documented behavior outside the supported precomputed domain.

#### Acceptance criteria

- Every result states whether it was simulated, interpolated, or retrieved.
- Data-package and model versions are independently identifiable.
- Out-of-domain queries fail or invoke an approved dynamic calculation rather than extrapolating silently.
- Native and GeoCLUSTER model-boundary differences are documented in comparison outputs.

## Phase 2 — GenGEO and FGEM pilots

### GenGEO modernization and integration pilot

#### Rationale

GenGEO offers useful coupled thermodynamic, reservoir, and cost calculations, but its upstream repository is based on a legacy Python environment and has not shown sustained maintenance since 2021. It therefore requires stricter isolation and validation than the Phase 1 engines.

#### Initial supported scope

- one water-based binary/ORC case family;
- advisory benchmark mode;
- native GEOPHIRES-X remains authoritative;
- no direct code merge into GEOPHIRES-X;
- CO2/CPG configurations deferred until the water-based adapter is validated.

#### Deliverables

- captured legacy GenGEO environment and golden results;
- modernization specification and dependency inventory;
- supported Python runtime or reproducible compatibility container;
- installable package or stable batch entry point;
- removal or isolation of shared mutable state, fixed paths, and shared temporary files;
- GEOPHIRES-X-to-GenGEO adapter;
- water-based ORC mapping document;
- legacy-versus-modernized regression report;
- GEOPHIRES-X/GenGEO comparison report;
- licensing and governance decision for upstream contribution versus compatibility fork.

#### Acceptance criteria

- Legacy reference cases are preserved before modernization.
- Modernized or containerized results match approved legacy tolerances.
- Independent worker processes run safely in parallel.
- Unsupported configurations are rejected explicitly.
- LGPL code and source-availability obligations remain clearly separated from GEOPHIRES-X core.

### FGEM operational extension pilot

#### Rationale

FGEM provides a distinctive hourly operational layer for flexible generation, storage, and market participation. Its best initial boundary is downstream of GEOPHIRES-X physical calculations rather than replacement of the full project model.

#### Initial supported scope

- GEOPHIRES-X hourly or representative production handoff;
- flexible dispatch and curtailment;
- thermal and battery storage where supported;
- wholesale energy, capacity, environmental-credit, and PPA revenue streams;
- chained mode as the default;
- full-FGEM comparison only as an optional research mode.

#### Deliverables

- FGEM chained adapter;
- time-series and market-input schema extensions;
- physical-to-operational boundary document;
- hourly dispatch and revenue output normalization;
- storage state and constraint mapping;
- flexible-operation golden cases;
- comparison against baseload GEOPHIRES-X economics;
- deterministic and batch execution tests.

#### Acceptance criteria

- Energy conservation and annual aggregation checks pass.
- Revenue streams and market assumptions are explicit and separable.
- Physical production limits from GEOPHIRES-X cannot be exceeded silently.
- Storage state, efficiency, and losses are reported consistently.
- Hourly outputs reproduce direct FGEM reference runs within approved tolerances.

## Deployment posture

### Public hosted service

The public calculator may expose supported extensions through:

- native GEOPHIRES-X;
- selected external engine;
- compare mode;
- asynchronous batch or Monte Carlo jobs where operational limits allow.

Unlimited anonymous cloud computation should not be assumed. Large studies may require quotas, authentication, or user-controlled cloud deployment.

### Local execution

Users should be able to run the same adapter and schema from:

- Python;
- the command line;
- a local container;
- local multicore workers.

This supports confidential studies, teaching, debugging, and exact reproduction of hosted results.

### User-controlled batch execution

The Phase 0 contract should remain compatible with:

- AWS Batch job arrays;
- S3-compatible input and output storage;
- fixed container versions;
- retries and resume;
- optional Spot compute;
- result aggregation;
- infrastructure-as-code examples.

This proposal does not require every adapter to ship an AWS deployment in its first implementation. It requires the framework contract not to prevent one.

## Licensing and dependency posture

GEOPHIRES-X is MIT-licensed. The candidate tools use different licenses and distribution models.

The framework should:

- preserve external license identity;
- avoid casual source copying into GEOPHIRES-X core;
- use process or container boundaries for LGPL/GPL tools where appropriate;
- publish source corresponding to distributed images when required;
- document all third-party dependencies and dataset terms;
- separate code licenses from data-package licenses;
- conduct a formal license review before production distribution.

This proposal is not legal advice.

## Principal risks

| Risk | Concern | Mitigation |
|---|---|---|
| Framework over-design | A generic API could become too abstract before real adapters exist | Build Phase 0 with a mock adapter, then harden it through Phase 1 |
| Unit and boundary mismatch | Similar fields may represent different physical boundaries | Canonical SI schema, explicit boundary documents, conversion tests |
| Double counting | Two plant, cost, or finance models may be applied to one boundary | Explicit replacement/chained/benchmark modes |
| External version drift | Upstream updates may change results | Pin versions, maintain compatibility matrices, rerun golden cases |
| Dataset drift | GeoCLUSTER or other data assets may change independently | Data manifests, checksums, immutable version references |
| Legacy GenGEO runtime | Old dependencies complicate hosting and security | Compatibility container, modernization pilot, regression suite |
| Monte Carlo overhead | Reinitialization may dominate runtime | Persistent worker initialization and chunked execution |
| Parallel safety | Research tools may use global state or shared files | Independent processes/containers and parallel-safety tests |
| Licensing | LGPL/GPL and dataset terms may constrain packaging | External boundaries and formal license review |
| False precision | Detailed models may appear more authoritative than validated | Advisory mode first, uncertainty reporting, documented limits |
| Maintenance burden | GEOPHIRES-X maintainers could inherit unsupported external code | Adapter ownership, capability manifests, optional dependencies |

## Governance

Each adapter should identify:

- technical owner;
- upstream project and contact path;
- supported versions;
- release and deprecation policy;
- validation owner;
- security-update process;
- license and data terms;
- expected response when the upstream project becomes inactive.

A tool should not be treated as a permanently supported extension merely because an experimental adapter exists.

## Out of scope for this PR

The following are not implementation commitments in PR #32:

- PyThermoNomics/OPM integration;
- REopt campus or hybrid optimization;
- reV regional supply-curve workflows;
- dGeo market-adoption modeling;
- a generalized shallow-geothermal component library;
- mandatory external-engine override behavior;
- CO2/CPG GenGEO integration before the water-based pilot passes;
- a commitment to indefinite hosting of third-party engines;
- any Phase 3 or later roadmap.

## Decision requested

The community is asked to decide whether to proceed with the bounded three-phase program:

1. establish the reusable extension framework;
2. implement PySAM/SSC and GeoCLUSTER as the first supported adapters;
3. pilot GenGEO and FGEM under stricter isolation and validation requirements.

This sequence provides immediate value from mature tools, creates an independent validation ecosystem, and tests the framework against both compiled software, large scientific datasets, legacy Python research code, and hourly operational models—without committing GEOPHIRES-X to an unlimited integration roadmap.
