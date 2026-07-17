# GEOPHIRES-X External Extension Framework Roadmap

This roadmap supports:

- `docs/proposals/geophires-gengeo-integration-proposal.md`
- `docs/proposals/geothermal-techno-economic-model-landscape.md`

The roadmap is intentionally limited to **Phase 0, Phase 1, and Phase 2**. It does not authorize or imply a later implementation phase.

## Guiding principles

- GEOPHIRES-X remains the host, canonical schema, default engine, and reporting layer.
- External tools remain independently governed and licensed.
- Extensions declare capabilities rather than inheriting tightly from GEOPHIRES-X internals.
- Replacement, chained, and benchmark modes are explicit.
- Native GEOPHIRES-X remains authoritative by default during initial adapter releases.
- Monte Carlo, batch execution, versioning, and provenance are framework requirements.
- Unsupported cases fail clearly and never alter model boundaries silently.

---

## Phase 0 — Extension framework foundation

### Goal

Create a reusable, testable interoperability layer before coupling GEOPHIRES-X to any particular external model.

### Decisions to resolve

- What is the minimum canonical case schema?
- Which physical and economic boundary definitions must be standardized?
- Which capabilities belong in the initial vocabulary?
- How are adapter-specific fields represented without contaminating the common schema?
- Which execution patterns must be supported in the first framework release?
- How are replacement, chained, and benchmark modes represented?
- What provenance is mandatory?
- What is the policy for unsupported cases, fallback, and failure?
- How are external licenses, data packages, and container images reviewed?

### Deliverables

#### Canonical schemas

- deterministic case schema;
- normalized result schema;
- explicit SI-unit convention;
- time-series artifact convention for Parquet or HDF5 data;
- schema-version and migration policy;
- financial-convention identifiers;
- physical-boundary definitions.

#### Capability and adapter contracts

- initial capability vocabulary;
- capability manifest format;
- adapter discovery mechanism;
- adapter version and compatibility manifest;
- validation and unsupported-mode result format;
- replacement/chained/benchmark execution semantics;
- reference adapter interface.

#### Execution layer

- in-process reference adapter;
- subprocess runner;
- local container runner;
- batch request/result contract compatible with AWS Batch and equivalent systems;
- deterministic seed and realization identifiers;
- chunking, retry, resume, and partial-failure conventions;
- persistent-worker initialization pattern;
- raw and aggregate result-bundle format.

#### Provenance and comparison

- mandatory provenance record;
- model, adapter, commit, dependency, and image identifiers;
- dataset version and checksum fields;
- simulated/interpolated/retrieved result classification;
- normalized comparison report;
- mapping-assumption and warning sections;
- runtime and resource-use fields.

#### Validation and governance

- golden-case test harness;
- mock external engine for framework tests;
- unit-conversion tests;
- chunk-size invariance tests;
- local-versus-container equivalence test;
- extension documentation template;
- adapter ownership and deprecation template;
- licensing and distribution review checklist.

### Exit criteria

- A mock adapter is discovered through the capability registry.
- A canonical case is validated, translated, executed, normalized, and compared with a native result.
- The same deterministic request produces equivalent results locally and in a container.
- A representative Monte Carlo request is chunked, resumed, and reconstructed reproducibly.
- Unsupported capabilities fail before external execution with actionable diagnostics.
- Provenance is sufficient to identify and reproduce the execution environment.
- Native-only GEOPHIRES-X users do not install external dependencies.
- Maintainers approve the schema and adapter contract for Phase 1 implementation.

### Suggested Phase 0 issues

1. Define canonical deterministic case schema.
2. Define normalized result and provenance schemas.
3. Define initial capability vocabulary.
4. Define replacement, chained, and benchmark semantics.
5. Implement adapter discovery and compatibility manifests.
6. Implement subprocess and local-container runners.
7. Define Monte Carlo and batch request/result contract.
8. Implement mock adapter and golden-case harness.
9. Implement normalized comparison report.
10. Define licensing, dataset, and adapter-governance checklists.

---

## Phase 1 — PySAM/SSC and GeoCLUSTER

### Goal

Validate the framework with two mature, high-value, institutionally supported integrations that exercise materially different technical patterns:

- a compiled scientific engine with supported Python bindings and extensive financial functionality;
- a closed-loop research platform using both dynamic calculations and large precomputed datasets.

## Phase 1A — PySAM/SSC adapter

### Initial supported scope

- geothermal electricity projects supported by the selected SAM geothermal module;
- hydrothermal and EGS cases where mappings are defensible;
- binary and flash conversion pathways;
- selected SAM financial models;
- advisory and benchmark modes first;
- replacement mode only after explicit validation and community approval.

### Integration modes

1. **Power-block mode** — GEOPHIRES-X supplies produced-fluid state and flow.
2. **SAM geothermal-project mode** — SAM runs the supported coupled geothermal calculation.
3. **Financial-postprocessor mode** — GEOPHIRES-X supplies production and cost data to a selected SAM financial model.
4. **Parallel benchmark mode** — GEOPHIRES-X and SAM results are normalized and compared.

### Deliverables

- PySAM/SSC adapter package;
- supported PySAM/SSC version matrix;
- dependency pinning and compatibility tests;
- input-boundary and unit-mapping document;
- binary-plant golden cases;
- flash-plant golden cases;
- selected financial-model mappings;
- deterministic adapter tests;
- Monte Carlo and batch tests;
- native-versus-SAM comparison report;
- GETEM/SAM assumption notes;
- public-calculator integration design, without requiring immediate hosted deployment.

### Exit criteria

- Adapter outputs reproduce direct PySAM reference runs within approved variable-specific tolerances.
- Each financial result identifies the SAM financial model and assumptions used.
- Unsupported direct-use and other non-SAM cases are rejected clearly.
- Native GEOPHIRES-X and SAM boundaries are visible in every comparison.
- Pinned-version changes automatically trigger the complete compatibility suite.
- Representative batch runs preserve deterministic seeds and result provenance.

## Phase 1B — GeoCLUSTER adapter

### Initial supported scope

- closed-loop water cases;
- selected geometries present in both GEOPHIRES-X and GeoCLUSTER;
- precomputed or surrogate queries;
- on-demand semi-analytical calculations where programmatically exposed;
- advisory comparison with GEOPHIRES-X SBT and U-loop calculations.

### Deliverables

- GeoCLUSTER adapter package;
- data-package manifest and checksum process;
- model/data compatibility matrix;
- geometry and operating-condition mapping document;
- surrogate/interpolation provenance fields;
- thermal-production normalization;
- pumping and parasitic-load normalization;
- plant-output and cost normalization where supported;
- closed-loop golden-case suite;
- out-of-domain behavior specification;
- native-versus-GeoCLUSTER comparison report;
- container and batch execution tests.

### Exit criteria

- Every result identifies whether it was simulated, interpolated, or retrieved.
- Model code and data package versions are independently identifiable.
- Out-of-domain requests fail or invoke an explicitly approved dynamic calculation.
- No silent extrapolation is permitted.
- GEOPHIRES-X and GeoCLUSTER model-boundary differences are documented in comparison outputs.
- Representative closed-loop batch runs are reproducible locally and in containers.

### Phase 1 integration review

Before Phase 2 begins, the community should review:

- whether the canonical schema remained stable under two real adapters;
- which adapter interfaces require refinement;
- whether the capability vocabulary is adequate;
- whether provenance and comparison reports are understandable;
- whether local and batch execution behavior is sufficiently reproducible;
- whether adapter ownership and maintenance responsibilities are credible.

### Suggested Phase 1 issues

1. Inventory current PySAM geothermal and finance interfaces.
2. Define PySAM supported-mode matrix.
3. Implement PySAM power-block adapter.
4. Implement full SAM geothermal-project benchmark mode.
5. Implement selected SAM financial-postprocessor modes.
6. Build PySAM golden cases and compatibility matrix.
7. Inventory GeoCLUSTER programmatic interfaces and datasets.
8. Define GeoCLUSTER supported geometries and operating domain.
9. Implement GeoCLUSTER dataset and surrogate adapter.
10. Implement GeoCLUSTER dynamic calculation path where available.
11. Build closed-loop golden cases and comparison reports.
12. Complete Phase 1 framework-retrospective and schema decision.

---

## Phase 2 — GenGEO and FGEM pilots

### Goal

Test the framework against two higher-risk, highly specialized tools:

- a scientifically valuable but operationally stale coupled model requiring modernization and license isolation;
- a recent hourly operational and market model best used as a chained downstream extension.

## Phase 2A — GenGEO modernization and adapter pilot

### Initial supported scope

- one water-based binary/ORC case family;
- advisory benchmark mode;
- native GEOPHIRES-X remains authoritative;
- no direct source merge into GEOPHIRES-X;
- CO2/CPG cases remain deferred;
- separate runtime or container boundary.

### Modernization deliverables

- inventory of GenGEO dependencies and runtime assumptions;
- captured legacy environment;
- legacy golden-case inputs and results;
- supported Python target or reproducible compatibility container;
- package or stable batch entry point;
- removal or isolation of fixed paths;
- removal or isolation of shared temporary files;
- removal or isolation of mutable module-level state;
- independent-process parallel-safety tests;
- documented numerical comparison with legacy GenGEO;
- upstream-contribution versus compatibility-fork decision.

### Adapter deliverables

- GEOPHIRES-X-to-GenGEO adapter;
- water-based ORC mapping document;
- explicit physical and cost boundary definitions;
- normalized plant and economic outputs;
- deterministic comparison cases;
- Monte Carlo and chunked batch tests;
- GEOPHIRES-X/GenGEO comparison report;
- optional three-way comparison with SAM where boundaries overlap;
- LGPL and source-distribution review.

### Exit criteria

- Legacy reference cases are preserved before modernization changes.
- Modernized or containerized results match approved legacy tolerances.
- Independent worker processes run without shared-state interference.
- Unsupported modes are rejected explicitly.
- Native-only GEOPHIRES-X users require no GenGEO installation.
- LGPL code and distribution obligations remain separate and traceable.
- Maintainers decide whether the bounded water-based adapter merits supported status.

## Phase 2B — FGEM operational extension pilot

### Initial supported scope

- GEOPHIRES-X hourly or representative production handoff;
- flexible generation and curtailment;
- thermal and battery storage where supported;
- wholesale energy prices;
- capacity revenues;
- environmental-credit revenues;
- PPA revenue structures;
- chained mode as the default integration boundary;
- full-FGEM comparison only as an optional research mode.

### Deliverables

- FGEM chained adapter;
- time-series schema extension;
- market-input schema extension;
- physical-to-operational boundary document;
- dispatch and curtailment normalization;
- storage state, efficiency, and loss normalization;
- revenue-stream normalization;
- hourly-to-annual aggregation tests;
- flexible-operation golden cases;
- comparison against baseload GEOPHIRES-X economics;
- local, container, and batch execution tests;
- market-assumption and provenance report.

### Exit criteria

- Direct FGEM reference cases are reproduced within approved tolerances.
- Energy-conservation and annual-aggregation checks pass.
- FGEM cannot exceed GEOPHIRES-X physical production limits silently.
- Storage state and losses reconcile over the modeled period.
- Every revenue stream and market assumption is explicit and separable.
- Batch results preserve time-series identity, seeds, and complete provenance.
- Maintainers decide whether the chained FGEM adapter merits supported status.

### Phase 2 completion review

Completion of Phase 2 should produce a decision report covering:

- framework stability across four materially different engines;
- supported versus experimental adapter status;
- ownership and maintenance commitments;
- numerical and model-boundary findings;
- performance and batch-execution findings;
- licensing and distribution findings;
- recommendations for deprecation, continued support, or future proposals.

The completion review may recommend future work, but this roadmap does not define or authorize a Phase 3.

### Suggested Phase 2 issues

1. Capture GenGEO legacy environment and golden cases.
2. Define GenGEO modernization and governance strategy.
3. Produce supported runtime or compatibility container.
4. Implement water-based ORC adapter.
5. Run GenGEO legacy and parallel-safety regression suite.
6. Complete GenGEO licensing and distribution review.
7. Define FGEM physical-to-operational boundary.
8. Extend canonical schema for hourly production and market inputs.
9. Implement FGEM chained adapter.
10. Build storage, dispatch, revenue, and aggregation tests.
11. Complete four-engine framework assessment.
12. Publish Phase 2 completion and support-status decision report.

---

## Explicitly deferred candidates

The landscape survey records several potentially useful tools, but this roadmap contains no implementation work for them:

- PyThermoNomics and OPM/Eclipse postprocessing;
- REopt campus and hybrid-system optimization;
- reV geospatial supply-curve workflows;
- dGeo market-adoption modeling;
- shallow-geothermal component libraries;
- additional commercial or proprietary engines;
- legacy GETEM spreadsheet execution.

A deferred tool requires a separate proposal and community decision. It should not be treated as an implicit next phase.

## Suggested labels

- `proposal`
- `extension-framework`
- `integration`
- `architecture`
- `capability-registry`
- `schema`
- `provenance`
- `comparison`
- `monte-carlo`
- `batch`
- `container`
- `PySAM`
- `SSC`
- `GeoCLUSTER`
- `GenGEO`
- `FGEM`
- `testing`
- `licensing`
- `external-engine`
- `advisory-mode`

## GitHub Project recommendation

A GitHub Project should be created only after Phase 0 is accepted and owners are identified for:

- framework schemas and adapter contracts;
- validation and comparison tooling;
- execution runners and batch behavior;
- PySAM/SSC;
- GeoCLUSTER;
- GenGEO modernization;
- FGEM operational integration;
- licensing and governance.

The Project should contain only the Phase 0–2 scope listed in this roadmap.
