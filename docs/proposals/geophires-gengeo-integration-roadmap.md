# GEOPHIRES–GenGEO Integration Roadmap

This roadmap supports `docs/proposals/geophires-gengeo-integration-proposal.md`.

The roadmap now assumes a **web-service-oriented, deployment-neutral architecture** with Monte Carlo, containerization, and AWS portability included in the MVP.

## Phase 0 — Community alignment

**Goal:** Decide whether the community supports the revised direction.

**Questions to resolve:**

- Is GenGEO modernization acceptable and who should own it?
- Can the hosted GEOPHIRES calculator support asynchronous jobs?
- Should the worker contain both GEOPHIRES-X and GenGEO?
- Is AWS Batch the appropriate reference deployment for large studies?
- What public-service limits and governance are acceptable?

**Exit criteria:**

- Agreement to proceed to specification.
- Initial working group identified.
- Existing hosted-service owner and constraints identified.

## Phase 1 — GenGEO modernization specification

**Goal:** Define the minimum work needed to move GenGEO to a supported runtime without changing scientific results unintentionally.

**Deliverables:**

- target Python version, initially evaluating Python 3.11;
- dependency inventory and upgrade plan;
- standard package structure and `pyproject.toml` design;
- stable programmatic entry point;
- parallel-safety assessment;
- legacy regression-case inventory;
- upstream contribution or compatibility-fork decision.

**Exit criteria:**

- Modernization scope approved.
- Legacy reference results captured.
- Ownership and repository strategy agreed.

## Phase 2 — Interoperability and execution specification

**Goal:** Define the contract shared by hosted, local, and AWS execution.

**Deliverables:**

- supported-mode matrix;
- deterministic request/result schemas;
- Monte Carlo batch request/result schemas;
- canonical unit convention;
- provenance and compatibility fields;
- error and unsupported-mode conventions;
- chunking, seed, retry, and resume conventions;
- advisory-report format;
- initial deterministic and Monte Carlo golden cases.

**Recommended initial modeling scope:**

- electricity only;
- water-based ORC only;
- advisory comparison only;
- native GEOPHIRES-X remains authoritative;
- fallback to native GEOPHIRES-X for unsupported cases.

**Exit criteria:**

- Schemas stable enough for implementation.
- Monte Carlo behavior specified.
- At least three deterministic cases and one representative Monte Carlo case selected.

## Phase 3 — Modernized GenGEO runtime

**Goal:** Produce a supported, testable GenGEO runtime suitable for public hosting and batch execution.

**Deliverables:**

- current supported Python runtime;
- updated dependencies;
- installable package;
- stable API;
- removal or isolation of fixed paths, shared temporary files, and mutable global state;
- deterministic batch entry point;
- cross-platform and container regression tests;
- documented numerical comparison with legacy GenGEO.

**Exit criteria:**

- Golden GenGEO cases pass within approved tolerances.
- Independent worker processes run safely in parallel.
- Container build succeeds reproducibly.

## Phase 4 — Containerized compare-only worker MVP

**Goal:** Build one execution worker usable by every deployment mode.

**Deliverables:**

- versioned container image;
- GEOPHIRES-X-to-GenGEO mapper;
- deterministic `evaluate` endpoint or command;
- batch `evaluate_batch` endpoint or command;
- native-versus-GenGEO advisory comparison;
- full provenance in outputs;
- explicit unsupported-mode fallback.

**Exit criteria:**

- Supported deterministic case runs end-to-end.
- Same request gives equivalent results locally and in CI.
- Native-only GEOPHIRES users require no GenGEO installation.

## Phase 5 — Monte Carlo MVP

**Goal:** Make uncertainty analysis efficient and reproducible from the first public release.

**Deliverables:**

- persistent worker initialization;
- configurable realization chunking;
- local multicore process pool;
- deterministic random seeds;
- raw realization output;
- aggregate statistics;
- partial-failure detection;
- chunk retry and resume;
- chunk-size invariance tests;
- performance benchmarks.

**Exit criteria:**

- Representative Monte Carlo study runs without reinitializing GenGEO for every realization.
- Equivalent seeds produce equivalent results locally and in containers.
- Failed chunks can be rerun independently.
- Performance is suitable for teaching and project use.

## Phase 6 — Hosted calculator integration

**Goal:** Make GenGEO available through the existing public GEOPHIRES interface.

**Deliverables:**

- native / GenGEO / compare selection;
- synchronous deterministic execution;
- asynchronous batch-job submission;
- job status and progress;
- downloadable CSV and JSON results;
- retention and expiration rules;
- quotas, rate limits, and cost controls;
- security and validation controls;
- operational logging and monitoring.

**Exit criteria:**

- Public deterministic run works end-to-end.
- Limited public Monte Carlo works asynchronously.
- Large anonymous jobs are prevented or redirected.
- Hosted results match the versioned worker's local results.

## Phase 7 — AWS reference deployment

**Goal:** Allow research and project users to run large studies in their own AWS accounts.

**Deliverables:**

- AWS Batch job definition;
- job-array and chunk orchestration;
- S3 input/output conventions;
- result aggregation;
- retry and resume workflow;
- optional Spot-compute guidance;
- least-privilege IAM guidance;
- infrastructure as code using Terraform, AWS CDK, or equivalent;
- deployment and teardown documentation;
- optional “prepare AWS run package” workflow from the public calculator.

**Exit criteria:**

- Representative Monte Carlo study runs reproducibly in AWS Batch.
- Successful chunks are preserved across retries.
- Results can be compared directly with local and hosted runs.
- Deployment can be recreated from the public repository.

## Phase 8 — Validation and community review

**Goal:** Decide whether advisory results are scientifically and operationally credible enough for supported release.

**Deliverables:**

- deterministic golden-case suite;
- Monte Carlo reproducibility suite;
- legacy-versus-modernized GenGEO report;
- hosted/local/AWS equivalence report;
- performance and cost benchmarks;
- modeling-assumption documentation;
- compatibility matrix.

**Exit criteria:**

- Numerical differences are understood and documented.
- Runtime and deployment differences do not alter results beyond accepted tolerances.
- Maintainers decide whether explicit override mode should be considered.

## Phase 9 — Optional override pilot

**Goal:** Allow explicit, limited use of selected GenGEO outputs in GEOPHIRES-X economics and reporting.

**Potential fields:**

- net electric power;
- gross electric power;
- parasitic load;
- cycle efficiency;
- selected plant-cost outputs.

**Rules:**

- Override is explicit, never default.
- Native GEOPHIRES-X remains available.
- Failed or unsupported GenGEO runs fall back or fail closed according to user policy.
- Reports identify every overridden field and its provenance.

## Suggested issue breakdown

1. Document hosted GEOPHIRES calculator architecture and operator constraints.
2. Define GenGEO modernization scope and legacy regression cases.
3. Decide upstream contribution versus compatibility fork.
4. Define supported-mode matrix.
5. Define deterministic and batch schemas.
6. Build modernized GenGEO package.
7. Build containerized compare-only worker.
8. Implement persistent Monte Carlo worker and chunking.
9. Implement local multicore execution.
10. Integrate synchronous hosted execution.
11. Integrate asynchronous hosted jobs.
12. Define public quotas, retention, and cost controls.
13. Build AWS Batch reference deployment.
14. Build golden-case and equivalence test suites.
15. Complete licensing and governance review.
16. Decide whether to pilot override mode.

## Suggested labels

- `proposal`
- `integration`
- `GenGEO`
- `architecture`
- `web-service`
- `monte-carlo`
- `aws`
- `container`
- `modernization`
- `testing`
- `licensing`
- `external-engine`
- `advisory-mode`

## GitHub Project recommendation

A GitHub Project is still premature until the community accepts the revised architecture and identifies owners for the modernization, worker, hosted-service, and AWS workstreams.

Once those decisions are made, a Project would be useful with views or workstreams for:

- GenGEO modernization;
- schemas and interoperability;
- execution worker;
- Monte Carlo;
- hosted service;
- AWS reference deployment;
- validation and release.
