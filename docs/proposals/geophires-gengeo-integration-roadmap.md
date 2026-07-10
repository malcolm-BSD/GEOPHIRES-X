# GEOPHIRES–GenGEO Integration Roadmap

This roadmap supports the companion proposal: `docs/proposals/geophires-gengeo-integration-proposal.md`.

The roadmap is intentionally lightweight. It is meant to help the GEOPHIRES-X community discuss scope before committing to implementation.

## Phase 0 — Community discussion

**Goal:** Decide whether the concept is worth exploring.

**Deliverables:**

- Proposal PR opened for review.
- Parent tracking issue created.
- Initial comments from GEOPHIRES-X and GenGEO stakeholders.
- Decision on whether the bridge should live in this repository or a separate repository.

**Exit criteria:**

- There is agreement on whether to proceed to a narrow specification phase.
- A maintainer or small working group is identified.

## Phase 1 — Interoperability specification

**Goal:** Define the bridge before writing production code.

**Deliverables:**

- Supported-mode matrix.
- Request schema.
- Result schema.
- Unit convention.
- Error convention.
- Advisory-report format.
- Initial golden-case list.

**Recommended initial scope:**

- electricity-only use case;
- water-based ORC only;
- advisory mode only;
- no required GenGEO dependency in GEOPHIRES-X core;
- fallback to native GEOPHIRES-X for unsupported cases.

**Exit criteria:**

- Schema is stable enough for prototype implementation.
- At least three candidate golden cases are selected.
- Unsupported cases are explicitly defined.

## Phase 2 — Compare-only bridge MVP

**Goal:** Demonstrate that GEOPHIRES-X boundary conditions can be mapped into a GenGEO-backed calculation and reported side-by-side.

**Deliverables:**

- Minimal bridge runner.
- Request/result JSON files.
- Unit-normalization tests.
- Advisory result comparison.
- Clear unsupported-mode handling.

**Non-goals:**

- No override of GEOPHIRES-X results.
- No broad mode coverage.
- No attempt to vendor GenGEO into GEOPHIRES-X.

**Exit criteria:**

- Supported case runs end-to-end.
- Unsupported cases fall back cleanly.
- Results are reproducible locally.
- The community can inspect GEOPHIRES-X versus GenGEO deltas.

## Phase 3 — Validation and golden cases

**Goal:** Determine whether advisory-mode results are credible enough to justify deeper integration.

**Deliverables:**

- Golden-case test suite.
- Field-level numerical tolerances.
- Delta reports for low-, medium-, and high-temperature ORC cases.
- Documentation of known differences in modeling assumptions.

**Exit criteria:**

- Numerical differences are understood and documented.
- CI or documented reproducibility workflow exists.
- Maintainers decide whether override mode should be piloted.

## Phase 4 — Optional override pilot

**Goal:** Allow explicit, limited use of selected GenGEO-backed outputs in GEOPHIRES-X economics/reporting.

**Potential override fields:**

- net electric power;
- gross electric power;
- parasitic load;
- cycle efficiency;
- selected plant-cost fields.

**Rules:**

- Override mode must be explicit.
- Native GEOPHIRES-X remains the default.
- Unsupported or failed bridge runs must fall back to native GEOPHIRES-X unless the user requests fail-closed behavior.
- Reports must clearly state when external-engine outputs were used.

**Exit criteria:**

- Override mode passes golden-case tests.
- Reports include provenance.
- Maintainers agree on user-facing documentation.

## Phase 5 — Productization decision

**Goal:** Decide whether this becomes a supported GEOPHIRES-X feature.

**Options:**

1. Keep as experimental external bridge.
2. Maintain as optional plugin package.
3. Add a formal external-engine hook to GEOPHIRES-X core.
4. Defer or close if the maintenance burden is not justified.

## Suggested issue breakdown

### Issue 1 — Define supported-mode matrix

Scope:

- Identify which GEOPHIRES-X cases can be mapped to GenGEO.
- Start with water-based ORC electricity cases.
- Explicitly list unsupported modes.

Acceptance criteria:

- Markdown table of supported and unsupported modes.
- First-pass list of required GEOPHIRES-X parameters.
- First-pass list of required GenGEO inputs.

### Issue 2 — Define bridge request/result schema

Scope:

- Create JSON schema or dataclass definitions.
- Include units and provenance.
- Define error codes.

Acceptance criteria:

- Versioned request schema.
- Versioned result schema.
- Example request and result files.

### Issue 3 — Identify golden cases

Scope:

- Select existing GEOPHIRES-X examples suitable for ORC comparison.
- Define expected output fields.
- Define tolerances.

Acceptance criteria:

- Three or more candidate cases selected.
- Each case includes input file, native GEOPHIRES-X result, and expected bridge fields.

### Issue 4 — Build compare-only bridge prototype

Scope:

- Build a prototype that runs outside GEOPHIRES-X core.
- Accept request JSON.
- Return result JSON.
- Produce advisory comparison.

Acceptance criteria:

- One supported case runs end-to-end.
- Unsupported case fails cleanly.
- No GenGEO dependency is added to GEOPHIRES-X core.

### Issue 5 — Licensing and dependency review

Scope:

- Document MIT/LGPL interaction.
- Decide whether subprocess, sidecar, package, or container is preferred.
- Document installation implications.

Acceptance criteria:

- Written license/dependency note.
- Maintainer decision on acceptable runtime boundary.

### Issue 6 — Decide whether to create a GitHub Project

Scope:

- Revisit once Issues 1–5 have initial agreement.

Acceptance criteria:

- If implementation proceeds, create a GitHub Project with columns such as Backlog, Spec, Prototype, Validation, Review, Done.
- If implementation does not proceed, close as not planned with rationale.

## Suggested labels

- `proposal`
- `integration`
- `GenGEO`
- `architecture`
- `testing`
- `licensing`
- `external-engine`
- `advisory-mode`

## Project-board recommendation

Do not create a GitHub Project yet.

A GitHub Project will be useful after the community agrees to proceed beyond the proposal stage. Until then, a parent tracking issue plus a small number of scoped issues is enough and avoids over-formalizing a concept that may still change substantially.
