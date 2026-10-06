# Baseline validation repairs before upstream reconciliation

Canonical repository: https://github.com/malcolm-BSD/GEOPHIRES-X

This repair branch starts at canonical main `ad63b20b5c208208452d8d3b638cd03a58938378`.
It is separate from upstream reconciliation PR #34, whose head is
`8565a13885f8b4d2a6e158042bb86dfaf6171551`. Main has not been changed.

## Confirmed defects repaired

- The example regression harness overrode the shared NaN sanitizer with a method
  returning `None`, then assigned its return value and accessed `.result`.
  This caused the failure in PR #34's GitHub Actions job
  [108528987583](https://github.com/malcolm-BSD/GEOPHIRES-X/actions/runs/36286801212/job/108528987583).
  Use the existing shared sanitizer, remove duplicate simulation calls, and
  select only `.txt` inputs. Numerical comparison tolerances are unchanged.
- The optional catalog optimizer invokes `pulp.PULP_CBC_CMD`, which is removed
  in PuLP 4. An unconstrained dependency silently falls back to the existing
  greedy algorithm. The existing optimal-cost test then returns 1000 instead of
  600. Bound PuLP to `>=2.6.0,<4` in both development dependency declarations.
  PuLP 3.3.2 passes the existing optimal-cost test.
- `example13b_well-integrity.out` was stale. Rerunning both example13 inputs
  produced identical parsed results after metadata removal. Example13's saved
  reference already matched its fresh result. Regenerate only the stale
  well-integrity output and apply the standard whitespace hooks. Both now have
  average net generation 3.77 MW and one redrilling event. No input, calculation,
  or numerical tolerance was changed to obtain this agreement.

## Validation

- Repair branch: 11 targeted tests passed (harness, catalog optimizer, wellbores).
- New harness checks verify a matching result passes, a numerical mismatch still
  fails, simulations run once per input, and generated artifacts are excluded.
- Three real example comparisons passed: example13, example13b_well-integrity,
  and Beckers_et_al_2023_Tabulated_Database_Coaxial_sCO2_heat (the example at which
  the GitHub job previously crashed).
- All pre-commit hooks passed for changed code, dependencies, and reference output.
- A separate local worktree merged these repairs into PR #34 without conflicts.
  Twenty targeted tests passed there, including the upstream reconciliation tests.
- This is targeted validation, not a claim that the full suite is green. The
  original sync audit records sandbox multiprocessing restrictions and resource
  limits. GitHub Actions and human review remain merge gates.

## Fervo modeling decision and approved repair

Fervo_Project_Cape-5.txt explicitly specifies ambient temperature 11.17 C and
states that hourly/seasonal fluctuations are not modeled. It also supplies
project coordinates. The fork's weather integration treats coordinates as an
implicit request for weather and the electricity model uses that profile even
when an ambient temperature was explicitly provided.

The prior baseline/sync runs reported minimum net output 563.3 MW, outside the
unchanged reference range 499 to 505 MW. A diagnostic run on the repair branch
with only weather loading disabled produced 499.10 MW, matching the saved
reference; maximum total generation was 599.80 MW. This diagnostic patch was
not committed; the approved parameter-based implementation is described below.

There is an additional time-alignment concern: `ambient_temperature_profile`
indexes hourly weather by simulation-array position, without converting model
elapsed years to weather time. With 30 years and 12 simulation steps per year,
the 360 model samples receive the first 360 hourly temperatures. Existing weather
tests verify temperature sensitivity but do not establish correct seasonal
alignment. Rebaselining the case study to the current weather output would be
premature.

Seasonal time alignment remains a separate follow-up, with no changes to its
calculations or scientific references in this repair.
On September 28, 2026, the owner chose to keep the fixed-temperature case with
an explicit weather opt-out.

### Implementation of the approved choice

- Add `Use Weather Data`, a boolean defaulting to `True`. Existing coordinate-based
  activation stays compatible; explicit `False` exits weather loading before cache
  or network access, clears the model weather object, and preserves supplied or
  default ambient/surface temperatures. Coordinates remain available as metadata.
- Add `Use Weather Data, False` to Cape-5 and its derived Cape-6 input; existing
  case generation copies this setting from Cape-5. No numerical references changed.
- Update the weather operating guide, parameter descriptions, and generated input
  schema. Validate both enabled and disabled behavior, including false/0 parsing,
  preservation of temperatures/coordinates, and disabled weather with one coordinate.
- Repair the Python 3.8-incompatible context-manager syntax in the harness test
  introduced in the first repair commit. GitHub job 109036630345 identified that
  collection error; all modified Python files now parse with Python 3.8 grammar.
- On the repair branch, 20 targeted tests and two subtests passed (weather parameter
  behavior, Fervo Cape-5 reference/documentation checks, generated schemas, harness).
- Cape-5 and Cape-6 full parsed results match their existing saved references,
  with weather loading mocked to raise if attempted. No weather load was attempted.

## Merge gate

Keep this PR and PR #34 in draft until remaining failures are understood and the
owner approves. The repository checklist requires a human to manually certify
correctness. No certification or approval is implied by automated checks.
