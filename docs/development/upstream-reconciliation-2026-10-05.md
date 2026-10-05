# Upstream reconciliation — October 5, 2026

Canonical fork: https://github.com/malcolm-BSD/GEOPHIRES-X

## Scope and inputs

- Laptop branch reported by the owner: `review/combined` at
  `41daf3bc5874ae531472267a1a45511c68e28aae`, with no tracked edits.
- Reported untracked content: the local virtual environment and six generated
  dispatch output files. None belongs in this merge.
- Reconstructed baseline parents: repair PR #35 at
  `a3ea2debafb40d2028243894213eb142b5321250` and sync PR #34 at
  `8565a13885f8b4d2a6e158042bb86dfaf6171551`.
- Reconstructed baseline tree: `ec3923dff2511a28878989751e14333863680023`.
  The laptop integration instructions verify this tree before proceeding; local
  history alone does not prove that a merge commit has identical file contents.
- Previous upstream revision: `937e8ed2c5e8350744d1ff206d532780536ab7f6`.
- New pinned upstream revision: `0e4af2f6aba2b33c4d834ec1bd4cebb8a6dd9487`.
- Increment: 17 commits, touching 14 files before this audit document.

## Changes and conflict decisions

- Import upstream's `Debt Tenor` parameter for the SAM Single Owner PPA model.
  When omitted, the tenor remains the plant lifetime. An explicit tenor cannot
  exceed the lifetime; upstream tests cover early repayment, default equivalence,
  and invalid tenor rejection.
- Import the request-schema entry, financing documentation, and guidance for
  regenerating example references.
- Update all upstream version declarations to 3.18.3.
- Resolve `docs/conf.py` by retaining fork formatting and accepting version 3.18.3.
- Resolve `setup.py` by retaining fork dependencies (including sympy, scikit-learn,
  tqdm, and jinja2), retaining numpy-financial 1.0.0, adding upstream tox and
  pre-commit development dependencies, and accepting version 3.18.3 and the
  corrected dependency issue link.
- Apply repository quote formatting to the imported debt-tenor test.
- Preserve the existing weather opt-out in Cape-5/Cape-6, fork costing behavior,
  PuLP compatibility bound, and previous reconciliation fixes. No example
  reference outputs or comparison tolerances were changed in the initial merge.
  The subsequent CI repair below updates four weather-dependent references.

## Validation

- 91 targeted tests passed, two existing tests were skipped, and six subtests
  passed on Python 3.12. This includes the complete SAM economics test module,
  general economics, schemas, weather parameters, Fervo Cape-5, upstream
  reconciliation checks, example harness, catalog optimizer, and wellbores.
- Changed files pass pre-commit hooks; imported test quotes were normalized.
- Both Cape-5 and Cape-6 complete parsed outputs match their saved references
  with weather loading configured to fail if attempted; neither loaded weather.
- No tracked numerical reference outputs were regenerated in the initial merge.

## CI follow-up: reproducible weather examples

The owner supplied the failed Python 3.8 CI log from build `37342191661`.
The first failing example was `example1_dispatchable_tess_weather.txt`: its saved
weather mean was 10.55 C, while the live API yielded 10.57 C. Examples 8, 9, and
SHR-3 also depended on live weather and had stale reference values.

Tests now use three checked-in historical API responses for these four examples.
The production normalizer still processes all 8,784 leap-year hours; temperatures
remain variable, and TESS dispatch still uses hourly weather. Each fixture is
checked against the input coordinates and year. Missing fixtures fail instead of
falling back to the network. API retrieval, caches, model calculations, the Fervo
opt-out, and numerical comparison tolerances are unchanged.

Before updating each reference, complete parsed outputs were compared between
the pre-increment baseline tree `ec3923dff2511a28878989751e14333863680023` and
the initial reconciled tree `19bf5a2deed3c541e849acc361dd1e85e7c01e7e` with
identical weather inputs. All four comparisons were exactly equal after the
existing metadata/NaN normalization. The references therefore capture existing
combined-branch behavior, not an unreviewed change in the 17 upstream commits.
See `tests/fixtures/weather/README.md` for source attribution, checksums, and the
offline reference regeneration command.

Validation of this repair on Python 3.12:

- 21 weather/fixture/harness tests passed.
- All four complete weather example comparisons passed with HTTP requests
  configured to raise if attempted.
- A diagnostic pass over later examples found an additional inherited S-DAC-GT
  reference mismatch: summary LCOH is 0.00 instead of -128.08 USD/MMBTU. Baseline
  and candidate complete outputs are identical; this single parsed field differs
  from the saved reference. The existing `_build_basis` helper returns zero for
  nonpositive output. This case has a discounted heat denominator of
  -74,174,793.41094875 and cost numerator of 32.41348270270895 MUSD; their ratio
  with the heat price factor would give the legacy negative value. Its reference
  and calculation were held unchanged until the owner's decision below.
- URL-backed examples 5c and SUTRAExample1a encountered remote-input failures.
  Example 5c passed on retry; SUTRAExample1a remained blocked by remote-file
  validation. These results are not a green full-suite claim.

## Owner-approved heat-price availability

On October 5, the owner directed: "explicitly mark heat price as unavailable when
net heat output is nonpositive."

The shared levelized-cost engine now uses a numeric NaN sentinel for an active
heat commodity with nonpositive discounted net output. Reports display `N/A`
and `Heat price status: Unavailable: net heat <= 0`. The client exposes a `None`
heat price and the status text; optional XLCOH and VALCOH prices and adjustments
also remain unavailable. Positive heat output still permits zero or negative
prices when lifecycle cost credits justify them. Inactive commodities keep their
existing behavior. The heat balance itself is not changed by this reporting fix.

Raw JSON contains a `Project Heat Price` object with `value: null`, units,
`status: unavailable`, and a reason. Unavailable heat output parameters use JSON
null. This separate project-price object avoids the legacy S-DAC flat-JSON
collision: S-DAC also names its distinct geothermal supply-cost metric `LCOH`.
That geothermal supply cost and its existing JSON key remain available.

Only the S-DAC saved reference's project heat-price line and new status line are
updated for this decision. All other parsed S-DAC results must match exactly.
Regression tests cover zero/negative output, positive output with positive/zero/
negative costs, and S-DAC text/HTML/JSON/client reporting with XLCOH and VALCOH.

The preceding weather-repair CI run `37349815174`, Python 3.11 example job
`111897522239`, finished with 45 tests passed, 101 subtests passed, and only the
S-DAC reference mismatch failed. The URL-backed examples passed in that CI job;
the earlier remote-file failures were not reproduced there.

Local validation of the availability change on Python 3.12:

- 50 levelized-cost, XLCO, and VALCO tests passed.
- All 37 report tests passed (including two subtests).
- Six availability tests and 18 client-result tests passed (including three
  subtests); the availability integration checks text, HTML, raw JSON, and client
  parsing together.
- The complete parsed S-DAC example matches its narrowly updated reference.
- Changed-file pre-commit checks passed. Cross-version CI remains required.

## Integration and review

The new branch retains both previous PR histories and the new upstream ancestry.
Its draft PR targets a snapshot of the combined baseline so the review displays
only the new increment. Existing PRs #34 and #35 remain open; canonical main is
not modified by preparing this branch.

On the laptop, first verify the clean tracked working tree and expected baseline
tree, create a backup branch, fetch the published candidate, and merge it on a new
local review branch. This preserves the owner's local merge commit. Git leaves
untracked files in place and refuses a merge if it would overwrite a collision.

Full CI and human review remain required before a main-branch merge. A successful
merge or targeted test run is not a claim that the complete test suite passes.
