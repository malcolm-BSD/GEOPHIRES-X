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
  reference outputs or comparison tolerances are changed in this increment.

## Validation

- 91 targeted tests passed, two existing tests were skipped, and six subtests
  passed on Python 3.12. This includes the complete SAM economics test module,
  general economics, schemas, weather parameters, Fervo Cape-5, upstream
  reconciliation checks, example harness, catalog optimizer, and wellbores.
- Changed files pass pre-commit hooks; imported test quotes were normalized.
- Both Cape-5 and Cape-6 complete parsed outputs match their saved references
  with weather loading configured to fail if attempted; neither loaded weather.
- No tracked numerical reference outputs were regenerated.

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
