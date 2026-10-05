# Upstream reconciliation — 26 September 2026

## Source of truth and scope

The canonical repository for this effort is https://github.com/malcolm-BSD/GEOPHIRES-X.
The upstream source is https://github.com/NatLabRockies/GEOPHIRES-X.
This change reconciles upstream into the canonical fork. Submission of the fork's
unique work to upstream is outside this change.

| Reference | Pinned commit |
| --- | --- |
| Canonical main before reconciliation | `ad63b20b5c208208452d8d3b638cd03a58938378` |
| Upstream main being incorporated | `937e8ed2c5e8350744d1ff206d532780536ab7f6` |
| Common ancestor | `f873f29bd44efee92a8c1d3e21877d2b1f3b35b2` |

The pinned histories contain 307 fork-only and 64 upstream-only commits. The GitHub
backup branch `backup/main-before-upstream-sync-2026-09-26` preserves the original
canonical main. Work is isolated on `sync/upstream-main-2026-09-26`.

The existing draft PR #31 targets an older July sync branch. Its base precedes the
July reconciliation already merged through PR #33. This new reconciliation starts
from current canonical main; it does not modify or close PR #31 or proposal PR #32.

## Integration decisions

The merge retains both parents, without rebasing or discarding fork history.
Fourteen files initially conflicted. Resolutions preserve fork functionality while
adapting the upstream changes to the fork's interfaces and module layout:

- Adopt upstream version 3.18.0, release notes, HIP-RA documentation, security
  documentation, and dependency changes. Preserve the fork's Monte Carlo dependencies.
- Retain split CI test jobs and Windows worker limits; incorporate Python 3.13
  coverage, Windows Agg backend selection, and upstream workflow updates.
- Retain dispatch design-capacity plant costing and add the upstream ORC
  temperature warning.
- Retain the return-value-based input reader and URL/comment parsing. Add upstream's
  warning-suppression argument and adapt HIP-RA-X callers to that interface.
- Keep the refactored output modules. Apply upstream's indirect drilling cost
  reporting and annual license fee reporting in `OutputsEconomics.py`, and the
  one-year annual-array fix in `OutputsProfiles.py`. The surface-results header
  already has upstream's desired blank line in `OutputsSurface.py`.
- Keep the fork's Tcl/Agg fallback alongside upstream's noninteractive CI guard.
- Preserve dispatch, weather, storage and other fork-specific parsed result fields.
- Keep the fork's Wanju example calculation. Re-running original main and the merged
  code gives the same total capital cost of 153.93 MUSD. Only the three drilling
  sub-items change to include the upstream 5% indirect cost factor (57.52, 8.76,
  and 12.10 MUSD). Upstream's unrelated 69.27 MUSD total is not substituted.

Other upstream changes include laterals-per-vertical-well inputs, the one-year
lifetime fix, reservoir cache limits, ending sale-price behavior, closed-loop pumping
energy unit labels, updated schemas and upstream example corrections.

## Validation status

Validation is in progress; this branch is not approved for main.

- Six added regression checks pass for the merged input-reader/HIP-RA interface,
  one-year profile sampling, ordinary annual sampling, CI plot suppression, and
  preservation of the Tcl/Agg fallback.
- Initial targeted tests: 180 passed, 3 skipped, 9 failed. Concurrent test processes
  interfered with repository-wide artifact cleanup, so that run is not an acceptance
  result. Tests are being rerun serially.
- The original canonical main also fails `test_redrilling_examples_equivalence`
  in the same environment. This is being tracked separately from merge regressions.
- Wanju before/after calculation comparison passed as described above.
- Complete serial tests, style/package checks and GitHub CI remain to be recorded.

The legacy HIP-RA tests interpret process arguments as output filenames. Local
multi-path runs therefore use `pytest.main(arguments)` with `sys.argv = ['pytest']`
to prevent test source paths from being treated as output destinations. Any test-
generated tracked-file changes are restored before committing.

## Review and acceptance

Do not merge until validation has been reviewed and Malcolm has explicitly approved
updating canonical main. The repository's PR checklist requires a human to certify
manual verification; an automated assistant must leave that checkbox unchecked.

Use a merge commit (or an approved fast-forward retaining this merge commit), not a
squash or rebase merge: upstream ancestry must remain intact for GitHub's behind count
to reach zero. Re-fetch both main branches immediately before acceptance. If either
has advanced, review and test the new delta before updating main.

No release tags, upstream PR, upstream push, or manual deployment is part of this
change. Updating canonical main will trigger the repository's existing main-branch
workflows, including documentation deployment if its test dependencies pass.
