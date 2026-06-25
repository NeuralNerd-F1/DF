---
name: spec-check
description: >
  Verify that a feature's implementation actually satisfies its spec. Invoke as
  /spec-check <feature-id>. Walks the spec's acceptance-criteria checklist and
  failure-mode table item by item and confirms each is implemented AND has a test
  that exercises it. Produces a conformance report and updates the INDEX status. Run
  this immediately before marking a feature DONE.
argument-hint: <feature-id>
---

Check implementation conformance for feature **$ARGUMENTS**.

1. Read the feature spec in `specs/`. Extract two lists: every acceptance criterion,
   and every documented failure mode (with its error code).
2. For EACH acceptance criterion, locate the implementing code and the test that
   proves it. Mark: MET (code + passing test), PARTIAL (code but no test, or test but
   code gap), or MISSING. Cite file:line for both code and test.
3. For EACH failure mode, confirm there is a test that triggers it and asserts the
   correct error code / typed state is returned. Mark MET / MISSING.
4. Run `poetry run pytest -q` scoped to this feature's tests and report pass/fail.
   Run `poetry run mypy app` on touched modules. Report any failures verbatim.
5. Produce a conformance table (criterion -> status -> evidence) and a separate
   failure-mode table. Then give a single verdict: DONE only if every criterion is MET
   and every failure mode has a passing test; otherwise NOT DONE with the exact gaps.
6. If and only if the verdict is DONE, update this feature's row in
   `specs/00-INDEX.md` to DONE with today's date. Never mark DONE with open gaps.
