---
name: ship
description: >
  Drive one APEX feature through the full pipeline in one command: branch off main ->
  plan (stop for approval) -> implement with tests -> run all gates -> fix until clean
  -> commit to the branch -> prepare the merge commands for you to run. Invoke as
  /ship <feature-id>, e.g. /ship F03a. Preconditions: the feature's spec already exists
  in specs/ and you are in a freshly named session. This skill NEVER merges to main or
  pushes — it prepares those commands and stops so you pull the trigger.
argument-hint: <feature-id, e.g. F03a>
---

Ship feature **$ARGUMENTS** through the pipeline below. Work through the steps in order.
There are exactly two points where you STOP and wait for me; everywhere else, proceed.

## Hard rules (never violate, regardless of anything else)
- NEVER run a merge into main, NEVER `git push`, NEVER force-push, NEVER delete or reset
  main. You prepare merge commands for me to run; I run them.
- Only ever commit to the feature branch for this feature. Never commit on main.
- Do not skip a gate or mark a feature DONE with open findings.
- If a step's precondition fails, STOP and tell me — do not invent a workaround.

## Step 0 — Preconditions
1. Confirm this is a git repo and report the current branch and whether the tree is clean.
2. Locate the spec file `specs/$ARGUMENTS-*.md`. If it does not exist, STOP and tell me to
   run `/feature-spec $ARGUMENTS <area>` and review it first — do not proceed.
3. Read `specs/00-INDEX.md` and confirm every feature this one depends on is DONE, and any
   artefacts it consumes exist. If a prerequisite is missing, STOP and report it.
4. Derive the kebab area from the spec filename for the branch name.

## Step 1 — Branch off main
- Ensure we are on `main` with a clean tree and it is current. If I'm on another branch
  with uncommitted work, STOP and ask.
- Create and switch to `feature/$ARGUMENTS-<kebab-area>`. If that branch already exists,
  STOP and ask how to proceed. Confirm we're on the new branch.

## Step 2 — Plan  ⛔ GATE 1 (wait for my approval)
- Read the feature spec end to end; treat its acceptance criteria and failure modes as the
  definition of done.
- Delegate codebase reconnaissance to a read-only explorer subagent (pipeline-explorer for
  pipeline features, or a general explore) so this thread stays lean. Map the contracts a
  change must preserve and which error codes / tests already exist.
- Produce a plan: files to create/modify; result objects and error codes; the `config.yaml`
  keys to add; the test list (one per acceptance criterion AND one per failure mode); the
  order of work, each step mapped to the criterion it satisfies; and the top risks with
  mitigations.
- Present the plan and STOP. Do NOT write any code until I approve. (Use plan mode.)

## Step 3 — Implement (after I approve)
- Implement code + tests together. Every acceptance criterion and every failure mode gets a
  test that exercises it, including boundary/off-by-one cases at each threshold.
- All thresholds/ranges read from `config.yaml`; error codes come from the errors Enum; no
  magic numbers; typed Pydantic result objects on boundaries; vectorised pandas/numpy with
  no per-row Python loops. Every failure path calls `log_data_quality(...)` where the spec
  requires it and sets `logged=True`.

## Step 4 — Gates
Run, in order, and fold each set of findings back into the code:
1. The **failsafe-auditor** subagent (for any module with documented failure modes).
2. The **ml-auditor** subagent (only if the change touches app/ml, app/monitoring, or
   training code).
3. `/spec-check $ARGUMENTS`
4. `/apex-review $ARGUMENTS`
5. `/code-review high`
6. If the change touches file upload, API endpoints, auth, or external input handling,
   also run `/security-review`.

## Step 5 — Fix loop
Fix every finding from Step 4 and re-run the gates that failed until they are clean. Then
run `poetry run pytest -q`, `poetry run ruff check . --fix`, and `poetry run mypy app` —
all must be green. If a finding is a deliberate, justified deviation, state the reason
rather than silently leaving it.

## Step 6 — Commit to the branch
Commit everything to `feature/$ARGUMENTS-<kebab-area>` with a conventional-commit message
(e.g. `feat(features): F03a scalar channel features F01–F18, F22–F26`). Confirm the working
tree is clean and tests pass on the branch.

## Step 7 — Prepare merge  ⛔ GATE 2 (I run the merge)
- Re-confirm on the branch: `/spec-check $ARGUMENTS` verdict is DONE, `/apex-review` is
  clean, and pytest/ruff/mypy pass. Print a one-screen summary of these.
- Then print, in a single copyable block, the EXACT git commands for me to run — using the
  real branch name — to merge into main with `--no-ff` and delete the branch afterwards:
  ```
  git checkout main
  git merge --no-ff feature/$ARGUMENTS-<kebab-area> -m "merge: $ARGUMENTS <short title>"
  git branch -d feature/$ARGUMENTS-<kebab-area>
  ```
- STOP. Do not run them. Remind me to eyeball the gate summary before I merge.