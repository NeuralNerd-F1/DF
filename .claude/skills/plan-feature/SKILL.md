---
name: plan-feature
description: >
  Start a disciplined planning pass for one APEX feature before any code is written.
  Invoke as /plan-feature <feature-id>. Loads the feature spec, delegates codebase
  reconnaissance to read-only explorer subagents (keeping main context clean), then
  produces an implementation plan tied to the spec's acceptance criteria. Use this at
  the very start of a feature session, in plan mode.
argument-hint: <feature-id, e.g. F06>
---

Plan the implementation of feature **$ARGUMENTS**. Do NOT write code yet.

1. Read this feature's spec in `specs/` end to end. Treat its acceptance criteria and
   failure modes as the definition of done.
2. Confirm prerequisites: check `specs/00-INDEX.md` that every upstream feature this
   one depends on is marked DONE, and that the artefacts it consumes
   (e.g. `artefacts/normalisation.json`, a trained model) exist. If a prerequisite is
   missing, stop and report it — do not stub around it.
3. Delegate reconnaissance to subagents so this thread stays lean:
   - Use the **pipeline-explorer** subagent for data-pipeline features, or read the
     relevant `app/ml`, `app/api` modules via a general explore, to map the contracts
     a change must preserve.
   - Ask the explorer which error codes already exist in `app/contracts/errors.py`
     and which tests already cover adjacent code.
4. Produce a written plan containing: files to create/modify; the result objects and
   error codes to add; the `config.yaml` keys needed; the test cases to write
   (one per acceptance criterion AND one per failure mode); and the exact order of
   work. Map each plan step to the spec criterion it satisfies.
5. List the top risks for this feature (pull from the spec's risk notes and the
   master risk register) and how the plan mitigates each.

Present the plan and wait for approval before exiting plan mode.
