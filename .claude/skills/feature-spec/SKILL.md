---
name: feature-spec
description: >
  Generate a single self-contained feature spec in specs/ from the APEX master
  specification. Invoke as /feature-spec <feature-id> <short area>, e.g.
  /feature-spec F06 driver classifier. Extracts only the relevant sections of the
  master PDF, fits them to specs/_TEMPLATE.md, and cross-links dependencies.
argument-hint: <feature-id> <area, e.g. "F06 driver classifier">
disable-model-invocation: true
---

Create the feature spec for: **$ARGUMENTS**

Steps:
1. Read `specs/_TEMPLATE.md` for the required structure and `specs/00-INDEX.md` to
   confirm the feature id, its dependencies, and which features depend on it.
2. Read the master specification at `docs/APEX_master_spec.pdf`. Pull ONLY the
   sections relevant to this feature (tables, error codes, failure modes, valid
   ranges, acceptance criteria). Do not summarise the whole document.
3. Write `specs/<feature-id>-<kebab-area>.md` filling every template section. Rules:
   - Copy error codes, thresholds, and valid ranges verbatim from the master — these
     are contracts, not suggestions.
   - The "Acceptance criteria" section must be a checklist that `/spec-check` can
     later verify mechanically (each item testable).
   - The "Failure modes" section must list every documented failure mode with its
     code, detection, response, and user-facing message.
   - State explicit inputs/outputs as Pydantic-style result objects.
   - Note upstream artefacts consumed and downstream consumers, by feature id.
4. Update the status row for this feature in `specs/00-INDEX.md` to `SPEC WRITTEN`.
5. Print the path of the file you created and a 3-line summary of what a session will
   need to build it. Do not start implementing.
