# APEX × Claude Code — Build Guide

How to build APEX end-to-end with Claude Code: spec-driven, one feature per session,
plan mode first, custom review gates. This guide is the "why and when". The scaffold
(`CLAUDE.md`, `.claude/agents/`, `.claude/skills/`, `specs/`) is the "what".

---

## 1. The mental model: four context layers

Claude Code gives you four ways to feed context and automate work. Map each to a job
and the system stays coherent over a 22-week build:

| Layer | File(s) | Lifetime | Job in APEX |
|-------|---------|----------|-------------|
| **Constitution** | `CLAUDE.md` | Every session, always loaded | Always-true rules: stack, the 5 non-negotiables, commands, repo map. |
| **Specs** | `specs/Fxx-*.md` | Loaded per feature session | The self-contained contract for the feature you're building right now. |
| **Skills (slash commands)** | `.claude/skills/<n>/SKILL.md` | Invoked on demand (`/cmd`) or auto | Repeatable rituals: `/feature-spec`, `/plan-feature`, `/apex-review`, `/spec-check`. |
| **Subagents** | `.claude/agents/<n>.md` | Spawned for a sub-task, own context | Heavy read-only work (explore, audit) that would otherwise pollute your main thread. |

> **Note (2026):** custom slash commands have been merged into skills. Put commands in
> `.claude/skills/<name>/SKILL.md`. The old `.claude/commands/` path still works but
> skills are canonical and add auto-invocation. Run `/help` in your version to confirm.

This directly answers your requirements: **per-feature spec docs** = `specs/`,
**shared context** = `CLAUDE.md` + the one spec, **plan mode + subagents** keep that
context clean, **custom review** = the `apex-review`/`spec-check` skills.

---

## 2. Phase 0 — one-time setup (do before any feature)

1. **Drop the scaffold into your repo root.** You get `CLAUDE.md`, `.claude/agents/`,
   `.claude/skills/`, `specs/00-INDEX.md`, `specs/_TEMPLATE.md`, and two worked specs
   (F01, F06).
2. **Put the master PDF at `docs/APEX_master_spec.pdf`** — the `/feature-spec` command
   reads it to generate the remaining specs.
3. **Build F00 first** (it's in the INDEX): the error-code Enum
   (`app/contracts/errors.py`), the Pydantic result objects, `config.yaml`, the repo
   skeleton, and `docker-compose.yml`. Everything downstream imports these. Do this in
   one short session.
4. **Initialise tooling so the gates work:** `poetry init` + add deps, `ruff`, `mypy`,
   `pytest`, `pre-commit`. Add the CI accuracy gate (macro-F1 > 0.75 or the build
   fails) as a GitHub Action — see §6.
5. **Verify the harness:** run `/help` to confirm your skills appear, and `/agents` to
   confirm the three subagents are registered. Run `/doctor` if skill descriptions look
   truncated.

---

## 3. Phase 1 — decompose the monolith into feature specs

You have one 42-page spec. You want ~22 small ones a session can each hold in working
memory. Don't paste the PDF into every session — that burns context and invites drift.

For each feature in `specs/00-INDEX.md`, run:

```
/feature-spec F03 feature extraction
```

The command reads only the relevant PDF sections, fills `_TEMPLATE.md`, copies error
codes/ranges verbatim (they're contracts), and writes `specs/F03-feature-extraction.md`.
Review it, tighten the acceptance criteria so each line is testable, commit. Two specs
(F01, F06) are already written as worked examples of the bar to hit.

You don't have to pre-generate all 22 up front. Generate a layer at a time, just ahead
of building it, so specs reflect what you actually learned building the layer below.

---

## 4. Phase 2 — the per-feature loop (the core ritual)

**One feature = one session.** Run this loop for each feature, in INDEX order:

```
# 0. Fresh context
/clear

# 1. Load the single source of truth for this feature
"Read specs/F03-feature-extraction.md. This is the feature we're building."

# 2. Plan first — enter plan mode (Shift+Tab cycles into it), then:
/plan-feature F03
#   -> loads the spec, delegates repo recon to the pipeline-explorer subagent
#      (its reading stays OUT of your main context), and proposes a plan mapped
#      to the spec's acceptance criteria. Review and approve before any code.

# 3. Implement (exit plan mode / auto-accept the approved plan)
#   The main session writes code + tests. It already knows the contracts from the
#   spec and the explorer's map. Build tests alongside code — one per acceptance
#   criterion AND one per failure mode.

# 4. Gate before committing
/spec-check F03      # every criterion MET + every failure mode has a passing test?
/apex-review F03     # leakage / silent-failure / seeds / gates / magic-number check
/code-review high    # built-in general bug hunt (run for anything non-trivial)
/security-review     # built-in, for upload/API-facing features (F05, F13, F14)

# 5. Fix findings, re-run gates, then commit
"feat(features): implement F03 28-feature extraction with NULL tolerance"
```

Then `/clear` and move to the next feature. The INDEX status row is your progress
ledger; `/spec-check` flips it to DONE only when there are zero gaps.

### Why this keeps "shared context" coherent
"Shared context" within a feature session = **CLAUDE.md (persistent) + the one spec
(loaded) + the conversation**. That's deliberately small and stable. The things that
would bloat it — scanning the repo, reading docs, auditing — are pushed into
**subagents**, which read in their own window and return only a summary. In plan mode
Claude Code does this automatically (Explore/Plan subagents skip `CLAUDE.md` and git
status to stay lean). That's the whole trick to fitting a full feature in one session.

### When a feature is too big for one session
Some features (F06 classifier+tuning, F13 the whole API) won't fit. Two options, both
spec-driven:
- **Split the spec.** e.g. F06a "training + calibration + MLflow", F06b "inference +
  confidence gates + serving". Each gets its own session and INDEX row.
- **Checkpoint with a handoff note.** Before context fills, ask Claude to append a
  "session handoff" block to the spec (decisions made, files touched, what's left).
  Next session: `/clear`, reload the spec, read the handoff, continue. Use `/compact`
  sparingly — a clean reload from the spec beats a compacted ramble.

---

## 5. The custom review commands (your quality gates)

Four skills do the heavy lifting; layer them with the built-ins:

- **`/feature-spec <id> <area>`** — turns master-PDF sections into a session-ready spec.
- **`/plan-feature <id>`** — disciplined plan-mode kickoff; delegates recon to subagents.
- **`/apex-review [id]`** — APEX-specific contract review: no silent failure, no
  normalisation leakage, fixed seeds, mandatory confidence gates, no magic numbers. It
  delegates ML-deep checks to `ml-auditor` and failure-mode coverage to
  `failsafe-auditor`.
- **`/spec-check <id>`** — conformance gate: each acceptance criterion and failure mode
  must have implementing code AND a passing test before a feature can be DONE.

Built-ins to combine with them: `/code-review high` (general correctness bug hunt),
`/review` (PR review), `/security-review` (for upload + API features). The custom ones
catch APEX's domain traps the generic ones can't (leakage, error-code discipline,
RQ5 failsafe coverage); the built-ins catch ordinary bugs. Use both.

---

## 6. Build sequencing (don't fight the dependency graph)

Follow `specs/00-INDEX.md` top to bottom. The ordering rules that matter most:

- **F00 before everything.** Contracts (error codes, result objects, config) are
  imported everywhere.
- **F04 is the leakage chokepoint.** F06, F08, F10, F11 all load
  `artefacts/normalisation.json`. Get the train-only fit contract bulletproof here, with
  the `ml-auditor` clean, before building any model on top.
- **Models before the API finishes.** Stub F13's Pydantic schemas around Wk 15 so the
  contract exists, but wire real outputs only after F06–F11 are DONE.
- **Held-out circuits (Belgium, Italy) are quarantined from F00.** They appear in
  exactly one place: the F20 evaluation. Any `.fit()` that sees them is a build-breaking
  bug your `ml-auditor` and `/apex-review` exist to catch.

**CI accuracy gate** (`.github/workflows/ci.yml`): on every push run `ruff`, `mypy`,
`pytest`, and a step that loads the champion model and fails the build if held-out
macro-F1 drops below 0.75. This makes "every training run is reproducible and gated" a
mechanical guarantee, not a hope.

---

## 7. APEX-specific gotchas to keep front-of-mind

- **Silent failure is the enemy the whole spec is built against.** If you ever see
  Claude write `except: pass` or drop laps without logging, stop and point it at
  non-negotiable #1. The `failsafe-auditor` is your safety net, but reviewing matters.
- **Magic numbers creep in fast.** 0.65, 30, 6, 20Hz, 115%, 2.9 bits — every one lives
  in `config.yaml`. `/apex-review` flags strays.
- **Vectorise.** The spec mandates no per-row Python loops in feature extraction. Review
  for `.iterrows()` / `.apply(axis=1)` and push back.
- **SHAP and UMAP have fallbacks by design** (prediction-without-SHAP; PCA when UMAP
  fails). Don't let "make it work" delete the fallback path — the fallback IS the
  feature.
- **Three.js (F18) is the riskiest UI piece.** The D3 2D fallback is required and must
  be invisible to the user. Build and test the fallback in the same session.
- **MCP / connectors:** if you want Claude Code to file issues or read your board while
  building, add an MCP server (`claude mcp add ...`). Optional, not required to build.

---

## 8. The 60-second version

1. Scaffold in, PDF at `docs/`, build F00.
2. `/feature-spec` a layer of features from the PDF.
3. Per feature: `/clear` -> read spec -> plan mode `/plan-feature` -> implement with
   tests -> `/spec-check` + `/apex-review` (+ built-in reviews) -> fix -> commit.
4. Subagents do the reading/auditing so your session stays lean.
5. Follow the INDEX order; guard F04 leakage and the Belgium/Italy quarantine; keep the
   CI accuracy gate green.
