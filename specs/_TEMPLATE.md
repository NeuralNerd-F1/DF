# Feature <ID> — <Name>

> One spec, one session. This document is the single source of truth a session loads
> to build this feature. It must be self-contained: a session should not need the
> 42-page master spec open, only this file plus CLAUDE.md.

**Status:** TODO | SPEC WRITTEN | IN PROGRESS | DONE
**Master spec sections:** <e.g. §5.1, §11.1, Risk Register row 1>
**Layer / Week:** <e.g. ML / Wk 5–6>

## 1. Purpose (one paragraph)
What this feature does and the one problem it solves. No more than 4 sentences.

## 2. Upstream / downstream contracts
- **Consumes:** <upstream feature ids + artefacts, e.g. F04 -> artefacts/normalisation.json>
- **Produces:** <typed output + artefacts, e.g. PredictionResult, artefacts/model.json>
- **Consumed by:** <downstream feature ids>

## 3. Inputs & outputs (typed)
- Input: <shape / Pydantic model>
- Output: <Pydantic result object with all fields, including the uncertainty/error states>

## 4. Method / algorithm
Step-by-step computation. Include exact library, parameters, and any formula. Pull
ranges and constants from the master spec verbatim; list the `config.yaml` keys they
map to.

## 5. config.yaml keys this feature adds
```yaml
# key: value  # meaning + source section
```

## 6. Error codes & failure modes (the failsafe contract)
| Failure mode | Detection | Error code | System response | Logged? | User-facing message |
|--------------|-----------|-----------|-----------------|---------|---------------------|
| ... | ... | ... | typed result returned | data_quality_log? | ... |

## 7. Acceptance criteria (each item must be mechanically testable by /spec-check)
- [ ] <criterion 1, with the exact threshold/range>
- [ ] <criterion 2>
- [ ] Every error code above has a unit test that triggers it and asserts the result.
- [ ] No `.fit()` touches held-out data (if applicable).
- [ ] All thresholds read from config.yaml; no magic numbers.

## 8. Test plan
- Happy-path fixtures: <describe synthetic data that passes>
- Failure fixtures: <one per error code / failure mode>
- Property/edge: <boundaries — off-by-one at the threshold, NaN, empty, degenerate>

## 9. Definition of done
All acceptance criteria MET with passing tests, `/apex-review` clean on the contract
checklist, mypy clean, INDEX row set to DONE. Note any limitation to record in the
dissertation's limitations section.

## 10. Risks & mitigations (from master risk register where relevant)
- <risk> -> <mitigation / contingency>
