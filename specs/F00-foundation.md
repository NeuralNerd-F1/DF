# Feature F00 — Foundation Contracts

> One spec, one session. This document is the single source of truth a session loads
> to build this feature. Authored during the F00 plan-feature pass (it did not exist
> in the original scaffold, which shipped only F01/F06 as worked examples).

**Status:** DONE
**Master spec sections:** §3.3, §3.4, §4.5, §5, §6.2, §7.2, §8.2–§8.5 (error tables); constitution non-negotiables #1–#5
**Layer / Week:** Foundation / Wk 1

## 1. Purpose
Establish the contracts every other feature imports: the canonical error-code
registry, the shared result/enum base, the `config.yaml` single-source-of-truth and
its typed loader, the `data_quality_log` writer, the repo skeleton, and the local
dev stores (`docker-compose.yml`). Nothing downstream can satisfy the "no silent
failure" rule without these, so F00 is built first. It has no upstream dependencies.

## 2. Upstream / downstream contracts
- **Consumes:** nothing (`—`).
- **Produces:** `app/contracts/{errors,enums,results,config,logging}.py`, `config.yaml`,
  `pyproject.toml`, `docker-compose.yml`, the `app/` package skeleton, `artefacts/`.
- **Consumed by:** every feature F01–F22.

## 3. Inputs & outputs (typed)
- `ErrorCode(str, Enum)` — complete registry seeded from the master spec.
- `Status(str, Enum)` = `OK | DEGRADED | FAILED`; `ConfidenceState(str, Enum)` =
  `HIGH | MEDIUM | LOW_CONFIDENCE | AMBIGUOUS | OUT_OF_DISTRIBUTION`.
- `BaseResult(BaseModel)` — `status`, `error_code`, `flags`, `logged`, `.ok`; `extra=forbid`.
- `load_config(path) -> AppConfig` — typed access; raises `ConfigError` (`CONFIG_001`).
- `log_data_quality(...) -> dict` — one row matching master-spec §8.4 columns
  (`id, created_at, entity_type, entity_id, error_code, error_detail, fallback_used`);
  raises `DataQualityLogError` (`DATA_QUALITY_LOG_UNWRITABLE`) if the sink is unwritable.

## 4. Method
The `ErrorCode` enum is the **complete canonical registry**: every code/flag/fallback
in the master spec (§3–§8) is enumerated and tagged with its owning feature, so no
downstream feature ever hardcodes a string. `BaseResult` provides the strict Pydantic
base each feature subclasses. `config.yaml` holds seeds, paths, held-out circuits, and
the named thresholds; the loader strictly validates the foundational blocks while
tolerating feature-owned sections (`extra="allow"` at the top level). The
`data_quality_log` writer is a JSONL append matching the §8.4 schema so F12 can swap
to PostgreSQL with zero caller changes.

## 5. config.yaml keys this feature adds
```yaml
seeds: { numpy: 42, xgboost: 42, optuna: 42 }
paths: { artefacts_dir, cache_dir, data_quality_log }
held_out_circuits: [Belgium, Italy]
thresholds: { confidence_gate: 0.65, entropy_ambiguous_bits: 2.9, null_ceiling: 6,
              min_clean_laps: 30, sample_rate_floor_hz: 20, laptime_bound_pct: 115 }
```

## 6. Error codes & failure modes (the failsafe contract)
| Failure mode | Detection | Error code | System response | Logged? | User-facing message |
|---|---|---|---|---|---|
| config.yaml missing / unreadable | `load_config` | `CONFIG_001` | raise `ConfigError`, no defaults invented | n/a (pre-log) | startup error |
| required config key absent/invalid | Pydantic validation | `CONFIG_001` | raise `ConfigError` | n/a | startup error |
| data_quality_log sink unwritable | `OSError` on write | `DATA_QUALITY_LOG_UNWRITABLE` | raise `DataQualityLogError`, never swallow | n/a (is the logger) | operator alert |

## 7. Acceptance criteria
- [ ] `ErrorCode` is the complete canonical registry (drift-guarded by `test_errors`);
      every code named in `CLAUDE.md`/F01/F06 is present; all values unique; `str` enum.
- [ ] `ConfidenceState` has the 5 mandatory states; `Status` has `OK/DEGRADED/FAILED`.
- [ ] `BaseResult` is a Pydantic v2 model with `status/error_code/flags/logged`,
      `extra=forbid`; subclasses inherit the strictness.
- [ ] `config.yaml` holds all seeds (=42) and every named threshold; loader reads them
      typed; no magic numbers in code.
- [ ] Missing config file/key → `ConfigError` (`CONFIG_001`), never a bare exception.
- [ ] `log_data_quality` writes a §8.4-schema row; an unwritable sink raises
      `DataQualityLogError` (`DATA_QUALITY_LOG_UNWRITABLE`).
- [ ] `held_out_circuits == [Belgium, Italy]`, read from config.
- [ ] Repo skeleton matches the CLAUDE.md repo map; `docker-compose.yml` defines
      Postgres 15 + Redis 7.
- [ ] Every error code / failure path above has a unit test that triggers it; `pytest`,
      `ruff`, `mypy` clean.

## 8. Test plan
- Happy path: load the shipped `config.yaml`; construct `BaseResult`; write a log row.
- Failure fixtures: missing file, missing key, unknown key, non-mapping, invalid YAML;
  unwritable log sink.
- Property/edge: enum registry equals the expected canonical set; name == value;
  values unique; feature-owned config section tolerated without breaking validation.

## 9. Definition of done
All acceptance criteria MET with passing tests, `/apex-review` clean, `mypy` clean,
INDEX row set to DONE. Limitation to record: `data_quality_log` is a JSONL file in F00;
the PostgreSQL-backed table arrives in F12 (drop-in via the same `log_data_quality`
signature and §8.4 schema).

## 10. Risks & mitigations
- **Enum drift** as later specs are written → `test_errors` asserts the canonical set;
  new codes are appended here and to that set, keeping a single registry.
- **Premature `data_quality_log` design vs F12 Postgres** → callers depend only on
  `log_data_quality()`; the JSONL backend swaps behind the §8.4 schema with no churn.
- **Config schema lock-in** → loader tolerates feature-owned nested sections (tested).
- **Held-out quarantine** must hold from F00 → `held_out_circuits` is a config constant
  from day one; downstream leakage guards read it, never hardcode Belgium/Italy.
