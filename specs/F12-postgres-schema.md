# Feature F12 — PostgreSQL Schema + Alembic Migrations (the persistence layer)

> One spec, one session. Self-contained: build this from this file + CLAUDE.md only.

**Status:** DONE
**Master spec sections:** §8.4 (Database Schema — the 10 tables), §8.3 (Component health / the
Supabase-PostgreSQL circuit breaker → `DB_UNAVAILABLE`), §9 (Tech stack — PostgreSQL 15 via
Supabase, JSONB, SQLAlchemy 2.0 + asyncpg, Alembic, RLS), §8.1 (every component has a health
check), §12.2 (shared-instance schema separation `apex_schema` / `strategy_schema`), Project Plan
Wk 15, Risk Register (database unavailability failsafe test)
**Layer / Week:** Platform / Wk 15
**Depends on:** F00 (error enum, `BaseResult`, `config.py`, `data_quality_log` JSONL writer,
`docker-compose.yml`)

## 1. Purpose
F12 is APEX's **persistence layer**: the PostgreSQL schema (10 tables from master §8.4), the
Alembic migration that materialises it, the async SQLAlchemy engine/session that talks to it, and
the DB **health probe + circuit breaker** that make a database outage a *typed, logged* failure
rather than a silent one. It also completes the promise `app/contracts/logging.py` already makes —
the `data_quality_log` JSONL row schema was built as a "drop-in for the F12 PostgreSQL swap", and
F12 delivers that swap (the async DB sink + repository). F12 defines and enforces the schema and its
integrity contracts; it does **not** build the FastAPI endpoints, the `/health` route, Redis, or
Celery — those are F13/F14, which consume F12's engine, health probe, and repositories.

## 2. Upstream / downstream contracts
- **Consumes:**
  - `app/contracts/errors.py` `ErrorCode` (stores `.value` strings in rows; reuses
    `DB_UNAVAILABLE`, `HEALTH_FAIL`, `DATA_QUALITY_LOG_UNWRITABLE`; adds `DB_INTEGRITY`).
  - `app/contracts/enums.py` `ConfidenceState` (the `predictions.confidence_state` ENUM domain),
    `Status`.
  - `app/contracts/results.py` `BaseResult` (every F12 result subclasses it).
  - `app/contracts/config.py` `BreakerConfig` (the §8.3 policy shape; F12 reimplements the breaker
    async-natively rather than importing F01's — see §4.0e), and the new `DatabaseConfig` block.
  - `docker-compose.yml` `postgres:15` service (F00) — the local dev/integration DB.
  - `config.yaml` → new `database:` block; the DSN itself comes from an **environment variable**
    (secret; never committed to YAML).
- **Produces:**
  - `app/db/` package: `base.py` (declarative `Base` + portable type helpers + Alembic naming
    convention), `models.py` (the 10 ORM models), `session.py` (async engine + session factory +
    breaker-wrapped access), `health.py` (`check_database` → `DbHealthResult`), `repository.py`
    (invariant-enforcing writers + the `data_quality_log` async sink + typed exceptions).
  - `alembic.ini` + `migrations/` (`env.py`, `script.py.mako`, `versions/0001_initial_schema.py`).
  - Typed results: `DbHealthResult`, `PersistResult`; typed exceptions `DatabaseUnavailableError`
    (`DB_UNAVAILABLE`), `DbIntegrityError` (`DB_INTEGRITY`).
  - **No committed database.** The migration is the artefact; there is no seeded `.db`/dump in the
    repo (mirrors F04: "machinery only, no committed artefact").
- **Consumed by:** F13 (FastAPI endpoints + `/health` — wraps `check_database` into the health
  contract, calls the repositories from route handlers, maps `DB_UNAVAILABLE` → HTTP 503), F14
  (replay engine reads `laps`/`lap_features`/`degradation_alerts`; routes the live
  `data_quality_log` writes to the DB sink with JSONL as the degraded fallback), F15/F20 (the RQ5
  "database unavailability" failsafe test asserts the 503 + no-partial-write behaviour F12
  guarantees).

## 3. Inputs & outputs (typed)
- **Engine / session** (`app/db/session.py`):
  - `make_engine(config: AppConfig, *, dsn: str | None = None) -> AsyncEngine` — DSN resolved from
    `dsn` arg else `os.environ[database.dsn_env_var]`; missing/empty env var → `ConfigError`
    (`CONFIG_001`, fail-loud at startup, never a silent `None` engine).
  - `make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]`.
  - `session_scope(factory, config) -> AbstractAsyncContextManager[AsyncSession]` — a
    breaker-wrapped context manager: commits on clean exit, rolls back on exception (**no partial
    writes**, §8.3), and re-raises infra faults as `DatabaseUnavailableError` when the breaker is open /
    connection fails.
- **Health probe** (`app/db/health.py`):
  - `check_database(factory, config, *, script_location: str | None = None) -> DbHealthResult` —
    `SELECT 1` inside the breaker + reads the applied Alembic revision, compares to migration head.
    Never raises (mirrors F06b/F10 health): every fault is a coded result.
- **Repository** (`app/db/repository.py`) — thin, invariant-enforcing, async:
  - `write_quality_log_row(session, *, error_code, entity_type, entity_id, error_detail,
    fallback_used, created_at=None) -> PersistResult` — the DB sink mirroring `log_data_quality`'s
    7 columns exactly. **Append-only**: the module exposes no delete for `data_quality_log`.
  - `insert_lap_features(session, *, lap_id, feature_vector, imputed_features, corner_type=None)
    -> PersistResult` — the worked example of app-layer enforcement: computes `null_count` from
    `feature_vector`, and any constraint/invariant violation → `DbIntegrityError` (`DB_INTEGRITY`).
  - `persist(session, obj) -> PersistResult` — generic commit wrapper translating SQLAlchemy
    `IntegrityError` → `DB_INTEGRITY` and `OperationalError`/breaker-open → `DB_UNAVAILABLE`.
- **Outputs:**
  - `DbHealthResult(BaseResult) { connected: bool, revision: str | None, head_revision: str | None,
    latency_ms: float | None, detail: str | None, status, error_code, flags, logged }`. `status=OK`
    only when `connected and revision == head_revision`; else `FAILED` + `HEALTH_FAIL` (detail says
    which: unreachable / stale-revision).
  - `PersistResult(BaseResult) { entity_type: str, entity_id: str | None, written: bool, status,
    error_code, flags, logged }`. Success → `OK` + `written=True` + `entity_id`; integrity violation
    → `FAILED` + `DB_INTEGRITY` + `written=False`; breaker-open → `FAILED` + `DB_UNAVAILABLE`.
  - `DatabaseUnavailableError(Exception)` carrying `ErrorCode.DB_UNAVAILABLE`; `DbIntegrityError(Exception)`
    carrying `ErrorCode.DB_INTEGRITY` (house pattern: infra faults are typed exceptions —
    `ConfigError`, `DataQualityLogError`, `FeatureComputationError`).

## 4. Method / algorithm

### 4.0 Design notes (the decisions a builder must not re-litigate)
- **(a) Async-first, SQLAlchemy 2.0.** `create_async_engine` + `AsyncSession` (master §9: "SQLAlchemy
  2.0 + asyncpg. Async DB access."). Postgres driver = `asyncpg` (deploy/integration); the offline
  unit suite uses `aiosqlite`. `greenlet` (already installed) backs SQLAlchemy's async bridge.
- **(b) Dialect-portable column types** so `Base.metadata.create_all()` works on SQLite for the
  offline unit suite AND the real Alembic migration emits Postgres-native DDL:
  - UUID PK → SQLAlchemy generic `Uuid` (native `UUID` on PG, `CHAR(32)` on SQLite).
  - JSONB columns → `postgresql.JSONB().with_variant(JSON, "sqlite")`.
  - `imputed_features TEXT[]` → `postgresql.ARRAY(Text).with_variant(JSON, "sqlite")`.
  - ENUMs → SQLAlchemy `Enum(PyEnum)` (native `CREATE TYPE` on PG, `VARCHAR + CHECK` on SQLite).
- **(c) Test strategy — hybrid (the house "runs anywhere" rule).** The default `pytest -q` suite
  runs entirely on an **in-memory aiosqlite** engine (`create_all`), covering models, invariants,
  repository codes, breaker, health, and the `data_quality_log` swap — no external service. A
  **separately marked** (`@pytest.mark.postgres`, auto-skipped unless `database.test_dsn_env_var`
  is set) integration test runs the *real* Alembic migration against the docker-compose Postgres and
  asserts `upgrade head` → `downgrade base` round-trips and the live schema matches `Base.metadata`.
- **(d) Error-code parsimony.** Reuse the three pre-seeded codes (`DB_UNAVAILABLE`, `HEALTH_FAIL`,
  `DATA_QUALITY_LOG_UNWRITABLE`) and add exactly **one** new code, `DB_INTEGRITY`, for a
  constraint/app-invariant violation on write (**divergence #24** — master §8.4 assigns the
  invariants to "the application layer" but names no code string; parsimony precedent: F06b added 1,
  F02 added 2). Domain error codes (`LAP_*`, `DATA_*`, …) are *stored as data* in row columns, not
  raised by F12.
- **(e) DB circuit breaker — same §8.3 policy, async-native reimplementation** (**build-time
  refinement**, divergence #26). The policy is the shared `BreakerConfig`
  (`timeout_s: 5, retries: 3, backoff_s: [1, 2, 4], open_after: 3, window_s: 30`), but F12 does NOT
  import F01's `call_with_breaker`: that lives in `app/pipeline/ingest.py`, which imports pandas +
  pyarrow at module top, so importing it would drag the data-pipeline stack — and a backwards
  `app.db → app.pipeline` dependency — into the persistence layer. `app/db/breaker.py` reimplements
  the identical state machine async-natively (`DbCircuitBreaker`), adding an `asyncio.wait_for(fn(),
  timeout_s)` per attempt (a DB call has no other timeout home). Breaker-open / retry-exhaustion →
  `DatabaseUnavailableError` (`DB_UNAVAILABLE`). Retryable = connection faults only
  (`OperationalError`, `InterfaceError`, both the builtin and `sqlalchemy.exc` `TimeoutError`); a
  data fault (`IntegrityError`) is never retried or counted.
- **(f) ENUM domains follow the CODE enums, not the master's prose paraphrase** (**divergence #25**,
  precedented by F06b/F08 which already ratified `LOW_CONFIDENCE`/`OUT_OF_DISTRIBUTION`):
  `confidence_state` = `ConfidenceState` (`HIGH, MEDIUM, LOW_CONFIDENCE, AMBIGUOUS,
  OUT_OF_DISTRIBUTION`) — a superset of §8.4's `HIGH/MEDIUM/LOW/AMBIGUOUS` (`LOW`→`LOW_CONFIDENCE`
  naming + the mandatory OOD state, non-negotiable #4). `alert_confidence` = `(LOW, MEDIUM, HIGH)`
  (F10 bands). `corner_type` = `(slow, medium, high_speed)` (F09). `circuit_type` = `config.circuit_types`
  (`street, high_speed, technical, mixed`, §8.4). Storing `'LOW'` when the code emits
  `'LOW_CONFIDENCE'` would break round-trips — the DB domain must match what the models produce.
- **(g) `laps.error_code` is VARCHAR, not a DB ENUM.** It stores an arbitrary `ErrorCode.value`
  (the set is large and grows per feature) exactly as `logging.py` does. The clean-lap invariant is a
  DB **CHECK**: `CHECK (NOT is_clean OR error_code = 'OK')` — fail-loud at the DB, not just the app.
- **(h) DSN from environment, never YAML.** `config.yaml` holds only the *name* of the env var
  (`dsn_env_var`) + non-secret knobs. This honours §9 "Row-level security enabled" / secret hygiene:
  no credentials in the repo.
- **(i) `data_quality_log` is the JSONL→Postgres drop-in.** F12 provides the async DB sink with the
  same 7-column row shape; the append-only "Never deleted" rule (§8.4) is enforced by the repository
  exposing no delete. Re-pointing the existing *synchronous* F01–F11 callers is F13/F14 integration
  work; until then the JSONL writer remains the degraded-mode fallback so an outage still means **no
  silent data loss** (§8.3).
- **(j) Deterministic constraint naming.** `Base.metadata` carries a SQLAlchemy naming convention
  (`ix`/`uq`/`ck`/`fk`/`pk` templates) so autogenerated constraint names are stable — this is what
  makes the "migration matches metadata / no drift" test meaningful.
- **(k) ORM class named `RaceSession`** (table `sessions`) to avoid confusion with
  SQLAlchemy `AsyncSession`; all other classes are the singular of the table name.
- **(l) `schema_name` optional.** Default `null` (default schema — SQLite has none). Set to
  `apex_schema` for the §12.2 shared-instance Postgres deploy; the migration and `MetaData(schema=…)`
  honour it. Kept out of the unit suite so aiosqlite stays simple.

### 4.1 The 10 tables (columns + integrity constraints copied verbatim from master §8.4)

| Table | Key columns (verbatim) | Purpose | Integrity constraints (verbatim) |
|-------|------------------------|---------|----------------------------------|
| `drivers` | id UUID PK, code VARCHAR(3), full_name, team, season | Driver master data | UNIQUE(code, season). 8 known + unlimited unknown drivers. |
| `circuits` | id UUID PK, name, type ENUM(street, high_speed, technical, mixed) | Circuit master data | UNIQUE(name). 10 circuits. |
| `sessions` | id UUID PK, circuit_id FK, season, round, session_type | Race session records | UNIQUE(circuit_id, season, round, session_type). |
| `laps` | id UUID PK, session_id FK, driver_id FK, lap_number, is_clean BOOL, error_code, tyre_compound, tyre_age, has_interpolation BOOL | Per-lap metadata with quality flags | NOT NULL on is_clean and error_code. error_code='OK' for clean laps. |
| `lap_features` | id UUID PK, lap_id FK UNIQUE, feature_vector JSONB, null_count INT, imputed_features TEXT[], corner_type ENUM | 28-feature vectors | null_count must equal count of null values in feature_vector. Enforced by application layer. |
| `fingerprints` | id UUID PK, driver_id FK, season, circuit_id, vector JSONB(5D), shap_global JSONB, radar_values JSONB(6D), stability_score FLOAT | Aggregate fingerprint per driver/season/circuit | UNIQUE(driver_id, season, circuit_id). |
| `predictions` | id UUID PK, lap_id FK, predicted_driver_id FK, confidence FLOAT, confidence_state ENUM, shap_values JSONB, shap_available BOOL | Classifier output | confidence_state must be HIGH/MEDIUM/LOW/AMBIGUOUS. No NULL confidence. |
| `setup_recommendations` | id UUID PK, driver_id FK, circuit_id FK, setup_vector JSONB, confidence_vector JSONB, constraint_flags JSONB, model_version VARCHAR | Personalisation output | constraint_flags records any constraint layer overrides. |
| `degradation_alerts` | id UUID PK, session_id FK, driver_id FK, lap_number, deviation_score FLOAT, alert_confidence ENUM, feature_attribution JSONB, lead_time_actual INT | Degradation monitor output | lead_time_actual populated post-race for validation. |
| `data_quality_log` | id UUID PK, created_at, entity_type, entity_id, error_code, error_detail, fallback_used BOOL | All pipeline errors and fallbacks | Every DATA_, LAP_, UPLOAD_, COMBO_, FEATURE_ logged here. Never deleted. |

**Encoded constraints (beyond the verbatim text above):**
- FKs: `sessions.circuit_id → circuits.id`; `laps.session_id → sessions.id`, `laps.driver_id →
  drivers.id`; `lap_features.lap_id → laps.id` (UNIQUE — 1:1); `fingerprints.driver_id → drivers.id`,
  `fingerprints.circuit_id → circuits.id`; `predictions.lap_id → laps.id`,
  `predictions.predicted_driver_id → drivers.id`; `setup_recommendations.driver_id → drivers.id`,
  `setup_recommendations.circuit_id → circuits.id`; `degradation_alerts.session_id → sessions.id`,
  `degradation_alerts.driver_id → drivers.id`.
- `laps`: `is_clean` NOT NULL, `error_code` NOT NULL (default `'OK'`), CHECK `(NOT is_clean OR
  error_code = 'OK')`.
- `predictions`: `confidence` NOT NULL (no NULL confidence), `confidence_state` NOT NULL ENUM.
- `lap_features`: `null_count` NOT NULL; the `null_count == count(nulls in feature_vector)` invariant
  is enforced in `insert_lap_features` (§8.4 "Enforced by application layer"), **not** as DDL.
- `data_quality_log`: `created_at` NOT NULL default `now()`; no `ondelete` cascade anywhere that
  could purge it; repository exposes no delete.
- Content of JSONB columns is opaque to the DB; shape is a Pydantic concern at the F13 boundary.

### 4.2 Alembic
- `migrations/env.py` (async): `target_metadata = Base.metadata`; online mode uses a `create_async_engine`
  on the env-var DSN. Honours `database.schema_name` and the naming convention;
  `render_as_batch=True` only under SQLite. **Offline (`--sql`) mode is unsupported** and fails loud
  with `ConfigError` — the create_all baseline (below) needs a live connection, not statement
  generation (**build-time refinement** vs the original "offline mode emits SQL").
- `migrations/versions/0001_initial_schema.py`: the baseline materialises `Base.metadata` directly
  via `create_all` (`upgrade`) / `drop_all` (`downgrade`) — **drift-free by construction**, dialect-aware
  through `.with_variant` (native `JSONB`/`ENUM`/`UUID`/`TEXT[]` on PostgreSQL; `JSON`/`VARCHAR+CHECK`/
  `CHAR(32)` on SQLite), and `create_all`/`drop_all` manage the PG ENUM-type lifecycle. Chosen over
  hand-written `op.create_table` calls (fragile without a live PG to validate names/enums against);
  F13+ changes are proper autogenerated diffs (**build-time refinement**, divergence #27).
  `alembic upgrade head` and `alembic downgrade base` both succeed.
- `alembic.ini`: `script_location = migrations`; `sqlalchemy.url` is left blank (resolved in `env.py`
  from the env var — never hard-code a DSN in the ini).

### 4.3 DB access + breaker
`session_scope` wraps every unit of work: acquire the session under the async `DbCircuitBreaker` (§8.3
policy), `yield` an `AsyncSession`, `commit` on success, `rollback` on any exception (no partial
writes), and translate driver connection/timeout errors + breaker-open into `DatabaseUnavailableError`.
`persist`/`insert_*` translate `IntegrityError` → `DbIntegrityError`. `check_database` calls the same
breaker so a DB outage is reported identically at health-check time and at serving time.

## 5. config.yaml keys this feature adds
```yaml
# --- F12 persistence layer (PostgreSQL schema + Alembic, §8.3/§8.4/§9) ---------
# The DSN is a secret sourced from the environment (dsn_env_var) — never committed here (§9 RLS /
# secret hygiene). The breaker mirrors the master §8.3 Supabase-PostgreSQL policy verbatim and is a
# BreakerConfig driven by F12's async DbCircuitBreaker (app/db/breaker.py). schema_name is null for the default
# schema (SQLite unit suite + single-instance deploy); §12.2 shared-instance deploys set apex_schema.
database:
  dsn_env_var: APEX_DATABASE_URL          # env var holding the async DSN (e.g. postgresql+asyncpg://…)
  test_dsn_env_var: APEX_TEST_DATABASE_URL # optional PG DSN; when set, enables the marked migration test
  schema_name: null                        # null = default schema; §12.2 sets apex_schema
  migrations_path: migrations              # Alembic script_location
  pool_size: 5                             # async engine pool size (design decision)
  max_overflow: 10                         # overflow connections beyond pool_size (design decision)
  pool_timeout_s: 30                       # wait for a pooled connection before erroring (design decision)
  statement_timeout_s: 5                   # §8.3 — server-side per-statement cap (Supabase timeout)
  echo: false                              # SQL echo (debug only)
  breaker: { timeout_s: 5, retries: 3, backoff_s: [1, 2, 4], open_after: 3, window_s: 30 }  # §8.3 verbatim
```
New typed model `DatabaseConfig` (in `app/contracts/config.py`, standing rule — feature config lives
in `config.py`, `extra="forbid"`), wired as `AppConfig.database: DatabaseConfig | None = None`.
Validators: `breaker` is a `BreakerConfig`; `dsn_env_var`/`migrations_path` non-empty; `pool_size`,
`max_overflow`, `pool_timeout_s`, `statement_timeout_s` all `> 0` (`max_overflow` `>= 0`); `echo`
bool. No cross-block validator needed (the block is self-contained + env-sourced).

## 6. Error codes & failure modes (the failsafe contract)

| Failure mode | Detection | Error code | System response | Logged? | User-facing message |
|--------------|-----------|-----------|-----------------|---------|---------------------|
| DB unreachable / timeout / breaker open | `call_with_breaker` open, or asyncpg/aiosqlite `OperationalError` in `session_scope`/`check_database` | `DB_UNAVAILABLE` | raise `DatabaseUnavailableError` (serving) / `DbHealthResult(connected=False)` (health); **rollback, no partial write** | yes (attempted; JSONL fallback if DB is the thing that's down) | "Database temporarily unavailable — request not persisted." (F13 → HTTP 503) |
| Constraint / app-invariant violation on write (UNIQUE, FK, NOT NULL, clean-lap CHECK, `null_count` mismatch) | `IntegrityError` from commit, or `null_count != count(nulls)` in `insert_lap_features` | `DB_INTEGRITY` *(new, div #24)* | raise `DbIntegrityError`; `PersistResult(status=FAILED, written=False)`; rollback | yes → `data_quality_log` | "Record rejected: violates a data-integrity constraint." |
| Health probe: DB up but schema at wrong Alembic revision (migrations not applied / stale) | `check_database` reads `alembic_version` ≠ migration head | `HEALTH_FAIL` | `DbHealthResult(status=FAILED, connected=True, revision≠head)` | yes | "Database schema out of date — run migrations." (F13 → 503) |
| Missing/empty DSN env var at engine build | `os.environ[dsn_env_var]` absent/empty in `make_engine` | `CONFIG_001` | raise `ConfigError` (fail-loud at startup; never a silent `None` engine) | n/a (startup) | "APEX_DATABASE_URL is not set." |
| `data_quality_log` DB sink itself unwritable | insert into `data_quality_log` fails | `DB_INTEGRITY` (constraint) or `DB_UNAVAILABLE` (connection) | `write_quality_log_row` raises the typed exception; the caller falls back to the F00 JSONL writer (`log_data_quality`) — which is where `DATA_QUALITY_LOG_UNWRITABLE` is raised if *that* sink also fails. No silent loss on any path. | — | "Audit log sink unavailable." |
| `database:` block malformed (bad breaker, non-positive pool, empty env-var name) | `DatabaseConfig` validator at load | `CONFIG_001` (reuse) | raise `ConfigError` at `load_config` | n/a | "config.database is invalid: …" |

## 7. Acceptance criteria (each item must be mechanically testable by /spec-check)
- [ ] `app/db/models.py` defines exactly the **10** master-§8.4 tables with the verbatim column
      names, PKs, FKs, UNIQUE constraints, and NOT-NULL columns in §4.1.
- [ ] `laps` carries CHECK `(NOT is_clean OR error_code = 'OK')` and `error_code` NOT NULL default
      `'OK'`; `predictions.confidence` is NOT NULL; `lap_features.lap_id` is UNIQUE (1:1).
- [ ] The DB ENUM domains equal the code enums: `confidence_state` = `ConfidenceState` (5 values incl.
      `LOW_CONFIDENCE` + `OUT_OF_DISTRIBUTION`), `circuit_type` = `config.circuit_types`,
      `alert_confidence` = `(LOW, MEDIUM, HIGH)`, `corner_type` = `(slow, medium, high_speed)`.
- [ ] `Base.metadata.create_all()` succeeds on an in-memory **aiosqlite** engine (portable types),
      and every table/column/constraint above is present.
- [ ] `alembic upgrade head` then `alembic downgrade base` round-trips cleanly against Postgres
      (marked `@pytest.mark.postgres`; skipped when `test_dsn_env_var` unset).
- [ ] `alembic revision --autogenerate` against a head-migrated DB produces **no** schema diff
      (models ≡ migration — no drift), asserted via Alembic's autogenerate comparison.
- [ ] `insert_lap_features` computes `null_count` from `feature_vector` and rejects a mismatch with
      `DB_INTEGRITY` (the §8.4 application-layer invariant).
- [ ] `session_scope` **rolls back and writes nothing** when the body raises (no partial write), and
      surfaces `DB_UNAVAILABLE` when the breaker is open / connection fails.
- [ ] `check_database` returns `DbHealthResult(connected=True, revision==head, status=OK)` on a
      migrated DB, and `connected=False`/`HEALTH_FAIL` on a down DB (never raises).
- [ ] `write_quality_log_row` inserts a row whose 7 columns match `logging.ROW_COLUMNS` /
      `log_data_quality`'s output exactly; the repository exposes **no** delete for `data_quality_log`.
- [ ] `make_engine` raises `ConfigError` (`CONFIG_001`) when the DSN env var is missing/empty.
- [ ] `DatabaseConfig` rejects a malformed `database:` block with `CONFIG_001`; the shipped
      `config.yaml` `database:` block loads clean.
- [ ] Every error code above has a unit test that triggers it and asserts the typed result/exception
      (`DB_UNAVAILABLE`, `DB_INTEGRITY`, `HEALTH_FAIL`, `CONFIG_001`).
- [ ] No `.fit()` / normalisation concern here (N/A — F12 touches no training data; note it explicitly).
- [ ] All thresholds/knobs (breaker policy, pool, timeout) read from `config.yaml`; no magic numbers;
      the DSN is read from the environment, never hard-coded.
- [ ] `mypy app` clean (SQLAlchemy 2.0 typed `Mapped[...]` models), `ruff` clean.

## 8. Test plan
- **Happy path (aiosqlite, offline):** `create_all` on in-memory async SQLite; insert one row per
  table honouring FKs (driver → circuit → session → lap → lap_features/prediction; fingerprint;
  setup_recommendation; degradation_alert; data_quality_log); assert round-trip reads, the UNIQUE and
  NOT-NULL constraints, and the clean-lap CHECK.
- **Failure fixtures (one per code):**
  - `DB_UNAVAILABLE`: a session factory whose connect raises `OperationalError`, and a breaker forced
    open → assert `DatabaseUnavailableError` / `DbHealthResult(connected=False)`; assert no partial write
    after a mid-transaction raise.
  - `DB_INTEGRITY`: duplicate `drivers(code, season)`; a `lap` with `is_clean=True, error_code='LAP_001'`
    (CHECK); `insert_lap_features` with `null_count` ≠ actual nulls → all raise `DbIntegrityError`.
  - `HEALTH_FAIL`: stamp `alembic_version` to a bogus revision → `check_database` reports stale.
  - `CONFIG_001`: unset DSN env var → `make_engine` raises; malformed `database:` block →
    `load_config` raises.
  - `DATA_QUALITY_LOG_UNWRITABLE`: DB sink insert fails → caller falls back to JSONL (assert the row
    still lands, no silent loss).
- **Property/edge:** empty `feature_vector` (`null_count == 0`); all-28-null vector; `imputed_features`
  empty list vs populated; `confidence_state` accepts every `ConfidenceState` member and rejects a
  bogus string; UUID PK default generates distinct ids; `downgrade base` leaves zero APEX tables.
- **Marked integration (`@pytest.mark.postgres`):** real migration `upgrade head`/`downgrade base`
  round-trip + `Base.metadata` ≡ live schema, against docker-compose Postgres.

## 9. Definition of done
All acceptance criteria MET with passing tests (offline suite green with **no** running Postgres;
marked PG test green when `APEX_TEST_DATABASE_URL` is set), `/apex-review F12` clean on the contract
checklist, `mypy app` + `ruff` clean, INDEX row set to DONE. `failsafe-auditor` run on `app/db/` (this
is the RQ5 "database unavailability" evidence: 503 + no partial write + no silent loss). Record in the
dissertation limitations: (1) full **Row-Level Security** policies (§9) are defined at deployment
(F19), not in the base migration — F12 ships RLS-ready tables, not the policies; (2) the running
suite proves the schema on SQLite semantics + one Postgres migration round-trip, not continuous
Postgres CI; (3) re-pointing the F01–F11 synchronous `data_quality_log` callers to the async DB sink
is F13/F14 integration work — F12 ships the sink and keeps JSONL as the degraded fallback.

## 10. Risks & mitigations (from master risk register where relevant)
- **`asyncpg` has no Python-3.14 (cp314) wheel** (the build env is 3.14; asyncpg is *not* currently
  installed). → **Mitigation/contingency:** the offline unit suite depends only on `aiosqlite` +
  `greenlet` (installed) and is unaffected. For the PG driver, prefer `asyncpg` if a cp314 wheel
  exists; else fall back to **`psycopg[binary]` v3 async** (`postgresql+psycopg://`), a ratified
  stack migration exactly like F04 (numpy/pandas), F06a (mlflow-skinny), F07 (numpy pin) — back-note
  it in `pyproject.toml` + CLAUDE.md. Only the marked PG test + real deploy touch this driver.
- **SQLite ≠ Postgres semantics** (no native JSONB/ARRAY/UUID/ENUM; CHECK enforcement differences)
  could let a bug pass offline that fails on PG. → Portable `.with_variant` types keep the models
  single-sourced; the marked migration test exercises the *real* PG DDL; the "no autogenerate drift"
  test guarantees models ≡ migration.
- **Normalisation leakage (master risk register).** N/A to F12 by construction — F12 persists
  vectors, it never fits a scaler or touches held-out circuits. Stated explicitly so `/apex-review`
  can tick it off rather than wonder.
- **Silent partial write on a mid-transaction failure** (would violate §8.3 "no partial writes"). →
  `session_scope` rolls back on any exception and is unit-tested for exactly that; the RQ5 failsafe
  test re-verifies it end-to-end in F15.
- **`data_quality_log` audit trail lost during a DB outage** (the log is the spine of non-negotiable
  #1). → The DB sink failing falls back to the JSONL writer (no silent loss); the table has no delete
  path ("Never deleted", §8.4).
- **Secret leakage** (DSN with credentials committed to `config.yaml`). → DSN is env-only;
  `config.yaml` stores just the env-var *name*; add an `.env.example` documenting `APEX_DATABASE_URL`.
```
