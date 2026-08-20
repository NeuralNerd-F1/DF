# Running APEX

Every command below was run end-to-end on a clean checkout before this file was written.
Commands are PowerShell, from the repo root (`D:\COMP702\apex-dissertation`).

You need **two terminals**: one for the API, one for the web app. Leave both open.

---

## What you do NOT need

- **No Postgres.** The demo data is a SQLite file, `artefacts/apex_demo.db`.
- **No Redis.** The cache is an optimisation, never a hard dependency. `/health` will
  report `cache_connected: false` and everything still works.
- **No Docker.** `docker-compose.yml` is only for a full Postgres/Redis setup.

---

## Terminal 1 — the API (port 8001)

```powershell
cd D:\COMP702\apex-dissertation
.venv\Scripts\Activate.ps1
$env:APEX_DATABASE_URL = "sqlite+aiosqlite:///artefacts/apex_demo.db"
python -m uvicorn app.api.main:create_app --factory --host 127.0.0.1 --port 8001
```

Two things people get wrong here, both fatal:

1. **`--factory` is required.** The app is built by `create_app()`, so plain
   `uvicorn app.api.main:app` fails with *"Attribute 'app' not found in module"*.
2. **`APEX_DATABASE_URL` must be set**, in *this* terminal, *before* uvicorn starts.
   The DSN is deliberately env-only (never stored in `config.yaml`), so without it the app
   boots but every database-backed screen is empty.

Wait for `Application startup complete.` — first boot takes ~20 s while the model artefacts load.

### Confirm it is actually healthy

```powershell
curl.exe http://127.0.0.1:8001/health
```

Good output — all four models `true`, `db_connected: true`:

```json
{"status":"healthy",
 "models_loaded":{"classifier":true,"stylespace":true,"degradation":true,"personalisation":true},
 "db_connected":true,"cache_connected":false,"version":"0.1.0"}
```

`cache_connected: false` is expected and fine. If any model says `false`, `artefacts/` is
missing or incomplete — see *Artefacts* below.

---

## Terminal 2 — the web app (port 3000)

```powershell
cd D:\COMP702\apex-dissertation\frontend
npm run dev
```

Then open **<http://localhost:3000>**.

The first visit to each screen compiles on demand and takes a few seconds; after that it is instant.

---

## The five screens

| Screen | Path | Needs |
|---|---|---|
| Driver Overview | `/` | `/drivers` |
| Comparison | `/compare` | `/compare/{a}/{b}` |
| Prediction + Setup Tendencies | `/ingest` | `/segments`, `/predict` |
| Tyre Analysis | `/tyres` | `/tyres` |
| Replay Monitor | `/replay` | `/sessions`, `/replay/...` |

---

## When it does not work

### Blank screens, or "Cannot reach the APEX API"

The API is not running, or not on 8001. Check terminal 1, then re-run the `/health` check.

### Every page 404s, or you see an old version of the UI

**A stale `next dev` from a previous session is still holding port 3000.** This is the most
common failure and it is silent — `npm run dev` prints `EADDRINUSE` and exits, but the old
server keeps answering, so the browser shows a broken or outdated app.

Find and kill it:

```powershell
netstat -ano | Select-String ":3000.*LISTENING"     # last column is the PID
Stop-Process -Id <PID> -Force
```

To see what a PID actually is before killing it:

```powershell
Get-CimInstance Win32_Process -Filter "ProcessId=<PID>" | Select-Object -Expand CommandLine
```

Same check for the API on port 8001 (`":8001.*LISTENING"`).

### A prediction says "LOW_CONFIDENCE" and names no driver

Working as designed, not a bug. When the top class probability falls below the 0.65 gate the
system returns an explicit uncertainty state instead of a forced answer. Pick a different
segment from the dropdown for a confident one.

### `npm run build` while `npm run dev` is running

Don't — the build overwrites `.next` underneath the dev server and the running app breaks.
Stop the dev server first.

---

## Artefacts

`artefacts/` (~112 MB) holds the trained model bundle — `classifier.pkl`,
`normalisation.json`, `labels.json`, `driver_profiles.json`, the style-space models, and the
demo database. **It is deliberately not in git.** A fresh `git clone` will not start until that
folder is copied in alongside the code.

---

## Rebuilding from scratch

Only needed on a new machine.

```powershell
cd D:\COMP702\apex-dissertation
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.lock
cd frontend; npm ci; cd ..
```

Then copy `artefacts/` in and follow the two-terminal steps above.

---

## Checks

```powershell
.venv\Scripts\Activate.ps1
pytest                       # 1017 passed, 3 skipped, 1 xfailed
ruff check .
mypy app
cd frontend; npm test        # 233 passed
```

On a **brand-new virtual environment the first `pytest` run can fail one evaluation test**
on timing — the RQ5 failsafe suite has wall-clock assertions and the first import of
pandas/pyarrow/xgboost is slow. Run it a second time before believing a failure.
