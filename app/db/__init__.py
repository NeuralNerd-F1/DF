"""APEX persistence layer (F12).

The 10 master-§8.4 tables as SQLAlchemy-2.0 async ORM models, the Alembic migration that
materialises them, the async engine/session + DB circuit breaker, the health probe, and the
``data_quality_log`` DB sink. Kept deliberately import-light: importing ``app.db`` (or the Alembic
env) must NOT pull the pandas/xgboost ML stack — the DB-domain enums are mirrored locally and
drift-guarded in tests rather than imported from the feature modules that own them.

Nothing heavy is re-exported here on purpose; import the concrete module you need
(``app.db.base``, ``app.db.models``, ``app.db.session``, ``app.db.repository``, ``app.db.health``).
"""

from __future__ import annotations
