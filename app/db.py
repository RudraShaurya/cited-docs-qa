from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config import settings

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

_pool: ConnectionPool | None = None


def _configure(conn) -> None:
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    except psycopg.errors.UniqueViolation:
        # Benign race: another pool connection created the extension first.
        conn.rollback()
    register_vector(conn)


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=10,
            configure=_configure,
            open=True,
        )
    return _pool


def run_migrations() -> None:
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename text PRIMARY KEY,
                applied_at timestamptz DEFAULT now()
            );
            """
        )
        applied_rows = conn.execute("SELECT filename FROM schema_migrations").fetchall()
        applied = {row[0] for row in applied_rows}

        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in applied:
                continue
            sql = path.read_text(encoding="utf-8")
            conn.execute(sql)
            conn.execute(
                "INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,)
            )
            conn.commit()


def health_check() -> bool:
    try:
        with get_pool().connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False
