import pytest

from app.db import get_pool, run_migrations


@pytest.fixture
def clean_db():
    """Run migrations, then truncate all tables so each test starts empty."""
    run_migrations()
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute("TRUNCATE TABLE chunks, documents RESTART IDENTITY CASCADE")
        conn.commit()
    yield pool
