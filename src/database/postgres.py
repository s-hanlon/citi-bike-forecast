from __future__ import annotations

import os

import psycopg


DEFAULT_DATABASE_URL = (
    "postgresql://citibike:citibike_dev_password@localhost:5432/citibike"
)


def get_database_url() -> str:
    """Return the configured database URL."""
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def connect(database_url: str | None = None) -> psycopg.Connection:
    """Create a Postgres connection."""
    if database_url is None:
        database_url = get_database_url()

    return psycopg.connect(database_url)