"""Load the named queries in queries.sql and run them.

    python -m pipeline.queries            # prints every query (uses P01 for parameterised ones)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SQL_FILE = Path(__file__).resolve().parent / "queries.sql"


def load() -> dict[str, dict]:
    out = {}
    for block in re.split(r"(?m)^-- name: ", SQL_FILE.read_text())[1:]:
        name, rest = block.split("\n", 1)
        comment = "\n".join(l[3:] for l in rest.splitlines() if l.startswith("-- "))
        sql = "\n".join(l for l in rest.splitlines() if not l.startswith("--")).strip().rstrip(";")
        out[name.strip()] = {"description": comment, "sql": sql}
    return out


def run(conn, name: str, **params) -> pd.DataFrame:
    return pd.read_sql_query(load()[name]["sql"], conn, params=params or None)


if __name__ == "__main__":
    from .build_db import connect
    conn = connect()
    pd.set_option("display.width", 160)
    for name in load():
        params = {"patient_id": "P01", "last_visit": "2026-11-06"} if ":patient_id" in load()[name]["sql"] else {}
        print(f"\n=== {name} ===")
        print(run(conn, name, **params).to_string(index=False))
