from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from config import SQLITE_ADMIN_CONFIG_TABLE, SNOWFLAKE_ADMIN_CONFIG_TABLE
from data_loader import DATA_SOURCE, SQLITE_DB_PATH, SNOWFLAKE_WAREHOUSE, get_snowflake_session


@dataclass(frozen=True)
class Scope:
    region: str
    location: str | None


DEFAULT_VISIBLE_COLUMNS = [
    "Date",
    "Location",
    "System",
    "Product",
    "Opening Inv",
    "Close Inv",
    "Batch In",
    "Batch Out",
    "Rack/Lifting",
    "Pipeline In",
    "Pipeline Out",
    "Gain/Loss",
    "Transfers",
    "Notes",
]


def _location_key(loc: str | None) -> str:
    return "*" if loc is None or str(loc).strip() == "" else str(loc).strip()


def _new_row(*, region: str, location: str | None) -> dict:
    return {
        "REGION": str(region).strip() or "Unknown",
        "LOCATION": _location_key(location),
        "VISIBLE_COLUMNS_JSON": json.dumps(DEFAULT_VISIBLE_COLUMNS),
        "BOTTOM": None,
        "SAFEFILL": None,
        "DEFAULT_START_DAYS": -10,
        "DEFAULT_END_DAYS": 30,
    }


def ensure_admin_config_table_sqlite():
    import sqlite3

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()

    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SQLITE_ADMIN_CONFIG_TABLE} (
            REGION TEXT NOT NULL,
            LOCATION TEXT NOT NULL,
            VISIBLE_COLUMNS_JSON TEXT,
            BOTTOM REAL,
            SAFEFILL REAL,
            DEFAULT_START_DAYS INTEGER,
            DEFAULT_END_DAYS INTEGER,
            UPDATED_AT TEXT,
            PRIMARY KEY (REGION, LOCATION)
        )
        """
    )

    cur.execute(f"PRAGMA table_info({SQLITE_ADMIN_CONFIG_TABLE})")
    existing = {r[1] for r in cur.fetchall()}
    desired = {
        "VISIBLE_COLUMNS_JSON": "TEXT",
        "BOTTOM": "REAL",
        "SAFEFILL": "REAL",
        "DEFAULT_START_DAYS": "INTEGER",
        "DEFAULT_END_DAYS": "INTEGER",
        "UPDATED_AT": "TEXT",
    }
    for col, typ in desired.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE {SQLITE_ADMIN_CONFIG_TABLE} ADD COLUMN {col} {typ}")

    conn.commit()
    conn.close()


@st.cache_data(ttl=60, show_spinner=False)
def load_admin_config_df() -> pd.DataFrame:
    cols = [
        "REGION",
        "LOCATION",
        "VISIBLE_COLUMNS_JSON",
        "BOTTOM",
        "SAFEFILL",
        "DEFAULT_START_DAYS",
        "DEFAULT_END_DAYS",
    ]

    if DATA_SOURCE == "sqlite":
        import sqlite3

        ensure_admin_config_table_sqlite()
        conn = sqlite3.connect(SQLITE_DB_PATH)
        df = pd.read_sql_query(f"SELECT {', '.join(cols)} FROM {SQLITE_ADMIN_CONFIG_TABLE}", conn)
        conn.close()
        return df

    session = get_snowflake_session()
    session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()
    query = f"SELECT {', '.join(cols)} FROM {SNOWFLAKE_ADMIN_CONFIG_TABLE}"
    return session.sql(query).to_pandas()


def _persist_sqlite(row: dict):
    import sqlite3

    ensure_admin_config_table_sqlite()
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        f"""
        INSERT INTO {SQLITE_ADMIN_CONFIG_TABLE}
        (REGION, LOCATION, VISIBLE_COLUMNS_JSON, BOTTOM, SAFEFILL, DEFAULT_START_DAYS, DEFAULT_END_DAYS, UPDATED_AT)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(REGION, LOCATION) DO UPDATE SET
            VISIBLE_COLUMNS_JSON=excluded.VISIBLE_COLUMNS_JSON,
            BOTTOM=excluded.BOTTOM,
            SAFEFILL=excluded.SAFEFILL,
            DEFAULT_START_DAYS=excluded.DEFAULT_START_DAYS,
            DEFAULT_END_DAYS=excluded.DEFAULT_END_DAYS,
            UPDATED_AT=datetime('now')
        """,
        (
            row["REGION"],
            row["LOCATION"],
            row.get("VISIBLE_COLUMNS_JSON"),
            row.get("BOTTOM"),
            row.get("SAFEFILL"),
            row.get("DEFAULT_START_DAYS"),
            row.get("DEFAULT_END_DAYS"),
        ),
    )
    conn.commit()
    conn.close()


def _persist_snowflake(row: dict):
    session = get_snowflake_session()
    session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()

    def _sql_str(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "NULL"
        return "'" + str(v).replace("'", "''") + "'"

    def _sql_num(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "NULL"
        return str(float(v))

    def _sql_int(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "NULL"
        return str(int(v))

    region = _sql_str(row["REGION"])
    location = _sql_str(row["LOCATION"])
    vis = _sql_str(row.get("VISIBLE_COLUMNS_JSON") or "[]")
    bottom = _sql_num(row.get("BOTTOM"))
    safefill = _sql_num(row.get("SAFEFILL"))
    start_days = _sql_int(row.get("DEFAULT_START_DAYS"))
    end_days = _sql_int(row.get("DEFAULT_END_DAYS"))

    sql = f"""
    MERGE INTO {SNOWFLAKE_ADMIN_CONFIG_TABLE} t
    USING (
        SELECT
            {region}::STRING AS REGION,
            {location}::STRING AS LOCATION,
            {vis}::STRING AS VISIBLE_COLUMNS_JSON,
            {bottom}::FLOAT AS BOTTOM,
            {safefill}::FLOAT AS SAFEFILL,
            {start_days}::INTEGER AS DEFAULT_START_DAYS,
            {end_days}::INTEGER AS DEFAULT_END_DAYS
    ) s
    ON t.REGION = s.REGION AND t.LOCATION = s.LOCATION
    WHEN MATCHED THEN UPDATE SET
        VISIBLE_COLUMNS_JSON = s.VISIBLE_COLUMNS_JSON,
        BOTTOM = s.BOTTOM,
        SAFEFILL = s.SAFEFILL,
        DEFAULT_START_DAYS = s.DEFAULT_START_DAYS,
        DEFAULT_END_DAYS = s.DEFAULT_END_DAYS,
        UPDATED_AT = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN INSERT (
        REGION, LOCATION, VISIBLE_COLUMNS_JSON, BOTTOM, SAFEFILL, DEFAULT_START_DAYS, DEFAULT_END_DAYS, UPDATED_AT
    ) VALUES (
        s.REGION, s.LOCATION, s.VISIBLE_COLUMNS_JSON, s.BOTTOM, s.SAFEFILL, s.DEFAULT_START_DAYS, s.DEFAULT_END_DAYS, CURRENT_TIMESTAMP()
    )
    """
    session.sql(sql).collect()


def save_admin_config(*, region: str, location: str | None, updates: dict):
    row = _new_row(region=region, location=location)
    row.update(updates or {})

    if DATA_SOURCE == "sqlite":
        _persist_sqlite(row)
    else:
        _persist_snowflake(row)

    load_admin_config_df.clear()


def _rows_for_scope(df: pd.DataFrame, scope: Scope) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    region = str(scope.region).strip() or "Unknown"
    loc = _location_key(scope.location)
    return df[(df["REGION"] == region) & (df["LOCATION"] == loc)].copy()


def get_effective_config(*, region: str, location: str | None) -> dict:
    df = load_admin_config_df()
    reg_row = _rows_for_scope(df, Scope(region=region, location=location))
    wildcard_row = _rows_for_scope(df, Scope(region=region, location=None))

    base = _new_row(region=region, location=location)
    if not wildcard_row.empty:
        base.update({k: wildcard_row.iloc[-1].get(k) for k in base.keys() if k in wildcard_row.columns})
    if not reg_row.empty:
        base.update({k: reg_row.iloc[-1].get(k) for k in base.keys() if k in reg_row.columns})

    return base


def get_visible_columns(*, region: str, location: str | None) -> list[str]:
    cfg = get_effective_config(region=region, location=location)
    raw = cfg.get("VISIBLE_COLUMNS_JSON")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return list(DEFAULT_VISIBLE_COLUMNS)

    cols = json.loads(str(raw) or "[]")
    if isinstance(cols, list) and cols:
        return [str(c) for c in cols]
    return list(DEFAULT_VISIBLE_COLUMNS)


def get_default_date_window(*, region: str, location: str | None) -> tuple[int, int]:
    cfg = get_effective_config(region=region, location=location)
    s = pd.to_numeric(pd.Series([cfg.get("DEFAULT_START_DAYS")]), errors="coerce").iloc[0]
    e = pd.to_numeric(pd.Series([cfg.get("DEFAULT_END_DAYS")]), errors="coerce").iloc[0]
    start = int(s) if pd.notna(s) else -10
    end = int(e) if pd.notna(e) else 30
    return start, end


def get_threshold_overrides(*, region: str, location: str | None) -> dict:
    cfg = get_effective_config(region=region, location=location)
    return {
        "BOTTOM": cfg.get("BOTTOM"),
        "SAFEFILL": cfg.get("SAFEFILL"),
    }
