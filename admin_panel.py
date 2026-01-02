from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from admin_config import DEFAULT_VISIBLE_COLUMNS, load_admin_config_df, save_admin_config, get_effective_config


def _to_float_or_none(x):
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    return None if pd.isna(v) else float(v)


def _to_int_or(x, fallback: int):
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    return fallback if pd.isna(v) else int(v)


def display_super_admin_panel(*, regions: list[str], active_region: str | None, all_data: pd.DataFrame):
    st.subheader("🛠️ Super Admin Configuration")

    region = st.selectbox(
        "Region",
        options=regions or ["Unknown"],
        index=(regions.index(active_region) if active_region in (regions or []) else 0),
    )

    locs = []
    if all_data is not None and not all_data.empty and "Region" in all_data.columns and "Location" in all_data.columns:
        locs = sorted(all_data[all_data["Region"] == region]["Location"].dropna().unique().tolist())

    scope_opts = ["(Region default)"] + locs
    scope = st.selectbox("Location (optional)", options=scope_opts)
    location = None if scope == "(Region default)" else scope

    cfg = get_effective_config(region=region, location=location)

    st.markdown("#### Column Visibility")

    all_cols = sorted(set(DEFAULT_VISIBLE_COLUMNS))
    current_cols = json.loads(str(cfg.get("VISIBLE_COLUMNS_JSON") or "[]"))
    if not isinstance(current_cols, list):
        current_cols = list(DEFAULT_VISIBLE_COLUMNS)

    selected_cols = st.multiselect("Visible columns", options=all_cols, default=[c for c in current_cols if c in all_cols])

    st.markdown("#### Thresholds")
    c1, c2 = st.columns(2)
    with c1:
        bottom = st.text_input(
            "Bottom",
            value="" if cfg.get("BOTTOM") is None or pd.isna(cfg.get("BOTTOM")) else str(cfg.get("BOTTOM")),
            placeholder="Leave blank for no override",
        )
    with c2:
        safefill = st.text_input(
            "SafeFill",
            value="" if cfg.get("SAFEFILL") is None or pd.isna(cfg.get("SAFEFILL")) else str(cfg.get("SAFEFILL")),
            placeholder="Leave blank for no override",
        )

    st.markdown("#### Default Date Range Selection")
    d1, d2 = st.columns(2)
    with d1:
        start_days = st.number_input(
            "Start offset (days from today)",
            value=_to_int_or(cfg.get("DEFAULT_START_DAYS"), -10),
            step=1,
        )
    with d2:
        end_days = st.number_input(
            "End offset (days from today)",
            value=_to_int_or(cfg.get("DEFAULT_END_DAYS"), 30),
            step=1,
        )

    if st.button("💾 Save configuration"):
        updates = {
            "VISIBLE_COLUMNS_JSON": json.dumps(selected_cols or DEFAULT_VISIBLE_COLUMNS),
            "BOTTOM": _to_float_or_none(bottom),
            "SAFEFILL": _to_float_or_none(safefill),
            "DEFAULT_START_DAYS": int(start_days),
            "DEFAULT_END_DAYS": int(end_days),
        }
        save_admin_config(region=region, location=location, updates=updates)
        st.success("Saved")

    st.markdown("#### Current stored rows")
    df = load_admin_config_df()
    if df is None or df.empty:
        st.write("(No config rows yet)")
    else:
        st.dataframe(df.sort_values(["REGION", "LOCATION"], kind="mergesort"), width="stretch", height=260)
