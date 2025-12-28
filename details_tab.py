import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

DETAILS_RENAME = {
    "Open Inv": "Opening Inv",
    "Batch In (RECEIPTS_BBL)": "Batch In",
    "Batch Out (DELIVERIES_BBL)": "Batch Out",
    "Rack/Liftings": "Rack/Lifting",
}

DETAILS_COLS = [
    "source",
    "Product",
    "Close Inv",
    "Opening Inv",
    "Batch In",
    "Batch Out",
    "Rack/Lifting",
    "Pipeline In",
    "Pipeline Out",
    "Gain/Loss",
    "Transfers",
    "Notes",
]

FORECAST_FLOW_COLS = [
    "Batch In (RECEIPTS_BBL)",
    "Batch Out (DELIVERIES_BBL)",
    "Rack/Liftings",
    "Pipeline In",
    "Pipeline Out",
    "Production",
    "Adjustments",
    "Gain/Loss",
    "Transfers",
]

INFLOW_COLS = [
    "Batch In (RECEIPTS_BBL)",
    "Pipeline In",
    "Production",
]

OUTFLOW_COLS = [
    "Batch Out (DELIVERIES_BBL)",
    "Rack/Liftings",
    "Pipeline Out",
]

NET_COLS = [
    "Adjustments",
    "Gain/Loss",
    "Transfers",
]


def _aggregate_daily_details(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    group_cols = ["Date", id_col, "Product"]
    agg_map: dict[str, str] = {}

    if "Open Inv" in df.columns:
        agg_map["Open Inv"] = "first"
    if "Close Inv" in df.columns:
        agg_map["Close Inv"] = "last"

    for c in _available_flow_cols(df):
        agg_map[c] = "sum"

    if "source" in df.columns:
        agg_map["source"] = "first"
    if "updated" in df.columns:
        agg_map["updated"] = "max"
    if "Notes" in df.columns:
        agg_map["Notes"] = "last"

    return df.groupby(group_cols, as_index=False).agg(agg_map)


def _available_flow_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in FORECAST_FLOW_COLS if c in df.columns]


def _ensure_lineage_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "source" not in out.columns:
        out["source"] = "system"
    else:
        out["source"] = out["source"].fillna("system")

    if "updated" not in out.columns:
        out["updated"] = 0
    else:
        out["updated"] = pd.to_numeric(out["updated"]).fillna(0).astype(int)

    return out


def _weekday_weighted_means(
    hist: pd.DataFrame,
    flow_cols: list[str],
    max_weeks: int = 6,
    decay: float = 0.70,
) -> dict[tuple[int, str], float]:
    if hist.empty or not flow_cols:
        return {}

    h = hist.sort_values("Date").copy()
    h["__weekday"] = h["Date"].dt.weekday

    recent = h.tail(21)
    fallback = {c: float(recent[c].mean()) if c in recent.columns and len(recent) else 0.0 for c in flow_cols}

    out: dict[tuple[int, str], float] = {}
    for wd in range(7):
        subset = h[h["__weekday"] == wd].sort_values("Date", ascending=False).head(max_weeks)
        if subset.empty:
            for c in flow_cols:
                out[(wd, c)] = fallback.get(c, 0.0)
            continue

        weights = np.array([decay**i for i in range(len(subset))], dtype=float)
        wsum = float(weights.sum()) or 1.0

        for c in flow_cols:
            vals = subset[c].astype(float).to_numpy() if c in subset.columns else np.zeros(len(subset), dtype=float)
            out[(wd, c)] = float((vals * weights).sum() / wsum)

    return out


def estimate_forecast_flows(
    group: pd.DataFrame,
    flow_cols: list[str],
    d: pd.Timestamp,
) -> dict[str, float]:
    means = _weekday_weighted_means(group, flow_cols=flow_cols)
    wd = int(d.weekday())
    return {c: float(means.get((wd, c), 0.0)) for c in flow_cols}


def _roll_inventory(prev_close: float, flows: dict[str, float], flow_cols: list[str]) -> tuple[float, float]:
    opening = float(prev_close)
    inflow = sum(float(flows.get(c, 0.0) or 0.0) for c in INFLOW_COLS if c in flow_cols)
    outflow = sum(float(flows.get(c, 0.0) or 0.0) for c in OUTFLOW_COLS if c in flow_cols)
    net = sum(float(flows.get(c, 0.0) or 0.0) for c in NET_COLS if c in flow_cols)
    closing = opening + inflow - outflow + net
    return opening, closing


def _forecast_dates(last_date: pd.Timestamp, forecast_end: pd.Timestamp | None, default_days: int) -> pd.DatetimeIndex:
    start = last_date + timedelta(days=1)
    if forecast_end is not None:
        if start > forecast_end:
            return pd.DatetimeIndex([])
        return pd.date_range(start=start, end=forecast_end, freq="D")
    return pd.date_range(start=start, periods=int(default_days), freq="D")


def _last_close_inv(group: pd.DataFrame) -> float:
    if "Close Inv" not in group.columns:
        return 0.0

    last_date = group["Date"].max()
    last_rows = group[group["Date"] == last_date]
    if last_rows.empty:
        return 0.0

    val = last_rows["Close Inv"].iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def _extend_with_30d_forecast(
    df: pd.DataFrame,
    *,
    id_col: str,
    forecast_end: pd.Timestamp | None = None,
    default_days: int = 30,
) -> pd.DataFrame:
    if df.empty:
        return df

    daily = _aggregate_daily_details(df, id_col=id_col)
    if daily.empty:
        return daily

    daily = _ensure_lineage_cols(daily).sort_values("Date")
    flow_cols = _available_flow_cols(daily)

    if forecast_end is not None:
        forecast_end = pd.Timestamp(forecast_end)

    forecast_rows: list[dict] = []

    for (id_val, product), group in daily.groupby([id_col, "Product"], dropna=False):
        group = group.sort_values("Date")
        last_date = pd.Timestamp(group["Date"].max())
        prev_close = _last_close_inv(group)
        for d in _forecast_dates(last_date, forecast_end, default_days):
            flows = estimate_forecast_flows(group, flow_cols=flow_cols, d=d)
            opening, closing = _roll_inventory(prev_close, flows, flow_cols)
            prev_close = closing

            row = {
                "Date": d,
                id_col: id_val,
                "Product": product,
                "source": "forecast",
                "updated": 0,
                "Notes": "Forecast",
                "Open Inv": opening,
                "Close Inv": closing,
                **flows,
            }
            forecast_rows.append(row)

    if not forecast_rows:
        return daily

    combined = pd.concat([daily, pd.DataFrame(forecast_rows)], ignore_index=True)
    for c in ["Open Inv", "Close Inv"] + flow_cols:
        if c in combined.columns:
            combined[c] = pd.to_numeric(combined[c]).fillna(0.0)

    return combined


def build_details_view(df: pd.DataFrame, id_col: str):
    df = df.sort_values("Date").rename(columns=DETAILS_RENAME)
    cols = ["Date", id_col] + DETAILS_COLS
    cols = [c for c in cols if c in df.columns]
    return df, cols


def display_midcon_details(df_filtered: pd.DataFrame, active_region: str, forecast_end: pd.Timestamp):
    st.subheader("🧾 Group Daily Details")

    source_cfg = st.column_config.SelectboxColumn(
        "source",
        help="system = pipeline row, forecast = generated, manual = user-added",
        options=["system", "forecast", "manual"],
        required=True,
    )
    updated_cfg = st.column_config.CheckboxColumn(
        "updated",
        help="Checked when row has been modified by user",
        default=False,
    )

    if df_filtered.empty:
        st.info("No data available for the selected filters.")
        return

    df_all = _extend_with_30d_forecast(df_filtered, id_col="System", forecast_end=forecast_end)
    df_display, cols = build_details_view(df_all, id_col="System")

    st.caption("Rows are editable. Use the 'source' column to distinguish system vs forecast vs manual.")

    st.data_editor(
        df_display[cols],
        num_rows="dynamic",
        width="stretch",
        key=f"{active_region}_edit",
        column_config={
            "source": source_cfg,
            "updated": updated_cfg,
        },
    )

    st.markdown('<div class="save-btn-bottom">', unsafe_allow_html=True)
    if st.button("💾 Save Changes", key=f"save_{active_region}"):
        st.success("✅ Changes saved successfully!")
    st.markdown("</div>", unsafe_allow_html=True)


def display_location_details(df_filtered: pd.DataFrame, active_region: str, forecast_end: pd.Timestamp):
    st.subheader("🏭 Locations")

    source_cfg = st.column_config.SelectboxColumn(
        "source",
        help="system = pipeline row, forecast = generated, manual = user-added",
        options=["system", "forecast", "manual"],
        required=True,
    )
    updated_cfg = st.column_config.CheckboxColumn(
        "updated",
        help="Checked when row has been modified by user",
        default=False,
    )

    if df_filtered.empty:
        st.info("No data available for the selected filters.")
        return

    region_locs = sorted(df_filtered["Location"].dropna().unique().tolist()) if "Location" in df_filtered.columns else []

    if not region_locs:
        st.write("*(No locations available in the current selection)*")
        return

    for i, loc in enumerate(st.tabs(region_locs)):
        with loc:
            st.markdown(f"### 📍 {region_locs[i]}")
            df_loc = df_filtered[df_filtered["Location"] == region_locs[i]]

            if df_loc.empty:
                st.write("*(No data for this location)*")
            else:
                df_all = _extend_with_30d_forecast(df_loc, id_col="Location", forecast_end=forecast_end)
                df_display, cols = build_details_view(df_all, id_col="Location")

                st.caption("Rows are editable. Use the 'source' column to distinguish system vs forecast vs manual.")

                st.data_editor(
                    df_display[cols],
                    num_rows="dynamic",
                    width="stretch",
                    key=f"{active_region}_{region_locs[i]}_edit",
                    column_config={
                        "source": source_cfg,
                        "updated": updated_cfg,
                    },
                )

            st.markdown('<div class="save-btn-bottom">', unsafe_allow_html=True)
            if st.button(f"💾 Save {region_locs[i]}", key=f"save_{active_region}_{region_locs[i]}"):
                st.success(f"✅ Changes for {region_locs[i]} saved successfully!")
            st.markdown("</div>", unsafe_allow_html=True)


def display_details_tab(df_filtered: pd.DataFrame, active_region: str, end_ts: pd.Timestamp):
    if active_region == "Group Supply Report (Midcon)":
        display_midcon_details(df_filtered, active_region, forecast_end=end_ts)
    else:
        display_location_details(df_filtered, active_region, forecast_end=end_ts)
