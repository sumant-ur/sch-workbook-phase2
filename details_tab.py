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

# Columns shown in the Details tab (order matters)
DETAILS_COLS = [
    "source",  # system | forecast | manual
    # "updated",  # 0/1 flag
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


# --- Forecasting helpers ----------------------------------------------------

# These are the columns we forecast directly via weekday-weighted averages.
# Open/Close inventory are derived (roll-forward) and are NOT directly averaged.
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
    """Normalize raw rows into 1 row per (Date, id_col, Product).

    The source data can contain multiple rows per date; Details view should show
    the daily total flows and a single opening/closing inventory snapshot.
    """
    if df.empty:
        return df

    # Ensure Date is datetime
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    group_cols = ["Date", id_col, "Product"]
    agg_map = {}

    # Inventory snapshots
    if "Open Inv" in df.columns:
        agg_map["Open Inv"] = "first"
    if "Close Inv" in df.columns:
        agg_map["Close Inv"] = "last"

    # Flow columns -> sum
    for c in [c for c in FORECAST_FLOW_COLS if c in df.columns]:
        agg_map[c] = "sum"

    # Lineage flags: when multiple rows exist for the same day, keep a stable tag.
    if "source" in df.columns:
        agg_map["source"] = "first"
    if "updated" in df.columns:
        agg_map["updated"] = "max"

    if "Notes" in df.columns:
        # Keep something readable (last note of the day)
        agg_map["Notes"] = "last"

    daily = df.groupby(group_cols, as_index=False).agg(agg_map)
    return daily


def _weighted_weekday_means(
    hist: pd.DataFrame,
    value_cols: list[str],
    max_weeks: int = 6,
    decay: float = 0.70,
) -> dict[tuple[int, str], float]:
    """Return {(weekday, col): weighted_mean} using last N occurrences.

    For a given weekday, we take up to `max_weeks` most recent occurrences and
    apply exponentially decaying weights (most recent gets the largest weight).
    """
    out: dict[tuple[int, str], float] = {}
    if hist.empty:
        return out

    h = hist.sort_values("Date")
    h["__weekday"] = h["Date"].dt.weekday

    # Fallback (overall) means from recent window, used when a weekday has no history
    recent = h.tail(21)
    fallback_means = {c: float(recent[c].mean()) if c in recent.columns and len(recent) else 0.0 for c in value_cols}

    for wd in range(7):
        subset = h[h["__weekday"] == wd].sort_values("Date", ascending=False)
        if subset.empty:
            for c in value_cols:
                out[(wd, c)] = fallback_means.get(c, 0.0)
            continue

        subset = subset.head(max_weeks)
        weights = np.array([decay**i for i in range(len(subset))], dtype=float)
        wsum = float(weights.sum()) if float(weights.sum()) != 0 else 1.0

        for c in value_cols:
            if c not in subset.columns:
                out[(wd, c)] = fallback_means.get(c, 0.0)
                continue
            vals = subset[c].astype(float).to_numpy()
            out[(wd, c)] = float((vals * weights).sum() / wsum)

    return out


def _extend_with_30d_forecast(
    df: pd.DataFrame,
    *,
    id_col: str,
    forecast_end: pd.Timestamp | None = None,
    default_days: int = 30,
) -> pd.DataFrame:
    """Append forecast rows per (id_col, Product) based on weekday averages.

    If `forecast_end` is provided, rows are generated ONLY from the day after the
    last available date through `forecast_end` (inclusive).

    If `forecast_end` is None, we fall back to the historical behavior and
    generate `default_days` rows.
    """

    if df.empty:
        return df

    # Aggregate to daily first, then forecast
    daily = _aggregate_daily_details(df, id_col=id_col)
    if daily.empty:
        return daily

    daily = daily.sort_values("Date")

    # Existing rows from the pipeline / DB are considered 'system' rows unless already tagged.
    if "source" not in daily.columns:
        daily["source"] = "system"
    else:
        daily["source"] = daily["source"].fillna("system")

    if "updated" not in daily.columns:
        daily["updated"] = 0
    else:
        daily["updated"] = pd.to_numeric(daily["updated"], errors="coerce").fillna(0).astype(int)

    # Which flow columns are available in this dataset?
    flow_cols = [c for c in FORECAST_FLOW_COLS if c in daily.columns]

    # Normalize forecast_end (if supplied) to a plain timestamp (no time component assumptions)
    if forecast_end is not None:
        forecast_end = pd.to_datetime(forecast_end, errors="coerce")
        if pd.isna(forecast_end):
            forecast_end = None

    forecast_rows: list[dict] = []

    for (id_val, product), g in daily.groupby([id_col, "Product"], dropna=False):
        g = g.sort_values("Date")
        last_date = pd.to_datetime(g["Date"].max())

        # Use last known closing inventory as the starting point
        if "Close Inv" in g.columns and g["Close Inv"].notna().any():
            prev_close = float(g.loc[g["Date"] == last_date, "Close Inv"].iloc[-1])
        else:
            prev_close = 0.0

        weekday_means = _weighted_weekday_means(g, value_cols=flow_cols)

        start = last_date + timedelta(days=1)

        if forecast_end is not None:
            # If user-selected range ends before (or on) the last actual date, no forecast needed.
            if start > forecast_end:
                continue
            dates = pd.date_range(start=start, end=forecast_end, freq="D")
        else:
            dates = pd.date_range(start=start, periods=int(default_days), freq="D")

        for d in dates:
            wd = int(d.weekday())

            row: dict = {
                "Date": d,
                id_col: id_val,
                "Product": product,
                "source": "forecast",
                "updated": 0,
                "Notes": "Forecast (weekday weighted avg)",
            }

            # Fill forecasted flows
            for c in flow_cols:
                row[c] = weekday_means.get((wd, c), 0.0)

            # Roll-forward inventory math
            opening = prev_close
            inflow = sum(float(row.get(c, 0.0) or 0.0) for c in INFLOW_COLS if c in flow_cols)
            outflow = sum(float(row.get(c, 0.0) or 0.0) for c in OUTFLOW_COLS if c in flow_cols)
            net = sum(float(row.get(c, 0.0) or 0.0) for c in NET_COLS if c in flow_cols)

            closing = opening + inflow - outflow + net

            row["Open Inv"] = opening
            row["Close Inv"] = closing

            prev_close = closing
            forecast_rows.append(row)

    if not forecast_rows:
        return daily

    forecast_df = pd.DataFrame(forecast_rows)

    # Combine and keep numeric columns numeric
    combined = pd.concat([daily, forecast_df], ignore_index=True)
    for c in ["Open Inv", "Close Inv"] + flow_cols:
        if c in combined.columns:
            combined[c] = pd.to_numeric(combined[c], errors="coerce").fillna(0.0)

    return combined


def build_details_view(df, id_col: str):
    df = df.sort_values("Date").rename(columns=DETAILS_RENAME)
    cols = ["Date", id_col] + DETAILS_COLS
    cols = [c for c in cols if c in df.columns]
    return df, cols


def display_midcon_details(df_filtered, active_region, forecast_end: pd.Timestamp):
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

    # Extend with forecast rows only up to the user-selected end date
    df_all = _extend_with_30d_forecast(df_filtered, id_col="System", forecast_end=forecast_end)

    df_display, cols = build_details_view(df_all, id_col="System")

    st.caption("Rows are editable. Use the 'source' column to distinguish system vs forecast vs manual.")

    # One combined editable table.
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


def display_location_details(df_filtered, active_region, forecast_end: pd.Timestamp):
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
                # Extend with forecast rows for this location only up to the user-selected end date
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


def display_details_tab(df_filtered, active_region, end_ts: pd.Timestamp):
    """Details tab.

    Forecast rows are generated only for missing days after the last available
    date in the filtered dataset, up to `end_ts`.
    """

    if active_region == "Group Supply Report (Midcon)":
        display_midcon_details(df_filtered, active_region, forecast_end=end_ts)
    else:
        display_location_details(df_filtered, active_region, forecast_end=end_ts)
