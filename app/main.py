import os

import pandas as pd
import streamlit as st

from services.bootstrap_data import BOOTSTRAP_SOURCE_LABEL, load_bootstrap_dataset
from services.herd_intelligence import build_validation_summary, filter_dataset_by_period, process_uploaded_dataset
from services.market_signals import get_market_signals
from services.overview import (
    build_cow_event_table,
    compute_cow_daily_trend,
    build_demo_milk_production_trend,
    build_demo_reproductive_trend,
    build_farms_cows_to_review,
    compute_cow_profile,
    compute_cow_ranking_table,
    compute_cow_vs_context,
    compute_data_completeness,
    compute_farm_comparison_table,
    compute_farm_daily_trends,
    compute_network_behaviour,
    compute_network_kpis,
    ensure_entity_columns,
    metric_drilldown,
)


st.set_page_config(page_title="Nigeria Dairy Intelligence", layout="wide")


@st.cache_data(show_spinner=False)
def load_uploaded_sensor_data(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    return process_uploaded_dataset(file_name, file_bytes)["processed_df"]


exchange_rate, dairy_indicator, retail_proxy, import_parity = get_market_signals()
market_status = "Partial (Fallback)" if any(s.status == "fallback" for s in [exchange_rate, dairy_indicator, retail_proxy]) else "Live"
latest_update = max(s.last_updated for s in [exchange_rate, dairy_indicator, retail_proxy])
environment_label = os.getenv("APP_ENV", "Development")

processed_df = pd.DataFrame()
filtered_df = pd.DataFrame()
sensor_data_source_label = "none"
uploaded_file_name = None
validation_summary = pd.DataFrame()

st.sidebar.title("Control Panel")
with st.sidebar.expander("Data Input", expanded=True):
    sensor_source_mode = st.radio(
        "Sensor data source",
        ["Bootstrap Demo", "Upload File"],
        horizontal=True,
        key="sensor_source_mode",
    )
    uploaded_file = st.file_uploader(
        "Upload sensor dataset (CSV/XLSX)",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
        key="herd_sensor_file",
        disabled=(sensor_source_mode != "Upload File"),
    )

if sensor_source_mode == "Bootstrap Demo":
    try:
        processed_df = load_bootstrap_dataset()
        sensor_data_source_label = BOOTSTRAP_SOURCE_LABEL
        uploaded_file_name = "sensor_bootstrap.csv"
    except Exception as exc:
        st.sidebar.error(f"Bootstrap load failed: {exc}")
elif uploaded_file is not None:
    uploaded_file_name = uploaded_file.name
    try:
        processed_df = load_uploaded_sensor_data(uploaded_file.name, uploaded_file.getvalue())
        sensor_data_source_label = "uploaded/real"
    except Exception as exc:
        st.sidebar.error(f"Upload failed: {exc}")

if not processed_df.empty:
    processed_df = ensure_entity_columns(processed_df)
    validation_summary = build_validation_summary(processed_df)

with st.sidebar.expander("Reporting Period", expanded=True):
    period_choice = st.radio("Summary window", ["7 days", "30 days", "full period", "custom range"], horizontal=False)

    custom_start = None
    custom_end = None
    if (
        period_choice == "custom range"
        and not processed_df.empty
        and "date" in processed_df.columns
        and processed_df["date"].notna().any()
    ):
        min_date = processed_df["date"].min().date()
        max_date = processed_df["date"].max().date()
        custom_start = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
        custom_end = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

if not processed_df.empty:
    filtered_df = filter_dataset_by_period(
        processed_df,
        period_choice,
        pd.Timestamp(custom_start) if custom_start else None,
        pd.Timestamp(custom_end) if custom_end else None,
    )

network_kpis = compute_network_kpis(filtered_df)
network_behaviour = compute_network_behaviour(filtered_df)
completeness_pct = compute_data_completeness(filtered_df) * 100.0
farm_table = compute_farm_comparison_table(filtered_df)
review_table = build_farms_cows_to_review(filtered_df)
milk_production_demo = build_demo_milk_production_trend()
reproductive_demo = build_demo_reproductive_trend()

header_left, header_right = st.columns([3, 2])
with header_left:
    st.title("Nigeria Dairy Intelligence")
    st.caption("Danone supply-chain intelligence and evidence platform")
with header_right:
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Market Status", market_status)
    h2.metric("Market Updated", latest_update)
    h3.metric("Environment", environment_label)
    h4.metric("Sensor Source", sensor_data_source_label)

with st.sidebar.expander("Market Inputs", expanded=False):
    st.selectbox("Commodity focus", ["Whole Milk Powder", "Skim Milk Powder", "Butter"], index=0)
    st.selectbox("Currency basis", ["NGN/USD", "NGN/EUR"], index=0)

with st.sidebar.expander("Scenario Tools", expanded=False):
    st.selectbox("Scenario set", ["Baseline", "Stress", "Upside"], index=0)
    st.slider("FX shock (%)", -30, 30, 0)
    st.slider("Demand shift (%)", -20, 20, 0)

exec_tab, farms_tab, cows_tab, metrics_tab, validation_tab, scenario_tab = st.tabs(
    [
        "Executive Overview",
        "Farms",
        "Cows",
        "Metrics",
        "Validation & Evidence",
        "Scenario & Scale",
    ]
)

with exec_tab:
    st.subheader("Executive Overview")
    st.caption(f"Active sensor dataset: `{sensor_data_source_label}`")

    st.markdown("#### International")
    intl_cols = st.columns(3)
    intl_cols[0].metric("Global Dairy Reference", f"{dairy_indicator.value:,.1f}")
    intl_cols[1].metric("Exchange Rate Context", f"{exchange_rate.value:,.2f} NGN/USD")
    intl_cols[2].metric("Import Parity Context", f"{import_parity.value:,.0f} NGN/kg")
    st.caption(f"International source type: placeholder adapter. Freshness: {latest_update}. Status: {market_status}.")

    st.markdown("#### National")
    nat_cols = st.columns(3)
    nat_cols[0].metric("Nigeria Retail Milk Proxy", f"{retail_proxy.value:,.0f} NGN/L")
    nat_cols[1].metric("Domestic Supply Coverage", "48% (Placeholder/Demo)")
    nat_cols[2].metric("Import Dependence", "52% (Placeholder/Demo)")
    st.caption("National context includes placeholders where live national data streams are not yet integrated.")

    st.markdown("#### Network KPI Cards")
    net_cols = st.columns(5)
    net_cols[0].metric("Farms", f"{network_kpis['farms']:,}")
    net_cols[1].metric("Cows", f"{network_kpis['cows']:,}")
    net_cols[2].metric(
        "Total Milk/Day",
        "N/A" if network_kpis["total_milk_per_day"] is None else f"{network_kpis['total_milk_per_day']:,.1f} L",
    )
    net_cols[3].metric(
        "Avg Milk/Cow/Day",
        "N/A" if network_kpis["avg_milk_per_cow_per_day"] is None else f"{network_kpis['avg_milk_per_cow_per_day']:,.2f} L",
    )
    net_cols[4].metric(
        "Avg Milk/Farm/Day",
        "N/A" if network_kpis["avg_milk_per_farm_per_day"] is None else f"{network_kpis['avg_milk_per_farm_per_day']:,.1f} L",
    )

    st.markdown("#### Network Behaviour Cards")
    beh_cols = st.columns(4)
    beh_cols[0].metric("Avg Rumination", "N/A" if network_behaviour["avg_rumination"] is None else f"{network_behaviour['avg_rumination']:,.2f}")
    beh_cols[1].metric("Avg Activity", "N/A" if network_behaviour["avg_activity"] is None else f"{network_behaviour['avg_activity']:,.2f}")
    beh_cols[2].metric("Avg Eating", "N/A" if network_behaviour["avg_eating"] is None else f"{network_behaviour['avg_eating']:,.2f}")
    beh_cols[3].metric("Avg Standing", "N/A" if network_behaviour["avg_standing"] is None else f"{network_behaviour['avg_standing']:,.2f}")

    rank_col, review_col = st.columns(2)
    with rank_col:
        st.markdown("#### Farm Ranking")
        if farm_table.empty:
            st.info("Select bootstrap/demo or upload data to view farm rankings.")
        else:
            st.dataframe(
                farm_table[
                    [
                        "farm_name",
                        "farm_type",
                        "avg_milk_yield_l",
                        "data_completeness_pct",
                        "behavioral_rating",
                        "behavioral_score",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
    with review_col:
        st.markdown("#### Farms/Cows To Review")
        if review_table.empty:
            st.info("Review panel populates when farm/cow records are available.")
        else:
            st.dataframe(review_table, use_container_width=True, hide_index=True)

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("#### Milk Production Trend (Generated Demo Data)")
        st.caption("Generated placeholder trend. Replace with validated production stream when available.")
        st.line_chart(milk_production_demo.set_index("month")[["milk_production_liters_demo"]], use_container_width=True)
    with p2:
        st.markdown("#### Reproductive Records Trend (Generated Demo Data)")
        st.caption("Generated placeholder trend. Replace with reproductive event feed when available.")
        st.line_chart(
            reproductive_demo.set_index("month")[["heat_detection_rate_demo", "pregnancy_confirmation_rate_demo"]],
            use_container_width=True,
        )

with farms_tab:
    st.subheader("Farm Explorer")
    if farm_table.empty:
        st.info("No farm-level data available for the selected source and period.")
    else:
        farm_choice = st.selectbox("Select farm", farm_table["farm_name"].tolist(), key="farm_select")
        farm_row = farm_table[farm_table["farm_name"] == farm_choice].iloc[0]
        farm_id = str(farm_row["farm_id"])

        score_cols = st.columns(6)
        score_cols[0].metric("Farm Type", str(farm_row["farm_type"]))
        score_cols[1].metric("Total Milk/Day", "N/A" if pd.isna(farm_row["avg_milk_yield_l"]) else f"{farm_row['avg_milk_yield_l'] * farm_row['cows']:,.1f} L")
        score_cols[2].metric("Avg Milk/Cow/Day", "N/A" if pd.isna(farm_row["avg_milk_yield_l"]) else f"{farm_row['avg_milk_yield_l']:,.2f} L")
        score_cols[3].metric("Herd Size", f"{int(farm_row['cows']):,}")
        score_cols[4].metric("Data Completeness", f"{farm_row['data_completeness_pct']:,.1f}%")
        score_cols[5].metric("Behavioural Rating", f"{farm_row['behavioral_rating']} ({farm_row['behavioral_score']:,.1f})")

        farm_trend = compute_farm_daily_trends(filtered_df, farm_id)
        ft1, ft2 = st.columns(2)
        with ft1:
            st.markdown("#### Milk Trend")
            if farm_trend.empty or "milk_yield_l" not in farm_trend.columns:
                st.info("Milk trend unavailable.")
            else:
                st.line_chart(farm_trend.set_index("date")[["milk_yield_l"]], use_container_width=True)
        with ft2:
            st.markdown("#### Behaviour Trends")
            cols = [c for c in ["rumination_min", "activity_rate", "eating_min", "standing_min"] if c in farm_trend.columns]
            if not cols:
                st.info("Behaviour trends unavailable.")
            else:
                st.line_chart(farm_trend.set_index("date")[cols], use_container_width=True)

        st.markdown("#### Cow Ranking")
        cow_rank = compute_cow_ranking_table(filtered_df, farm_id)
        st.dataframe(cow_rank, use_container_width=True, hide_index=True)

        st.markdown("#### Cows To Review")
        farm_review = review_table[review_table["farm_id"].astype(str) == farm_id] if not review_table.empty else pd.DataFrame()
        if farm_review.empty:
            st.info("No flagged cows for review in this farm for selected period.")
        else:
            st.dataframe(farm_review, use_container_width=True, hide_index=True)

with cows_tab:
    st.subheader("Cow Explorer")
    if filtered_df.empty:
        st.info("No cow-level data available for selected source and period.")
    else:
        farm_options = sorted(filtered_df["farm_name"].dropna().astype(str).unique().tolist())
        selected_farm_name = st.selectbox("Filter by farm", farm_options, key="cow_farm_filter")
        farm_df = filtered_df[filtered_df["farm_name"].astype(str) == selected_farm_name]
        cow_options = sorted(farm_df["animal_id"].dropna().astype(str).unique().tolist())

        selected_cow = st.selectbox("Select cow", cow_options, key="cow_select")
        cow_profile = compute_cow_profile(filtered_df, selected_cow)

        if not cow_profile:
            st.warning("No profile available for selected cow.")
        else:
            ck = st.columns(4)
            ck[0].metric("Litres/Day", "N/A" if cow_profile["lpd"] is None else f"{cow_profile['lpd']:,.2f} L")
            ck[1].metric("Behavioural Rating", f"{cow_profile['behavioral_rating']} ({cow_profile['behavioral_score']:,.1f})")
            ck[2].metric("Data Completeness", f"{cow_profile['data_completeness_pct']:,.1f}%")
            ck[3].metric("Farm", cow_profile["farm_name"])

            st.markdown("#### Cow vs Farm and Network")
            compare_df = compute_cow_vs_context(filtered_df, selected_cow)
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

            cow_trend = compute_cow_daily_trend(filtered_df[filtered_df["animal_id"].astype(str) == str(selected_cow)])
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Milk Trend")
                if cow_trend.empty or "milk_yield_l" not in cow_trend.columns:
                    cow_daily = metric_drilldown(filtered_df, "milk_yield_l", "Cow", cow_id=selected_cow)
                    if cow_daily.empty:
                        st.info("Milk trend unavailable.")
                    else:
                        st.line_chart(cow_daily.set_index("date")[["avg_milk_yield_l"]], use_container_width=True)
                else:
                    st.line_chart(cow_trend.set_index("date")[["milk_yield_l"]], use_container_width=True)
            with c2:
                st.markdown("#### Behaviour Timelines")
                cols = [c for c in ["rumination_min", "activity_rate", "eating_min", "standing_min"] if c in cow_trend.columns]
                if not cols:
                    st.info("Behaviour timeline unavailable.")
                else:
                    st.line_chart(cow_trend.set_index("date")[cols], use_container_width=True)

            st.markdown("#### Deviation/Event Table")
            event_df = build_cow_event_table(filtered_df, selected_cow)
            if event_df.empty:
                st.info("No flagged events for selected cow in this period.")
            else:
                st.dataframe(event_df, use_container_width=True, hide_index=True)

with metrics_tab:
    st.subheader("Metric Explorer")
    metric = st.selectbox(
        "Select metric",
        ["milk_yield_l", "rumination_min", "activity_rate", "eating_min", "standing_min", "resting_min"],
        key="metric_select",
    )
    level = st.radio("View level", ["Network", "Farm", "Cow"], horizontal=True)

    selected_farm_id = None
    selected_cow_id = None
    if level == "Farm" and not farm_table.empty:
        farm_pick = st.selectbox("Farm", farm_table["farm_name"].tolist(), key="metric_farm")
        selected_farm_id = str(farm_table[farm_table["farm_name"] == farm_pick]["farm_id"].iloc[0])
    if level == "Cow" and not filtered_df.empty:
        cow_options = sorted(filtered_df["animal_id"].dropna().astype(str).unique().tolist())
        selected_cow_id = st.selectbox("Cow", cow_options, key="metric_cow")

    metric_series = metric_drilldown(filtered_df, metric, level, farm_id=selected_farm_id, cow_id=selected_cow_id)
    if metric_series.empty:
        st.info("Metric series unavailable for selected level/source.")
    else:
        value_col = [c for c in metric_series.columns if c != "date"][0]
        st.line_chart(metric_series.set_index("date")[[value_col]], use_container_width=True)

        mcols = st.columns(3)
        mcols[0].metric("Latest", f"{metric_series[value_col].iloc[-1]:,.2f}")
        mcols[1].metric("Mean", f"{metric_series[value_col].mean():,.2f}")
        mcols[2].metric("Min/Max", f"{metric_series[value_col].min():,.2f} / {metric_series[value_col].max():,.2f}")

        st.dataframe(metric_series.tail(30), use_container_width=True, hide_index=True)

with validation_tab:
    st.subheader("Validation & Evidence")
    st.caption("Data quality, source traceability, and operational evidence layer.")

    vc1, vc2, vc3 = st.columns(3)
    vc1.metric("Sensor Source", sensor_data_source_label)
    vc2.metric("Uploaded File", uploaded_file_name if uploaded_file_name else "none")
    vc3.metric("Data Completeness", f"{completeness_pct:,.1f}%")

    st.markdown("#### Source & Freshness")
    source_df = pd.DataFrame(
        {
            "source": [
                f"Sensor dataset ({sensor_data_source_label})",
                exchange_rate.source_label,
                dairy_indicator.source_label,
                retail_proxy.source_label,
            ],
            "last_updated": [
                "selected" if sensor_data_source_label != "none" else "not selected",
                exchange_rate.last_updated,
                dairy_indicator.last_updated,
                retail_proxy.last_updated,
            ],
            "notes": [
                "bootstrap/demo or uploaded/real",
                exchange_rate.note,
                dairy_indicator.note,
                retail_proxy.note,
            ],
        }
    )
    st.dataframe(source_df, use_container_width=True, hide_index=True)

    st.markdown("#### Validation Summary")
    if validation_summary.empty:
        st.info("Validation summary will populate when a dataset is selected.")
    else:
        st.dataframe(validation_summary, use_container_width=True, hide_index=True)

    st.markdown("#### Missingness by Variable")
    if filtered_df.empty:
        st.info("Missingness summary unavailable for current source/period.")
    else:
        miss = (
            filtered_df.isna().mean().mul(100).round(2).rename_axis("variable").reset_index(name="missing_pct")
            .sort_values("missing_pct", ascending=False)
        )
        st.dataframe(miss, use_container_width=True, hide_index=True)

    op1, op2, op3 = st.columns(3)
    op1.info("Uptime metrics placeholder")
    op2.info("Device reliability placeholder")
    op3.info("Operational validation placeholder")

with scenario_tab:
    st.subheader("Scenario & Scale")
    st.caption("Structured placeholder for expansion and performance scenarios.")

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("#### Scenario Inputs")
        st.number_input("Target farms added", min_value=0, value=25, step=5)
        st.number_input("Target cows added", min_value=0, value=1000, step=100)
        st.slider("Import pressure change (%)", -30, 30, 0)
    with sc2:
        st.markdown("#### Scale Assumptions")
        st.number_input("Milk yield growth (%)", min_value=-20, max_value=50, value=8)
        st.number_input("Feed cost change (%)", min_value=-20, max_value=50, value=5)
        st.number_input("Coverage target (%)", min_value=0, max_value=100, value=90)

    out1, out2 = st.columns(2)
    with out1:
        st.info("Commercial expansion output placeholder")
    with out2:
        st.info("Performance scenario output placeholder")
