import os

import pandas as pd
import streamlit as st

from services.herd_intelligence import compute_cow_metrics, process_uploaded_dataset
from services.market_signals import get_market_signals


st.set_page_config(page_title="Nigeria Dairy Intelligence", layout="wide")


@st.cache_data(show_spinner=False)
def build_market_history(
    exchange_rate: float,
    dairy_reference: float,
    retail_price: float,
    import_parity: float,
) -> pd.DataFrame:
    periods = pd.date_range(end=pd.Timestamp.today().normalize(), periods=12, freq="M")
    trend = list(range(12))

    return pd.DataFrame(
        {
            "month": periods,
            "global_wmp_price_usd_mt": [dairy_reference * 25 + i * 3 for i in trend],
            "exchange_rate_ngn_usd": [exchange_rate - 40 + i * 4 for i in trend],
            "import_parity_ngn_kg": [import_parity - 180 + i * 16 for i in trend],
            "retail_milk_price_ngn_l": [retail_price - 90 + i * 8 for i in trend],
        }
    )


@st.cache_data(show_spinner=False)
def load_uploaded_herd_data(file_name: str, file_bytes: bytes) -> dict:
    return process_uploaded_dataset(file_name, file_bytes)


exchange_rate, dairy_indicator, retail_proxy, import_parity = get_market_signals()
market_signals = [exchange_rate, dairy_indicator, retail_proxy]
latest_update = max(signal.last_updated for signal in market_signals)
data_status = "Partial (Fallback)" if any(signal.status == "fallback" for signal in market_signals) else "Live"
environment_label = os.getenv("APP_ENV", "Development")

header_left, header_right = st.columns([3, 2])
with header_left:
    st.title("Nigeria Dairy Intelligence")
    st.caption("Control-panel dashboard for market, economics, herd operations, and scenario planning.")
with header_right:
    c1, c2, c3 = st.columns(3)
    c1.metric("Data Status", data_status)
    c2.metric("Updated", latest_update)
    c3.metric("Environment", environment_label)

market_tab, farm_tab, herd_tab, scenario_tab = st.tabs(
    ["Market Signals", "Farm Economics", "Herd Intelligence", "Scenario Lab"]
)

st.sidebar.title("Control Panel")
with st.sidebar.expander("Market Inputs", expanded=True):
    st.selectbox("Commodity focus", ["Whole Milk Powder", "Skim Milk Powder", "Butter"], index=0)
    st.selectbox("Currency basis", ["NGN/USD", "NGN/EUR"], index=0)
    st.slider("Import duty assumption (%)", 0, 30, 10)

with st.sidebar.expander("Farm Inputs", expanded=True):
    st.selectbox("Farm size", ["Smallholder", "Mid-scale", "Commercial"], index=1)
    st.slider("Daily feed cost (NGN/head)", 1000, 12000, 4500, step=100)
    st.slider("Average milk yield (L/day)", 5, 40, 18)

with st.sidebar.expander("Scenario Tools", expanded=True):
    st.selectbox("Scenario set", ["Baseline", "Stress", "Upside"], index=0)
    st.slider("FX shock (%)", -30, 30, 0)
    st.slider("Demand shift (%)", -20, 20, 0)

with market_tab:
    st.subheader("Market Signals")
    st.caption("Track global and local indicators shaping import pressure and dairy competitiveness in Nigeria.")

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Global WMP Price", f"{dairy_indicator.value * 25:,.0f} USD/MT")
    kpi_cols[1].metric("Import Parity Price", f"{import_parity.value:,.0f} NGN/kg")
    kpi_cols[2].metric("Exchange Rate", f"{exchange_rate.value:,.2f} NGN/USD")
    kpi_cols[3].metric("Nigerian Retail Milk Price", f"{retail_proxy.value:,.0f} NGN/L")

    st.caption(
        "Sources: "
        f"FX ({exchange_rate.source_label}), "
        f"Dairy reference ({dairy_indicator.source_label}), "
        f"Retail proxy ({retail_proxy.source_label})."
    )

    market_history = build_market_history(
        exchange_rate.value,
        dairy_indicator.value,
        retail_proxy.value,
        import_parity.value,
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("#### FX vs Import Parity Trend")
        st.line_chart(
            market_history.set_index("month")[["exchange_rate_ngn_usd", "import_parity_ngn_kg"]],
            use_container_width=True,
        )
    with chart_col2:
        st.markdown("#### WMP vs Retail Price Trend")
        st.line_chart(
            market_history.set_index("month")[["global_wmp_price_usd_mt", "retail_milk_price_ngn_l"]],
            use_container_width=True,
        )

    st.markdown("#### Market Data History")
    st.dataframe(market_history, use_container_width=True, hide_index=True)

with farm_tab:
    st.subheader("Farm Economics")
    st.caption("Structured placeholder for production costs, margin decomposition, and farm profitability diagnostics.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Unit Economics Panel")
        st.info("Placeholder panel for cost per liter, feed conversion, and gross margin calculations.")
    with col_b:
        st.markdown("#### Cashflow & Risk Panel")
        st.info("Placeholder panel for monthly cashflow outlook and downside risk scenarios.")

with herd_tab:
    st.subheader("Herd Intelligence")
    st.caption("Upload sensor data to generate herd summaries, behaviour trends, and simple rumination anomaly flags.")

    uploaded_file = st.file_uploader(
        "Upload herd sensor dataset",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
        key="herd_sensor_file",
    )

    if uploaded_file is None:
        st.info(
            "No dataset uploaded yet. Upload a CSV or XLSX sensor file to view herd KPIs, trends, validation checks, "
            "and cow-level anomaly screening."
        )
    else:
        try:
            herd_output = load_uploaded_herd_data(uploaded_file.name, uploaded_file.getvalue())
        except Exception as exc:
            st.error(f"Failed to process uploaded file: {exc}")
        else:
            herd_metrics = herd_output["herd_metrics"]
            activity_ts = herd_output["activity_timeseries"]
            rumination_ts = herd_output["rumination_timeseries"]
            animal_counts = herd_output["animal_counts"]
            validation_summary = herd_output["validation_summary"]
            processed_df = herd_output["processed_df"]

            kpi_cols = st.columns(4)
            kpi_cols[0].metric("Animals detected", f"{herd_metrics['animals_detected']:,}")
            kpi_cols[1].metric("Records loaded", f"{herd_metrics['records_loaded']:,}")
            kpi_cols[2].metric("Date range", herd_metrics["date_range"])
            kpi_cols[3].metric("Variables available", f"{herd_metrics['variables_available']:,}")

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("#### Herd Avg Activity Over Time")
                if activity_ts.empty:
                    st.warning("Activity chart unavailable: `activity_rate` and/or valid `date` not found.")
                else:
                    st.line_chart(
                        activity_ts.set_index("date")[["avg_activity_rate"]],
                        use_container_width=True,
                    )

            with chart_col2:
                st.markdown("#### Herd Avg Rumination Over Time")
                if rumination_ts.empty:
                    st.warning("Rumination chart unavailable: `rumination_min` and/or valid `date` not found.")
                else:
                    st.line_chart(
                        rumination_ts.set_index("date")[["avg_rumination_min"]],
                        use_container_width=True,
                    )

            st.markdown("#### Per-Animal Record Counts")
            if animal_counts.empty:
                st.warning("Per-animal count chart unavailable: `animal_id` column not found.")
            else:
                top_counts = animal_counts.head(30).set_index("animal_id")
                st.bar_chart(top_counts[["record_count"]], use_container_width=True)

            st.markdown("#### Validation Summary")
            st.dataframe(validation_summary, use_container_width=True, hide_index=True)

            st.markdown("#### Selected Cow Summary")
            if animal_counts.empty:
                st.info("Cow-level metrics unavailable because no `animal_id` values were detected.")
            else:
                selected_cow = st.selectbox("Select cow", animal_counts["animal_id"].tolist(), key="selected_cow")
                cow_metrics, cow_df = compute_cow_metrics(processed_df, selected_cow)

                if not cow_metrics:
                    st.warning("No records found for selected cow.")
                else:
                    cow_kpis = st.columns(4)
                    cow_kpis[0].metric("Cow records", f"{cow_metrics['record_count']:,}")
                    cow_kpis[1].metric("Avg rumination (min)", str(cow_metrics["avg_rumination_min"]))
                    cow_kpis[2].metric("Avg activity rate", str(cow_metrics["avg_activity_rate"]))
                    cow_kpis[3].metric("Rumination anomalies", f"{cow_metrics['anomaly_count']:,}")

                    if "rumination_anomaly" in cow_df.columns:
                        anomaly_rows = cow_df[cow_df["rumination_anomaly"]].copy()
                        st.caption("Baseline anomaly rule: rumination_min < 80% of selected cow's mean rumination.")
                        st.dataframe(anomaly_rows.head(20), use_container_width=True, hide_index=True)

with scenario_tab:
    st.subheader("Scenario Lab")
    st.caption("Structured placeholder for simulation controls, assumptions, and strategy comparisons.")

    sc_col1, sc_col2 = st.columns(2)
    with sc_col1:
        st.markdown("#### Assumption Builder")
        st.info("Placeholder panel for defining macro, farm, and herd assumptions.")
    with sc_col2:
        st.markdown("#### Scenario Results")
        st.info("Placeholder panel for comparing baseline, stress, and upside outcomes.")
