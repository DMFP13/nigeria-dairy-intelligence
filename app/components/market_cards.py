import streamlit as st

from services.market_signals import SignalValue


def _format_value(signal: SignalValue) -> str:
    if "Rate" in signal.label:
        return f"{signal.value:,.2f}"
    if "Reference" in signal.label:
        return f"{signal.value:,.1f}"
    return f"{signal.value:,.2f}"


def _render_single_card(column, signal: SignalValue) -> None:
    column.metric(signal.label, _format_value(signal), help=signal.unit)
    column.caption(f"Source: {signal.source_label}")
    column.caption(f"Last updated: {signal.last_updated}")
    if signal.source_url:
        column.caption(f"Reference: {signal.source_url}")

    if signal.status == "fallback":
        column.warning("Fallback/demo value")
    else:
        column.caption(f"Status: {signal.status}")

    column.caption(signal.note)


def render_market_cards(signals: list[SignalValue]) -> None:
    for idx in range(0, len(signals), 2):
        cols = st.columns(2)
        for col, signal in zip(cols, signals[idx : idx + 2]):
            _render_single_card(col, signal)
