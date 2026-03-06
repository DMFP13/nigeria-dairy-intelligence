from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.error import URLError
from urllib.request import Request, urlopen

import streamlit as st


FX_CACHE_TTL_SECONDS = 24 * 60 * 60
MONTHLY_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60

CBN_NFEM_URL = os.getenv("CBN_NFEM_URL", "https://www.cbn.gov.ng/")
FAO_DAIRY_URL = os.getenv("FAO_DAIRY_URL", "https://www.fao.org/worldfoodsituation/foodpricesindex/en/")
NBS_RETAIL_MILK_URL = os.getenv("NBS_RETAIL_MILK_URL", "https://nigerianstat.gov.ng/")


@dataclass
class SignalValue:
    label: str
    value: float
    unit: str
    source_label: str
    source_url: str
    last_updated: str
    status: str
    note: str


def _utc_today() -> str:
    return datetime.now(UTC).date().isoformat()


def _fetch_json(url: str, timeout_seconds: int = 8) -> dict | list:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_date(payload: dict | list) -> str:
    candidate_keys = ("date", "as_of", "last_updated", "updated_at", "period")

    if isinstance(payload, dict):
        for key in candidate_keys:
            if key in payload and payload[key]:
                return str(payload[key])[:10]
        for value in payload.values():
            if isinstance(value, (dict, list)):
                nested = _extract_date(value)
                if nested:
                    return nested

    if isinstance(payload, list):
        for item in reversed(payload):
            if isinstance(item, dict):
                nested = _extract_date(item)
                if nested:
                    return nested

    return _utc_today()


def _parse_ngn_usd_rate(payload: dict | list) -> float:
    if isinstance(payload, dict):
        for key in ("ngn_usd", "NGN_USD", "usd_ngn", "USDNGN", "rate"):
            if key in payload:
                return float(payload[key])
        for value in payload.values():
            if isinstance(value, (dict, list)):
                try:
                    return _parse_ngn_usd_rate(value)
                except (ValueError, TypeError):
                    continue

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                currency_code = str(item.get("currency") or item.get("code") or "").upper()
                if currency_code == "USD":
                    for rate_key in ("rate", "value", "mid", "sell", "buy"):
                        if rate_key in item:
                            return float(item[rate_key])
                try:
                    return _parse_ngn_usd_rate(item)
                except (ValueError, TypeError):
                    continue

    raise ValueError("Could not parse NGN/USD rate payload")


def _parse_fao_dairy_value(payload: dict | list) -> float:
    if isinstance(payload, dict):
        for key in ("value", "index", "dairy_index", "latest_value"):
            if key in payload:
                return float(payload[key])
        for nested_key in ("data", "results", "items"):
            if nested_key in payload:
                return _parse_fao_dairy_value(payload[nested_key])

    if isinstance(payload, list):
        for item in reversed(payload):
            if isinstance(item, dict):
                for key in ("value", "index", "dairy_index"):
                    if key in item:
                        return float(item[key])

    raise ValueError("Could not parse FAO dairy indicator payload")


def _parse_nbs_retail_milk_proxy(payload: dict | list) -> float:
    if isinstance(payload, dict):
        for key in ("retail_milk_price", "milk_price", "value", "price"):
            if key in payload:
                return float(payload[key])
        for nested_key in ("data", "results", "items"):
            if nested_key in payload:
                return _parse_nbs_retail_milk_proxy(payload[nested_key])

    if isinstance(payload, list):
        for item in reversed(payload):
            if isinstance(item, dict):
                for key in ("retail_milk_price", "milk_price", "value", "price"):
                    if key in item:
                        return float(item[key])

    raise ValueError("Could not parse Nigeria retail milk proxy payload")


@st.cache_data(ttl=FX_CACHE_TTL_SECONDS, show_spinner=False)
def get_ngn_usd_signal() -> SignalValue:
    fallback_value = 1580.0
    fallback_note = "Fallback demo value while CBN NFEM parser is being hardened."

    try:
        payload = _fetch_json(CBN_NFEM_URL)
        return SignalValue(
            label="NGN/USD Exchange Rate",
            value=_parse_ngn_usd_rate(payload),
            unit="NGN per USD",
            source_label="CBN NFEM (official adapter)",
            source_url=CBN_NFEM_URL,
            last_updated=_extract_date(payload),
            status="live",
            note="Daily refresh target (24h cache).",
        )
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError):
        return SignalValue(
            label="NGN/USD Exchange Rate",
            value=fallback_value,
            unit="NGN per USD",
            source_label="CBN NFEM (fallback)",
            source_url=CBN_NFEM_URL,
            last_updated=_utc_today(),
            status="fallback",
            note=fallback_note,
        )


@st.cache_data(ttl=MONTHLY_CACHE_TTL_SECONDS, show_spinner=False)
def get_global_dairy_signal() -> SignalValue:
    fallback_value = 129.4
    fallback_note = "Fallback demo value while FAO/FAOSTAT live retrieval is being hardened."

    try:
        payload = _fetch_json(FAO_DAIRY_URL)
        return SignalValue(
            label="Global Dairy Reference",
            value=_parse_fao_dairy_value(payload),
            unit="FAO dairy index points",
            source_label="FAO Dairy Price Index / FAOSTAT adapter",
            source_url=FAO_DAIRY_URL,
            last_updated=_extract_date(payload),
            status="live",
            note="Monthly refresh target (30-day cache).",
        )
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError):
        return SignalValue(
            label="Global Dairy Reference",
            value=fallback_value,
            unit="FAO dairy index points",
            source_label="FAO Dairy Price Index (fallback)",
            source_url=FAO_DAIRY_URL,
            last_updated=_utc_today(),
            status="fallback",
            note=fallback_note,
        )


@st.cache_data(ttl=MONTHLY_CACHE_TTL_SECONDS, show_spinner=False)
def get_nigeria_retail_milk_proxy_signal() -> SignalValue:
    fallback_value = 1450.0
    fallback_note = "Fallback demo proxy while NBS retail milk adapter is being stabilized."

    try:
        payload = _fetch_json(NBS_RETAIL_MILK_URL)
        return SignalValue(
            label="Nigeria Retail Milk Price Proxy",
            value=_parse_nbs_retail_milk_proxy(payload),
            unit="NGN per liter",
            source_label="NBS retail proxy adapter",
            source_url=NBS_RETAIL_MILK_URL,
            last_updated=_extract_date(payload),
            status="live",
            note="Monthly refresh target (30-day cache).",
        )
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError):
        return SignalValue(
            label="Nigeria Retail Milk Price Proxy",
            value=fallback_value,
            unit="NGN per liter",
            source_label="NBS retail proxy (fallback)",
            source_url=NBS_RETAIL_MILK_URL,
            last_updated=_utc_today(),
            status="fallback",
            note=fallback_note,
        )


def calculate_import_parity_benchmark(
    exchange_rate: SignalValue,
    dairy_indicator: SignalValue,
    retail_proxy: SignalValue,
) -> SignalValue:
    import_cost_proxy = (dairy_indicator.value * exchange_rate.value) / 40.0
    benchmark_value = (import_cost_proxy + retail_proxy.value) / 2.0

    return SignalValue(
        label="Import Parity Benchmark",
        value=benchmark_value,
        unit="NGN per kg (proxy)",
        source_label="Derived from CBN FX + FAO dairy + NBS retail proxy",
        source_url="",
        last_updated=max(exchange_rate.last_updated, dairy_indicator.last_updated, retail_proxy.last_updated),
        status="derived",
        note="Formula: average of ((FAO * FX)/40) and retail proxy. Replace with landed-cost model later.",
    )


def get_market_signals() -> tuple[SignalValue, SignalValue, SignalValue, SignalValue]:
    exchange_rate = get_ngn_usd_signal()
    dairy_indicator = get_global_dairy_signal()
    retail_proxy = get_nigeria_retail_milk_proxy_signal()
    import_parity = calculate_import_parity_benchmark(exchange_rate, dairy_indicator, retail_proxy)
    return exchange_rate, dairy_indicator, retail_proxy, import_parity
