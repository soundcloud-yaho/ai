"""Prometheus query_range JSON → ds/y DataFrame 변환"""

from __future__ import annotations

from typing import Any

import pandas as pd


def load_prometheus_payload(payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    results = payload.get("data", {}).get("result", [])
    if not results:
        raise ValueError("Prometheus 응답에 시계열 데이터가 없습니다.")

    values = results[0]["values"]
    df = pd.DataFrame(values, columns=["timestamp", "y"])
    df["ds"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["y"] = df["y"].astype(float)
    df = df[["ds", "y"]].sort_values("ds").reset_index(drop=True)

    meta = payload.get("meta", {})
    return df, meta