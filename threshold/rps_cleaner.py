"""
환경 전제 (동적 임계치 + 새벽 저트래픽 방어 버전)
  - 평상시 rps 범위: 0 ~ 100
  - 공격 트래픽: 대략 500 rps 수준
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass

WINDOW_MINUTES = 12
MIN_PERIODS = 3
THRESHOLD_MULTIPLIER = 5
FALLBACK_THRESHOLD = 500

MIN_THRESHOLD = 50
LOW_TRAFFIC_GUARD = 20
LOW_TRAFFIC_MULTIPLIER = 10

CONSECUTIVE_OVERRIDE = 3
HARD_CEILING = 300


@dataclass
class CleaningResult:
    cleaned: pd.DataFrame
    audit_log: pd.DataFrame


def clean_rps_series_dynamic(
    df: pd.DataFrame,
    window_minutes: int = WINDOW_MINUTES,
    min_periods: int = MIN_PERIODS,
    multiplier: float = THRESHOLD_MULTIPLIER,
    fallback_threshold: float = FALLBACK_THRESHOLD,
    min_threshold: float = MIN_THRESHOLD,
    low_traffic_guard: float = LOW_TRAFFIC_GUARD,
    low_traffic_multiplier: float = LOW_TRAFFIC_MULTIPLIER,
    consecutive_override: int = CONSECUTIVE_OVERRIDE,
    hard_ceiling: float = HARD_CEILING,
) -> CleaningResult:
    """
    df: columns = ["ds", "y"], ds 오름차순 가정.
    """
    df = df.sort_values("ds").reset_index(drop=True).copy()

    cleaned_values = []
    audit_rows = []
    last_valid = None
    consecutive_over_count = 0

    for i, row in df.iterrows():
        window = cleaned_values[-window_minutes:]

        if len(window) < min_periods:
            threshold = fallback_threshold
            baseline = None
        else:
            baseline = sum(window) / len(window)
            effective_multiplier = (
                low_traffic_multiplier if baseline < low_traffic_guard else multiplier
            )
            threshold = max(baseline * effective_multiplier, min_threshold)

        is_over = row["y"] > threshold

        if is_over:
            consecutive_over_count += 1

            if consecutive_over_count >= consecutive_override and row["y"] <= hard_ceiling:
                cleaned_value = row["y"]
                last_valid = row["y"]
                consecutive_over_count = 0
                audit_rows.append({
                    "ds": row["ds"],
                    "original_y": row["y"],
                    "replaced_with": row["y"],
                    "baseline": baseline,
                    "threshold": threshold,
                    "reason": (
                        f"threshold 초과했지만 연속 {consecutive_override}회 이상 지속 & "
                        f"hard_ceiling({hard_ceiling}) 이하 -> 실제 트래픽 증가로 인정, 통과"
                    ),
                })
            else:
                if last_valid is None:
                    last_valid = threshold
                cleaned_value = last_valid
                reason = (
                    f"y({row['y']:.1f}) > threshold({threshold:.1f}) "
                    f"[baseline={'N/A' if baseline is None else f'{baseline:.1f}'}]"
                )
                if row["y"] > hard_ceiling:
                    reason += f" | hard_ceiling({hard_ceiling}) 초과 -> 연속 지속돼도 정상 인정 안 함"
                audit_rows.append({
                    "ds": row["ds"],
                    "original_y": row["y"],
                    "replaced_with": last_valid,
                    "baseline": baseline,
                    "threshold": threshold,
                    "reason": reason,
                })
        else:
            consecutive_over_count = 0
            cleaned_value = row["y"]
            last_valid = row["y"]

        cleaned_values.append(cleaned_value)

    result_df = df.copy()
    result_df["y"] = cleaned_values
    result_df["is_cleaned"] = df["y"].values != result_df["y"].values

    return CleaningResult(cleaned=result_df, audit_log=pd.DataFrame(audit_rows))