#!/usr/bin/env python3
"""
[정제] 2분 주기: ALB(YACE) RPS 조회 -> 동적 임계치로 이상치 정제 -> Pushgateway push.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.prometheus_client import fetch_training_payload
from common.pushgateway import push_single_gauge
from config import (
    CONSECUTIVE_OVERRIDE,
    FALLBACK_THRESHOLD,
    HARD_CEILING,
    LOOKBACK_MINUTES,
    LOW_TRAFFIC_GUARD,
    LOW_TRAFFIC_MULTIPLIER,
    METRIC_NAME,
    MIN_PERIODS,
    MIN_THRESHOLD,
    OUTPUT_DIR,
    PROMETHEUS_QUERY,
    PROMETHEUS_URL,
    PUSHGATEWAY_DRY_RUN,
    PUSHGATEWAY_INSTANCE,
    PUSHGATEWAY_JOB,
    PUSHGATEWAY_URL,
    THRESHOLD_MULTIPLIER,
    WINDOW_MINUTES,
)
from preprocess import load_prometheus_payload
from rps_cleaner import clean_rps_series_dynamic


def main() -> None:
    """정제 파이프라인 진입점."""
    parser = argparse.ArgumentParser(description="ALB RPS 이상치 정제 후 Pushgateway push")
    parser.add_argument("--prometheus-url", default=PROMETHEUS_URL)
    parser.add_argument("--query", default=PROMETHEUS_QUERY)
    parser.add_argument("--lookback-minutes", type=int, default=LOOKBACK_MINUTES)
    parser.add_argument("--pushgateway-url", default=PUSHGATEWAY_URL)
    parser.add_argument("--pushgateway-job", default=PUSHGATEWAY_JOB)
    parser.add_argument("--pushgateway-instance", default=PUSHGATEWAY_INSTANCE)
    parser.add_argument(
        "--push",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pushgateway에 메트릭 push (기본: push)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=args.lookback_minutes)

    print("[1/4] Fetching raw RPS from Prometheus...")
    print(f"      url={args.prometheus_url}")
    print(f"      query={args.query}")
    print(f"      range={start.isoformat()} .. {end.isoformat()}")

    payload = fetch_training_payload(
        prometheus_url=args.prometheus_url,
        query=args.query,
        start=start,
        end=end,
        step="1m",
    )
    df, meta = load_prometheus_payload(payload)

    print("[2/4] Cleaning anomalies (dynamic threshold)...")
    result = clean_rps_series_dynamic(
        df,
        window_minutes=WINDOW_MINUTES,
        min_periods=MIN_PERIODS,
        multiplier=THRESHOLD_MULTIPLIER,
        fallback_threshold=FALLBACK_THRESHOLD,
        min_threshold=MIN_THRESHOLD,
        low_traffic_guard=LOW_TRAFFIC_GUARD,
        low_traffic_multiplier=LOW_TRAFFIC_MULTIPLIER,
        consecutive_override=CONSECUTIVE_OVERRIDE,
        hard_ceiling=HARD_CEILING,
    )

    n_cleaned = int(result.audit_log.shape[0])
    print(f"      cleaned={n_cleaned}/{len(df)}")
    if n_cleaned > 0:
        print(result.audit_log.to_string(index=False))

    latest = result.cleaned.iloc[-1]

    print("[3/4] Pushing metric to Pushgateway...")
    dry_run = PUSHGATEWAY_DRY_RUN or not args.push
    push_result = push_single_gauge(
        args.pushgateway_url,
        metric_name=METRIC_NAME,
        value=float(latest["y"]),
        job=args.pushgateway_job,
        instance=args.pushgateway_instance,
        help_text="Dynamically cleaned RPS (rolling baseline x multiplier, hard-ceiling guarded)",
        dry_run=dry_run,
    )

    audit_path = OUTPUT_DIR / "rps_cleaner_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "latest_ds": latest["ds"].isoformat(),
                "latest_y": float(latest["y"]),
                "n_cleaned": n_cleaned,
                "pushgateway": push_result,
                "data_meta": meta,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print("[4/4] Done")
    if push_result.get("pushed"):
        print(f"Pushgateway: {push_result['url']}")


if __name__ == "__main__":
    main()