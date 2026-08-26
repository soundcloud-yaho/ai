#!/usr/bin/env python3
"""
[정제] 2분 주기: ALB(YACE) RPS 조회 -> 동적 임계치로 이상치 정제 -> Pushgateway push.

[수정 사유] 기존 코드는 "최신 1개 값(iloc[-1])"만 Pushgateway에 push했음.
Pushgateway는 job/instance 조합당 마지막 값 하나만 보관하는 구조라서,
2분 CronJob 주기 안에서 정제 이벤트가 발생해도 다음 실행이 그 값을 덮어써버리면
Prometheus가 우연히 그 사이(스크랩 주기 내)에 값을 긁어가지 못하는 한
정제 이력이 시계열에서 통째로 사라짐 (query_range로 실측 확인됨 - 정제된
274.14가 한 번도 스크랩되지 못하고 다음 실행의 0으로 덮어써짐).

[해결] lookback 구간(22분치, 1분 스텝) 전체를 순회하며 "원본값"과 "정제값"을
각각 별도 게이지로, ds(각 분)를 매번 함께 push. Pushgateway는 동일 job/instance
안에서도 라벨이 다르면 별도 시계열로 취급하므로, ds를 라벨에 넣어 각 분(minute)을
서로 다른 시계열로 만들어 다음 실행이 서로를 덮어쓰지 않게 함.
이렇게 하면 Prometheus가 몇 초 만에 스크랩하든 상관없이, 22분치 이력이
매 실행마다 통째로 다시 채워져 시계열이 끊기지 않음.
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

    print("[3/4] Pushing metrics to Pushgateway...")
    dry_run = PUSHGATEWAY_DRY_RUN or not args.push

    # (기존 유지) "현재 최신 정제값" - 실시간 단일 스탯 패널(숫자 카드)용.
    # 여기서 push되는 값은 다음 실행 때 덮어써지는 게 정상 동작(원래 목적이 "지금 값"이므로).
    push_result = push_single_gauge(
        args.pushgateway_url,
        metric_name=METRIC_NAME,
        value=float(latest["y"]),
        job=args.pushgateway_job,
        instance=args.pushgateway_instance,
        help_text="Dynamically cleaned RPS (rolling baseline x multiplier, hard-ceiling guarded)",
        dry_run=dry_run,
    )

    # (신규) lookback 구간 전체(원본 df 길이만큼)를 분(minute) 단위로 각각 push.
    # instance 라벨에 "분(HHMM)"을 함께 넣어 서로 다른 시계열로 분리 -
    # 이렇게 하면 2분 뒤 다음 실행이 와도 "이번에 새로 계산된 분(minute)"만
    # 갱신되고, 그 이전 분들의 값은 이번 실행에서도 동일하게 다시 push되므로
    # 시계열이 끊기지 않고 lookback 구간만큼(기본 22분) 항상 유지됨.
    series_pushed = 0
    for i in range(len(df)):
        ts = df.iloc[i]["ds"]
        minute_key = ts.strftime("%Y%m%d%H%M")
        raw_y = float(df.iloc[i]["y"])
        cleaned_y = float(result.cleaned.iloc[i]["y"])
        is_cleaned = bool(result.cleaned.iloc[i]["is_cleaned"])

        push_single_gauge(
            args.pushgateway_url,
            metric_name="rps_raw_series",
            value=raw_y,
            job=args.pushgateway_job,
            instance=f"{args.pushgateway_instance}_{minute_key}",
            help_text="Raw RPS per minute (lookback window, keyed by minute to avoid overwrite)",
            dry_run=dry_run,
        )
        push_single_gauge(
            args.pushgateway_url,
            metric_name="rps_cleaned_series",
            value=cleaned_y,
            job=args.pushgateway_job,
            instance=f"{args.pushgateway_instance}_{minute_key}",
            help_text="Cleaned RPS per minute (lookback window, keyed by minute to avoid overwrite)",
            dry_run=dry_run,
        )
        push_single_gauge(
            args.pushgateway_url,
            metric_name="rps_anomaly_flag",
            value=1.0 if is_cleaned else 0.0,
            job=args.pushgateway_job,
            instance=f"{args.pushgateway_instance}_{minute_key}",
            help_text="1 if this minute's value was identified as anomaly and cleaned, else 0",
            dry_run=dry_run,
        )
        series_pushed += 1

    audit_path = OUTPUT_DIR / "rps_cleaner_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "latest_ds": latest["ds"].isoformat(),
                "latest_y": float(latest["y"]),
                "n_cleaned": n_cleaned,
                "series_pushed": series_pushed,
                "pushgateway": push_result,
                "data_meta": meta,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"      series_pushed={series_pushed} (raw/cleaned/anomaly-flag x {series_pushed} minutes)")
    print("[4/4] Done")
    if push_result.get("pushed"):
        print(f"Pushgateway: {push_result['url']}")


if __name__ == "__main__":
    main()