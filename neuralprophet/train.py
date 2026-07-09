#!/usr/bin/env python3
"""
[학습] 일 1회: Prometheus 시계열 조회 -> NeuralProphet 재학습 -> 모델 저장.

CronJob으로 하루에 한 번 실행되며, predict.py가 사용할 학습된 모델(.np)을 생성한다.
"""

from __future__ import annotations  # 타입 힌트 forward reference 허용

import argparse  # CLI 인자 파싱
import json  # 결과·지표 JSON 저장
import sys  # sys.path 조작(공통 모듈 import)
from datetime import datetime  # 날짜별 모델 파일명·ISO 파싱
from pathlib import Path  # 경로 처리
from zoneinfo import ZoneInfo  # 학습 타임존

import matplotlib.pyplot as plt  # 검증 그래프 생성
import pandas as pd  # 시계열 DataFrame
from neuralprophet import save  # 학습된 모델 .np 저장

ROOT_DIR = Path(__file__).resolve().parent.parent  # ai/ 레포 루트 (common 모듈 위치)
if str(ROOT_DIR) not in sys.path:  # 아직 path에 없으면
    sys.path.insert(0, str(ROOT_DIR))  # common.prometheus_client import 가능하게 추가

from common.prometheus_client import default_train_window, fetch_training_payload  # Prometheus 조회
from config import (  # 경로·Prometheus·학습 설정
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    PROMETHEUS_QUERY,
    PROMETHEUS_URL,
    TRAIN_DATA_START,
    TRAIN_LOOKBACK_DAYS,
    TRAIN_TIMEZONE,
    build_model,
)
from preprocess import (  # 전처리·검증 지표
    compute_metrics,
    load_prometheus_json,
    load_prometheus_payload,
    resample_to_5min,
    train_val_split,
)

plt.rcParams["font.family"] = ["DejaVu Sans"]  # 그래프 폰트(한글 미지원 환경 대비)
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지


def _normalize_ds(df: pd.DataFrame) -> pd.DataFrame:
    """그래프 병합 시 타임존 불일치를 막기 위해 ds를 naive datetime으로 통일한다."""
    out = df.copy()  # 원본 보존
    out["ds"] = pd.to_datetime(out["ds"], utc=True).dt.tz_localize(None)  # UTC → naive
    return out


def _parse_train_start(value: str | None) -> datetime | None:
    """--train-start ISO 문자열을 datetime으로 변환한다."""
    if not value:  # 미지정이면 None
        return None
    return datetime.fromisoformat(value)  # 예: 2026-06-01T00:00:00+09:00


def load_training_data(
    source: str,
    data_path: Path,
    prometheus_url: str,
    query: str,
    lookback_days: int,
    timezone_name: str,
    train_start: datetime | None,
    train_end: datetime | None,
) -> tuple[pd.DataFrame, dict]:
    """운영(Prometheus) 또는 로컬 파일에서 학습 데이터를 로드한다."""
    if source == "file":  # 로컬 테스트 모드
        print(f"      source=file path={data_path}")
        return load_prometheus_json(data_path)  # test_dataset.json 등

    start, end = default_train_window(  # 운영: 어제까지 구간 계산
        timezone_name=timezone_name,
        lookback_days=lookback_days,
        train_start=train_start,
        now=train_end,
    )
    print(f"      source=prometheus url={prometheus_url}")
    print(f"      query={query}")
    print(f"      range={start.isoformat()} .. {end.isoformat()} ({timezone_name})")

    payload = fetch_training_payload(  # Prometheus HTTP query_range
        prometheus_url=prometheus_url,
        query=query,
        start=start,
        end=end,
        step="5m",
    )
    return load_prometheus_payload(payload)  # ds/y DataFrame 변환


def plot_validation(val_df: pd.DataFrame, forecast: pd.DataFrame, output_path: Path) -> None:
    """검증 구간 실측 RPS vs 1-step 예측 그래프."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)  # 상하 2단 subplot

    val_df = _normalize_ds(val_df)  # 타임존 통일
    forecast = _normalize_ds(forecast)
    merged = val_df.merge(forecast[["ds", "yhat1"]], on="ds", how="inner")  # 실측·예측 병합

    axes[0].plot(merged["ds"], merged["y"], label="Actual RPS", color="#1f77b4", linewidth=1.2)  # 실측
    axes[0].plot(merged["ds"], merged["yhat1"], label="Predicted RPS (1-step)", color="#ff7f0e", linewidth=1.2)  # 예측
    axes[0].set_title("Validation: Actual vs Predicted HTTP Request Rate")
    axes[0].set_ylabel("req/s")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    error = merged["y"] - merged["yhat1"]  # 오차 = 실측 - 예측
    axes[1].bar(merged["ds"], error, width=0.003, color="#d62728", alpha=0.7)  # 오차 막대
    axes[1].axhline(0, color="black", linewidth=0.8)  # 0 기준선
    axes[1].set_title("Prediction Error (Actual - Predicted)")
    axes[1].set_xlabel("Time (UTC)")
    axes[1].set_ylabel("req/s")
    axes[1].grid(alpha=0.3)

    fig.autofmt_xdate()  # x축 날짜 라벨 자동 회전
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)  # PNG 저장
    plt.close(fig)  # 메모리 해제


def plot_training_overview(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, output_path: Path) -> None:
    """전체 데이터에서 학습/검증 구간이 어떻게 나뉘었는지 시각화한다."""
    fig, ax = plt.subplots(figsize=(14, 5))  # 단일 subplot
    ax.plot(df["ds"], df["y"], color="#cccccc", linewidth=0.8, label="Full dataset")  # 전체(회색)
    ax.plot(train_df["ds"], train_df["y"], color="#1f77b4", linewidth=1.0, label="Train")  # 학습(파랑)
    ax.plot(val_df["ds"], val_df["y"], color="#2ca02c", linewidth=1.2, label="Validation")  # 검증(초록)
    ax.set_title("Dataset Split for Training (last 2 days = validation)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("req/s")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """학습 파이프라인 진입점."""
    parser = argparse.ArgumentParser(description="Train NeuralProphet model for World Cup traffic")
    parser.add_argument(
        "--source",
        choices=["prometheus", "file"],
        default="prometheus",  # 운영 기본값: Prometheus
        help="데이터 소스. 운영=CronJob에서 prometheus, 로컬 검증=file",
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="--source file 일 때 JSON 경로")
    parser.add_argument("--prometheus-url", default=PROMETHEUS_URL)  # 클러스터 내부 Prometheus URL
    parser.add_argument("--query", default=PROMETHEUS_QUERY)  # HTTP rate PromQL
    parser.add_argument("--timezone", default=TRAIN_TIMEZONE, help="학습 구간 계산 타임존")
    parser.add_argument("--lookback-days", type=int, default=TRAIN_LOOKBACK_DAYS)  # 조회 일수
    parser.add_argument(
        "--train-start",
        default=TRAIN_DATA_START,
        help="학습 시작 시각 ISO8601. 미지정 시 lookback-days 사용",
    )
    parser.add_argument(
        "--train-end",
        default=None,
        help="학습 종료 기준 시각 ISO8601. 미지정 시 해당 타임존 오늘 00:00(=어제까지 포함)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="README epochs 덮어쓰기")
    parser.add_argument("--val-days", type=int, default=2)  # holdout 검증 일수
    args = parser.parse_args()  # CLI 인자 파싱 완료

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # output 디렉터리 생성

    train_start = _parse_train_start(args.train_start)  # 고정 시작일(선택)
    train_end = _parse_train_start(args.train_end) if args.train_end else None  # 종료 기준(선택)

    print("[1/5] Loading training data...")
    df, meta = load_training_data(  # Prometheus 또는 파일에서 로드
        source=args.source,
        data_path=args.data,
        prometheus_url=args.prometheus_url,
        query=args.query,
        lookback_days=args.lookback_days,
        timezone_name=args.timezone,
        train_start=train_start,
        train_end=train_end,
    )
    df = resample_to_5min(df)  # 5분 격자 정렬·결측 보간
    train_df, val_df = train_val_split(df, val_days=args.val_days)  # 마지막 N일 holdout
    print(f"      total={len(df)}, train={len(train_df)}, val={len(val_df)}")

    print("[2/5] Training model...")
    model = build_model()  # README 옵션으로 NeuralProphet 생성
    if args.epochs is not None:  # 빠른 로컬 테스트용 epoch 덮어쓰기
        model.config_train.epochs = args.epochs
    metrics = model.fit(train_df, freq="5min")  # 학습(재학습 아님, 매일 새 모델)

    save(model, str(MODEL_PATH))  # predict.py가 로드할 최신 모델
    dated_model_path = OUTPUT_DIR / f"worldcup_model_{datetime.now(ZoneInfo(args.timezone)).date()}.np"  # 날짜별 백업
    save(model, str(dated_model_path))
    print(f"      model saved: {MODEL_PATH}")
    print(f"      dated copy : {dated_model_path}")

    print("[3/5] Validating on holdout...")
    val_forecast = model.predict(val_df)  # 검증 구간 1-step 예측
    val_metrics = compute_metrics(val_forecast["y"], val_forecast["yhat1"])  # MAE/RMSE/MAPE

    result = {  # train_metrics.json에 저장할 결과
        "data_source": args.source,
        "prometheus_url": args.prometheus_url if args.source == "prometheus" else None,
        "query": args.query if args.source == "prometheus" else None,
        "timezone": args.timezone,
        "train_points": len(train_df),
        "val_points": len(val_df),
        "validation_metrics": val_metrics,
        "fit_metrics": metrics if isinstance(metrics, dict) else str(metrics),
        "model_path": str(MODEL_PATH),
        "dated_model_path": str(dated_model_path),
        "data_meta": meta,
    }
    METRICS_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("[4/5] Saving plots...")
    plot_training_overview(df, train_df, val_df, OUTPUT_DIR / "01_dataset_split.png")
    plot_validation(val_df, val_forecast, OUTPUT_DIR / "02_validation_forecast.png")

    print("[5/5] Done")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"\nGraphs: {OUTPUT_DIR}/01_dataset_split.png, {OUTPUT_DIR}/02_validation_forecast.png")


if __name__ == "__main__":  # 스크립트 직접 실행 시
    main()
