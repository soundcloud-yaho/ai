#!/usr/bin/env python3
"""
[추론] 5분 주기: train.py가 저장한 모델 로드 -> 트래픽 예측 -> Pushgateway push.

처리 흐름:
  1. PVC의 worldcup_model.np 로드 (train CronJob 산출물)
  2. Prometheus에서 최근 4시간 HTTP rate 조회 (AR 입력용, 재학습 아님)
  3. 60분 앞 RPS 예측 및 KEDA/Karpenter 스케일 신호 산출
  4. Pushgateway에 gauge 메트릭 push
"""

from __future__ import annotations  # 타입 힌트 forward reference 허용

import argparse  # CLI 인자 파싱
import json  # 예측 결과 JSON 저장
import sys  # sys.path 조작
from datetime import datetime, timezone  # 추론 시각
from pathlib import Path  # 경로 처리

import matplotlib.pyplot as plt  # 추론 그래프(로컬/디버그용)
import pandas as pd  # 시계열 DataFrame

ROOT_DIR = Path(__file__).resolve().parent.parent  # ai/ 레포 루트
if str(ROOT_DIR) not in sys.path:  # common 모듈 import 경로
    sys.path.insert(0, str(ROOT_DIR))

from common.prometheus_client import fetch_inference_payload  # 최근 이력 Prometheus 조회
from common.pushgateway import push_scale_signals  # Pushgateway PUT
from config import (  # 경로·Prometheus·Pushgateway·모델 설정
    DATA_PATH,
    MODEL_PATH,
    N_FORECASTS,
    N_LAGS,
    OUTPUT_DIR,
    PREDICT_HISTORY_HOURS,
    PROMETHEUS_QUERY,
    PROMETHEUS_URL,
    PUSHGATEWAY_DRY_RUN,
    PUSHGATEWAY_INSTANCE,
    PUSHGATEWAY_JOB,
    PUSHGATEWAY_URL,
    STEP_MINUTES,
    load_model,
)
from preprocess import (  # 전처리·스케일 신호·예측 스텝 추출
    derive_scale_signals,
    extract_forecast_steps,
    load_prometheus_json,
    load_prometheus_payload,
    resample_to_5min,
)

plt.rcParams["font.family"] = ["DejaVu Sans"]  # 그래프 폰트
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지


def _normalize_ds(df: pd.DataFrame) -> pd.DataFrame:
    """그래프용 ds 타임존을 naive datetime으로 통일한다."""
    out = df.copy()
    out["ds"] = pd.to_datetime(out["ds"], utc=True).dt.tz_localize(None)
    return out


def load_inference_data(
    source: str,
    data_path: Path,
    prometheus_url: str,
    query: str,
    history_hours: float,
    inference_at: datetime | None,
) -> tuple[pd.DataFrame, dict]:
    """운영(Prometheus) 또는 로컬 파일에서 추론용 최근 이력을 로드한다."""
    if source == "file":  # 로컬 테스트: JSON 파일 사용
        print(f"      source=file path={data_path}")
        return load_prometheus_json(data_path)

    now = inference_at or datetime.now(timezone.utc)  # 추론 기준 시각(기본: 지금)
    print(f"      source=prometheus url={prometheus_url}")
    print(f"      query={query}")
    print(f"      history_hours={history_hours}, inference_at={now.isoformat()}")

    payload = fetch_inference_payload(  # 최근 4시간만 조회(n_lags=36 입력용)
        prometheus_url=prometheus_url,
        query=query,
        history_hours=history_hours,
        now=now,
    )
    return load_prometheus_payload(payload)  # ds/y 변환


def plot_forecast_horizon(
    history_df: pd.DataFrame,
    forecast_steps: pd.DataFrame,
    inference_time: datetime,
    output_path: Path,
) -> None:
    """최근 이력 + 60분 앞 예측 그래프를 저장한다."""
    fig, ax = plt.subplots(figsize=(14, 6))

    recent = _normalize_ds(history_df).tail(N_LAGS)  # AR 입력 구간(최근 3시간)
    ax.plot(
        recent["ds"],
        recent["y"],
        color="#1f77b4",
        linewidth=1.5,
        label=f"Recent {N_LAGS * STEP_MINUTES}min history",
    )

    future_times = [  # 각 예측 스텝의 미래 시각
        inference_time + pd.Timedelta(minutes=step * STEP_MINUTES)
        for step in forecast_steps["step"]
    ]
    ax.plot(
        future_times,
        forecast_steps["predicted_rps"],
        color="#ff7f0e",
        marker="o",
        linewidth=2,
        label=f"Forecast next {N_FORECASTS * STEP_MINUTES}min",
    )
    ax.axvline(inference_time, color="#d62728", linestyle="--", linewidth=1, label="Inference time")  # 추론 시점
    ax.set_title("5-min Inference: 60-minute Ahead HTTP Rate Forecast")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("req/s")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_scale_signals(scale_signals: dict, output_path: Path) -> None:
    """KEDA/Karpenter 스케일 신호 막대 그래프를 저장한다."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Pod / Node 2열

    keda = scale_signals["keda"]
    karp = scale_signals["karpenter"]

    axes[0].bar(
        ["Current Pods", "Recommended Pods"],
        [keda["current_pods"], keda["recommended_pods"]],
        color=["#aec7e8", "#ff7f0e"],
    )
    axes[0].set_title(f"KEDA Scale Signal (trigger={keda['scale_out_trigger']})")
    axes[0].set_ylabel("Pod count")

    axes[1].bar(
        ["Current Nodes", "Recommended Nodes"],
        [karp["current_nodes"], karp["recommended_nodes"]],
        color=["#aec7e8", "#2ca02c"],
    )
    axes[1].set_title(f"Karpenter Scale Signal (trigger={karp['scale_out_trigger']})")
    axes[1].set_ylabel("Node count")

    fig.suptitle(
        f"Predicted peak RPS (1h): {scale_signals['predicted_rps_peak_1h']} | "
        f"Current RPS: {scale_signals['current_rps']}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """추론 파이프라인 진입점."""
    parser = argparse.ArgumentParser(description="Predict traffic and push scale signals to Pushgateway")
    parser.add_argument(
        "--source",
        choices=["prometheus", "file"],
        default="prometheus",  # 운영 CronJob 기본값
        help="데이터 소스. 운영=CronJob에서 prometheus, 로컬 검증=file",
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="--source file 일 때 JSON 경로")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="train.py가 저장한 모델 경로")
    parser.add_argument("--prometheus-url", default=PROMETHEUS_URL)  # 최근 이력 조회 URL
    parser.add_argument("--query", default=PROMETHEUS_QUERY)  # HTTP rate PromQL
    parser.add_argument("--history-hours", type=float, default=PREDICT_HISTORY_HOURS)  # AR 입력 이력 길이
    parser.add_argument(
        "--inference-at",
        type=str,
        default=None,
        help="ISO datetime (UTC). 미지정 시 now",
    )
    parser.add_argument("--pushgateway-url", default=PUSHGATEWAY_URL)
    parser.add_argument("--pushgateway-job", default=PUSHGATEWAY_JOB)
    parser.add_argument("--pushgateway-instance", default=PUSHGATEWAY_INSTANCE)
    parser.add_argument(
        "--push",
        action=argparse.BooleanOptionalAction,
        default=True,  # 운영에서는 기본 push
        help="Pushgateway에 메트릭 push (기본: push)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",  # 5분 CronJob에서 그래프 생략
        help="그래프 저장 생략 (5분 CronJob용)",
    )
    parser.add_argument("--tag", type=str, default=None, help="출력 파일 접미사 (로컬 테스트용)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 출력 디렉터리 보장

    if not args.model.exists():  # train CronJob이 아직 안 돌았으면 실패
        raise FileNotFoundError(
            f"Model not found: {args.model}. Run train.py first or wait for train CronJob."
        )

    inference_time: pd.Timestamp | None = None  # 추론 시각(Timestamp)
    if args.inference_at:  # 로컬 테스트용 고정 시각
        inference_time = pd.Timestamp(args.inference_at)
        if inference_time.tzinfo is None:  # naive면 UTC로 간주
            inference_time = inference_time.tz_localize("UTC")
    inference_at = inference_time.to_pydatetime() if inference_time is not None else None  # datetime 변환

    print(f"[1/6] Loading model: {args.model}")
    model = load_model(args.model)  # train.py 산출물 로드(재학습 없음)

    print("[2/6] Loading recent history...")
    df, meta = load_inference_data(  # Prometheus 최근 이력 또는 파일
        source=args.source,
        data_path=args.data,
        prometheus_url=args.prometheus_url,
        query=args.query,
        history_hours=args.history_hours,
        inference_at=inference_at,
    )
    df = resample_to_5min(df)  # 5분 격자 정렬

    if inference_time is None:  # 미지정 시 데이터 마지막 시점을 추론 시각으로
        inference_time = df["ds"].max()
        if inference_time.tzinfo is None:
            inference_time = inference_time.tz_localize("UTC")

    history_df = df[df["ds"] <= inference_time].copy()  # 추론 시각까지의 이력만 사용
    if len(history_df) < N_LAGS:  # AR에 필요한 최소 36포인트(3시간) 확인
        raise ValueError(f"Need at least {N_LAGS} history points, got {len(history_df)}")

    print(f"[3/6] Predicting from {inference_time} ...")
    forecast_df = model.predict(history_df)  # 60분 앞 예측(n_forecasts=12)
    forecast_steps = extract_forecast_steps(forecast_df)  # yhat1~12 정리

    current_rps = float(history_df.iloc[-1]["y"])  # 마지막 실측 RPS
    predicted_peak = float(forecast_steps["predicted_rps"].max())  # 1시간 내 예측 피크
    scale_signals = derive_scale_signals(predicted_peak, current_rps)  # KEDA/Karpenter 신호
    scale_signals["inference_time"] = inference_time.isoformat()
    scale_signals["model_path"] = str(args.model)

    print("[4/6] Pushing metrics to Pushgateway...")
    dry_run = PUSHGATEWAY_DRY_RUN or not args.push  # 환경변수 또는 --no-push
    push_result = push_scale_signals(
        args.pushgateway_url,
        scale_signals,
        job=args.pushgateway_job,
        instance=args.pushgateway_instance,
        dry_run=dry_run,
    )
    scale_signals["pushgateway"] = push_result  # push 결과를 JSON에도 기록

    prediction_result = {  # predictions.json 내용
        "inference_time": inference_time.isoformat(),
        "forecast_horizon_minutes": N_FORECASTS * STEP_MINUTES,  # 60분
        "forecast_steps": forecast_steps.to_dict(orient="records"),
        "scale_signals": scale_signals,
        "data_meta": meta,
    }

    tag = f"_{args.tag}" if args.tag else ""  # 로컬 테스트용 파일 접미사
    pred_path = OUTPUT_DIR / f"predictions{tag}.json"
    scale_path = OUTPUT_DIR / f"scale_signals{tag}.json"
    pred_path.write_text(
        json.dumps(prediction_result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    scale_path.write_text(
        json.dumps(scale_signals, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    if not args.skip_plots:  # CronJob은 --skip-plots로 생략
        print("[5/6] Saving plots...")
        plot_forecast_horizon(
            history_df,
            forecast_steps,
            inference_time.to_pydatetime(),
            OUTPUT_DIR / f"03_forecast_horizon{tag}.png",
        )
        plot_scale_signals(scale_signals, OUTPUT_DIR / f"04_scale_signals{tag}.png")
    else:
        print("[5/6] Skipping plots")

    print("[6/6] Done")
    print(json.dumps(scale_signals, indent=2, ensure_ascii=False, default=str))
    print(f"\nOutputs: {pred_path}, {scale_path}")
    if push_result.get("pushed"):  # 실제 push 성공 시 URL 출력
        print(f"Pushgateway: {push_result['url']}")


if __name__ == "__main__":  # 스크립트 직접 실행 시
    main()
