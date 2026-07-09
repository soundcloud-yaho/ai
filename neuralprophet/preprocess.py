"""Prometheus query_range JSON → NeuralProphet 입력 및 스케일 신호 변환."""

from __future__ import annotations  # 타입 힌트 forward reference 허용

import json  # 로컬 JSON 파일 로드
from datetime import datetime, timedelta, timezone  # 경기 이벤트·구간 분리 계산
from pathlib import Path  # 파일 경로 처리
from typing import Any  # dict 등 유연한 타입 표기

import numpy as np  # 수치 연산·올림(ceil)
import pandas as pd  # 시계열 DataFrame 처리

from config import (  # 스케일링·이벤트·예측 상수
    BASELINE_NODES,
    BASELINE_PODS,
    CPU_CORES_PER_NODE,
    EVENT_OFFSETS_MINUTES,
    FORECAST_MINUTES,
    HEADROOM_FACTOR,
    MATCH_EVENTS,
    N_FORECASTS,
    RPS_PER_CPU_CORE,
    RPS_PER_POD,
    STEP_MINUTES,
)


def load_prometheus_payload(payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prometheus query_range 응답 dict를 ds/y DataFrame으로 변환한다."""
    results = payload.get("data", {}).get("result", [])  # matrix result 배열 추출
    if not results:  # 빈 응답이면 학습/추론 불가
        raise ValueError("Prometheus 응답에 시계열 데이터가 없습니다.")

    values = results[0]["values"]  # 첫 번째 시계열의 [timestamp, value] 쌍
    df = pd.DataFrame(values, columns=["timestamp", "y"])  # 원시 DataFrame 생성
    df["ds"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)  # Unix초 → UTC datetime
    df["y"] = df["y"].astype(float)  # 문자열 RPS 값을 float으로 변환
    df = df[["ds", "y"]].sort_values("ds").reset_index(drop=True)  # NeuralProphet 필수 컬럼만 정렬

    meta = payload.get("meta", {})  # query, matches 등 부가 정보
    return df, meta  # (시계열, 메타) 튜플 반환


def load_prometheus_json(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """로컬 Prometheus query_range JSON 파일을 ds/y DataFrame으로 변환한다."""
    with path.open(encoding="utf-8") as f:  # UTF-8로 파일 열기
        payload = json.load(f)  # JSON 파싱
    return load_prometheus_payload(payload)  # 공통 변환 함수 재사용


def _parse_kickoff(match: dict[str, Any]) -> datetime:
    """경기 메타 한 건에서 킥오프 UTC datetime을 만든다."""
    hour, minute = map(int, match["kickoff_utc"].split(":"))  # "21:00" → hour=21, minute=0
    kickoff = datetime.strptime(match["date"], "%Y-%m-%d").replace(  # 날짜 문자열 파싱
        hour=hour, minute=minute, tzinfo=timezone.utc  # UTC 시각으로 조합
    )
    return kickoff  # 킥오프 datetime 반환


def build_events_df(meta: dict[str, Any]) -> pd.DataFrame:
    """경기 일정(meta.matches)을 NeuralProphet events_df(ds, event)로 변환한다."""
    rows: list[dict[str, Any]] = []  # 이벤트 행 누적 리스트
    for match in meta.get("matches", []):  # meta에 정의된 경기 목록 순회
        kickoff = _parse_kickoff(match)  # 해당 경기 킥오프 시각
        for event_name in MATCH_EVENTS:  # match_start, half_time 등
            offset = EVENT_OFFSETS_MINUTES[event_name]  # 킥오프 기준 분 오프셋
            rows.append(  # 이벤트 한 건 추가
                {
                    "ds": kickoff + timedelta(minutes=offset),  # 이벤트 발생 시각
                    "event": event_name,  # 이벤트 이름
                }
            )
    events_df = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)  # 시간순 정렬
    return events_df  # add_events() 입력용 DataFrame


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """5분 격자에 맞춰 리샘플하고 결측을 선형 보간한다."""
    indexed = df.set_index("ds").sort_index()  # datetime 인덱스로 변환·정렬
    resampled = indexed.resample(f"{STEP_MINUTES}min").mean()  # 5분 평균으로 리샘플
    resampled["y"] = resampled["y"].interpolate(method="time").ffill().bfill()  # 결측 보간
    return resampled.reset_index()[["ds", "y"]]  # 인덱스 복원 후 컬럼만 반환


def train_val_split(
    df: pd.DataFrame, val_days: int = 2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """마지막 val_days일을 검증 구간으로 분리한다."""
    cutoff = df["ds"].max() - timedelta(days=val_days)  # 검증 시작 시각 = 끝 - N일
    train_df = df[df["ds"] <= cutoff].copy()  # cutoff 이전 = 학습
    val_df = df[df["ds"] > cutoff].copy()  # cutoff 이후 = 검증(holdout)
    return train_df, val_df  # (학습, 검증) 튜플


def extract_forecast_steps(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """n_forecasts 스텝 예측값(yhat1~yhatN)을 한 행으로 정리한다."""
    yhat_cols = [c for c in forecast_df.columns if c.startswith("yhat") and c[4:].isdigit()]  # yhat1~12
    if not yhat_cols:  # predict() 결과에 yhat 컬럼이 없으면 오류
        raise ValueError("forecast DataFrame에 yhat 컬럼이 없습니다.")

    latest = forecast_df.dropna(subset=["yhat1"]).iloc[-1]  # 마지막 유효 예측 행
    rows = []  # 스텝별 예측값 누적
    for col in sorted(yhat_cols, key=lambda c: int(c[4:])):  # yhat1, yhat2, ... 순서
        step = int(col[4:])  # 컬럼명에서 스텝 번호 추출
        rows.append(
            {
                "step": step,  # 1~12 스텝
                "minutes_ahead": step * STEP_MINUTES,  # 몇 분 앞인지(5, 10, ... 60)
                "predicted_rps": max(0.0, float(latest[col])),  # 음수 예측은 0으로 클리핑
            }
        )
    return pd.DataFrame(rows)  # 60분 앞 예측 테이블


def derive_scale_signals(predicted_rps_peak: float, current_rps: float) -> dict[str, Any]:
    """예측 RPS 피크를 KEDA Pod / Karpenter Node 스케일 신호로 변환한다."""
    target_rps = predicted_rps_peak * HEADROOM_FACTOR  # 20% 여유(headroom) 반영 목표 RPS

    recommended_pods = max(BASELINE_PODS, int(np.ceil(target_rps / RPS_PER_POD)))  # 필요 Pod 수
    required_cpu = max(1.0, target_rps / RPS_PER_CPU_CORE)  # 필요 CPU 코어(최소 1)
    recommended_nodes = max(BASELINE_NODES, int(np.ceil(required_cpu / CPU_CORES_PER_NODE)))  # 필요 Node 수

    current_pods = max(BASELINE_PODS, int(np.ceil(current_rps * HEADROOM_FACTOR / RPS_PER_POD)))  # 현재 기준 Pod
    current_nodes = max(  # 현재 기준 Node
        BASELINE_NODES,
        int(np.ceil(current_rps * HEADROOM_FACTOR / RPS_PER_CPU_CORE / CPU_CORES_PER_NODE)),
    )

    return {  # KEDA/Karpenter가 사용할 스케일 신호 dict
        "current_rps": round(current_rps, 2),  # 현재 실측 RPS
        "predicted_rps_peak_1h": round(predicted_rps_peak, 2),  # 1시간 내 예측 피크
        "predicted_rps_with_headroom": round(target_rps, 2),  # 여유 반영 목표 RPS
        "forecast_horizon_minutes": FORECAST_MINUTES,  # 예측 구간(60분)
        "keda": {  # Pod 스케일 아웃 신호
            "metric_name": "worldcup_predicted_rps_peak",
            "recommended_pods": recommended_pods,
            "current_pods": current_pods,
            "scale_out_trigger": recommended_pods > current_pods,  # 권장 > 현재면 scale out
            "rps_per_pod": RPS_PER_POD,
        },
        "karpenter": {  # Node 스케일 아웃 신호
            "metric_name": "worldcup_predicted_cpu_cores_required",
            "required_cpu_cores": round(required_cpu, 2),
            "recommended_nodes": recommended_nodes,
            "current_nodes": current_nodes,
            "scale_out_trigger": recommended_nodes > current_nodes,
            "cpu_cores_per_node": CPU_CORES_PER_NODE,
        },
    }


def compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """MAE, RMSE, MAPE 검증 지표를 계산한다."""
    mask = actual.notna() & predicted.notna()  # NaN 제외 마스크
    y = actual[mask].astype(float)  # 실측값
    yhat = predicted[mask].astype(float)  # 예측값
    if len(y) == 0:  # 유효 비교 샘플 없음
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}

    mae = float(np.mean(np.abs(y - yhat)))  # Mean Absolute Error
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))  # Root Mean Squared Error
    mape = float(  # Mean Absolute Percentage Error (%)
        np.mean(
            np.abs((y - yhat) / y.replace(0, np.nan)).dropna()  # 0으로 나누기 방지
        ) * 100
    )
    return {"mae": mae, "rmse": rmse, "mape": mape}  # 검증 지표 dict
