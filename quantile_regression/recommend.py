"""Quantile Regression 기반 Pod 리소스 사이징 권고.

Prometheus에서 수집한 CPU·메모리 시계열을 P50/P90/P99 분위수로 분석하고,
P99 + 버퍼 기준으로 Request/Limit 권고값을 산출한다.

자동 반영하지 않는다 — 권고 결과는 Slack 리포트로만 전달되며,
운영자가 검토 후 deployment.yaml에 수동으로 반영한다.

로컬 개발(클러스터 밖)에서는 Prometheus 내부 DNS가 해석되지 않으므로
JSON 파일을 직접 넘기는 방식(--source file)으로 테스트한다.

CronJob 진입점: python -m quantile_regression.recommend
"""

import argparse   # CLI 인자 파싱
import json       # 결과 JSON 저장
import os         # 환경변수 읽기
import sys        # sys.path 조작
from pathlib import Path   # 경로 처리
from typing import Any, Dict  # Python 3.8 호환 타입 힌트

import numpy as np   # 수치 연산
import pandas as pd  # 시계열 DataFrame 처리
from sklearn.linear_model import QuantileRegressor  # 분위수 회귀 모델

ROOT_DIR = Path(__file__).resolve().parent.parent  # ai/ 레포 루트
if str(ROOT_DIR) not in sys.path:  # common 모듈 import 가능하게 추가
    sys.path.insert(0, str(ROOT_DIR))


# ── 환경변수 기본값 ────────────────────────────────────────────────────────────

PROMETHEUS_URL    = os.environ.get(
    "PROMETHEUS_URL",
    "http://kube-prometheus-stack-prometheus.monitoring.svc:9090",
)
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
QR_OUTPUT_DIR     = Path(os.environ.get("QR_OUTPUT_DIR", Path(__file__).parent / "output"))
QR_LOOKBACK_DAYS  = int(os.environ.get("QR_LOOKBACK_DAYS", "30"))
QR_DRY_RUN        = os.environ.get("QR_DRY_RUN", "").lower() in {"1", "true", "yes"}

POD_NAME     = os.environ.get("QR_POD_NAME", "backend")
CPU_QUERY    = os.environ.get(
    "QR_CPU_QUERY",
    'rate(container_cpu_usage_seconds_total{namespace="app",container="backend"}[5m])',
)
MEMORY_QUERY = os.environ.get(
    "QR_MEMORY_QUERY",
    'container_memory_working_set_bytes{namespace="app",container="backend"}',
)

CURRENT_CPU_REQUEST = os.environ.get("CURRENT_CPU_REQUEST", "200m")
CURRENT_CPU_LIMIT   = os.environ.get("CURRENT_CPU_LIMIT",   "1000m")
CURRENT_MEM_REQUEST = os.environ.get("CURRENT_MEM_REQUEST", "256Mi")
CURRENT_MEM_LIMIT   = os.environ.get("CURRENT_MEM_LIMIT",   "512Mi")


# ── 상수 ──────────────────────────────────────────────────────────────────────

QUANTILES = {  # 분석할 분위수 목록
    "p50": 0.50,
    "p90": 0.90,
    "p99": 0.99,
}

CPU_BUFFER_FACTOR = 1.2   # P99 CPU × 1.2 → Request (20% 여유)
CPU_LIMIT_FACTOR  = 2.0   # P99 CPU × 2.0 → Limit
MEM_BUFFER_FACTOR = 1.2   # P99 Memory × 1.2 → Request
MEM_LIMIT_FACTOR  = 2.0   # P99 Memory × 2.0 → Limit

BYTES_PER_MIB = 1024 * 1024  # bytes → MiB 변환 상수
RESAMPLE_STEP = "5min"        # 리샘플링 간격 — Prometheus step과 일치


# ── Prometheus payload 파싱 ────────────────────────────────────────────────────

def load_prometheus_payload(payload):
    # type: (Dict[str, Any]) -> pd.DataFrame
    """Prometheus query_range 응답 dict를 timestamp/value DataFrame으로 변환한다."""
    results = payload.get("data", {}).get("result", [])  # matrix result 배열 추출
    if not results:  # 빈 응답이면 분석 불가
        raise ValueError("Prometheus 응답에 시계열 데이터가 없습니다.")

    rows = []
    for result in results:  # 여러 Pod가 있을 경우 전체 합산
        for ts, val in result["values"]:  # [timestamp, value] 쌍 순회
            rows.append({"timestamp": float(ts), "value": float(val)})

    df = (
        pd.DataFrame(rows)
        .sort_values("timestamp")  # 시간순 정렬
        .reset_index(drop=True)
    )
    return df  # timestamp / value 컬럼 DataFrame


# ── 전처리 ────────────────────────────────────────────────────────────────────

def preprocess(df):
    # type: (pd.DataFrame) -> pd.DataFrame
    """5분 격자 리샘플링 + 결측치 보간.

    QR 모델은 행 순번(0, 1, 2...)을 시간 특징으로 쓰기 때문에
    샘플 간격이 균일하지 않으면 분위수 계산이 왜곡된다.
    neuralprophet/preprocess.py의 resample_to_5min과 동일한 전략을 따른다.

    이상치 제거는 하지 않는다 —
    P99 자체가 이상치를 포함한 최악의 경우를 반영하는 것이 목적이기 때문이다.
    """
    df = df.copy()  # 원본 보존
    df["ds"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)  # Unix초 → UTC datetime

    # 5분 격자로 리샘플 — scrape 지연으로 생긴 중복 해소
    resampled = (
        df.set_index("ds")["value"]
        .resample(RESAMPLE_STEP)
        .mean()
        .interpolate(method="time")  # 시간 가중 선형 보간
        .ffill()                     # 앞쪽 끝단 NaN 처리
        .bfill()                     # 뒤쪽 끝단 NaN 처리
    )

    return resampled.reset_index().rename(columns={"ds": "ds", "value": "value"})


# ── Quantile Regression 분석 ──────────────────────────────────────────────────

def analyze_quantiles(df):
    # type: (pd.DataFrame) -> Dict[str, float]
    """P50 / P90 / P99 분위수를 Quantile Regression으로 산출한다.

    특징(X): 행 순번(0, 1, 2, ...) — 단순 시간 추세 근사
    목적(y): 리소스 사용량(CPU 코어 또는 bytes)
    마지막 시점(len(values))의 예측값을 각 분위수의 대표값으로 사용한다.
    """
    col = "value" if "value" in df.columns else df.columns[-1]  # preprocess 후 컬럼 자동 감지
    values = df[col].dropna().values  # NaN 제거 후 numpy 배열
    if len(values) == 0:  # 유효 데이터 없으면 분석 불가
        raise ValueError("분석할 데이터가 없습니다.")

    X = np.arange(len(values)).reshape(-1, 1)  # 순번 특징 행렬

    result = {}
    for label, q in QUANTILES.items():
        model = QuantileRegressor(
            quantile=q,
            alpha=0,          # 정규화 없음 — 소규모 데이터에 적합
            solver="highs",   # 대용량에도 안정적인 LP 솔버
        )
        model.fit(X, values)  # 분위수 회귀 학습
        pred = model.predict([[len(values)]])[0]  # 다음 시점 예측값
        result[label] = max(0.0, round(float(pred), 6))  # 음수 방지

    return result  # {"p50": ..., "p90": ..., "p99": ...}


# ── 권고값 생성 ───────────────────────────────────────────────────────────────

def _cpu_recommendation(stats):
    # type: (Dict[str, float]) -> Dict[str, str]
    """CPU 분위수(코어 단위)를 millicores Request/Limit 권고값으로 변환한다."""
    def to_m(cores):
        return "{0}m".format(max(1, int(cores * 1000)))

    p99 = stats["p99"]
    return {
        "p50":                 to_m(stats["p50"]),
        "p90":                 to_m(stats["p90"]),
        "p99":                 to_m(p99),
        "recommended_request": to_m(p99 * CPU_BUFFER_FACTOR),  # P99 + 20%
        "recommended_limit":   to_m(p99 * CPU_LIMIT_FACTOR),   # P99 × 2
    }


def _memory_recommendation(stats):
    # type: (Dict[str, float]) -> Dict[str, str]
    """메모리 분위수(bytes 단위)를 MiB Request/Limit 권고값으로 변환한다."""
    def to_mi(b):
        return "{0}Mi".format(max(1, int(b / BYTES_PER_MIB)))

    p99 = stats["p99"]
    return {
        "p50":                 to_mi(stats["p50"]),
        "p90":                 to_mi(stats["p90"]),
        "p99":                 to_mi(p99),
        "recommended_request": to_mi(p99 * MEM_BUFFER_FACTOR),  # P99 + 20%
        "recommended_limit":   to_mi(p99 * MEM_LIMIT_FACTOR),   # P99 × 2
    }


def generate_recommendation(cpu_stats, mem_stats, pod_name="backend"):
    # type: (Dict[str, float], Dict[str, float], str) -> Dict[str, Any]
    """CPU·메모리 분위수를 받아 Pod 리소스 권고 dict를 반환한다.

    자동 반영하지 않는다.
    반환값은 report.py로 전달되어 Slack 메시지로만 출력된다.
    """
    return {
        "pod_name": pod_name,
        "cpu":    _cpu_recommendation(cpu_stats),    # millicores 권고
        "memory": _memory_recommendation(mem_stats),  # MiB 권고
        "note": (
            "자동 반영 아님 — 운영자 검토 후 "
            "k8s/manifests/backend/deployment.yaml에 수동 적용하세요."
        ),
    }


# ── 데이터 로드 (Prometheus 또는 로컬 파일) ───────────────────────────────────

def load_training_data(source, data_path, query):
    # type: (str, Path, str) -> Dict[str, Any]
    """Prometheus 또는 로컬 파일에서 query_range 응답을 반환한다."""
    if source == "file":  # 로컬 테스트 모드
        print("      source=file  path={0}".format(data_path))
        with open(str(data_path), encoding="utf-8") as f:
            return json.load(f)

    # 운영 모드: Prometheus HTTP query_range
    from common.prometheus_client import default_train_window, fetch_training_payload
    start, end = default_train_window(lookback_days=QR_LOOKBACK_DAYS)
    print("      source=prometheus  query={0}".format(query))
    print("      range={0} .. {1}".format(start.isoformat(), end.isoformat()))

    return fetch_training_payload(
        prometheus_url=PROMETHEUS_URL,
        query=query,
        start=start,
        end=end,
        step="5m",
    )


# ── 메인 파이프라인 ────────────────────────────────────────────────────────────

def main():
    """QR 권고 파이프라인 진입점. CronJob: python -m quantile_regression.recommend"""
    parser = argparse.ArgumentParser(description="Quantile Regression Pod Sizing Report")
    parser.add_argument(
        "--source",
        choices=["prometheus", "file"],
        default="prometheus",  # 운영 기본값: Prometheus
        help="데이터 소스. CronJob=prometheus, 로컬 검증=file",
    )
    parser.add_argument(
        "--cpu-data",
        type=Path,
        default=Path(__file__).parent / "test_cpu.json",
        help="--source file 일 때 CPU JSON 경로",
    )
    parser.add_argument(
        "--mem-data",
        type=Path,
        default=Path(__file__).parent / "test_mem.json",
        help="--source file 일 때 Memory JSON 경로",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=QR_DRY_RUN,
        help="Slack 전송 없이 권고값만 출력",
    )
    args = parser.parse_args()

    QR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # output 디렉터리 생성

    # ── 1. CPU 메트릭 수집 + 분석 ─────────────────────────────────────────────
    print("[1/4] CPU 메트릭 수집 중...")
    cpu_payload = load_training_data(args.source, args.cpu_data, CPU_QUERY)
    cpu_df      = preprocess(load_prometheus_payload(cpu_payload))
    cpu_stats   = analyze_quantiles(cpu_df)
    print("      CPU P50={0:.4f}  P90={1:.4f}  P99={2:.4f} (cores)".format(
        cpu_stats["p50"], cpu_stats["p90"], cpu_stats["p99"]))

    # ── 2. Memory 메트릭 수집 + 분석 ──────────────────────────────────────────
    print("[2/4] Memory 메트릭 수집 중...")
    mem_payload = load_training_data(args.source, args.mem_data, MEMORY_QUERY)
    mem_df      = preprocess(load_prometheus_payload(mem_payload))
    mem_stats   = analyze_quantiles(mem_df)
    print("      MEM P50={0}Mi  P90={1}Mi  P99={2}Mi".format(
        int(mem_stats["p50"] // 1024 // 1024),
        int(mem_stats["p90"] // 1024 // 1024),
        int(mem_stats["p99"] // 1024 // 1024)))

    # ── 3. 권고값 생성 ─────────────────────────────────────────────────────────
    print("[3/4] 권고값 생성 중...")
    recommendation = generate_recommendation(cpu_stats, mem_stats, pod_name=POD_NAME)

    print("      [현재] CPU Request={0}  Limit={1}".format(
        CURRENT_CPU_REQUEST, CURRENT_CPU_LIMIT))
    print("      [권고] CPU Request={0}  Limit={1}".format(
        recommendation["cpu"]["recommended_request"],
        recommendation["cpu"]["recommended_limit"]))
    print("      [현재] MEM Request={0}  Limit={1}".format(
        CURRENT_MEM_REQUEST, CURRENT_MEM_LIMIT))
    print("      [권고] MEM Request={0}  Limit={1}".format(
        recommendation["memory"]["recommended_request"],
        recommendation["memory"]["recommended_limit"]))

    # 결과 JSON 저장
    result = {
        "pod_name": POD_NAME,
        "current": {
            "cpu_request": CURRENT_CPU_REQUEST,
            "cpu_limit":   CURRENT_CPU_LIMIT,
            "mem_request": CURRENT_MEM_REQUEST,
            "mem_limit":   CURRENT_MEM_LIMIT,
        },
        "recommendation": recommendation,
        "cpu_stats":      cpu_stats,
        "mem_stats":      mem_stats,
    }
    output_path = QR_OUTPUT_DIR / "qr_report.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("      결과 저장: {0}".format(output_path))

    # ── 4. Slack 전송 ──────────────────────────────────────────────────────────
    print("[4/4] Slack 리포트 전송 중...")
    from quantile_regression.report import send_slack_report
    if not SLACK_WEBHOOK_URL and not args.dry_run:
        print("      SLACK_WEBHOOK_URL 미설정 — 전송 건너뜀")
    else:
        send_result = send_slack_report(
            slack_webhook_url=SLACK_WEBHOOK_URL,
            recommendation=recommendation,
            dry_run=args.dry_run,
        )
        status = "전송 완료" if send_result.get("sent") else "dry-run"
        print("      Slack {0}".format(status))

    print("[Done]")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()