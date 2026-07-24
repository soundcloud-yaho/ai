"""Quantile Regression 기반 Pod 리소스 사이징 권고.

Prometheus에서 수집한 CPU·메모리 시계열을 경기 시간대/평시로 분리해
Segmented Quantile(np.quantile) 방식으로 분위수를 산출한다.

컨테이너 이름을 Prometheus에서 자동 탐지 — 환경변수 입력 불필요.
분석 대상 네임스페이스:
  app        → backend, sync-matches (Worker)
  monitoring → prometheus (System)
  argocd     → argocd-server (System)
  kube-system→ karpenter (System)
  ai         → np-predict, np-train (AI)

CronJob 진입점: python -m quantile_regression.recommend
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen
from urllib.parse import urlencode

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ── 상수 ──────────────────────────────────────────────────────────────────────

POD_QUANTILES = {"p50": 0.50, "p90": 0.90, "p99": 0.99}
NODE_QUANTILES = {"p50": 0.50, "p90": 0.90, "p95": 0.95}

# 경기 시간대 — KST 18~02시/ 프로젝트 테스트를 위한 시간 12~19시까지
#MATCH_HOURS_KST = list(range(18, 24)) + list(range(0, 3))
MATCH_HOURS_KST = list(range(12, 19))

BYTES_PER_MIB = 1024 * 1024
RESAMPLE_STEP = "5min"

# 분석 대상 정의 — (label, namespace, 후보 컨테이너 이름 목록)
# 후보 목록 중 Prometheus에 실제 존재하는 첫 번째 이름을 자동 선택
ANALYSIS_TARGETS = [
    # label              namespace      후보 컨테이너 이름
    ("backend",          "app",         ["backend"]),
    ("sync-matches",     "app",         ["sync", "sync-matches"]),
    ("prometheus",       "monitoring",  ["prometheus", "prometheus-server"]),
    ("argocd",          "argocd",      ["argocd-server", "server"]),
    ("karpenter",        "kube-system", ["controller", "karpenter"]),
    ("np-predict",       "ai",          ["predict"]),
    ("np-train",         "ai",          ["neuralprophet-train"]),
]


# ── 단위 변환 유틸 ────────────────────────────────────────────────────────────

def _cores_to_millicores(cores):
    # type: (float) -> str
    return "{0}m".format(max(1, int(cores * 1000)))


def _bytes_to_mib(b):
    # type: (float) -> str
    return "{0}Mi".format(max(1, int(b / BYTES_PER_MIB)))


# ── Prometheus 컨테이너 자동 탐지 ─────────────────────────────────────────────

def discover_container(prometheus_url, namespace, candidates, timeout=10):
    # type: (str, str, List[str], int) -> str
    """Prometheus에서 실제 존재하는 컨테이너 이름을 자동 탐지한다.

    후보 목록을 순서대로 조회해 데이터가 있는 첫 번째 이름을 반환한다.
    모두 없으면 빈 문자열을 반환 → 해당 Pod 건너뜀.
    """
    for candidate in candidates:
        query = (
            'absent(container_cpu_usage_seconds_total'
            '{{namespace="{ns}",container="{c}"}})'
        ).format(ns=namespace, c=candidate)
        url = "{base}/api/v1/query?{params}".format(
            base=prometheus_url,
            params=urlencode({"query": query}),
        )
        try:
            with urlopen(url, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            # absent()가 결과 없음 = 컨테이너가 존재함
            if not data.get("data", {}).get("result"):
                return candidate
        except Exception:
            continue  # 조회 실패 시 다음 후보 시도
    return ""  # 모든 후보 없음


def discover_all_containers(prometheus_url, targets, source):
    # type: (str, list, str) -> Dict[str, str]
    """모든 분석 대상의 컨테이너 이름을 자동 탐지한다.

    --source file 모드에서는 탐지를 건너뛰고 후보 목록 첫 번째를 사용한다.
    """
    container_map = {}

    for label, namespace, candidates in targets:
        if source == "file":
            # 로컬 테스트: 탐지 없이 첫 번째 후보 사용
            container_map[label] = candidates[0]
            continue

        print("      [{0}] 컨테이너 탐지 중... (ns={1})".format(label, namespace))
        found = discover_container(prometheus_url, namespace, candidates)
        if found:
            print("        → 발견: {0}".format(found))
        else:
            print("        → 없음 — 건너뜀")
        container_map[label] = found

    return container_map


# ── Prometheus payload 파싱 ────────────────────────────────────────────────────

def load_prometheus_payload(payload):
    # type: (Dict[str, Any]) -> pd.DataFrame
    results = payload.get("data", {}).get("result", [])
    if not results:
        raise ValueError("Prometheus 응답에 시계열 데이터가 없습니다.")

    rows = []
    for result in results:
        for ts, val in result["values"]:
            rows.append({"timestamp": float(ts), "value": float(val)})

    return (
        pd.DataFrame(rows)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# ── 전처리 ────────────────────────────────────────────────────────────────────

def preprocess(df):
    # type: (pd.DataFrame) -> pd.DataFrame
    """5분 격자 리샘플링 + 결측치 보간."""
    df = df.copy()
    df["ds"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    resampled = (
        df.set_index("ds")["value"]
        .resample(RESAMPLE_STEP)
        .mean()
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    result = resampled.reset_index()
    result.columns = ["timestamp", "value"]
    return result


# ── Segmented Quantile 분석 ───────────────────────────────────────────────────

def analyze_quantiles(df, quantiles=None, match_hours=None):
    # type: (pd.DataFrame, Optional[Dict[str, float]], Optional[List[int]]) -> Dict[str, Any]
    """경기 시간대 vs 평시로 분리해 np.quantile()로 분위수를 산출한다."""
    if quantiles is None:
        quantiles = POD_QUANTILES
    if match_hours is None:
        match_hours = MATCH_HOURS_KST

    df = df.copy()
    df["hour"] = (
        pd.to_datetime(df["timestamp"], unit="s", utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.hour
    )

    match_vals  = df[df["hour"].isin(match_hours)]["value"].dropna().values
    normal_vals = df[~df["hour"].isin(match_hours)]["value"].dropna().values
    all_vals    = df["value"].dropna().values

    result = {}
    
    for label, q in quantiles.items():
        # 경기 시간대 분위수
        src = match_vals if len(match_vals) >= 10 else all_vals
        result[label] = float(round(np.quantile(src, q), 6))

        # 평시 분위수 — 별도 키로 저장
        src_normal = normal_vals if len(normal_vals) >= 10 else all_vals
        result["{0}_normal".format(label)] = float(round(np.quantile(src_normal, q), 6))

    result["match_sample_count"]  = int(len(match_vals))
    result["normal_sample_count"] = int(len(normal_vals))
    result["total_sample_count"]  = int(len(all_vals))
    result["used_segment"] = "match" if len(match_vals) >= 10 else "all"
    return result


# ── 권고값 생성 ───────────────────────────────────────────────────────────────

def _cpu_rec(stats, buf, lim):
    # type: (Dict[str, Any], float, float) -> Dict[str, str]
    p99 = stats["p99"]
    p99_normal = stats.get("p99_normal", p99)  # 평시 P99
    return {
        # 경기 시간대 권고
        "p50": _cores_to_millicores(stats["p50"]),
        "p90": _cores_to_millicores(stats["p90"]),
        "p99": _cores_to_millicores(p99),
        "recommended_request": _cores_to_millicores(p99 * buf),
        "recommended_limit":   _cores_to_millicores(p99 * lim),
        # 평시 권고 (KEDA minReplica 기준)
        "p99_normal":                  _cores_to_millicores(p99_normal),
        "recommended_request_normal":  _cores_to_millicores(p99_normal * buf),
        "recommended_limit_normal":    _cores_to_millicores(p99_normal * lim),
        "used_segment":  stats.get("used_segment", "all"),
        "match_samples": stats.get("match_sample_count", 0),
    }


def _mem_rec(stats, buf, lim):
    # type: (Dict[str, Any], float, float) -> Dict[str, str]
    p99 = stats["p99"]
    return {
        "p50": _bytes_to_mib(stats["p50"]),
        "p90": _bytes_to_mib(stats["p90"]),
        "p99": _bytes_to_mib(p99),
        "recommended_request": _bytes_to_mib(p99 * buf),
        "recommended_limit":   _bytes_to_mib(p99 * lim),
        "used_segment":  stats.get("used_segment", "all"),
        "match_samples": stats.get("match_sample_count", 0),
    }


def generate_recommendation(cpu_stats, mem_stats, pod_name,
                             cpu_buf, cpu_lim, mem_buf, mem_lim):
    # type: (Dict, Dict, str, float, float, float, float) -> Dict[str, Any]
    return {
        "pod_name": pod_name,
        "cpu":    _cpu_rec(cpu_stats, cpu_buf, cpu_lim),
        "memory": _mem_rec(mem_stats, mem_buf, mem_lim),
        "note":   "자동 반영 아님 — 운영자 검토 후 수동 적용하세요.",
    }


def generate_node_recommendation(node_cpu_stats, node_buf=1.3):
    # type: (Dict[str, Any], float) -> Dict[str, Any]
    p95_cores = node_cpu_stats.get("p95", 0.0)
    rec_cores = p95_cores * node_buf

    if rec_cores <= 2:
        instance = "t3.medium (2 vCPU)"
    elif rec_cores <= 4:
        instance = "t3.xlarge (4 vCPU)"
    elif rec_cores <= 8:
        instance = "t3.2xlarge (8 vCPU)"
    else:
        instance = "m5.2xlarge 이상 검토 필요"

    return {
        "target":               "spot-worker 노드그룹",
        "p50":                  _cores_to_millicores(node_cpu_stats.get("p50", 0)),
        "p90":                  _cores_to_millicores(node_cpu_stats.get("p90", 0)),
        "p95":                  _cores_to_millicores(p95_cores),
        "recommended_cores":    round(rec_cores, 2),
        "recommended_instance": instance,
        "used_segment":         node_cpu_stats.get("used_segment", "all"),
        "note": "자동 반영 아님 — Karpenter NodePool instanceType을 수동 조정하세요.",
    }


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_training_data(source, data_path, query, prometheus_url, lookback_days):
    # type: (str, Path, str, str, int) -> Dict[str, Any]
    if source == "file":
        with open(str(data_path), encoding="utf-8") as f:
            return json.load(f)

    from common.prometheus_client import default_train_window, fetch_training_payload
    start, end = default_train_window(lookback_days=lookback_days)
    return fetch_training_payload(
        prometheus_url=prometheus_url, query=query,
        start=start, end=end, step="5m",
    )


# ── Pod 분석 헬퍼 ─────────────────────────────────────────────────────────────

def _analyze_pod(label, namespace, container, source, data_path,
                 prometheus_url, lookback_days, cpu_buf, cpu_lim, mem_buf, mem_lim):
    # type: (str, str, str, str, Path, str, int, float, float, float, float) -> Dict[str, Any]
    """단일 Pod CPU·Memory를 분석하고 권고값을 반환한다."""
    if not container:
        return {"skipped": "컨테이너 미탐지"}

    cpu_q = (
        'rate(container_cpu_usage_seconds_total'
        '{{namespace="{ns}",container="{c}"}}[5m])'
    ).format(ns=namespace, c=container)
    mem_q = (
        'container_memory_working_set_bytes'
        '{{namespace="{ns}",container="{c}"}}'
    ).format(ns=namespace, c=container)

    try:
        cpu_df    = preprocess(load_prometheus_payload(
            load_training_data(source, data_path, cpu_q, prometheus_url, lookback_days)))
        cpu_stats = analyze_quantiles(cpu_df)

        mem_df    = preprocess(load_prometheus_payload(
            load_training_data(source, data_path, mem_q, prometheus_url, lookback_days)))
        mem_stats = analyze_quantiles(mem_df)
    except Exception as e:
        print("        [{0}] 분석 실패: {1}".format(label, e))
        return {"skipped": str(e)}

    print("        [{0}] CPU P99={1:.4f} [{2}구간]  MEM P99={3}Mi".format(
        label, cpu_stats["p99"], cpu_stats["used_segment"],
        int(mem_stats["p99"] // 1024 // 1024)))

    return generate_recommendation(
        cpu_stats, mem_stats, pod_name=label,
        cpu_buf=cpu_buf, cpu_lim=cpu_lim,
        mem_buf=mem_buf, mem_lim=mem_lim,
    )


# ── 메인 파이프라인 ────────────────────────────────────────────────────────────

def main():
    """QR 권고 파이프라인 진입점. CronJob: python -m quantile_regression.recommend"""

    prometheus_url    = os.environ.get(
        "PROMETHEUS_URL",
        "http://kube-prometheus-stack-prometheus.monitoring.svc:9090",
    )
    slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    output_dir        = Path(os.environ.get("QR_OUTPUT_DIR",
                             str(Path(__file__).parent / "output")))
    lookback_days     = int(os.environ.get("QR_LOOKBACK_DAYS",
                            os.environ.get("ANALYSIS_WINDOW_DAYS", "7")))
    dry_run_env       = os.environ.get("QR_DRY_RUN", "").lower() in {"1", "true", "yes"}

    cpu_buf  = float(os.environ.get("QR_CPU_BUFFER_FACTOR",  "1.2"))
    cpu_lim  = float(os.environ.get("QR_CPU_LIMIT_FACTOR",   "2.0"))
    mem_buf  = float(os.environ.get("QR_MEM_BUFFER_FACTOR",  "1.2"))
    mem_lim  = float(os.environ.get("QR_MEM_LIMIT_FACTOR",   "2.0"))
    node_buf = float(os.environ.get("QR_NODE_BUFFER_FACTOR", "1.3"))

    node_cpu_query = os.environ.get(
        "QR_NODE_CPU_QUERY",
        'sum(rate(node_cpu_seconds_total{mode!="idle"}[5m])) by (node)',
    )

    parser = argparse.ArgumentParser(description="Quantile Regression Pod Sizing Report")
    parser.add_argument("--source", choices=["prometheus", "file"], default="prometheus")
    parser.add_argument("--data",      type=Path,
        default=Path(__file__).parent / "test_cpu.json")
    parser.add_argument("--node-data", type=Path,
        default=Path(__file__).parent / "test_node_cpu.json")
    parser.add_argument("--dry-run", action="store_true", default=dry_run_env)
    args = parser.parse_args()

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # ── 0. 컨테이너 이름 자동 탐지 ────────────────────────────────────────────
    print("[0/4] 컨테이너 이름 자동 탐지 중...")
    container_map = discover_all_containers(prometheus_url, ANALYSIS_TARGETS, args.source)

    # ── 1~3. 노드그룹별 Pod 분석 ───────────────────────────────────────────────
    groups = [
        ("[1/4] Worker 파드", ["backend", "sync-matches"]),
        ("[2/4] System 파드", ["prometheus", "argocd", "karpenter"]),
        ("[3/4] AI 파드",     ["np-predict", "np-train"]),
    ]

    target_map = {label: (ns, cands)
                  for label, ns, cands in ANALYSIS_TARGETS}

    for group_label, labels in groups:
        print("{0} 분석 중... (lookback={1}일)".format(group_label, lookback_days))
        for label in labels:
            ns, _ = target_map[label]
            container = container_map.get(label, "")
            rec = _analyze_pod(
                label, ns, container, args.source, args.data,
                prometheus_url, lookback_days,
                cpu_buf, cpu_lim, mem_buf, mem_lim,
            )
            results[label] = rec

    # ── 4. Spot Worker 노드 CPU 분석 ──────────────────────────────────────────
    print("[4/4] Spot Worker 노드 CPU 분석 중...")
    try:
        node_payload   = load_training_data(
            args.source, args.node_data, node_cpu_query,
            prometheus_url, lookback_days)
        node_df        = preprocess(load_prometheus_payload(node_payload))
        node_cpu_stats = analyze_quantiles(node_df, NODE_QUANTILES)
        node_rec       = generate_node_recommendation(node_cpu_stats, node_buf)
        results["spot-node"] = node_rec
        print("      [spot-node] P95={0}  권고: {1}".format(
            node_rec["p95"], node_rec["recommended_instance"]))
    except Exception as e:
        print("      [spot-node] 분석 실패: {0}".format(e))
        results["spot-node"] = {"skipped": str(e)}

    # ── 결과 JSON 저장 ─────────────────────────────────────────────────────────
    output = {"lookback_days": lookback_days, "recommendations": results}
    output_path = output_dir / "qr_report.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print("      결과 저장: {0}".format(output_path))

    # ── Slack 전송 (실패해도 Job 성공 처리) ───────────────────────────────────
    print("[Slack] 리포트 전송 중...")
    from quantile_regression.report import send_slack_report
    try:
        send_result = send_slack_report(
            slack_webhook_url=slack_webhook_url,
            recommendation=results,
            lookback_days=lookback_days,
            dry_run=args.dry_run,
        )
        status = "전송 완료" if send_result.get("sent") else "dry-run"
        print("      Slack {0}".format(status))
    except Exception as e:
        print("      [WARNING] Slack 전송 실패 (분석은 성공 처리): {0}".format(e))

    print("[Done]")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()