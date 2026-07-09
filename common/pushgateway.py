"""Pushgateway push 유틸 — 메트릭 이름/라벨 규약의 단일 소스.

변경 시 worldcup-infra KEDA ScaledObject 트리거 쿼리도 함께 수정해야 한다.
"""

from __future__ import annotations  # 타입 힌트 forward reference 허용

from typing import Any  # dict 등 유연한 타입 표기
from urllib.error import HTTPError, URLError  # HTTP/네트워크 오류 처리
from urllib.request import Request, urlopen  # Pushgateway PUT 요청

PUSHGATEWAY_JOB = "neuralprophet-predict"  # Pushgateway job 라벨 기본값

# KEDA / Karpenter가 참조하는 gauge 메트릭 이름 (인프라 ScaledObject와 동기화 필요)
METRIC_PREDICTED_RPS_PEAK = "worldcup_predicted_rps_peak"  # 1시간 내 예측 RPS 피크
METRIC_RECOMMENDED_PODS = "worldcup_recommended_pods"  # KEDA 권장 Pod 수
METRIC_PREDICTED_CPU_CORES = "worldcup_predicted_cpu_cores_required"  # Karpenter 필요 CPU 코어
METRIC_RECOMMENDED_NODES = "worldcup_recommended_nodes"  # Karpenter 권장 Node 수
METRIC_CURRENT_RPS = "worldcup_current_rps"  # 현재 실측 RPS


def build_exposition(scale_signals: dict[str, Any]) -> str:
    """scale_signals dict를 Prometheus text exposition 형식으로 변환한다."""
    keda = scale_signals["keda"]  # KEDA 관련 신호 추출
    karp = scale_signals["karpenter"]  # Karpenter 관련 신호 추출
    lines = [  # Prometheus가 scrape하는 텍스트 포맷 라인 목록
        f"# TYPE {METRIC_PREDICTED_RPS_PEAK} gauge",  # 메트릭 타입 선언
        f'{METRIC_PREDICTED_RPS_PEAK}{{target="keda"}} {scale_signals["predicted_rps_peak_1h"]}',  # 예측 피크 RPS
        f"# TYPE {METRIC_RECOMMENDED_PODS} gauge",
        f'{METRIC_RECOMMENDED_PODS}{{target="keda"}} {keda["recommended_pods"]}',  # 권장 Pod 수
        f"# TYPE {METRIC_PREDICTED_CPU_CORES} gauge",
        f'{METRIC_PREDICTED_CPU_CORES}{{target="karpenter"}} {karp["required_cpu_cores"]}',  # 필요 CPU 코어
        f"# TYPE {METRIC_RECOMMENDED_NODES} gauge",
        f'{METRIC_RECOMMENDED_NODES}{{target="karpenter"}} {karp["recommended_nodes"]}',  # 권장 Node 수
        f"# TYPE {METRIC_CURRENT_RPS} gauge",
        f'{METRIC_CURRENT_RPS}{{target="keda"}} {scale_signals["current_rps"]}',  # 현재 RPS
    ]
    return "\n".join(lines) + "\n"  # 줄바꿈으로 연결해 exposition body 완성


def push_scale_signals(
    pushgateway_url: str,
    scale_signals: dict[str, Any],
    *,
    job: str = PUSHGATEWAY_JOB,
    instance: str = "neuralprophet",
    timeout: int = 30,
    dry_run: bool = False,
) -> dict[str, Any]:
    """예측 스케일 신호를 Pushgateway에 push한다."""
    body = build_exposition(scale_signals)  # Prometheus 텍스트 포맷 생성
    base = pushgateway_url.rstrip("/")  # URL 끝 슬래시 제거
    url = f"{base}/metrics/job/{job}/instance/{instance}"  # Pushgateway grouping URL

    result = {  # push 결과/디버그 정보를 담을 dict
        "pushgateway_url": pushgateway_url,  # 사용한 Pushgateway 주소
        "job": job,  # job 라벨
        "instance": instance,  # instance 라벨
        "url": url,  # 실제 PUT URL
        "body": body,  # 전송한 exposition 본문
        "pushed": False,  # push 성공 여부(초기값 False)
    }

    if dry_run:  # 로컬 테스트: HTTP 전송 없이 body만 반환
        return result

    request = Request(  # PUT 요청 객체 생성
        url,
        data=body.encode("utf-8"),  # 본문을 bytes로 인코딩
        method="PUT",  # Pushgateway는 PUT으로 메트릭 등록
        headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"},  # exposition MIME
    )
    try:
        with urlopen(request, timeout=timeout) as resp:  # HTTP PUT 실행
            result["pushed"] = 200 <= resp.status < 300  # 2xx면 성공
            result["status_code"] = resp.status  # HTTP 상태 코드 기록
    except HTTPError as exc:  # 4xx/5xx 응답
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Pushgateway HTTP {exc.code}: {error_body}") from exc
    except URLError as exc:  # 연결 실패 등
        raise RuntimeError(f"Pushgateway request failed: {exc}") from exc

    return result  # push 결과 dict 반환
