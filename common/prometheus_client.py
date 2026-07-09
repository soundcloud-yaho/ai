"""Prometheus HTTP API client — query_range 조회.

운영(CronJob)에서는 외부 URL이 아니라 Kubernetes 클러스터 내부 DNS를 사용한다.
기본값은 config.PROMETHEUS_URL:

    http://kube-prometheus-stack-prometheus.monitoring.svc:9090

형식은 ``<service>.<namespace>.svc:<port>`` 이며, kube-prometheus-stack Helm 차트가
만드는 Service 이름·네임스페이스(monitoring)로부터 프로비저닝 전에 미리 정할 수 있다.
CronJob Pod는 클러스터 안에서 실행되므로 LoadBalancer/Ingress 없이 이 주소로 바로 접근한다.

로컬 개발(클러스터 밖)에서는 내부 DNS가 해석되지 않으므로 train.py --source file 을 사용한다.
"""

from __future__ import annotations  # 타입 힌트에서 아직 정의되지 않은 클래스 이름 참조 허용

import json  # Prometheus HTTP 응답 JSON 파싱
from datetime import datetime, timedelta, timezone  # 시계열 구간(start/end) 계산
from typing import Any  # dict 등 유연한 타입 표기
from urllib.error import HTTPError, URLError  # HTTP/네트워크 오류 처리
from urllib.parse import urlencode  # query string 인코딩
from urllib.request import urlopen  # 표준 라이브러리 HTTP GET
from zoneinfo import ZoneInfo  # IANA 타임존(Asia/Seoul 등) 처리


def _to_utc(dt: datetime) -> datetime:
    """aware datetime을 UTC로 변환한다."""
    if dt.tzinfo is None:  # naive datetime은 허용하지 않음(구간 해석 오류 방지)
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(timezone.utc)  # UTC 기준 시각으로 변환 후 반환


def query_range(
    prometheus_url: str,
    query: str,
    start: datetime,
    end: datetime,
    step: str = "5m",
    timeout: int = 120,
) -> dict[str, Any]:
    """Prometheus /api/v1/query_range 응답을 반환한다."""
    base = prometheus_url.rstrip("/")  # URL 끝 슬래시 제거
    params = urlencode(  # GET 파라미터를 URL-safe 문자열로 변환
        {
            "query": query,  # PromQL (예: sum(rate(...[5m])))
            "start": _to_utc(start).timestamp(),  # 구간 시작 Unix timestamp
            "end": _to_utc(end).timestamp(),  # 구간 종료 Unix timestamp
            "step": step,  # 샘플 간격(5m = 5분)
        }
    )
    url = f"{base}/api/v1/query_range?{params}"  # 최종 요청 URL 조립

    try:
        with urlopen(url, timeout=timeout) as resp:  # HTTP GET 실행
            payload = json.loads(resp.read().decode("utf-8"))  # 응답 body JSON 파싱
    except HTTPError as exc:  # 4xx/5xx 응답
        body = exc.read().decode("utf-8", errors="replace")  # 에러 본문 읽기
        raise RuntimeError(f"Prometheus HTTP {exc.code}: {body}") from exc  # 호출자에게 전달
    except URLError as exc:  # DNS 실패, 연결 거부 등
        raise RuntimeError(f"Prometheus request failed: {exc}") from exc

    if payload.get("status") != "success":  # Prometheus API 실패 응답 검사
        raise RuntimeError(f"Prometheus query failed: {payload}")

    return payload  # matrix 형식 시계열 응답 반환


def default_train_window(
    *,
    timezone_name: str = "Asia/Seoul",
    lookback_days: int = 30,
    train_start: datetime | None = None,
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    """일 1회 학습용 기본 구간(start, end)을 계산한다."""
    tz = ZoneInfo(timezone_name)  # 학습 구간 기준 타임존 객체
    reference = now.astimezone(tz) if now else datetime.now(tz)  # 기준 시각(기본: 지금)
    end = reference.replace(hour=0, minute=0, second=0, microsecond=0)  # 오늘 00:00 = 어제까지 포함
    start = train_start if train_start else end - timedelta(days=lookback_days)  # 고정 시작일 또는 lookback
    return start, end  # (학습 시작, 학습 종료) 튜플


def fetch_training_payload(
    prometheus_url: str,
    query: str,
    start: datetime,
    end: datetime,
    step: str = "5m",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """학습용 Prometheus matrix 응답에 meta 정보를 붙여 반환한다."""
    payload = query_range(prometheus_url, query, start, end, step=step)  # 실제 시계열 조회
    if meta:  # 호출자가 meta를 직접 넘긴 경우
        payload["meta"] = meta  # 그대로 첨부
    else:  # 기본 meta 자동 생성
        payload["meta"] = {
            "query": query,  # 사용한 PromQL
            "step": step,  # 샘플 간격
            "start": _to_utc(start).isoformat(),  # 조회 시작 시각(ISO8601)
            "end": _to_utc(end).isoformat(),  # 조회 종료 시각(ISO8601)
        }
    return payload  # preprocess.load_prometheus_payload() 입력 형식


def default_inference_window(
    *,
    history_hours: float = 4.0,
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    """5분 추론용 최근 이력 구간을 계산한다 (n_lags=36 + 여유)."""
    end = now if now else datetime.now(timezone.utc)  # 추론 기준 시각(기본: 현재 UTC)
    if end.tzinfo is None:  # naive datetime이 들어오면 UTC로 간주
        end = end.replace(tzinfo=timezone.utc)
    start = end - timedelta(hours=history_hours)  # 최근 N시간 전부터 조회
    return start, end  # (이력 시작, 추론 시각) 튜플


def fetch_inference_payload(
    prometheus_url: str,
    query: str,
    history_hours: float = 4.0,
    step: str = "5m",
    now: datetime | None = None,
) -> dict[str, Any]:
    """추론용 Prometheus matrix 응답을 반환한다."""
    start, end = default_inference_window(history_hours=history_hours, now=now)  # 최근 구간 계산
    return fetch_training_payload(  # 학습과 동일 API, 구간만 짧음
        prometheus_url=prometheus_url,
        query=query,
        start=start,
        end=end,
        step=step,
        meta={
            "query": query,  # PromQL
            "step": step,  # 5분 간격
            "start": _to_utc(start).isoformat(),  # 이력 시작
            "end": _to_utc(end).isoformat(),  # 추론 시각
            "purpose": "inference",  # 용도 구분용 태그
        },
    )
