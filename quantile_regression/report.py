"""QR 권고 결과 → Slack 전송 유틸.

recommend.py가 산출한 Pod 리소스 권고값을 Slack Incoming Webhook으로 전송한다.
자동 반영하지 않는다 — 운영자가 Slack 메시지를 보고 수동으로 deployment.yaml에 적용한다.
"""



import json  # Slack payload JSON 직렬화
from datetime import datetime, timezone  # 리포트 발행 시각
from typing import Any, Dict 
from urllib.error import HTTPError, URLError  # HTTP/네트워크 오류 처리
from urllib.request import Request, urlopen  # 표준 라이브러리 HTTP POST


def _now_kst() -> str:
    """현재 시각을 KST ISO8601 문자열로 반환한다."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M KST")

def build_slack_message(recommendation: Dict[str, Any]) -> Dict[str, Any]:
    """recommendation dict를 Slack Block Kit 메시지로 변환한다."""
    pod   = recommendation["pod_name"]
    cpu   = recommendation["cpu"]
    mem   = recommendation["memory"]
    note  = recommendation["note"]

    text = (
        f"*[QR 권고 리포트] `{pod}`* — {_now_kst()}\n\n"
        f"*CPU 분포*\n"
        f"  P50: `{cpu['p50']}`  P90: `{cpu['p90']}`  P99: `{cpu['p99']}`\n"
        f"*CPU 권고*\n"
        f"  Request: `{cpu['recommended_request']}`  "
        f"Limit: `{cpu['recommended_limit']}`\n\n"
        f"*Memory 분포*\n"
        f"  P50: `{mem['p50']}`  P90: `{mem['p90']}`  P99: `{mem['p99']}`\n"
        f"*Memory 권고*\n"
        f"  Request: `{mem['recommended_request']}`  "
        f"Limit: `{mem['recommended_limit']}`\n\n"
        f"> ⚠️  {note}"
    )

    return {  # Slack API payload
        "text": text,  # 알림 미리보기 텍스트(폴백)
        "blocks": [  # Block Kit 본문
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": text},
            }
        ],
    }


def send_slack_report(
    slack_webhook_url: str,
    recommendation: Dict[str, Any],
    *,
    timeout: int = 30,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Slack Incoming Webhook으로 권고 리포트를 전송한다."""
    payload = build_slack_message(recommendation)  # Block Kit 메시지 생성
    body    = json.dumps(payload).encode("utf-8")  # JSON bytes 직렬화

    result = {  # 전송 결과·디버그 정보
        "webhook_url": slack_webhook_url,
        "payload":     payload,
        "sent":        False,  # 전송 성공 여부 초기값
    }

    if dry_run:  # 로컬 테스트: HTTP 전송 없이 payload만 반환
        print("[dry-run] Slack payload:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return result

    request = Request(  # POST 요청 객체
        slack_webhook_url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as resp:  # HTTP POST 실행
            result["sent"]        = 200 <= resp.status < 300  # 2xx면 성공
            result["status_code"] = resp.status
    except HTTPError as exc:  # 4xx/5xx 응답
        body_err = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Slack webhook HTTP {exc.code}: {body_err}") from exc
    except URLError as exc:  # 연결 실패 등
        raise RuntimeError(f"Slack webhook request failed: {exc}") from exc

    return result  # 전송 결과 dict