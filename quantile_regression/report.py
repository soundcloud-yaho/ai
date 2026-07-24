"""QR 권고 결과 → Slack 전송 유틸.

recommend.py가 산출한 전체 Pod/노드 권고 dict를
Slack Incoming Webhook으로 전송한다.
자동 반영하지 않는다 — 운영자가 Slack 메시지를 보고 수동으로 적용한다.
"""

import json
from datetime import datetime
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _now_kst():
    # type: () -> str
    """현재 시각을 KST 문자열로 반환한다."""
    try:
        from zoneinfo import ZoneInfo           # Python 3.9+
    except ImportError:
        from backports.zoneinfo import ZoneInfo  # Python 3.8 폴백
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M KST")


def _pod_section(label, rec):
    # type: (str, Dict[str, Any]) -> str
    """단일 Pod 권고 섹션 문자열을 반환한다.

    rec 구조:
      {"cpu": {...}, "memory": {...}, "note": "..."}
      또는 {"skipped": "사유"}
    """
    # 컨테이너 미탐지 또는 분석 실패
    if rec.get("skipped"):
        return "*[{0}]* — {1}\n".format(label, rec["skipped"])

    cpu = rec.get("cpu", {})
    mem = rec.get("memory", {})
    seg = cpu.get("used_segment", "all")
    smp = cpu.get("match_samples", 0)

    return (
        "*[{label}]* `{seg}구간 / {smp}샘플`\n"
        "  *경기 시간대* CPU P99: `{cp99}` → Request: `{creq}`  Limit: `{clim}`\n"
        "  *평시*        CPU P99: `{cp99n}` → Request: `{creqn}`  Limit: `{climn}`\n"
        "  *경기 시간대* MEM P99: `{mp99}` → Request: `{mreq}`  Limit: `{mlim}`\n"
        "  *평시*        MEM P99: `{mp99n}` → Request: `{mreqn}`  Limit: `{mlimn}`\n"
        "  *KEDA* 평시 minReplica 기준: CPU `{creqn}` 이하 설정 권장\n"
    ).format(
        label=label, seg=seg, smp=smp,
        cp99=cpu.get("p99", "-"),
        creq=cpu.get("recommended_request", "-"),
        clim=cpu.get("recommended_limit", "-"),
        cp99n=cpu.get("p99_normal", "-"),
        creqn=cpu.get("recommended_request_normal", "-"),
        climn=cpu.get("recommended_limit_normal", "-"),
        mp99=mem.get("p99", "-"),
        mreq=mem.get("recommended_request", "-"),
        mlim=mem.get("recommended_limit", "-"),
        mp99n=mem.get("p99_normal", "-"),
        mreqn=mem.get("recommended_request_normal", "-"),
        mlimn=mem.get("recommended_limit_normal", "-"),
    )


def build_slack_message(recommendations, lookback_days=7):
    # type: (Dict[str, Any], int) -> Dict[str, Any]
    """recommendations dict를 Slack Block Kit 메시지로 변환한다.

    recommendations 구조 (recommend.py의 results dict):
      {
        "backend":       {"cpu": {...}, "memory": {...}},
        "sync-matches":  {"cpu": {...}, "memory": {...}},
        "prometheus":    {"cpu": {...}, "memory": {...}},
        "argocd":        {"cpu": {...}, "memory": {...}},
        "karpenter":     {"cpu": {...}, "memory": {...}},
        "np-predict":    {"cpu": {...}, "memory": {...}},
        "np-train":      {"cpu": {...}, "memory": {...}},
        "spot-node":     {"p95": "...", "recommended_instance": "..."},
      }
    """
    text = "*[QR 권고 리포트]* — {now}  (분석 기간: {days}일)\n\n".format(
        now=_now_kst(), days=lookback_days,
    )

    # ── Worker 노드 파드 ──────────────────────────────────────────────────────
    text += "*── Worker 노드 ──*\n"
    for label in ("backend", "sync-matches"):
        if label in recommendations:
            text += _pod_section(label, recommendations[label])

    # ── System 노드 파드 ──────────────────────────────────────────────────────
    text += "\n*── System 노드 ──*\n"
    for label in ("prometheus", "argocd", "karpenter"):
        if label in recommendations:
            text += _pod_section(label, recommendations[label])

    # ── AI 노드 파드 ──────────────────────────────────────────────────────────
    text += "\n*── AI 노드 ──*\n"
    for label in ("np-predict", "np-train"):
        if label in recommendations:
            text += _pod_section(label, recommendations[label])

    # ── Spot Worker 노드 ──────────────────────────────────────────────────────
    node = recommendations.get("spot-node", {})
    if node and not node.get("skipped"):
        text += (
            "\n*── Spot Worker 노드 ──*\n"
            "  CPU P95: `{p95}` → 권고 인스턴스: `{inst}` ({cores} vCPU)\n"
        ).format(
            p95=node.get("p95", "-"),
            inst=node.get("recommended_instance", "-"),
            cores=node.get("recommended_cores", "-"),
        )
    elif node.get("skipped"):
        text += "\n*── Spot Worker 노드 ──*\n  {0}\n".format(node["skipped"])

    text += "\n> ⚠️  자동 반영 아님 — 운영자 검토 후 수동 적용하세요."

    return {
        "text": text,  # 알림 미리보기 폴백
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": text}}
        ],
    }


def send_slack_report(slack_webhook_url, recommendation,
                      lookback_days=7, timeout=30, dry_run=False):
    # type: (str, Dict[str, Any], int, int, bool) -> Dict[str, Any]
    """Slack Incoming Webhook으로 권고 리포트를 전송한다."""
    payload = build_slack_message(recommendation, lookback_days=lookback_days)
    body    = json.dumps(payload).encode("utf-8")

    result = {
        "webhook_url": slack_webhook_url,
        "payload":     payload,
        "sent":        False,
    }

    if dry_run:  # 로컬 테스트: 전송 없이 payload만 출력
        print("[dry-run] Slack payload:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return result

    request = Request(
        slack_webhook_url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as resp:
            result["status_code"] = resp.status
            body_text = resp.read().decode("utf-8", errors="replace")
            result["response_body"] = body_text
            # Slack 정상 응답은 "ok", 실패는 "no_service" 등
            result["sent"] = (200 <= resp.status < 300) and (body_text.strip() == "ok")
    except HTTPError as exc:
        body_err = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "Slack webhook HTTP {0}: {1}".format(exc.code, body_err)) from exc
    except URLError as exc:
        raise RuntimeError(
            "Slack webhook request failed: {0}".format(exc)) from exc

    return result