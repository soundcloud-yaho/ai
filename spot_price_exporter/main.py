import logging
import os
import sys

import boto3
import httpx
from kubernetes import client, config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spot_price_exporter")

PUSHGATEWAY_URL = os.environ.get(
    "PUSHGATEWAY_URL", "http://pushgateway-prometheus-pushgateway.monitoring.svc:9091"
)
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
CAPACITY_TYPE_LABEL = "karpenter.sh/capacity-type"
INSTANCE_TYPE_LABEL = "node.kubernetes.io/instance-type"
ZONE_LABEL = "topology.kubernetes.io/zone"
JOB_NAME = "spot-price-exporter"


def get_running_spot_nodes() -> list[dict]:
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    nodes = v1.list_node(label_selector=f"{CAPACITY_TYPE_LABEL}=spot")

    parsed = []
    for node in nodes.items:
        labels = node.metadata.labels or {}
        instance_type = labels.get(INSTANCE_TYPE_LABEL)
        zone = labels.get(ZONE_LABEL)
        node_name = node.metadata.name

        if not instance_type or not zone:
            logger.warning("노드 %s에 instance-type 또는 zone 라벨이 없어 건너뜀", node_name)
            continue

        parsed.append({"node_name": node_name, "instance_type": instance_type, "zone": zone})
    return parsed


def fetch_spot_price(ec2_client, instance_type: str, zone: str):
    try:
        resp = ec2_client.describe_spot_price_history(
            InstanceTypes=[instance_type],
            AvailabilityZone=zone,
            ProductDescriptions=["Linux/UNIX"],
            MaxResults=1,
        )
    except Exception:
        logger.exception("스팟 가격 조회 실패: instance_type=%s zone=%s", instance_type, zone)
        return None

    history = resp.get("SpotPriceHistory", [])
    if not history:
        logger.warning("스팟 가격 이력 없음: instance_type=%s zone=%s", instance_type, zone)
        return None

    return float(history[0]["SpotPrice"])


def push_to_pushgateway(records: list) -> None:
    if not records:
        logger.warning("push할 레코드가 없음")
        return

    lines = [
        "# HELP worldcup_spot_realtime_price_usd Realtime AWS spot price (USD/hr) "
        "fetched via DescribeSpotPriceHistory, bypassing Kubecost's delayed spot data feed",
        "# TYPE worldcup_spot_realtime_price_usd gauge",
    ]
    for r in records:
        lines.append(
            'worldcup_spot_realtime_price_usd{node="%s",instance_type="%s",zone="%s"} %s'
            % (r["node_name"], r["instance_type"], r["zone"], r["price"])
        )
    body = "\n".join(lines) + "\n"

    url = f"{PUSHGATEWAY_URL}/metrics/job/{JOB_NAME}"
    resp = httpx.post(url, content=body.encode("utf-8"), timeout=10)
    resp.raise_for_status()
    logger.info("Pushgateway로 %d개 레코드 전송 완료", len(records))


def main() -> int:
    nodes = get_running_spot_nodes()
    if not nodes:
        logger.info("현재 실행 중인 스팟 노드가 없음. 종료.")
        return 0

    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    records = []
    for node in nodes:
        price = fetch_spot_price(ec2, node["instance_type"], node["zone"])
        if price is None:
            continue
        records.append({**node, "price": price})
        logger.info(
            "%s (%s, %s) -> $%.4f/hr", node["node_name"], node["instance_type"], node["zone"], price
        )

    push_to_pushgateway(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())