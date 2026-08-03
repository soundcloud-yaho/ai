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
PRICING_API_REGION = "us-east-1"
REGION_LOCATION_NAME = {
    "ap-northeast-2": "Asia Pacific (Seoul)",
}

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
    """실시간 스팟 가격 조회 (기존 로직 그대로)."""
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


def fetch_ondemand_price(pricing_client, instance_type: str) -> float | None:
    """
    AWS Pricing API로 온디맨드 정가(USD/hr, Linux, Shared tenancy 기준) 조회.
    결과가 없으면 None.
    """
    location = REGION_LOCATION_NAME.get(AWS_REGION)
    if location is None:
        logger.error("REGION_LOCATION_NAME에 %s 매핑이 없음", AWS_REGION)
        return None

    try:
        resp = pricing_client.get_products(
            ServiceCode="AmazonEC2",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
                {"Type": "TERM_MATCH", "Field": "location", "Value": location},
                {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
            ],
            MaxResults=1,
        )
    except Exception:
        logger.exception("온디맨드 정가 조회 실패: instance_type=%s", instance_type)
        return None

    price_list = resp.get("PriceList", [])
    if not price_list:
        logger.warning("온디맨드 정가 없음: instance_type=%s", instance_type)
        return None

    import json

    product = json.loads(price_list[0])
    on_demand_terms = product.get("terms", {}).get("OnDemand", {})
    for term in on_demand_terms.values():
        for dim in term.get("priceDimensions", {}).values():
            usd = dim.get("pricePerUnit", {}).get("USD")
            if usd:
                return float(usd)
    return None


def push_to_pushgateway(spot_records: list, ondemand_records: list) -> None:
    if not spot_records and not ondemand_records:
        logger.warning("push할 레코드가 없음")
        return

    lines = [
        "# HELP worldcup_spot_realtime_price_usd Realtime AWS spot price (USD/hr) "
        "fetched via DescribeSpotPriceHistory, bypassing Kubecost's delayed spot data feed",
        "# TYPE worldcup_spot_realtime_price_usd gauge",
    ]
    for r in spot_records:
        lines.append(
            'worldcup_spot_realtime_price_usd{node="%s",instance_type="%s",zone="%s"} %s'
            % (r["node_name"], r["instance_type"], r["zone"], r["price"])
        )

    lines += [
        "# HELP worldcup_ondemand_realtime_price_usd Realtime AWS on-demand price (USD/hr) "
        "fetched via Pricing API, used as the accurate baseline for spot savings comparison",
        "# TYPE worldcup_ondemand_realtime_price_usd gauge",
    ]
    for r in ondemand_records:
        lines.append(
            'worldcup_ondemand_realtime_price_usd{node="%s",instance_type="%s",zone="%s"} %s'
            % (r["node_name"], r["instance_type"], r["zone"], r["price"])
        )

    body = "\n".join(lines) + "\n"
    url = f"{PUSHGATEWAY_URL}/metrics/job/{JOB_NAME}"
    resp = httpx.post(url, content=body.encode("utf-8"), timeout=10)
    resp.raise_for_status()
    logger.info(
        "Pushgateway로 스팟 %d개, 온디맨드 %d개 레코드 전송 완료",
        len(spot_records),
        len(ondemand_records),
    )


def main() -> int:
    nodes = get_running_spot_nodes()
    if not nodes:
        logger.info("현재 실행 중인 스팟 노드가 없음. 종료.")
        return 0

    ec2 = boto3.client("ec2", region_name=AWS_REGION)
    pricing = boto3.client("pricing", region_name=PRICING_API_REGION)

    ondemand_cache: dict[str, float | None] = {}

    spot_records = []
    ondemand_records = []
    for node in nodes:
        spot_price = fetch_spot_price(ec2, node["instance_type"], node["zone"])
        if spot_price is not None:
            spot_records.append({**node, "price": spot_price})
            logger.info(
                "[spot] %s (%s, %s) -> $%.4f/hr",
                node["node_name"], node["instance_type"], node["zone"], spot_price,
            )

        instance_type = node["instance_type"]
        if instance_type not in ondemand_cache:
            ondemand_cache[instance_type] = fetch_ondemand_price(pricing, instance_type)
        ondemand_price = ondemand_cache[instance_type]
        if ondemand_price is not None:
            ondemand_records.append({**node, "price": ondemand_price})
            logger.info(
                "[on-demand] %s (%s, %s) -> $%.4f/hr",
                node["node_name"], node["instance_type"], node["zone"], ondemand_price,
            )

    push_to_pushgateway(spot_records, ondemand_records)
    return 0


if __name__ == "__main__":
    sys.exit(main())