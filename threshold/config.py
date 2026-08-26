import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("THRESHOLD_OUTPUT_DIR", BASE_DIR / "output"))

PROMETHEUS_URL = os.environ.get(
    "PROMETHEUS_URL",
    "http://kube-prometheus-stack-prometheus.monitoring.svc:9090",
)
PROMETHEUS_QUERY = os.environ.get(
    "PROMETHEUS_QUERY",
    'sum(rate(aws_applicationelb_request_count_sum{...}[1m]))',  # YACE 실제 라벨
)
LOOKBACK_MINUTES = int(os.environ.get("LOOKBACK_MINUTES", "22"))

PUSHGATEWAY_URL = os.environ.get(
    "PUSHGATEWAY_URL",
    "http://pushgateway-prometheus-pushgateway.monitoring.svc:9091",
)
PUSHGATEWAY_JOB = os.environ.get("PUSHGATEWAY_JOB", "rps-cleaner")
PUSHGATEWAY_INSTANCE = os.environ.get("PUSHGATEWAY_INSTANCE", "rps-cleaner")
PUSHGATEWAY_DRY_RUN = os.environ.get("PUSHGATEWAY_DRY_RUN", "").lower() in {"1", "true", "yes"}
METRIC_NAME = os.environ.get("METRIC_NAME", "rps_cleaned")

# 동적 임계치 파라미터
WINDOW_MINUTES = int(os.environ.get("THRESHOLD_WINDOW_MINUTES", "12"))
MIN_PERIODS = int(os.environ.get("THRESHOLD_MIN_PERIODS", "3"))
THRESHOLD_MULTIPLIER = float(os.environ.get("THRESHOLD_MULTIPLIER", "5"))
FALLBACK_THRESHOLD = float(os.environ.get("THRESHOLD_FALLBACK", "500"))
MIN_THRESHOLD = float(os.environ.get("THRESHOLD_MIN", "50"))
LOW_TRAFFIC_GUARD = float(os.environ.get("THRESHOLD_LOW_TRAFFIC_GUARD", "20"))
LOW_TRAFFIC_MULTIPLIER = float(os.environ.get("THRESHOLD_LOW_TRAFFIC_MULTIPLIER", "10"))
CONSECUTIVE_OVERRIDE = int(os.environ.get("THRESHOLD_CONSECUTIVE_OVERRIDE", "3"))
HARD_CEILING = float(os.environ.get("THRESHOLD_HARD_CEILING", "300"))