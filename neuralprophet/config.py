"""NeuralProphet 모델·스케일링 상수 및 경로 설정."""

import os
from pathlib import Path

from neuralprophet import NeuralProphet, load

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "test_dataset.json"
OUTPUT_DIR = Path(os.environ.get("NP_OUTPUT_DIR", BASE_DIR / "output"))
MODEL_PATH = OUTPUT_DIR / "worldcup_model.np"
METRICS_PATH = OUTPUT_DIR / "train_metrics.json"

PROMETHEUS_URL = os.environ.get(
    "PROMETHEUS_URL",
    "http://kube-prometheus-stack-prometheus.monitoring.svc:9090",
)
PROMETHEUS_QUERY = os.environ.get(
    "PROMETHEUS_QUERY",
    'sum(rate(http_requests_total{namespace="app",service="backend"}[5m]))',
)
TRAIN_TIMEZONE = os.environ.get("TRAIN_TIMEZONE", "Asia/Seoul")
TRAIN_LOOKBACK_DAYS = int(os.environ.get("TRAIN_LOOKBACK_DAYS", "30"))
TRAIN_DATA_START = os.environ.get("TRAIN_DATA_START")  # 예: 2026-06-01T00:00:00+09:00

PUSHGATEWAY_URL = os.environ.get(
    "PUSHGATEWAY_URL",
    "http://pushgateway-prometheus-pushgateway.monitoring.svc:9091",
)
PUSHGATEWAY_JOB = os.environ.get("PUSHGATEWAY_JOB", "neuralprophet-predict")
PUSHGATEWAY_INSTANCE = os.environ.get("PUSHGATEWAY_INSTANCE", "neuralprophet")
PUSHGATEWAY_DRY_RUN = os.environ.get("PUSHGATEWAY_DRY_RUN", "").lower() in {"1", "true", "yes"}
PREDICT_HISTORY_HOURS = float(os.environ.get("PREDICT_HISTORY_HOURS", "4"))

PREDICTION_PATH = OUTPUT_DIR / "predictions.json"
SCALE_SIGNAL_PATH = OUTPUT_DIR / "scale_signals.json"

STEP_MINUTES = 5
N_FORECASTS = 12
N_LAGS = 36
FORECAST_MINUTES = N_FORECASTS * STEP_MINUTES

# KEDA / Karpenter 스케일 변환 (Pushgateway 메트릭 규약 초안)
RPS_PER_POD = 500
RPS_PER_CPU_CORE = 200
CPU_CORES_PER_NODE = 8
HEADROOM_FACTOR = 1.2
BASELINE_PODS = 2
BASELINE_NODES = 1

MATCH_EVENTS = ["match_start", "half_time", "second_half", "match_end"]
EVENT_OFFSETS_MINUTES = {
    "match_start": 0,
    "half_time": 45,
    "second_half": 60,
    "match_end": 105,
}


def load_model(path: Path):
    """PyTorch 2.6+ weights_only 기본값 이슈를 우회해 모델을 로드한다."""
    import torch

    try:
        model = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        model = load(str(path), map_location="cpu")
        return model

    model.restore_trainer(accelerator="cpu")
    return model


def build_model() -> NeuralProphet:
    """README에 정의된 옵션으로 NeuralProphet 모델을 생성한다."""
    model = NeuralProphet(
        growth="linear",
        changepoints=None,
        n_changepoints=20,
        changepoints_range=0.9,
        trend_reg=1.0,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="multiplicative",
        seasonality_reg=0.5,
        n_forecasts=N_FORECASTS,
        n_lags=N_LAGS,
        ar_layers=[64, 32],
        ar_reg=0.01,
        learning_rate=0.001,
        epochs=150,
        batch_size=256,
        optimizer="AdamW",
        loss_func="SmoothL1Loss",
        normalize="standardize",
        impute_missing=True,
        drop_missing=False,
        newer_samples_weight=2,
        newer_samples_start=0.8,
        collect_metrics=True,
    )
    return model
