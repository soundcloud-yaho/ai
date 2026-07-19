# [Docker] python-slim + neuralprophet/scikit-learn — 무거우니 의존성 캐시 레이어 순서 중요

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 소스보다 의존성을 먼저 복사해 Docker 레이어 캐시 활용
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup appuser

COPY --chown=appuser:appgroup common/ ./common/
COPY --chown=appuser:appgroup neuralprophet/ ./neuralprophet/
COPY --chown=appuser:appgroup quantile_regression/ ./quantile_regression/
COPY --chown=appuser:appgroup isolation_forest/ ./isolation_forest/
COPY --chown=appuser:appgroup report/ ./report/
COPY --chown=appuser:appgroup loadtest/ ./loadtest/

USER appuser

CMD ["python", "neuralprophet/predict.py"]
