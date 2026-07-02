# worldcup-ai

Sound_Cloud 프로젝트의 AI 파이프라인 레포. 트래픽 예측, 파드 사이즈 권고, 비용 분석 리포트, 부하 테스트 시나리오를 포함한다.

이미지는 **하나로 빌드**하고, 각 CronJob이 실행 커맨드만 다르게 지정해 공유한다. 학습/추론/권고/리포트마다 이미지를 따로 만들지 않는다 (관리 부담 최소화).

---

## 아키텍처 상 위치 — 예측·스케일링 루프

```
Prometheus(지표 수집)
      │
      ▼ (5분마다 과거 시계열 조회)
NeuralProphet 추론 CronJob
      │
      ▼ (예측값 push)
Pushgateway
      │
      ▼ (Prometheus가 scrape)
KEDA ScaledObject (예측 메트릭 + 실측 CPU/RPS 이중 트리거)
      │
      ▼ (Pod Pending 발생 시)
Karpenter → Spot Worker 노드 프로비저닝
```

- **추론(5분)과 학습(일 1회)은 다른 잡이다.** 추론은 이미 학습된 모델로 예측만 하고, 재학습을 별도로 돌리지 않으면 모델은 최초 학습 시점에 고정된다.
- KEDA가 예측 메트릭만 보면 모델이 틀렸을 때 실제 트래픽 폭증에 대응하지 못한다. 그래서 실측 CPU/RPS를 보험 트리거로 항상 병행한다.
- 이 레포의 모든 CronJob은 **AI/MLOps 노드그룹(On-Demand)** 에서만 실행된다 — 학습 도중 Spot이 회수되어 연산이 날아가는 것을 방지하기 위해서다. 노드에는 taint가 걸려 있어 이 레포의 워크로드만 `toleration`으로 진입 가능하다.

---

## 모듈 구성

| 경로 | 역할 | 실행 주기 |
|---|---|---|
| `neuralprophet/train.py` | Prometheus 시계열 조회 → NeuralProphet 재학습 → 모델 저장 | 일 1회 |
| `neuralprophet/predict.py` | 저장된 모델 로드 → 다음 구간 트래픽 예측 → Pushgateway push | 5분 |
| `quantile_regression/recommend.py` | P99 자원 분포 분석 → 파드 사이즈 권고 리포트 생성 (Grafana/Slack 발송, **자동 반영 아님**) | 일 1회 |
| `isolation_forest/detect.py` | (확장) 봇/매크로 이상 트래픽 탐지 → 학습 데이터 정제 룰 | 여유 시 |
| `report/bedrock_report.py` | Kubecost API 지표 수집 → Bedrock 호출 → Slack 비용 분석 리포트 발송 | 주 1회 |
| `common/prometheus_client.py` | Prometheus 조회 공용 클라이언트 — 쿼리/기간 파라미터화 | - |
| `common/pushgateway.py` | Pushgateway push 유틸 — **메트릭 이름/라벨 규약의 단일 소스.** 이 파일을 바꾸면 `worldcup-infra`의 KEDA ScaledObject 트리거 쿼리도 함께 수정해야 한다 | - |
| `loadtest/locustfile.py` | 경기 시작 시점 트래픽 급증 패턴 시나리오 — **AI ON/OFF 비교 실험의 기준 코드.** 임의 수정 시 두 실험 간 비교가 무효화된다 | 수동 실행 |

---

## 배포 흐름

1. `master` 브랜치에 push
2. CI(`.github/workflows/build-push.yaml`)가 Docker 이미지 빌드 → ECR에 `ai:<git-sha-7자리>` 태그로 push
3. 실제 배포는 `worldcup-infra`의 `k8s/manifests/ai/*.yaml`에서 이미지 태그를 갱신해 커밋 → ArgoCD가 반영
4. 각 CronJob의 `command` 필드로 어떤 스크립트를 실행할지 지정 (예: `["python", "neuralprophet/predict.py"]`)

---

## 측정 지표 (발표용 핵심 산출물)

- NeuralProphet 예측 정확도 (MAE / MAPE)
- 선제 스케일링 리드타임 (예측 → 실제 트래픽 도달까지 시간차)
- AI 예측 ON vs OFF 비교 — 동일 `loadtest/locustfile.py` 시나리오로 2회 실행한 결과
- Spot 절감액: (On-Demand 단가 − Spot 단가) × 사용 시간

## 로컬 실행 / 테스트

```bash
pip install -r requirements.txt
python neuralprophet/train.py --local-mock   # 더미 시계열로 로컬 검증
```
