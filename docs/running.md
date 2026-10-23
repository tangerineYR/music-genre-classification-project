# 설치와 실행

Python 3.10 이상. 저장소 루트에서 실행한다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

## Historical dataset 검증

대상은 [AI Hub 음악 유사성 판별 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71544)에서
2025년에 생성한 historical 60-second sampled WAV다. 파일명과 폴더 구조를 유지하고
데이터 디렉터리는 읽기 전용으로 취급한다. 현재 historical dataset에는 `Org`가 없다.

예시 구조:

```text
<dataset>/Hiphop/00873/Cover_Timbre_B/Similar_Hiphop_00873_Cover_Timbre_B.wav
```

Historical WAV에 `sample_audio.py`를 다시 실행하지 않는다. 먼저 소수 파일의 존재·크기·decode를
확인한다. OneDrive 동기화 폴더라면 online-only placeholder를 실제 로컬 사용 가능 상태로 만든다.
전체 inventory는 모든 audio file을 끝까지 decode하여 duration, sample rate, channels, 유한값,
원본 파일 SHA-256, decoded PCM hash와 중복을 기록한다. 결과 경로는 dataset-relative다.

```bash
python scripts/inventory_dataset.py --input "<dataset>" --output artifacts/inventory --summary-output results
```

`dataset_inventory.csv`는 전체 file-level audit, `dataset_inventory.json`은 분포·제외 사유와
complete-case song IDs를 담는다. 교차 song 중복은 fold leakage 위험 때문에 평가 전에 검토해야 한다.
파일의 실제 decode나 local availability 실패는 결과에서 숨기지 않는다.
`--summary-output results`는 장르·source family·variant별 aggregate CSV를 별도 생성한다.
공개 결과에는 이 aggregate를 사용하고 전체 file-level audit은 local artifact로 보관한다.

## Single-source Cover_Timbre baseline vs feature-aware source mapping

Baseline은 `Cover_Timbre` 한 음원에서 모든 96D feature를 추출한다. Feature-aware는
[정의된 mapping](lineage.md#exact-feature-schema)에 따라 `Cover_Timbre`, `Cover_Rhythm`,
`Cover_Instrument`, `Cover_Genre`를 사용한다. 네 family가 모두 있는 complete-case 곡 전체를
추출한다. 누락된 source를 다른 source로 대체하지 않는다.

각 family에서 dataset-relative path 기준 사전순 첫 matching file을 선택한다. Variant에 대한
성능 탐색은 없으며, 선택된 실제 경로·hash를 `source_manifest.json`에 기록한다.
`Cover_Tempo`와 `Cover_Arrangement`는 이 96D 평가에 사용하지 않는다.

```bash
python scripts/build_features.py --input "<dataset>" --output artifacts/features60 --baseline-source Cover_Timbre --seconds 60
python scripts/evaluate.py --baseline artifacts/features60/baseline.npz --feature-aware artifacts/features60/feature_aware.npz --inventory artifacts/inventory/dataset_inventory.json --output results
```

`load_audio()`는 mono 16 kHz로 representation을 통일한다. Historical segment는 유지하며
60초에서 10 ms 이내의 길이 차이는 원래 길이 그대로 허용한다. 짧은 파일을 padding하지 않는다.
Feature 출력은 unscaled 96D NPZ이며 기존 archive/manifest는 덮어쓰지 않는다.
두 조건은 한 곡의 특징이 모두 성공해야 함께 추가된다. Inventory의 complete-case IDs와
feature archive의 IDs가 다르면 평가를 중단한다.

## 평가와 해석

Complete-case 특징 추출 후, 최소 class count에 맞추어 seed 42로 deterministic undersampling한다.
두 조건에 동일 곡·label·balancing 결과·10개 StratifiedKFold(shuffle=True, random_state=42)를
사용한다. GMM은 7 components, diagonal covariance, max_iter=400, random_state=42,
init_params=kmeans다. MinMaxScaler는 매 fold의 training data에만 fit한다.

생성 파일:

- `results/cv_metrics.json`: protocol·버전·hash·class counts·validation IDs·confusion counts·paired 차이
- `results/fold_metrics.csv`: 조건별 fold accuracy, macro F1, convergence
- `results/feature_source_ablation.csv`: 조건별 accuracy/macro F1 mean ± std, nonconverged folds
- `results/confusion_matrix.png`: 두 조건의 confusion matrix

점수는 0–1 단위, 표준편차는 ddof=0이다. Feature-aware에서 baseline을 뺀 차이를 보고한다.
이 비교는 균형화된 historical complete-case cohort에 한정된다. Source별 sampled segment의
정확한 timestamp alignment가 보장되지 않으므로 source type의 causal effect나 AI Hub 전체
benchmark로 해석하지 않는다. 실측 결과와 한계는 [결과](../results/STATUS.md)에 기록한다.

음원·feature matrix·학습 모델은 로컬 artifact로 관리하고 Git에는 평가 결과만 저장한다.

## 테스트

```bash
python -m unittest discover -s tests -v
```

테스트는 sampling, 96D schema, lexical source 선택, historical duration tolerance,
inventory의 decode·중복 검사, sample 정렬, fold별 scaling을 검증한다.
합성 데이터 테스트 통과는 실제 음악 분류 성능을 의미하지 않는다.
