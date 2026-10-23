# 설치와 실행

Python 3.10 이상. 저장소 루트에서 실행한다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 데이터 준비

[AI Hub 음악 유사성 판별 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71544)의
Similar 음원을 사용한다. 파일명과 하위 폴더 구조를 유지한다.
예: `Similar_Trot_00610_Org.wav`, `Similar_Trot_00610_Cover_Timbre_A.wav`.
WAV를 권장한다. 그 외 지원 확장자는 FLAC/MP3/OGG/AIFF/M4A이며 decoder가 필요할 수 있다.

원천 음원은 다음 명령으로 onset 기준 60초, mono 16 kHz PCM16 WAV로 변환한다.
이미 60초로 처리한 음원이 있다면 sampling을 생략하고 해당 폴더를 feature build 입력으로 지정한다.

```bash
python scripts/sample_audio.py --input data/raw --output data/processed
```

출력은 입력과 중첩되지 않는 새 디렉터리를 사용한다. 기존 파일은 덮어쓰지 않는다.
`sampling_manifest.json`은 입력 hash, 시작 위치, padding과 출력 subtype을 기록한다.
PCM16은 60초 파일당 약 1.92 MB이며 양자화로 인해 float WAV와 수치가 완전히 같지는 않다.

## Org와 feature-aware 비교

Baseline은 **Org 한 음원에서 모든 feature를 추출**한다. Feature-aware는 같은 곡의
`Cover_Timbre_A`, `Cover_Rhythm_A`, `Cover_Instrument_A`, `Cover_Genre_A`를 사용한다.
`Cover_Timbre` baseline은 음색 강조 음원만 사용하는 별도 실험에 해당한다.

```bash
python scripts/build_features.py --input data/processed --output artifacts/features60 --baseline-source Org --variant A --songs-per-genre 50
python scripts/evaluate.py --baseline artifacts/features60/baseline.npz --feature-aware artifacts/features60/feature_aware.npz --output results
```

`--songs-per-genre 50`은 Ballade, Dance, Hiphop, RnB, Rock, Trot 각각에서 필요한 다섯
source가 있는 곡 50개를 seed 42로 선택한다. 파일명 정렬 후 선택하므로 탐색 순서에 의존하지 않는다.
**선택은 audio load와 feature extraction 전에 수행**한다. 최대 300곡·1,500개 source를 읽는다.
`--variant A`는 A 파일을 요구하며 C/B로 대체하지 않는다. 동일 source의 중복 파일은 상대 경로
사전순 첫 파일을 선택한다. 어느 장르든 곡이 부족하면 추출 전에 오류가 발생한다.
선택된 곡에서 추출 실패가 발생하면 manifest를 기록하고 NPZ를 생성하지 않는다.

`source_manifest.json`에서 선택된 song IDs, source 경로·hash, 실패 항목을 확인한다.
입력은 60초 길이를 요구하며 loading 시 mono 16 kHz로 맞춘다. 출력은 unscaled 96D NPZ이고
기존 archive/manifest를 덮어쓰지 않는다. 옵션을 생략하면 전체 후보를 처리하므로
위 subset 명령을 사용해 불필요한 추출을 피할 수 있다.

## 평가와 해석

두 조건은 같은 song IDs와 같은 10개 stratified fold를 사용한다. 300곡이 모두 성공하면
각 fold의 장르별 training/validation 곡 수는 45/5이다. GMM은 7 components, diagonal
covariance, max_iter=400, random_state=42, init_params=kmeans로 고정한다.
MinMaxScaler는 training fold에서만 fit한다.

생성 파일:

- `results/cv_metrics.json`: 설정·버전·입력 hash·class counts·validation song IDs·confusion counts
- `results/fold_metrics.csv`: 조건별 fold accuracy, macro F1, convergence
- `results/feature_source_ablation.csv`: 조건별 accuracy/macro F1 mean ± std, nonconverged folds
- `results/confusion_matrix.png`: 두 조건의 장르별 confusion matrix

점수는 0–1 단위이고 표준편차는 ddof=0이다. Feature-aware에서 baseline을 뺀 차이,
미수렴 fold 수, 장르별 혼동을 함께 해석한다. 결과가 낮거나 개선이 없어도 그대로 보고한다.
이 실험은 **균형화된 300곡 부분집합의 검증**이며 AI Hub 전체 데이터의 성능으로 일반화하지 않는다.
현재 공개된 실데이터 평가 지표는 없다. 5초 비교는 이 실험에 포함하지 않는다.

음원·feature matrix·학습 모델은 로컬 artifact로 관리하고 Git에는 평가 요약만 저장한다.

## 테스트

```bash
python -m unittest discover -s tests -v
```

테스트는 sampling, feature 차원·유한값, source 선택·sample 정렬, fold별 scaling을 검증한다.
합성 데이터의 테스트 통과는 실제 음악의 분류 성능을 의미하지 않는다.
