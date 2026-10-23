# Historical intent and Portfolio Refactoring

이 문서는 2026-10-23 Portfolio Refactoring의 기록이다. README, repository name,
About, Topics는 이번 작업 범위에서 제외했다. 아래 변경을 2025년 구현으로 소급하지 않는다.

## Paper → Reference → My Adaptation

1. G. Tzanetakis & P. Cook, *Musical Genre Classification of Audio Signals* (2002).
   논문 리뷰 자료의 19D는 original paper의 **timbral texture** vector이며,
   별도의 rhythmic/pitch feature sets도 설명한다. 현재 프로젝트의 전체 차원이 아니다.
2. [SamanKhamesian reference implementation](https://github.com/SamanKhamesian/Music-Genre-Classification-of-Audio-Signals).
   이번 확인 snapshot: `c315bf9bba83581ac2de618f60b48c945880c985`.
   이것은 현재 확인한 upstream 버전이지, 2025년 당시 사용한 commit이 확인됐다는 뜻이 아니다.
   `Source/Utilities.py`, `Source/CreateDataset.py`, `config.py`에서 Librosa 기반 96D,
   5초 load, 22050 Hz, Source import 및 Model 설정을 확인했다.
   Reference의 classifier 선택과 이 프로젝트의 장르별 GMM 구현은 구분한다.
3. 이 프로젝트의 2025년 adaptation은 AI Hub 한국 음악 데이터, onset-based 60초 공통 샘플,
   특징에 따라 다른 음원을 선택하는 아이디어, 장르별 GMM 및 10-fold CV이다.
   기존 repository는 그 의도가 일부만 실행 가능한 상태로 남아 있었다.

| 부분 | 출처 / historical evidence | 이번 변경 |
| --- | --- | --- |
| spectral, MFCC, Mel, tempo, chroma, tonnetz 통계 | Reference Utilities의 expanded 96D | 기존 프로젝트 순서 유지, coefficient별 mean/std 교차 순서와 names 일치, matrix 1회 계산 |
| 5초 / 22050 Hz | Reference CreateDataset 및 config | 기본 60초 전체 / 16000 Hz로 통일. 5초는 선택적 duration 비교 |
| Source imports / Main / Model 저장 | Reference 구조에서 유래한 잔재 | package relative import, scripts 3개, dangling persistence 및 Main 제거 |
| 60초 onset sampling | 첨부 `music_sampling.py`, 주간보고서 3주차, 팀 최종보고서 공통 데이터 설명 | Windows 경로 제거, CLI/재귀 탐색/manifest/mono resampling 통합 |
| feature별 source | 주간보고서 7주차(2025.05.05–05.11), 기존 config FEATURE_MAP | mapping 유지, lexical variant 선택 및 source hash 기록 |
| 장르별 GMM 및 고정 설정 | 주간보고서 6주차, 기존 GMM_Classification | 7 / diag / 400 / 42 / kmeans 고정. 새 탐색 없음 |
| 장르별 sample 수 동일화 | 주간보고서 6주차 | 최소 class count로 seed 42 deterministic undersampling. 구체 seed/선택 알고리즘은 이번 명시화 |
| 10-fold CV | 기존 EvaluateModel, 주간보고서 8주차, 최종보고서 | fold 내부 scaler fit, 동일 곡·fold ablation, macro F1·provenance 추가 |

## Sampling restoration and newly specified edge cases

첨부 `music_sampling.py`의 핵심 동작을 보존한다:
`librosa.load(sr=16000)` → `onset_detect(backtrack=True)` →
`onset_strength(aggregate=np.median)`를 detected frame에서 비교 → 최대 강도 시점부터
60초 → 뒤가 부족하면 마지막 60초. FRAME hop은 기존 Librosa default인 512이다.
각 source 파일은 원래 코드처럼 독립적으로 sampling한다(서로 시간 정렬한 것이 아니다).

이번에 보완한 정책: onset이 없으면 처음부터, 60초 미만이면 오른쪽 zero padding,
빈/비정상 음원은 오류, 출력은 mono 16 kHz float WAV. 원본은 짧은 음원을 그대로 저장하고
onset이 없으면 argmax에서 실패할 수 있었으므로 이 edge case를 historical 동작으로 주장하지 않는다.
정수 frame/sample 연산으로 60초 길이를 보장하며 원본 시간→정수 변환의 반올림 차이는 최대 1 sample이다.

## Exact feature schema

벡터 순서는 기존 프로젝트의 feature 파일 결합 순서를 따른다(reference 순서와는 다르다).
각 coefficient의 mean, std가 교차한다. `config.FEATURE_NAMES`가 같은 순서로 생성된다.

| Feature | Dimension | Feature-aware source |
| --- | ---: | --- |
| spectral centroid, bandwidth, contrast, rolloff | 8 | Cover_Timbre |
| spectral flux (onset strength) | 2 | Cover_Rhythm |
| zero crossing | 2 | Cover_Timbre |
| MFCC (first 13 default coefficients) | 26 | Cover_Timbre |
| log Mel (first 10 default 128 bands) | 20 | Cover_Timbre |
| tempo mean/std | 2 | Cover_Rhythm |
| chroma | 24 | Cover_Instrument |
| tonnetz | 12 | Cover_Genre |
| Total | **96** | |

Spectral contrast는 기존처럼 전체 band/frame의 mean/std 2개이다. Mel을 새로운 10-band
filterbank로 바꾸지 않았다. Tempo는 reference와 동일한 global estimate의 mean/std이며
std는 singleton 통계이므로 0이다. 시간별 tempo 추정이라는 새 feature를 추가하지 않았다.
기존 adaptation은 tempo std가 빠져 실제 95D였으며 이번에 복구했다.

## Evaluation interpretation

A는 명시적으로 지정한 하나의 source에서 모든 특징을 추출한다. B는 위 mapping을 쓴다.
두 조건의 모든 특징이 성공한 곡만 함께 commit한다. 곡 ID는 `Similar_<genre>_<id>`이며
동일 곡의 variant를 별도 sample로 취급하지 않는다. 누락/실패 곡은 manifest에 기록한다.
source prefix는 token 단위로 일치시키고 variant는 상대 경로 사전순 첫 파일을 선택한다.
이는 재현성 정책이며 특정 variant가 가장 좋다는 주장이 아니다.

동일 complete-case cohort를 class별 최소 수로 undersampling한 뒤, 같은 StratifiedKFold
(10, shuffle=True, seed=42)를 두 조건에 적용한다. 따라서 평가는 원 데이터의 자연 발생
장르 비율이 아닌 **균형화된 비교 집합**에 대한 결과다. 제외·균형화 이전/이후 counts와
validation song IDs, archive hash 및 dependency 버전을 결과에 남긴다.
MinMaxScaler는 각 train fold에서만 fit한다. Validation 값은 [0,1] 밖에 있을 수 있다.
Accuracy와 macro F1의 fold mean ± population std(ddof=0), convergence 상태를 기록한다.
기존 전체 데이터 scaling 결과는 새 성능 근거로 사용하지 않는다.

5초 옵션은 **60초 onset sample의 첫 5초**를 사용하는 duration ablation이다. 원본 reference의
원천 파일 첫 5초 / 22050 Hz 실행을 그대로 재현한 실험이라고 표현하지 않는다.

## Contribution to Team Final Model (separate context)

주간보고서 10주차는 개인 GMM 작업 이후 팀 SuCo-HRNE 모델의 repeat & thumbs 제거와
구간별 평균·표준편차 도입을 기록한다. 이는 이 repository GMM의 성능 개선으로 포함하지 않는다.
팀 최종보고서의 **99.08%**는 여러 팀원의 개선을 합친 팀 최종 모델 결과이며 개인 GMM accuracy가 아니다.

## Evidence and artifact limits

근거: 제공된 `music_sampling.py`, 황유리 2025-1 주간학습보고서(3/6/7/8/10주차),
팀 최종결과보고서, 논문 리뷰 발표자료, 작업 시작 commit `e3fecc8`의 코드.
서명·개인정보가 있는 PDF 원본은 Git에 추가하지 않았다.
기존 tracked `.npz` 11개와 `model.pkl`은 provenance 및 재평가 적합성을 확인할 수 없어
working tree에서 제거했다. Git 과거 이력은 다시 쓰지 않았다.
이번 환경에서 AI Hub 접근이 network proxy 403으로 차단되어 실제 음원 평가를 수행하지 못했다.
Synthetic 검증은 실행 검증이며 장르 분류 성능 증거가 아니다.
