# 연구 출처와 방법

## Paper → Reference → My Adaptation

1. G. Tzanetakis & P. Cook, [*Musical Genre Classification of Audio Signals* (2002)](https://doi.org/10.1109/TSA.2002.800560).
   논문의 19D는 **timbral texture** vector이며,
   별도의 rhythmic/pitch feature sets도 설명한다. 현재 프로젝트의 전체 차원이 아니다.
2. [SamanKhamesian reference implementation](https://github.com/SamanKhamesian/Music-Genre-Classification-of-Audio-Signals).
   검토한 [upstream snapshot](https://github.com/SamanKhamesian/Music-Genre-Classification-of-Audio-Signals/tree/c315bf9bba83581ac2de618f60b48c945880c985).
   이것은 현재 확인한 upstream 버전이지, 2025년 당시 사용한 commit이 확인됐다는 뜻이 아니다.
   [Utilities](https://github.com/SamanKhamesian/Music-Genre-Classification-of-Audio-Signals/blob/c315bf9bba83581ac2de618f60b48c945880c985/Source/Utilities.py),
   [CreateDataset](https://github.com/SamanKhamesian/Music-Genre-Classification-of-Audio-Signals/blob/c315bf9bba83581ac2de618f60b48c945880c985/Source/CreateDataset.py),
   [config](https://github.com/SamanKhamesian/Music-Genre-Classification-of-Audio-Signals/blob/c315bf9bba83581ac2de618f60b48c945880c985/config.py)에서 Librosa 기반 96D,
   5초 load, 22050 Hz, Source import 및 Model 설정을 확인했다.
   Reference의 classifier 선택과 이 프로젝트의 장르별 GMM 구현은 구분한다.
3. 이 프로젝트의 2025년 adaptation은 AI Hub 한국 음악 데이터, onset-based 60초 공통 샘플,
   특징에 따라 다른 음원을 선택하는 아이디어, 장르별 GMM 및 10-fold CV이다.
   기존 repository는 그 의도가 일부만 실행 가능한 상태로 남아 있었다.

| 부분 | 출처 / historical evidence | 2026년 재현성 수정 |
| --- | --- | --- |
| spectral, MFCC, Mel, tempo, chroma, tonnetz 통계 | Reference Utilities의 expanded 96D | 기존 프로젝트 순서 유지, coefficient별 mean/std 교차 순서와 names 일치, matrix 1회 계산 |
| 5초 / 22050 Hz | Reference CreateDataset 및 config | 기본 60초 전체 / 16000 Hz로 통일. 5초는 선택적 duration 비교 |
| Source imports / Main / Model 저장 | Reference 구조와 historical code | package relative import, 실행 CLI, dangling persistence 및 Main 제거 |
| 60초 sampling | [historical README][historical-readme]의 비트 변화 최고점 기준 1분 설명 | 현재 sampling 구현은 [audio.py](../gmm_classifier_feature/audio.py). Historical WAV 평가에서는 실행하지 않음 |
| feature별 source | [historical config][historical-config]의 FEATURE_MAP | mapping 유지, lexical variant 선택 및 source hash 기록 |
| 장르별 GMM 및 고정 설정 | [historical GMM_Classification][historical-gmm] | 7 / diag / 400 / 42 / kmeans 고정. 새 탐색 없음 |
| 장르별 sample 수 동일화 | [현재 balanced_indices](../gmm_classifier_feature/EvaluateModel.py) | 최소 class count로 seed 42 deterministic undersampling. 구체 선택 알고리즘은 2026년 protocol |
| 10-fold CV | [historical EvaluateModel][historical-evaluate] | fold 내부 scaler fit, 동일 곡·fold ablation, macro F1·provenance 추가 |

## Historical segment와 현재 sampling 구현

[Historical README][historical-readme]에는 비트 변화 최고점 기준 1분 sampling이 설명되어 있다.
현재 [audio.py](../gmm_classifier_feature/audio.py)의 sampling 함수는 16 kHz load,
backtracked onset과 median onset strength, 최대 강도 시점 이후 60초, 뒤가 부족하면 마지막
60초를 사용한다. Onset이 없으면 처음부터, 짧으면 오른쪽 zero padding하는 예외 처리는
2026년 구현이다. 이 구현의 세부 동작 전부를 historical README만으로 입증할 수는 없다.

이번 평가는 이미 생성된 historical WAV를 대상으로 하므로 sampling 함수를 실행하지 않는다.
각 source의 정확한 시작 timestamp가 기록되어 있지 않아 source 간 시간 정렬을 보장하지 않는다.

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
기존 adaptation은 tempo std가 빠져 실제 95D였으며 2026년에 복구했다.

## Evaluation interpretation

현재 historical dataset에는 `Org`가 없다. 비교는 **Single-source Cover_Timbre baseline
vs feature-aware source mapping**이다. A는 `Cover_Timbre` 한 음원에서 96개 특징을 모두
추출하며, B는 위 표의 mapping을 사용한다. 네 required source family가 모두 존재하고
선택된 파일이 정상 decode·duration 검증을 통과한 모든 곡을 추출한다. Tempo와 Arrangement
source family는 inventory에 포함하지만 현재 평가에는 사용하지 않는다.

곡 ID는 `Similar_<genre>_<id>`이며 동일 곡의 variant를 별도 sample로 취급하지 않는다.
source prefix는 token 단위로 일치시키며, 동일 family에서는 dataset-relative path 사전순
첫 matching file을 선택한다. 성능에 따라 variant를 선택하지 않는다. 선택된 경로와 SHA-256,
누락·실패 사유를 manifest에 남긴다. 이 deterministic selection은 2026년 재현성 정책이며
특정 variant가 우수하다는 주장이 아니다. 이번 전체 historical 평가에는 variant 제한이나
장르별 사전 subset 옵션을 적용하지 않는다.

2025년 생성된 sampled WAV를 그대로 읽고 onset sampling을 반복하지 않는다. Feature
extraction의 representation만 `load_audio()`로 mono 16 kHz에 통일한다. 60초에서 10 ms 이내의
길이 차이는 원래 길이 그대로 허용하며 padding이나 구간 재선택을 하지 않는다.
그 이상의 차이, 빈 데이터, 비유한 값, decode 실패는 오류로 기록한다.

동일 complete-case cohort를 class별 최소 수로 undersampling한 뒤, 같은 StratifiedKFold
(10, shuffle=True, seed=42)를 두 조건에 적용한다. 따라서 평가는 원 데이터의 자연 발생
장르 비율이 아닌 **균형화된 비교 집합**에 대한 결과다. 제외·균형화 이전/이후 counts와
validation song IDs, archive hash 및 dependency 버전을 결과에 남긴다.
MinMaxScaler는 각 train fold에서만 fit한다. Validation 값은 [0,1] 밖에 있을 수 있다.
Accuracy와 macro F1의 fold mean ± population std(ddof=0), convergence 상태를 기록한다.
기존 전체 데이터 scaling 결과는 새 성능 근거로 사용하지 않는다.

5초 옵션은 **60초 onset sample의 첫 5초**를 사용하는 duration ablation이다. 원본 reference의
원천 파일 첫 5초 / 22050 Hz 실행을 그대로 재현한 실험이라고 표현하지 않는다.

각 Cover source는 2025년에 독립적으로 sampling되어 정확히 같은 timestamp 구간이라고
보장할 수 없다. 결과는 **historical single-source pipeline과 feature-aware pipeline의
paired comparison**이며 source type의 causal effect가 아니다. 평가 모집단은 required source가
모두 존재하는 2025 historical complete-case cohort이며 AI Hub 전체 benchmark가 아니다.

## 공개 코드 근거

Historical source mapping, GMM 설정, 10-fold CV는 commit
[`e3fecc8`](https://github.com/tangerineYR/music-genre-classification-project/tree/e3fecc8cbce09e85479f3bf05dd6bbd3f1e95ed4)의
코드로 확인할 수 있다. 현재 96D 정의는 [config.py](../gmm_classifier_feature/config.py)와
[Utilities.py](../gmm_classifier_feature/Utilities.py), paired protocol은
[EvaluateModel.py](../gmm_classifier_feature/EvaluateModel.py)를 따른다.
2026년 재현성 수정과 실측 결과를 2025년 구현·성능으로 소급하지 않는다.
별도 팀 모델의 성능은 이 저장소 GMM의 성능 근거에 포함하지 않는다.

[historical-readme]: https://github.com/tangerineYR/music-genre-classification-project/blob/e3fecc8cbce09e85479f3bf05dd6bbd3f1e95ed4/README.md
[historical-config]: https://github.com/tangerineYR/music-genre-classification-project/blob/e3fecc8cbce09e85479f3bf05dd6bbd3f1e95ed4/gmm_classifier_feature/config.py
[historical-gmm]: https://github.com/tangerineYR/music-genre-classification-project/blob/e3fecc8cbce09e85479f3bf05dd6bbd3f1e95ed4/gmm_classifier_feature/GMM_Classification.py
[historical-evaluate]: https://github.com/tangerineYR/music-genre-classification-project/blob/e3fecc8cbce09e85479f3bf05dd6bbd3f1e95ed4/gmm_classifier_feature/EvaluateModel.py
