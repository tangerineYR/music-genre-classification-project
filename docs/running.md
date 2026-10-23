# Portfolio pipeline 실행

Python 3.10 이상. 저장소 루트에서 실행한다. README는 요청에 따라 기존 상태를 보존했으므로
이번 실행 방법은 이 문서를 따른다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

[AI Hub 음악 유사성 판별 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71544)의
원천 음원을 제공 절차에 따라 내려받아 압축을 해제한다. 링크 자체가 로컬 음원은 아니다.
MIDI나 라벨 파일 대신 waveform 파일을 사용하고, 원래의
`Similar_Trot_00610_Cover_Instrument_B.wav` 같은 파일명을 유지한다.
하위 폴더는 재귀적으로 읽는다. 기본 지원 형식은 WAV/FLAC/MP3/OGG/AIFF/M4A이며,
비-WAV 코덱의 사용 가능 여부는 로컬 decoder에 달려 있으므로 WAV를 권장한다.

```bash
python scripts/sample_audio.py --input data/raw --output data/processed
python scripts/build_features.py --input data/processed --output artifacts/features60 --baseline-source Cover_Timbre
python scripts/evaluate.py --baseline artifacts/features60/baseline.npz --feature-aware artifacts/features60/feature_aware.npz --output results
```

`Cover_Timbre`는 실행 예시로 선택한 공통 single-source baseline이다. 원음 source가 있다면
실제 파일명의 source prefix로 명시한다(예: `--baseline-source Original`). 두 실험 모두
동일 곡을 쓰기 때문에 baseline source 및 Cover_Timbre/Rhythm/Instrument/Genre가 모두
있는 곡만 평가 대상이다. 이 조건으로 유효 곡이 부족하면 원천 데이터 구성을 먼저 점검한다.
필요 source를 다른 source로 자동 대체하지 않는다.

Sampling 출력은 원래 폴더 구조를 유지하며, 기존 파일을 덮어쓰거나 입력과 중첩된 출력
폴더를 허용하지 않는다. 새 출력 폴더를 지정한다. Feature build도 기존 archive/manifest를 덮어쓰지 않으므로 새 출력 폴더를 사용한다. `sampling_manifest.json`에는 시작 위치,
padding 및 입력 hash가 남는다. Feature 추출은 60초 processed audio를 요구한다.
`artifacts/features60/source_manifest.json`에서 선택한 파일과 실패·누락 곡을 확인한다.
중간 NPZ는 unscaled 96D이고 pickle을 사용하지 않는다. 이전 NPZ와 호환되지 않으므로 다시 생성한다.

Evaluation은 조건별 완전한 곡이 class당 최소 10개, class가 최소 2개 있어야 실행된다.
동일 곡을 두 fold에 중복 배치하지 않는다. historical GMM 설정은 고정하며 각 training class에
최소 7곡이 있어야 한다. 10-fold에서 조건을 만족하지 못하면 fold 수나 모델을 몰래 바꾸지 않고 실패한다.

결과는 `cv_metrics.json`, `fold_metrics.csv`, `confusion_matrix.png`,
`feature_source_ablation.csv` 네 파일이다. CSV 점수는 0–1 단위이며 mean/std를 함께 비교한다.
`nonconverged_folds`가 있으면 convergence 경고도 함께 해석한다.
개선이 없거나 제한적이어도 그대로 보고한다. 소규모 평가 결과만 Git으로 관리하고
raw audio, processed audio, feature matrix, fitted model은 올리지 않는다.

선택적 5초 비교는 build 명령에 `--seconds 5 --output artifacts/features5`를 사용하고
별도 `--output results/5sec`에서 평가한다. 60초와 비교할 때 cv_metrics의 validation song IDs가
동일한지 반드시 확인한다. 다르면 duration 비교 결과로 사용하지 않는다.

```bash
python -m unittest discover -s tests -v
```
