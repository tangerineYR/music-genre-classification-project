# Portfolio validation — 2026-10-23

Python 3.12 환경에서 다음을 확인했다. 이는 2025년 검증 기록이 아니다.

- `python -m unittest discover -s tests -v`: 네 범주의 7개 회귀 테스트 통과.
- 세 scripts의 `--help` 실행 및 package imports 정상.
- `pip check`: dependency 충돌 없음.
- CLI smoke: 22.05 kHz stereo 합성 WAV 4개 → 16 kHz mono 60초 sampling →
  baseline/feature-aware 각각 `(1, 96)` archive 생성/검증.
- 별도 합성 96D 데이터(두 class, 총 30곡)로 실제 fixed GMM 10-fold 평가 및
  네 결과 파일(JSON, CSV 2개, PNG) 생성 확인. 두 조건에 같은 X를 주었을 때
  동일 결과이며 confusion matrix 총합이 30임을 확인. 임시 결과는 삭제됨.
- `git diff --check` 통과, README 변경 없음, dangling Source/Model 참조 없음.

합성 데이터 수치는 음악 장르 분류 성능으로 사용하지 않는다.
실제 AI Hub 음원 평가 및 feature-source 가설 검증은 다운로드 차단으로 미실행 상태다.
외부 codec의 audioread 경로에서 Python deprecated-module 경고가 있었고, 제한된 환경의
fontconfig cache 경고가 있었으나 WAV 처리 및 PNG 생성은 성공했다.
