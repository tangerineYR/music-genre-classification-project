참고 논문:
  이 project는 G.Tzanetakis and Perry R.Cook - "Musical Genre Classification of Audio Signals" 논문 내용을 기반으로 리프로덕션을 진행하였습니다. 

----------------------------------------------------------------------------------------------------------------------------------------

이 project를 실행하기 위해서는 다음과 같은 파이썬 패키지가 필요합니다:
  - librosa
  - numpy
  - os
  - sklearn
  - soundfile
  - pydub
  - joblib

----------------------------------------------------------------------------------------------------------------------------------------


Dataset:
  다음 링크의 원천데이터 데이터셋을 비트 변화 최고점 기준 1분 샘플링 및 전처리를 적용하여 사용하였습니다.
  (https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71544) 

----------------------------------------------------------------------------------------------------------------------------------------


py 파일들에 대한 설명:
  - config.py : 음악 특징 데이터셋을 만드는 데 필요한 설정값, 데이터 경로, 하이퍼파라미터 등을 모아 놓은 파일이다. (FEATURE_MAP, MFCC_SIZE, SAMPLING_RATE 등)
  - Utilities.py : 본 프로젝트에서 주로 반복적으로 사용되는 함수들을 모아 놓은 파일이다. (get_subdirectories, filter_files_by_feature, get_feature_extractor 등)
  - CreateFeatureDataset.py : 특징 데이터셋(.npz)을 만드는 파일이다. 주로 Utilities.py에서 함수를 가져와 사용한다.
  - GMM_Classification.py : GMM 분류기를 구현하는 데 사용되는 함수들을 모아 놓은 파일이다. GMM 모델을 학습시키는 train_gmm_models, 학습된 GMM 모델을 이용하여 장르를 예측하는 predict_with_gmms, GMM 모델들을 파일로 저장하는 save_gmm_models 함수가 있다.
  - EvaluateModel.py : CreateFeatureDataset.py를 통해 생성된 특징 별 데이터셋을 불러와 하나로 병합하고, GMM_Classification의 함수들을 불러와 GMM 모델을 구현하여 장르 분류를 실행하는 파일이다. 또한 데이터셋을 k개의 fold로 나누고, 한 fold씩 번갈아가며 검증 셋으로 사용하여 총 k번의 학습 및 평가를 진행한 뒤 평균 정확도를 도출하는 k-fold cross-validation을 이용하여 모델을 평가한다.
  - Main.py : CreateFeatureDataset.py, EvaluateModel.py를 순서대로 실행하여 데이터셋에 대한 장르 분류 및 정확도 측정을 실행하는 파일이다.
