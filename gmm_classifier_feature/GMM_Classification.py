import numpy as np
from sklearn.mixture import GaussianMixture
import joblib
from config import CreateDataset, Model


def train_gmm_models(data_x, data_y, n_components=5, covariance_type='full', max_iter=200, random_state=42):
    """
    각 장르별로 GMM 모델을 학습하고 딕셔너리 형태로 반환한다.
    """
    trained_gmms = {}
    genres = CreateDataset.Genres
    print(f"\n총 {len(genres)}개의 장르별 GMM 모델 학습 시작...")

    for genre in genres:
        print(f"'{genre}' 장르 GMM 모델 학습 중...")

        genre_data_x = data_x[data_y == genre]

        if genre_data_x.shape[0] == 0:
            print(f"경고: '{genre}' 장르의 데이터가 없습니다. 건너뜁니다.")
            continue

        print(f"  '{genre}' 장르 데이터 형태: {genre_data_x.shape}")

        gmm = GaussianMixture(n_components=n_components,
                              covariance_type=covariance_type,
                              random_state=random_state,
                              max_iter=max_iter)

        gmm.fit(genre_data_x)
        trained_gmms[genre] = gmm

        print(f"  '{genre}' 장르 GMM 모델 학습 완료.")

    print("\n모든 장르 GMM 모델 학습 완료!")
    print(f"학습된 GMM 모델 개수: {len(trained_gmms)}")

    return trained_gmms


def predict_with_gmms(gmm_models, test_x):
    """
    학습된 GMM 모델들로부터 각 샘플의 장르를 예측
    """
    predictions = []

    for sample in test_x:
        scores = {genre: gmm.score_samples(sample.reshape(1, -1))[0]
                  for genre, gmm in gmm_models.items()}
        predicted_genre = max(scores, key=scores.get)
        predictions.append(predicted_genre)

    return np.array(predictions)


def save_gmm_models(gmm_models, filename=None):
    """
    GMM 모델들을 파일로 저장
    """
    if filename is None:
        filename = Model.NAME

    print(f"\nGMM 모델을 '{filename}'에 저장 중...")
    try:
        joblib.dump(gmm_models, filename)
        print("GMM 모델 저장 완료!")
    except Exception as e:
        print(f"오류: GMM 모델 저장 실패 - {e}")
