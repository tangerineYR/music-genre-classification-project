"""Historical selected GMM configuration; no model persistence or new tuning."""
import numpy as np
from sklearn.mixture import GaussianMixture
from .config import GMM_PARAMS


def train_gmm_models(data_x, data_y):
    models = {}
    for genre in np.unique(data_y):
        rows = data_x[data_y == genre]
        if len(rows) < GMM_PARAMS['n_components']:
            raise ValueError(f'{genre}: at least 7 training songs required per fold')
        models[genre] = GaussianMixture(**GMM_PARAMS).fit(rows)
    return models


def predict_with_gmms(gmm_models, test_x):
    genres = list(gmm_models)
    scores = np.column_stack([gmm_models[g].score_samples(test_x) for g in genres])
    return np.asarray(genres)[scores.argmax(axis=1)]
