"""Vectorized statistics of the project's reference-derived Librosa features."""
import librosa
import numpy as np

from .config import FRAME_SIZE, HOP_SIZE, TARGET_SAMPLE_RATE, FEATURE_SIZES


def extract_feature(y, feature, sr=TARGET_SAMPLE_RATE):
    if sr != TARGET_SAMPLE_RATE:
        raise ValueError(f'Expected {TARGET_SAMPLE_RATE} Hz audio, got {sr}')
    if np.ndim(y) != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('Audio must be nonempty, finite and mono')
    kwargs = dict(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    if feature in ('spectral_centroid', 'spectral_bandwidth',
                   'spectral_contrast', 'spectral_rolloff'):
        matrix = getattr(librosa.feature, feature)(**kwargs).reshape(1, -1)
    elif feature == 'spectral_flux':
        matrix = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_SIZE)[None, :]
    elif feature == 'zero_crossing':
        matrix = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_SIZE, hop_length=HOP_SIZE)
    elif feature == 'mfcc':
        # Historical definition: first 13 of the default 20 MFCCs.
        matrix = librosa.feature.mfcc(**kwargs)[:13]
    elif feature == 'mel_scale':
        # First 10 of the default 128 Mel bands, NOT a new 10-band filterbank.
        matrix = librosa.power_to_db(librosa.feature.melspectrogram(**kwargs))[:10]
    elif feature == 'tempo':
        # Preserve global tempo estimation. Its one-element std is zero.
        matrix = librosa.feature.tempo(y=y, sr=sr, hop_length=HOP_SIZE)[None, :]
    elif feature == 'chroma':
        matrix = librosa.feature.chroma_stft(**kwargs)
    elif feature == 'tonnetz':
        matrix = librosa.feature.tonnetz(y=y, sr=sr)
    else:
        raise ValueError(f'Unknown feature: {feature}')
    vector = np.column_stack((matrix.mean(axis=1), matrix.std(axis=1))).ravel()
    if vector.shape != (2 * FEATURE_SIZES[feature],) or not np.isfinite(vector).all():
        raise ValueError(f'Invalid feature: {feature}')
    return vector


def get_feature_extractor(feature_name):
    return lambda y: extract_feature(y, feature_name)
