"""Historical feature/source choices; portfolio execution defaults (see docs/lineage.md)."""
TARGET_SAMPLE_RATE = 16000
SAMPLE_SECONDS = 60
FRAME_SIZE = 2048
HOP_SIZE = 512
RANDOM_STATE = 42
GMM_PARAMS = dict(n_components=7, covariance_type='diag', max_iter=400,
                  random_state=RANDOM_STATE, init_params='kmeans')
# Insertion order is the feature vector order. Preserve the project's mapping.
FEATURE_SOURCES = {
    'spectral_centroid': 'Cover_Timbre',
    'spectral_bandwidth': 'Cover_Timbre',
    'spectral_contrast': 'Cover_Timbre',
    'spectral_rolloff': 'Cover_Timbre',
    'spectral_flux': 'Cover_Rhythm',
    'zero_crossing': 'Cover_Timbre',
    'mfcc': 'Cover_Timbre',
    'mel_scale': 'Cover_Timbre',
    'tempo': 'Cover_Rhythm',
    'chroma': 'Cover_Instrument',
    'tonnetz': 'Cover_Genre',
}
FEATURE_SIZES = dict(zip(FEATURE_SOURCES, [1, 1, 1, 1, 1, 1, 13, 10, 1, 12, 6]))
FEATURE_NAMES = [f'{stat}_{feature}_{i + 1:02d}'
                 for feature, size in FEATURE_SIZES.items()
                 for i in range(size) for stat in ('mean', 'std')]
SCHEMA_VERSION = 'portfolio-96d-v1'
