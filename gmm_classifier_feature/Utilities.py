import librosa
import numpy as np
from config import CreateDataset
import os
import librosa.beat


SAMPLING_RATE = CreateDataset.SAMPLING_RATE
FRAME_SIZE = CreateDataset.FRAME_SIZE
HOP_SIZE = CreateDataset.HOP_SIZE
MFCC_SIZE = CreateDataset.MFCC_SIZE
CHROMA_SIZE = CreateDataset.CHROMA_SIZE
MEL_SCALE_SIZE = CreateDataset.MEL_SCALE_SIZE
TONNETZ_SIZE = CreateDataset.TONNETZ_SIZE


def get_subdirectories(directory):
    return [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]


def get_feature_prefix(feature_name):
    return CreateDataset.FEATURE_MAP[feature_name]['prefix']  # 예: 'Cover_Instrument_'


def filter_files_by_feature(file_list, feature_name):
    """
    파일명에 Cover_특징_이 '포함'되어 있으면 선택
    """
    prefix = get_feature_prefix(feature_name)  # 예: 'Cover_Instrument_'
    return [f for f in file_list if prefix in os.path.basename(f)]


def get_feature_extractor(feature_name: str):
    """개별 특징 추출 함수 반환 (완전한 구현)"""
    feature_map = {
        # 음색 관련 특징
        'spectral_centroid': lambda s: [
            np.mean(librosa.feature.spectral_centroid(y=s, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)),
            np.std(librosa.feature.spectral_centroid(y=s, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE))
        ],
        'spectral_bandwidth': lambda s: [
            np.mean(librosa.feature.spectral_bandwidth(y=s, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)),
            np.std(librosa.feature.spectral_bandwidth(y=s, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE))
        ],
        'spectral_contrast': lambda s: [
            np.mean(librosa.feature.spectral_contrast(y=s, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)),
            np.std(librosa.feature.spectral_contrast(y=s, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE))
        ],
        'spectral_rolloff': lambda s: [
            np.mean(librosa.feature.spectral_rolloff(y=s, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)),
            np.std(librosa.feature.spectral_rolloff(y=s, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE))
        ],
        'spectral_flux': lambda s: [
            np.mean(librosa.onset.onset_strength(y=s, sr=SAMPLING_RATE, hop_length=HOP_SIZE)),
            np.std(librosa.onset.onset_strength(y=s, sr=SAMPLING_RATE, hop_length=HOP_SIZE))
        ],
        'zero_crossing': lambda s: [
            np.mean(librosa.feature.zero_crossing_rate(y=s, frame_length=FRAME_SIZE, hop_length=HOP_SIZE)),
            np.std(librosa.feature.zero_crossing_rate(y=s, frame_length=FRAME_SIZE, hop_length=HOP_SIZE))
        ],

        # MFCC
        'mfcc': lambda s: [
            val for i in range(MFCC_SIZE)
            for val in [
                np.mean(librosa.feature.mfcc(y=s, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)[i]),
                np.std(librosa.feature.mfcc(y=s, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)[i])
            ]
        ],

        # Mel Spectrogram
        'mel_scale': lambda s: [
            val for i in range(MEL_SCALE_SIZE)
            for val in [
                np.mean(librosa.power_to_db(
                    librosa.feature.melspectrogram(y=s, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
                )[i]),
                np.std(librosa.power_to_db(
                    librosa.feature.melspectrogram(y=s, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
                )[i])
            ]
        ],
        
        # 템포/리듬
        'tempo': lambda s: [
            np.mean(librosa.beat.tempo(y=s, sr=SAMPLING_RATE, hop_length=HOP_SIZE))
        ],
        
        # Chroma
        'chroma': lambda s: [
            val for i in range(CHROMA_SIZE)
            for val in [
                np.mean(librosa.feature.chroma_stft(y=s, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)[i]),
                np.std(librosa.feature.chroma_stft(y=s, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)[i])
            ]
        ],
        
        # Tonnetz
        'tonnetz': lambda s: [
            val for i in range(TONNETZ_SIZE)
            for val in [
                np.mean(librosa.feature.tonnetz(y=s, sr=SAMPLING_RATE)[i]),
                np.std(librosa.feature.tonnetz(y=s, sr=SAMPLING_RATE)[i])
            ]
        ]
    }
    
    if feature_name.lower() not in feature_map:
        raise ValueError(f"지원하지 않는 특징: {feature_name}")
        
    return feature_map[feature_name.lower()]
