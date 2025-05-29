import os
import numpy as np
import sklearn
import soundfile as sf
import librosa
from pydub import AudioSegment
from Source.Utilities import *
from config import CreateDataset

DATASET_DIR = CreateDataset.DIRECTORY
SAMPLING_RATE = CreateDataset.SAMPLING_RATE

def extract_song_id(file_path):
    """파일명에서 곡 ID 추출 (예: 'Similar_Trot_00610_Cover_Instrument_B.wav' → 'Similar_Trot_00610')"""
    parts = os.path.basename(file_path).split('_')
    return '_'.join(parts[:3])  # 첫 3부분 조합 (Similar + 장르 + ID)


def robust_audio_load(file_path):
    """개선된 오디오 로딩 시스템"""
    try:
        y, sr = sf.read(file_path)
    except Exception:
        try:
            audio = AudioSegment.from_file(file_path)
            y = np.array(audio.get_array_of_samples(), dtype=np.float32)
            y /= np.iinfo(audio.array_type).max
            sr = audio.frame_rate
        except Exception:
            raise
    return y, sr

def process_individual_features(target_features):
    # 1. 모든 파일 수집 및 그룹화
    all_files = librosa.util.find_files(DATASET_DIR, recurse=True)
    
    # {song_id: {feature: file_path}}
    song_db = {}
    for file in all_files:
        song_id = extract_song_id(file)
        parts = os.path.basename(file).split('_')
        feature_type = '_'.join(parts[3:5])  # 'Cover_Instrument', 'Cover_Tempo' 등

        if song_id not in song_db:
            song_db[song_id] = {}
        song_db[song_id][feature_type] = file
    
    # 디버깅: song_db 샘플 출력
    print("\n=== song_db 샘플 (최대 3개) ===")
    for song_id, features in list(song_db.items())[:3]:
        print(f"  {song_id}: {list(features.keys())}")
    
    # 2. 필수 특징 접두사 추출
    feature_prefixes = [get_feature_prefix(f) for f in target_features]
    print(f"\n필요한 특징 접두사: {feature_prefixes}")
    
    # 3. 모든 특징을 가진 곡 필터링
    valid_songs = []
    for song_id, features in song_db.items():
        if all(prefix in features for prefix in feature_prefixes):
            valid_songs.append(song_id)
    
    print(f"\n유효한 곡 수: {len(valid_songs)}")
    if len(valid_songs) == 0:
        raise ValueError("유효한 곡이 없습니다! 파일명 구조를 확인하세요.")
    
    # 3. 피처별 데이터 저장소 초기화
    feature_data = {f: [] for f in target_features}
    labels = []
    
    # 4. 곡별 처리
    for song_id in valid_songs:
        genre = song_id.split('_')[1]  # 예: 'Trot'
        labels.append(genre)
        
        # 피처별 추출
        for feature in target_features:
            prefix = get_feature_prefix(feature)
            file_path = song_db[song_id][prefix]
            
            try:
                y, _ = robust_audio_load(file_path)
                y = y[:SAMPLING_RATE*5] if len(y) >= SAMPLING_RATE*5 else np.pad(y, (0, SAMPLING_RATE*5 - len(y)))
                
                # 특징 추출
                feature_extractor = get_feature_extractor(feature)
                row = feature_extractor(y)
                feature_data[feature].append(row)
                
            except Exception as e:
                print(f"[오류] {file_path} 처리 실패: {str(e)}")
                break
    
    # 5. 피처별 저장
    for feature in target_features:
        data = np.array(feature_data[feature])
        scaler = sklearn.preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data)
        
        np.savez(
            f'FEATURE_{feature.upper()}.npz',
            x=data,
            y=np.array(labels),
            files=np.array(valid_songs)
        )
        print(f"✅ FEATURE_{feature.upper()}.npz 저장 완료")

if __name__ == '__main__':
    target_features = [
        'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast',
        'spectral_rolloff', 'spectral_flux', 'zero_crossing',
        'mfcc', 'mel_scale', 'tempo', 'chroma', 'tonnetz'
    ]
    process_individual_features(target_features)
