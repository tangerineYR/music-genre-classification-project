import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from Source.GMM_Classification import train_gmm_models, predict_with_gmms

# 특징 파일 리스트
feature_files = [
    'FEATURE_SPECTRAL_CENTROID.npz',
    'FEATURE_SPECTRAL_BANDWIDTH.npz',
    'FEATURE_SPECTRAL_CONTRAST.npz',
    'FEATURE_SPECTRAL_ROLLOFF.npz',
    'FEATURE_SPECTRAL_FLUX.npz',
    'FEATURE_ZERO_CROSSING.npz',
    'FEATURE_MFCC.npz',
    'FEATURE_MEL_SCALE.npz',
    'FEATURE_TEMPO.npz',
    'FEATURE_CHROMA.npz',
    'FEATURE_TONNETZ.npz'
]

def load_and_validate_features(feature_files):
    """특징 파일 로드 및 검증"""
    all_data = []
    for file in feature_files:
        try:
            data = np.load(file)
            assert 'files' in data, f"'files' 키 누락: {file}"
            assert 'x' in data and 'y' in data, f"데이터 형식 오류: {file}"
            all_data.append(data)
            print(f"✅ {file} 로드 완료 ({data['x'].shape[0]} samples)")
        except Exception as e:
            print(f"⚠️ {file} 처리 실패: {str(e)}")
            raise
    return all_data

def get_common_samples(all_data):
    """공통 샘플 추출"""
    # 1. 모든 파일명 추출
    file_sets = [set(data['files']) for data in all_data]
    
    # 2. 공통 파일 찾기
    common_files = set.intersection(*file_sets)
    if not common_files:
        raise ValueError("공통 파일이 존재하지 않습니다!")
    common_files = sorted(list(common_files))
    
    # 3. 인덱스 매핑
    indices_list = []
    for data in all_data:
        index_map = {f: i for i, f in enumerate(data['files'])}
        indices = [index_map[f] for f in common_files]
        indices_list.append(indices)
    
    return common_files, indices_list

# 1. 특징 파일 로드
print("=== 특징 파일 로드 시작 ===")
all_data = load_and_validate_features(feature_files)

# 2. 공통 샘플 추출
print("\n=== 공통 샘플 추출 ===")
common_files, indices_list = get_common_samples(all_data)
print(f"공통 파일 수: {len(common_files)}")
print(f"샘플 파일: {common_files[:2]}")

# 3. 데이터 병합
X = np.hstack([data['x'][indices] for data, indices in zip(all_data, indices_list)])
y = all_data[0]['y'][indices_list[0]]

# 4. 라벨 일관성 검증
for i, data in enumerate(all_data[1:]):
    if not np.array_equal(y, data['y'][indices_list[i+1]]):
        raise ValueError(f"라벨 불일치: {feature_files[i+1]}")

print("\n=== 데이터 병합 결과 ===")
print(f"X 형태: {X.shape}")
print(f"y 형태: {y.shape}")
print(f"고유 라벨: {np.unique(y)}")

# 5. K-Fold 교차검증
print("\n=== 모델 평가 시작 ===")
k = 10
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n[Fold {fold}]")
    
    # 데이터 분할
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # GMM 모델 학습
    gmm_models = train_gmm_models(X_train, y_train)
    
    # 예측 및 평가
    y_pred = predict_with_gmms(gmm_models, X_test)
    acc = accuracy_score(y_test, y_pred)
    fold_accuracies.append(acc)
    print(f"정확도: {acc:.4f}")

# 최종 결과
print("\n=== 최종 결과 ===")
print(f"Fold 별 정확도: {[round(acc,4) for acc in fold_accuracies]}")
print(f"평균 정확도: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
