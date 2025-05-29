class CreateDataset:
    # Path of Resource Files
    DIRECTORY = r"C:\Users\dbfl0\OneDrive\Desktop\세종대\3-1\창의학기제\Project\sepotify\Sepotify_Resource"

    # Sampling Rate (Hz)
    SAMPLING_RATE = 22050

    # Frame Size (Samples)
    FRAME_SIZE = 2048

    # Hop Size (Samples)
    HOP_SIZE = 512

    # Number of MFCC Coefficients
    MFCC_SIZE = 13

    # Number of Chroma Coefficients
    CHROMA_SIZE = 12

    # Number of Mel-Scale Coefficients
    MEL_SCALE_SIZE = 10

    # Tonal Centroid Size
    TONNETZ_SIZE = 6

    # Name of Genres
    Genres = ['Ballade', 'Dance', 'Hiphop', 'RnB', 'Rock', 'Trot']

    # 특징명 리스트
    Feature_Names = [
        # Timbral Texture Features (8개)
        'meanSpecCentroid', 'stdSpecCentroid',
        'meanSpecBandwidth', 'stdSpecBandwidth',
        'meanSpecContrast', 'stdSpecContrast',
        'meanSpecRolloff', 'stdSpecRolloff',
        'meanSpecFlux', 'stdSpecFlux',

        # Zero Crossing Rate (2개)
        'meanZCR', 'stdZCR',

        # MFCC (13개 계수)
        *[f'meanMFCC_{i:02d}' for i in range(1,14)],
        *[f'stdMFCC_{i:02d}' for i in range(1,14)],

        # Mel Spectrogram (10개)
        *[f'meanMelScale_{i:02d}' for i in range(1,11)],
        *[f'stdMelScale_{i:02d}' for i in range(1,11)],

        # Rhythmic Content Features (2개)
        'meanTempo', 'stdTempo',
    
        # Chroma STFT (12개)
        *[f'meanChromaSTFT_{i:02d}' for i in range(1,13)],
        *[f'stdChromaSTFT_{i:02d}' for i in range(1,13)],
    
        # Tonnetz (6개)
        *[f'meanTonnetz_{i:02d}' for i in range(1,7)],
        *[f'stdTonnetz_{i:02d}' for i in range(1,7)],
    ]

    # 폴더-특징 매핑 시스템
    FEATURE_MAP = {
        # 음색 관련 (Cover_Timbre)
        'spectral_centroid': {'prefix': 'Cover_Timbre', 'stats': ['mean', 'std']},
        'spectral_bandwidth': {'prefix': 'Cover_Timbre', 'stats': ['mean', 'std']},
        'spectral_contrast': {'prefix': 'Cover_Timbre', 'stats': ['mean', 'std']},
        'spectral_rolloff': {'prefix': 'Cover_Timbre', 'stats': ['mean', 'std']},
        'spectral_flux': {'prefix': 'Cover_Rhythm', 'stats': ['mean', 'std']},
        'zero_crossing': {'prefix': 'Cover_Timbre', 'stats': ['mean', 'std']},
        'mfcc': {'prefix': 'Cover_Timbre', 'stats': ['mean', 'std']},
        'mel_scale': {'prefix': 'Cover_Timbre', 'stats': ['mean', 'std']},
    
        # 리듬 관련 (Cover_Rhythm)
        'tempo': {'prefix': 'Cover_Rhythm', 'stats': ['mean', 'std']},
    
        # 악기 특성 (Cover_Instrument)
        'chroma': {'prefix': 'Cover_Instrument', 'stats': ['mean', 'std']},
    
        # 조성 특성 (Cover_Genre)
        'tonnetz': {'prefix': 'Cover_Genre', 'stats': ['mean', 'std']}
    }
