"""Small regression suite: sampling, feature schema, dataset, evaluation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import librosa
import numpy as np
import soundfile as sf
from sklearn.preprocessing import MinMaxScaler
from gmm_classifier_feature import audio
from gmm_classifier_feature.config import FEATURE_SIZES, FEATURE_NAMES, TARGET_SAMPLE_RATE as SR
from gmm_classifier_feature.Utilities import extract_feature
from gmm_classifier_feature.CreateFeatureDataset import build_features, select_sources, select_subset, GENRES
from gmm_classifier_feature.EvaluateModel import evaluate_cv, balanced_indices, load_dataset


class SamplingTests(unittest.TestCase):
    def test_length_end_fallback_and_no_onset(self):
        y = np.arange(70 * SR, dtype=float)
        with patch.object(librosa.onset, 'onset_detect', return_value=np.array([2100])), \
             patch.object(librosa.onset, 'onset_strength', return_value=np.ones(2200)):
            segment, start = audio.sample_audio(y)
        self.assertEqual(start, 10 * SR)
        np.testing.assert_array_equal(segment, y[-60 * SR:])
        with patch.object(librosa.onset, 'onset_detect', return_value=np.array([], dtype=int)), \
             patch.object(librosa.onset, 'onset_strength', return_value=np.zeros(2200)):
            segment, start = audio.sample_audio(y)
        self.assertEqual(start, 0)
        self.assertEqual(len(segment), 60 * SR)
        short, start = audio.sample_audio(np.ones(SR))
        self.assertEqual(len(short), 60 * SR)
        self.assertFalse(short[SR:].any())

    def test_matches_supplied_sampling_algorithm(self):
        y = np.zeros(70 * SR)
        rng = np.random.default_rng(42)
        for second in (2, 8, 40, 66):
            y[second*SR:second*SR+2000] = rng.normal(size=2000)
        frames = librosa.onset.onset_detect(y=y, sr=SR, backtrack=True)
        strength = librosa.onset.onset_strength(y=y, sr=SR, aggregate=np.median)
        start_time = librosa.frames_to_time(frames, sr=SR)[np.argmax(strength[frames])]
        start_time = min(start_time, 10)
        segment, start = audio.sample_audio(y)
        self.assertAlmostEqual(start, int(start_time * SR), delta=1)
        self.assertEqual(len(segment), 60 * SR)

    def test_load_resamples_stereo(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'stereo.wav'
            sf.write(path, np.zeros((22050, 2)), 22050)
            y, sr = audio.load_audio(path)
        self.assertEqual((y.shape, sr), ((SR,), SR))


class FeatureTests(unittest.TestCase):
    def test_finite_96d_and_matrix_statistics(self):
        y = np.random.default_rng(42).normal(0, .1, 3 * SR)
        vectors = {name: extract_feature(y, name) for name in FEATURE_SIZES}
        vector = np.concatenate(list(vectors.values()))
        self.assertEqual(vector.shape, (96,))
        self.assertEqual(len(FEATURE_NAMES), 96)
        self.assertEqual(len(set(FEATURE_NAMES)), 96)
        self.assertTrue(np.isfinite(vector).all())
        # Direct coefficient-wise reference calculation verifies values and interleaving.
        matrix = librosa.feature.mfcc(y=y, sr=SR, n_fft=2048, hop_length=512)
        expected = [value for row in matrix[:13] for value in (np.mean(row), np.std(row))]
        np.testing.assert_allclose(vectors['mfcc'], expected)
        self.assertEqual(vectors['tempo'][1], 0)


class DatasetTests(unittest.TestCase):
    def test_org_and_strict_a_variant_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for genre in GENRES:
                for song in ('00001', '00002', '00003'):
                    stem = f'Similar_{genre}_{song}'
                    (root / f'{stem}_Org.wav').touch()
                    for source in ('Timbre', 'Rhythm', 'Instrument', 'Genre'):
                        for variant in ('A', 'C'):
                            (root / f'{stem}_Cover_{source}_{variant}.wav').touch()
            songs = select_sources(root, 'Org', variant='A')
            subset = select_subset(songs, 'Org', 2)
            self.assertEqual(len(subset), 12)
            self.assertEqual(list(subset), list(select_subset(dict(reversed(list(songs.items()))), 'Org', 2)))
            self.assertTrue(all(mapping['Org'].name.endswith('_Org.wav') for mapping in subset.values()))
            self.assertTrue(all(path.name.endswith('_A.wav') for mapping in subset.values()
                                for source, path in mapping.items() if source.startswith('Cover_')))
            (root / 'Similar_Rock_00001_Cover_Timbre_A.wav').unlink()
            songs = select_sources(root, 'Org', variant='A')
            self.assertNotIn('Cover_Timbre', songs['Similar_Rock_00001'])
            with self.assertRaisesRegex(ValueError, 'Rock: need 3'):
                select_subset(songs, 'Org', 3)

    def test_deterministic_mapping_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / 'audio', Path(tmp) / 'features'
            root.mkdir()
            for song in ('00001', '00002'):
                for source in ('Timbre', 'Rhythm', 'Instrument', 'Genre'):
                    for variant in ('B', 'A'):
                        (root / f'Similar_Rock_{song}_Cover_{source}_{variant}.wav').touch()
            mapping = select_sources(root, 'Cover_Timbre')
            self.assertTrue(all(p.name.endswith('_A.wav') for m in mapping.values() for p in m.values()))
            def fake_load(path):
                return np.full(60 * SR, 1 if '00001' in str(path) else 2), SR
            def fake_feature(y, feature):
                if y[0] == 1 and feature == 'tempo':
                    raise ValueError('injected mid-song failure')
                return np.zeros(2 * FEATURE_SIZES[feature])
            with patch('gmm_classifier_feature.CreateFeatureDataset.load_audio', side_effect=fake_load), \
                 patch('gmm_classifier_feature.CreateFeatureDataset.extract_feature', side_effect=fake_feature):
                self.assertEqual(build_features(root, out, 'Cover_Timbre'), (1, 1))
            for name in ('baseline', 'feature_aware'):
                data = load_dataset(out / f'{name}.npz')
                self.assertEqual(data['X'].shape, (1, 96))
                self.assertEqual(data['ids'].tolist(), ['Similar_Rock_00002'])
            self.assertEqual(len(json.loads((out / 'source_manifest.json').read_text())['excluded']), 1)


class EvaluationTests(unittest.TestCase):
    def test_scaler_fits_only_train_and_keeps_validation_extremes(self):
        X = np.array([[0.], [2.], [100.], [102.]])
        y = np.array(['A', 'B', 'A', 'B'])
        splits = [(np.array([0, 1]), np.array([2, 3])),
                  (np.array([2, 3]), np.array([0, 1]))]
        fit_rows, validation_rows = [], []
        class RecordingScaler(MinMaxScaler):
            def fit(self, values, y=None):
                fit_rows.append(values.copy())
                return super().fit(values, y)
        def predict(models, values):
            validation_rows.append(values.copy())
            return np.array(['A', 'B'])
        with patch('gmm_classifier_feature.EvaluateModel.MinMaxScaler', RecordingScaler), \
             patch('gmm_classifier_feature.EvaluateModel.train_gmm_models', return_value={}), \
             patch('gmm_classifier_feature.EvaluateModel.predict_with_gmms', side_effect=predict):
            evaluate_cv(X, y, splits)
        np.testing.assert_array_equal(fit_rows[0], X[:2])
        np.testing.assert_array_equal(fit_rows[1], X[2:])
        self.assertGreater(validation_rows[0].min(), 1)
        self.assertLess(validation_rows[1].max(), 0)

    def test_balancing_is_order_independent(self):
        y = np.array(['A']*12 + ['B']*10)
        ids = np.array([f'song{i:02d}' for i in range(22)])
        indices = balanced_indices(y, ids)
        self.assertEqual(np.unique(y[indices], return_counts=True)[1].tolist(), [10, 10])
        reversed_indices = balanced_indices(y[::-1], ids[::-1])
        self.assertEqual(ids[indices].tolist(), ids[::-1][reversed_indices].tolist())


if __name__ == '__main__':
    unittest.main()
