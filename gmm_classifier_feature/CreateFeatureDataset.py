"""Build paired, unscaled 96D datasets and record all source selections."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .audio import audio_files, load_audio
from .config import (FEATURE_SOURCES, FEATURE_NAMES, TARGET_SAMPLE_RATE,
                     SAMPLE_SECONDS, SCHEMA_VERSION, RANDOM_STATE)
from .Utilities import extract_feature


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract_song_id(path):
    parts = Path(path).stem.split('_')
    if len(parts) < 4 or parts[0] != 'Similar' or not parts[2].isdigit():
        raise ValueError(f'Expected Similar_<genre>_<id>_<source>: {path}')
    return '_'.join(parts[:3])


def select_sources(root, baseline_source, variant=None):
    """Exact source token prefix, then lexical relative-path priority for variants."""
    songs = {}
    required = set(FEATURE_SOURCES.values()) | {baseline_source}
    for path in audio_files(root):
        song_id = extract_song_id(path)
        source_tokens = Path(path).stem.split('_')[3:]
        song = songs.setdefault(song_id, {})
        for source in sorted(required):
            tokens = source.split('_')
            if source_tokens[:len(tokens)] == tokens:
                if variant is not None and source.startswith('Cover_'):
                    if source_tokens[len(tokens):] != [variant]:
                        continue
                song.setdefault(source, path)
    return songs


GENRES = ('Ballade', 'Dance', 'Hiphop', 'RnB', 'Rock', 'Trot')


def select_subset(songs, baseline_source, songs_per_genre):
    """Select complete source sets before audio loading; fixed seed, six genres."""
    if songs_per_genre < 1:
        raise ValueError('songs_per_genre must be positive')
    required = set(FEATURE_SOURCES.values()) | {baseline_source}
    rng = np.random.default_rng(RANDOM_STATE)
    chosen = {}
    for genre in GENRES:
        candidates = sorted(song for song, sources in songs.items()
                            if song.split('_')[1] == genre and required <= sources.keys())
        if len(candidates) < songs_per_genre:
            raise ValueError(f'{genre}: need {songs_per_genre} complete songs, found {len(candidates)}')
        for song in rng.choice(candidates, songs_per_genre, replace=False):
            chosen[str(song)] = songs[song]
    return dict(sorted(chosen.items()))


def build_features(input_dir, output_dir, baseline_source, seconds=60,
                   variant=None, songs_per_genre=None):
    if seconds not in (5, SAMPLE_SECONDS):
        raise ValueError('Feature duration must be 5 or 60 seconds')
    root, output = Path(input_dir), Path(output_dir)
    if any((output / name).exists() for name in
           ('baseline.npz', 'feature_aware.npz', 'source_manifest.json')):
        raise ValueError('Feature outputs already exist; use a fresh output directory')
    if not baseline_source or any(not token for token in baseline_source.split('_')):
        raise ValueError('Baseline source must be a nonempty filename token prefix')
    selected = select_sources(root, baseline_source, variant)
    discovered_count = len(selected)
    if songs_per_genre is not None:
        selected = select_subset(selected, baseline_source, songs_per_genre)
    rows = {'baseline': [], 'feature_aware': []}
    labels, ids, records, failures = [], [], [], []
    required = set(FEATURE_SOURCES.values()) | {baseline_source}
    for song_id, mapping in sorted(selected.items()):
        missing = sorted(required - mapping.keys())
        if missing:
            failures.append(dict(song_id=song_id, error=f'Missing sources: {missing}'))
            continue
        try:
            audio, sources = {}, {}
            for source in sorted(required):
                path = mapping[source]
                y, sr = load_audio(path)
                if len(y) != SAMPLE_SECONDS * sr:
                    raise ValueError(f'Expected processed 60s audio: {path}; run sample_audio.py')
                audio[source] = y[:seconds * sr]
                sources[source] = dict(path=path.relative_to(root).as_posix(),
                                       sha256=file_digest(path))
            cache = {}
            pending = {}
            for condition in rows:
                vectors = []
                for feature, aware_source in FEATURE_SOURCES.items():
                    source = baseline_source if condition == 'baseline' else aware_source
                    key = (source, feature)
                    if key not in cache:
                        cache[key] = extract_feature(audio[source], feature)
                    vectors.append(cache[key])
                pending[condition] = np.concatenate(vectors)
            # Atomic paired commit: neither condition nor labels survive partial failure.
            for condition in rows:
                rows[condition].append(pending[condition])
            labels.append(song_id.split('_')[1])
            ids.append(song_id)
            records.append(dict(song_id=song_id, sources=sources))
        except Exception as exc:
            failures.append(dict(song_id=song_id, error=str(exc)))
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(schema=SCHEMA_VERSION, sample_rate=TARGET_SAMPLE_RATE,
                    seconds=seconds, scaled=False, baseline_source=baseline_source,
                    feature_sources=FEATURE_SOURCES,
                    variant_policy='lexicographically_first_relative_path', variant=variant,
                    songs_per_genre=songs_per_genre, subset_seed=RANDOM_STATE,
                    selected_song_ids=list(sorted(selected)),
                    discovered_song_count=discovered_count, excluded_song_count=len(failures),
                    complete_song_count=len(ids),
                    samples=records, excluded=failures)
    (output / 'source_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if songs_per_genre is not None and failures:
        raise ValueError('Subset extraction failed; see source_manifest.json. No archives written.')
    if not ids:
        raise ValueError('No complete paired samples; see source_manifest.json')
    manifest_hash = file_digest(output / 'source_manifest.json')
    for condition, vectors in rows.items():
        np.savez_compressed(output / f'{condition}.npz', X=np.asarray(vectors),
                            y=np.asarray(labels), song_id=np.asarray(ids),
                            feature_names=np.asarray(FEATURE_NAMES),
                            metadata=json.dumps({**{k: v for k, v in metadata.items()
                                                    if k not in ('samples', 'excluded')},
                                                 'condition': condition,
                                                 'source_manifest_sha256': manifest_hash}))
    return len(ids), len(failures)
