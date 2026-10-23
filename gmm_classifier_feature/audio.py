"""Mono/resampling and onset-based 60-second sampling."""
from pathlib import Path
import librosa
import numpy as np
from .config import TARGET_SAMPLE_RATE, SAMPLE_SECONDS, HOP_SIZE

AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3', '.ogg', '.aif', '.aiff', '.m4a'}


def audio_files(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'Audio directory does not exist: {root}')
    return sorted((p for p in root.rglob('*') if p.suffix.lower() in AUDIO_EXTENSIONS),
                  key=lambda p: p.relative_to(root).as_posix())


def load_audio(path):
    y, sr = librosa.load(path, sr=TARGET_SAMPLE_RATE, mono=True)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError(f'Empty or non-finite audio: {path}')
    return y, sr


def sample_audio(y, sr=TARGET_SAMPLE_RATE):
    """Strongest detected onset; last 60s if needed; start at zero without onsets.

    Short audio is right-padded with silence.
    Returns the sample and its start offset in samples.
    """
    if sr != TARGET_SAMPLE_RATE or np.ndim(y) != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('Expected nonempty finite mono audio at TARGET_SAMPLE_RATE')
    length = SAMPLE_SECONDS * sr
    if len(y) <= length:
        return np.pad(y, (0, length - len(y))), 0
    # Preserve supplied music_sampling.py: backtracked onsets, median strength.
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_SIZE, backtrack=True)
    strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_SIZE, aggregate=np.median)
    start = int(onsets[np.argmax(strength[onsets])] * HOP_SIZE) if len(onsets) else 0
    start = min(start, len(y) - length)
    return y[start:start + length], start
