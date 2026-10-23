"""Restore onset-based sampling using explicit input/output directories."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import soundfile as sf
from gmm_classifier_feature.audio import audio_files, load_audio, sample_audio
from gmm_classifier_feature.config import TARGET_SAMPLE_RATE, SAMPLE_SECONDS
from gmm_classifier_feature.CreateFeatureDataset import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error('Input and output directories must be separate, non-nested trees')
    paths = audio_files(source)
    if not paths:
        parser.error('No audio found')
    targets = [output / p.relative_to(source).with_suffix('.wav') for p in paths]
    if len(set(targets)) != len(targets) or any(p.exists() for p in targets):
        parser.error('Output collision; use a fresh output directory')
    records = []
    for path, target in zip(paths, targets):
        y, sr = load_audio(path)
        sampled, start = sample_audio(y, sr)
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, sampled, sr, subtype='PCM_16')
        records.append(dict(input=path.relative_to(source).as_posix(),
                            output=target.relative_to(output).as_posix(),
                            input_sha256=file_digest(path), start_sample=start,
                            input_samples=len(y), padded_samples=max(0, len(sampled)-len(y))))
    (output / 'sampling_manifest.json').write_text(json.dumps(dict(
        sample_rate=TARGET_SAMPLE_RATE, seconds=SAMPLE_SECONDS, subtype='PCM_16',
        policy='strongest_detected_onset_then_end_fallback', samples=records), indent=2))
    print(f'Sampled {len(records)} files to {output}')


if __name__ == '__main__':
    main()
