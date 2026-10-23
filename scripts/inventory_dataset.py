"""Decode a historical dataset and audit paths, content, and complete source sets."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from gmm_classifier_feature.audio import audio_files
from gmm_classifier_feature.config import FEATURE_SOURCES, SAMPLE_SECONDS, DURATION_TOLERANCE_SECONDS
from gmm_classifier_feature.CreateFeatureDataset import extract_song_id, file_digest, select_sources, GENRES


def inspect_file(path, root):
    relative = path.relative_to(root).as_posix()
    row = dict(relative_path=relative, genre='', song_id='', source_family='', variant='',
               file_size=None, duration=None, sample_rate=None, channels=None,
               decode_success=False, finite=False, empty=True, duration_valid=False,
               sha256='', pcm_sha256='', duplicate=False,
               cross_song_duplicate=False, error='')
    try:
        row['song_id'] = extract_song_id(path)
        parts = path.stem.split('_')
        row.update(genre=parts[1], source_family='_'.join(parts[3:5]), variant='_'.join(parts[5:]))
        row.update(file_size=path.stat().st_size, sha256=file_digest(path))
        pcm = hashlib.sha256()
        with sf.SoundFile(path) as audio:
            row.update(duration=audio.frames / audio.samplerate,
                       sample_rate=audio.samplerate, channels=audio.channels)
            pcm.update(f'{audio.samplerate}:{audio.channels}:'.encode())
            frames, finite = 0, True
            for block in audio.blocks(blocksize=65536, dtype='float32', always_2d=True):
                frames += len(block)
                finite &= bool(np.isfinite(block).all())
                pcm.update(block.astype('<f4', copy=False).tobytes())
            if frames != audio.frames:
                raise ValueError('Decoded frame count differs from header')
        row.update(decode_success=True, finite=finite, empty=frames == 0,
                   duration_valid=abs(row['duration'] - SAMPLE_SECONDS) <= DURATION_TOLERANCE_SECONDS,
                   pcm_sha256=pcm.hexdigest())
    except Exception as exc:
        row['error'] = str(exc).replace(str(root), '<dataset>')
    return row


def write_aggregate(rows, output):
    """Publish compact counts; keep the complete path/hash audit in local artifacts."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['genre'], row['source_family'], row['variant']].append(row)
    aggregates = []
    for (genre, family, variant), group in sorted(groups.items()):
        durations = [r['duration'] for r in group if r['duration'] is not None]
        aggregates.append(dict(
            genre=genre, source_family=family, variant=variant,
            audio_file_count=len(group), song_count=len({r['song_id'] for r in group}),
            total_bytes=sum(r['file_size'] or 0 for r in group),
            decode_success_count=sum(r['decode_success'] for r in group),
            decode_failure_count=sum(not r['decode_success'] for r in group),
            duration_min=min(durations) if durations else None,
            duration_max=max(durations) if durations else None,
            sample_rates=';'.join(sorted({str(r['sample_rate']) for r in group})),
            channels=';'.join(sorted({str(r['channels']) for r in group})),
            duplicate_file_count=sum(r['duplicate'] for r in group),
            cross_song_duplicate_file_count=sum(r['cross_song_duplicate'] for r in group)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'dataset_inventory.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(aggregates)


def inventory(root, output):
    root, output = Path(root), Path(output)
    rows = [inspect_file(p, root) for p in audio_files(root)]
    if not rows:
        raise ValueError('No audio files found; verify local dataset availability')
    by_hash = defaultdict(list)
    for row in rows:
        if row['pcm_sha256']:
            by_hash[row['pcm_sha256']].append(row)
    for group in by_hash.values():
        for row in group:
            row['duplicate'] = len(group) > 1
            row['cross_song_duplicate'] = len({r['song_id'] for r in group}) > 1
    by_path = {row['relative_path']: row for row in rows}
    selections = select_sources(root, 'Cover_Timbre')
    cohort, excluded = [], []
    required = set(FEATURE_SOURCES.values())
    for song_id, mapping in sorted(selections.items()):
        reasons = [f'Missing {s}' for s in sorted(required - mapping.keys())]
        for source, path in sorted(mapping.items()):
            row = by_path[path.relative_to(root).as_posix()]
            if not row['decode_success']: reasons.append(f'Decode failure: {source}')
            elif row['empty'] or not row['finite']: reasons.append(f'Invalid samples: {source}')
            elif not row['duration_valid']: reasons.append(f'Invalid duration: {source}')
        # Duplicate diagnostics do not silently change the predefined cohort.
        # Cross-song content must be reviewed before ordinary song-level CV.
        if reasons:
            excluded.append(dict(song_id=song_id, reasons=reasons))
        else:
            cohort.append(song_id)
    counts = lambda values: dict(sorted(Counter(values).items()))
    duration = np.array([r['duration'] for r in rows if r['duration'] is not None])
    summary = dict(audio_file_count=len(rows), discovered_song_count=len(selections),
                   complete_song_count=len(cohort), excluded_song_count=len(excluded),
                   discovered_class_counts=counts(s.split('_')[1] for s in selections),
                   complete_class_counts=counts(s.split('_')[1] for s in cohort),
                   source_family_counts=counts(r['source_family'] for r in rows),
                   source_variant_counts=counts(r['source_family']+'_'+r['variant'] for r in rows),
                   sample_rate_counts=counts(str(r['sample_rate']) for r in rows),
                   channel_counts=counts(str(r['channels']) for r in rows),
                   duration_seconds=(dict(zip(('min','p25','median','p75','max'),
                                              np.quantile(duration,[0,.25,.5,.75,1]).tolist()))
                                     if len(duration) else None),
                   duration_tolerance_seconds=DURATION_TOLERANCE_SECONDS,
                   invalid_duration_count=sum(not r['duration_valid'] for r in rows),
                   decode_failure_count=sum(not r['decode_success'] for r in rows),
                   nonfinite_count=sum(r['decode_success'] and not r['finite'] for r in rows),
                   empty_count=sum(r['empty'] for r in rows),
                   duplicate_file_count=sum(r['duplicate'] for r in rows),
                   cross_song_duplicate_file_count=sum(r['cross_song_duplicate'] for r in rows),
                   unexpected_genres=sorted({r['genre'] for r in rows}-set(GENRES)),
                   exclusion_reasons=counts(reason for x in excluded for reason in x['reasons']),
                   complete_song_ids=cohort, excluded=excluded)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'dataset_inventory.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    summary['file_inventory_sha256']=file_digest(output/'dataset_inventory.csv')
    (output/'dataset_inventory.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('complete_song_ids','excluded')},indent=2))
    return rows, summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--summary-output', help='Separate directory for a compact publishable CSV')
    args=parser.parse_args()
    if args.summary_output and Path(args.summary_output).resolve() == Path(args.output).resolve():
        parser.error('--summary-output must differ from --output to preserve the file-level audit')
    rows, summary = inventory(args.input,args.output)
    if args.summary_output:
        write_aggregate(rows, args.summary_output)
