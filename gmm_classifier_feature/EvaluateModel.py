"""Paired song-level CV with training-fold-only scaling and fixed GMMs."""
import csv
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from .config import FEATURE_NAMES, GMM_PARAMS, RANDOM_STATE, SCHEMA_VERSION, TARGET_SAMPLE_RATE
from .GMM_Classification import train_gmm_models, predict_with_gmms
from .CreateFeatureDataset import file_digest


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        X, y, ids = archive['X'], archive['y'], archive['song_id']
        metadata = json.loads(str(archive['metadata']))
        names = archive['feature_names'].tolist()
    if X.ndim != 2 or X.shape != (len(y), 96) or y.ndim != 1 or ids.shape != y.shape:
        raise ValueError('X / y / song_id shape mismatch')
    if not len(y) or len(set(ids)) != len(ids) or not np.isfinite(X).all():
        raise ValueError('Empty, duplicate-song or non-finite dataset')
    if names != FEATURE_NAMES or metadata.get('schema') != SCHEMA_VERSION:
        raise ValueError('Feature schema mismatch; rebuild features')
    if metadata.get('scaled') is not False or metadata.get('sample_rate') != TARGET_SAMPLE_RATE:
        raise ValueError('Expected unscaled features at TARGET_SAMPLE_RATE')
    return dict(X=X, y=y, ids=ids, metadata=metadata)


def balanced_indices(y, ids):
    """Reproduce equal-class intent with a documented deterministic subset."""
    rng = np.random.default_rng(RANDOM_STATE)
    classes, counts = np.unique(y, return_counts=True)
    chosen = []
    for label in classes:
        candidates = np.flatnonzero(y == label)
        candidates = candidates[np.argsort(ids[candidates])]
        chosen.extend(rng.choice(candidates, size=int(counts.min()), replace=False))
    return np.asarray(sorted(chosen, key=lambda i: ids[i]), dtype=int)


def evaluate_cv(X, y, splits):
    rows, prediction = [], np.empty_like(y)
    for fold, (train, validation) in enumerate(splits, 1):
        scaler = MinMaxScaler()
        train_X = scaler.fit_transform(X[train])
        validation_X = scaler.transform(X[validation])
        models = train_gmm_models(train_X, y[train])
        predicted = predict_with_gmms(models, validation_X)
        prediction[validation] = predicted
        rows.append(dict(fold=fold, accuracy=float(accuracy_score(y[validation], predicted)),
                         macro_f1=float(f1_score(y[validation], predicted, average='macro', zero_division=0)),
                         train_count=len(train), validation_count=len(validation),
                         all_gmms_converged=all(m.converged_ for m in models.values())))
    return rows, prediction


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(baseline_path, feature_aware_path, output_dir):
    datasets = {name: load_dataset(path) for name, path in
                [('baseline', baseline_path), ('feature_aware', feature_aware_path)]}
    baseline, aware = datasets.values()
    for name, data in datasets.items():
        if data['metadata'].get('condition') != name:
            raise ValueError(f'Wrong condition archive: {name}')
    for key in ('seconds', 'baseline_source', 'feature_sources', 'source_manifest_sha256'):
        if baseline['metadata'].get(key) != aware['metadata'].get(key):
            raise ValueError(f'Paired provenance mismatch: {key}')
    if not np.array_equal(baseline['ids'], aware['ids']) or not np.array_equal(baseline['y'], aware['y']):
        raise ValueError('Both experiments must contain identical ordered songs and labels')
    indices = balanced_indices(baseline['y'], baseline['ids'])
    y, ids = baseline['y'][indices], baseline['ids'][indices]
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < 10:
        raise ValueError('10-fold evaluation requires at least 2 classes and 10 complete songs per class')
    splits = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE).split(ids, y))
    # Import plotting before expensive fitting so a missing dependency fails early.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    all_rows, summaries, matrices = [], [], {}
    for name, data in datasets.items():
        rows, predicted = evaluate_cv(data['X'][indices], y, splits)
        all_rows.extend(dict(condition=name, **row) for row in rows)
        summary = dict(condition=name)
        for metric in ('accuracy', 'macro_f1'):
            values = [row[metric] for row in rows]
            summary[f'{metric}_mean'] = float(np.mean(values))
            summary[f'{metric}_std'] = float(np.std(values))
        summary['nonconverged_folds'] = sum(not row['all_gmms_converged'] for row in rows)
        summaries.append(summary)
        matrices[name] = confusion_matrix(y, predicted, labels=classes).tolist()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    original_classes, original_counts = np.unique(baseline['y'], return_counts=True)
    metrics = dict(status='measured', protocol='portfolio_refactoring', n_splits=10,
                   random_state=RANDOM_STATE, gmm=GMM_PARAMS,
                   balancing='deterministic undersampling to minimum class count before CV',
                   original_class_counts=dict(zip(original_classes.tolist(), original_counts.tolist())),
                   evaluated_class_counts=dict(zip(classes.tolist(), counts.tolist())),
                   std_ddof=0, metadata=baseline['metadata'],
                   input_sha256={'baseline': file_digest(baseline_path),
                                 'feature_aware': file_digest(feature_aware_path)},
                   versions={name: importlib.metadata.version(name) for name in
                             ('numpy', 'librosa', 'scikit-learn', 'soundfile', 'matplotlib')},
                   metrics=summaries, classes=classes.tolist(), confusion_matrices=matrices,
                   validation_song_ids={str(i): ids[validation].tolist()
                                        for i, (_, validation) in enumerate(splits, 1)})
    (output / 'cv_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    write_csv(output / 'fold_metrics.csv', all_rows)
    write_csv(output / 'feature_source_ablation.csv', summaries)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, matrix) in zip(axes, matrices.items()):
        ConfusionMatrixDisplay(np.asarray(matrix), display_labels=classes).plot(ax=ax, colorbar=False)
        ax.set_title(name)
        ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=150)
    plt.close(fig)
    return summaries
