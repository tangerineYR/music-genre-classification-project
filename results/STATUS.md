# Historical 60-second dataset evaluation

Evaluation date: 2026-10-23. Protocol: `portfolio_refactoring`.

Single-source Cover_Timbre baseline vs feature-aware source mapping. Both conditions use the same songs, labels, deterministic balancing result, and 10 stratified folds.

## Dataset

The complete historical inventory contains **5,300 WAV files** and **590 songs**. All files were decoded before feature extraction. **587 songs** have all four required source families and passed validation; **3 songs** were excluded.

| Genre | Discovered | Complete | Excluded | Evaluated |
| --- | ---: | ---: | ---: | ---: |
| Ballade | 217 | 216 | 1 | 39 |
| Dance | 128 | 127 | 1 | 39 |
| Hiphop | 39 | 39 | 0 | 39 |
| RnB | 39 | 39 | 0 | 39 |
| Rock | 39 | 39 | 0 | 39 |
| Trot | 128 | 127 | 1 | 39 |

Excluded songs:

- `Similar_Ballade_00056`: Missing Cover_Genre, Missing Cover_Rhythm.
- `Similar_Dance_00087`: Missing Cover_Genre.
- `Similar_Trot_00610`: Missing Cover_Genre.

All 5,300 files are mono at 16,000 Hz. Duration range: 59.9999375–60.0000625 seconds; median 60.0000000 seconds. A 10 ms tolerance accepts sample rounding without padding or segment reselection.

Decode failures: 0; invalid durations: 0; empty files: 0; non-finite audio: 0. 77 files belong to within-song decoded-content duplicate groups; cross-song duplicates: 0. Within-song duplicates remain diagnostics and do not create additional samples.

### Source family distribution

| Source family | Files |
| --- | ---: |
| Cover_Arrangement | 586 |
| Cover_Genre | 587 |
| Cover_Instrument | 1180 |
| Cover_Rhythm | 589 |
| Cover_Tempo | 1180 |
| Cover_Timbre | 1178 |

`Cover_Tempo` and `Cover_Arrangement` are inventoried but are not used by the 96D evaluation. `Org` is absent. Each required family uses its lexicographically first dataset-relative path; variants were not selected using performance. Source family and variant counts are available in [dataset_inventory.csv](dataset_inventory.csv).

## Protocol and validation

Feature arrays contain 587 paired songs × 96 unscaled finite values. Feature names are unique and both conditions have identical ordered song IDs and labels. Extraction uses mono 16 kHz through `load_audio()` and retains the historical segments.

Deterministic undersampling selects 234 songs (39 per genre). Validation fold sizes: 24, 24, 24, 24, 23, 23, 23, 23, 23, 23. Each validation song appears exactly once. Each fold fits MinMaxScaler only on its training rows.

GMM: 7 components, diagonal covariance, max_iter=400, random_state=42, init_params=kmeans. CV: StratifiedKFold(n_splits=10, shuffle=True, random_state=42). No hyperparameter or variant search was performed.

## Measured results

| Condition | Accuracy mean ± std | Macro F1 mean ± std | Nonconverged folds |
| --- | ---: | ---: | ---: |
| Cover_Timbre baseline | 0.294746 ± 0.102069 | 0.270577 ± 0.093847 | 0 / 10 |
| Feature-aware mapping | 0.303080 ± 0.071768 | 0.290179 ± 0.065378 | 0 / 10 |

Δ Accuracy (feature-aware − baseline): **+0.008333** (+0.833 percentage points). Δ Macro F1: **+0.019602**. Standard deviations use ddof=0; metrics use the 0–1 scale.

### Fold results

| Fold | Baseline accuracy | Feature-aware accuracy | Baseline macro F1 | Feature-aware macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.500000 | 0.291667 | 0.464286 | 0.287458 |
| 2 | 0.166667 | 0.333333 | 0.165079 | 0.311111 |
| 3 | 0.208333 | 0.375000 | 0.164286 | 0.359007 |
| 4 | 0.333333 | 0.291667 | 0.288360 | 0.288781 |
| 5 | 0.434783 | 0.391304 | 0.411447 | 0.368254 |
| 6 | 0.173913 | 0.260870 | 0.198413 | 0.280952 |
| 7 | 0.304348 | 0.260870 | 0.276150 | 0.220238 |
| 8 | 0.260870 | 0.347826 | 0.222222 | 0.298413 |
| 9 | 0.304348 | 0.347826 | 0.263034 | 0.348942 |
| 10 | 0.260870 | 0.130435 | 0.252493 | 0.138636 |

![Paired confusion matrices](confusion_matrix.png)

Both matrices use the same color scale and report counts; rows are true labels.

Largest off-diagonal confusion counts:

- Baseline: Dance → Hiphop: 15; Rock → Hiphop: 11; RnB → Ballade: 11.
- Feature-aware: Ballade → RnB: 14; Rock → Hiphop: 10; RnB → Trot: 10.

## Interpretation and limitations

Feature-aware mapping has slightly higher mean accuracy and macro F1 on this historical balanced cohort. The accuracy increase is only 0.83 percentage points, compared with fold standard deviations of 10.21 and 7.18 percentage points. The observed difference is limited and does not provide strong support for a reliable pipeline improvement. No causal effect or statistical significance is claimed.

The audio consists of 2025 historical sampled segments. Different Cover sources are not guaranteed to share the same exact timestamps. This is a paired comparison of historical single-source and feature-aware pipelines, not a causal estimate of source type. The complete-case requirement and deterministic balancing restrict the population. These results are not an AI Hub full-dataset benchmark or a result for a separate team model.

## Reproducibility

Exact class counts, validation song IDs, paired fold differences, input/manifest hashes, dependency versions, schema and model parameters are recorded in [cv_metrics.json](cv_metrics.json). [fold_metrics.csv](fold_metrics.csv) contains per-fold convergence and scores; [feature_source_ablation.csv](feature_source_ablation.csv) contains condition summaries. The inventory CSV is an aggregate by genre, family and variant. Full file-level audits and source manifests are generated as local artifacts by the [documented commands](../docs/running.md).
