"""Paired protocol checks using synthetic arrays, never reported as real metrics."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from gmm_classifier_feature.EvaluateModel import evaluate
from gmm_classifier_feature.config import FEATURE_SOURCES


class PairedEvaluationTests(unittest.TestCase):
    def test_shared_folds_and_audited_cohort_guard(self):
        ids=np.array([f'Similar_{g}_{i:05d}' for g in ('Rock','Trot') for i in range(10)])
        labels=np.array(['Rock']*10+['Trot']*10)
        metadata=dict(baseline_source='Cover_Timbre',seconds=60,sample_rate=16000,
                      feature_sources=FEATURE_SOURCES,source_manifest_sha256='test-manifest')
        datasets={name:dict(X=np.zeros((20,96)),y=labels,ids=ids,
                           metadata=dict(metadata,condition=name))
                  for name in ('baseline','feature_aware')}
        observed=[]
        def cv(X,y,splits):
            observed.append(splits)
            rows=[dict(fold=i+1,accuracy=1.,macro_f1=1.,train_count=len(train),
                       validation_count=len(val),all_gmms_converged=True)
                  for i,(train,val) in enumerate(splits)]
            return rows,y.copy()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in datasets:(root/name).write_bytes(b'test input hash')
            audit=root/'inventory.json'
            summary=dict(complete_song_ids=ids.tolist(),cross_song_duplicate_file_count=0,
                         discovered_song_count=20,complete_song_count=20,excluded_song_count=0,
                         file_inventory_sha256='test-inventory')
            audit.write_text(json.dumps(summary))
            with patch('gmm_classifier_feature.EvaluateModel.load_dataset',
                       side_effect=lambda path:datasets[Path(path).name]), \
                 patch('gmm_classifier_feature.EvaluateModel.evaluate_cv',side_effect=cv):
                evaluate(root/'baseline',root/'feature_aware',root/'results',audit)
                self.assertIs(observed[0],observed[1])
                validation=np.concatenate([v for _,v in observed[0]])
                self.assertEqual(sorted(validation.tolist()),list(range(20)))
                for train,val in observed[0]:self.assertFalse(set(train)&set(val))
                results=json.loads((root/'results'/'cv_metrics.json').read_text())
                self.assertEqual(results['delta'],{'accuracy':0.,'macro_f1':0.})
                summary['complete_song_ids']=ids.tolist()[:-1]
                audit.write_text(json.dumps(summary))
                with self.assertRaisesRegex(ValueError,'audited complete-case cohort'):
                    evaluate(root/'baseline',root/'feature_aware',root/'bad',audit)
