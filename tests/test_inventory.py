"""Content-level audit regression tests; no classification performance claims."""
import tempfile
import csv
from pathlib import Path
import unittest
import numpy as np
import soundfile as sf
from scripts.inventory_dataset import inspect_file, inventory, write_aggregate


class InventoryTests(unittest.TestCase):
    def test_decode_failure_and_short_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path=root/'Similar_Rock_00001_Cover_Timbre_B.wav'
            path.write_bytes(b'broken WAV')
            row=inspect_file(path,root)
            self.assertFalse(row['decode_success'])
            self.assertNotIn(str(root),row['error'])
            sf.write(path,np.zeros(16000),16000)
            row=inspect_file(path,root)
            self.assertTrue(row['decode_success'])
            self.assertFalse(row['duration_valid'])
            self.assertEqual(row['duration'],1)

    def test_duplicates_and_missing_source_are_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'audio';root.mkdir()
            for source in ('Timbre','Rhythm','Instrument'):
                sf.write(root/f'Similar_Rock_00001_Cover_{source}_B.wav',np.zeros(960000),16000)
            rows,summary=inventory(root,Path(tmp)/'results')
            self.assertEqual(summary['complete_song_count'],0)
            self.assertEqual(summary['exclusion_reasons'],{'Missing Cover_Genre':1})
            self.assertEqual(summary['duplicate_file_count'],3)
            self.assertEqual(summary['cross_song_duplicate_file_count'],0)
            self.assertTrue(all(not Path(r['relative_path']).is_absolute() for r in rows))
            published = Path(tmp)/'published'
            write_aggregate(rows, published)
            text = (published/'dataset_inventory.csv').read_text()
            self.assertNotIn(str(root), text)
            aggregate = list(csv.DictReader(text.splitlines()))
            self.assertEqual(sum(int(r['audio_file_count']) for r in aggregate),3)
            self.assertEqual(sum(int(r['duplicate_file_count']) for r in aggregate),3)
            self.assertNotIn('relative_path',aggregate[0])
