import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import resilience

class ResilienceTests(unittest.TestCase):
    def test_backup_restore_and_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in resilience.ROOT_FILES:
                (root / name).write_text('x = 1' if name.endswith('.py') else 'gradio', encoding='utf-8')
            archive = resilience.backup(root)
            (root / 'app.py').write_text('x = 2')
            restored = resilience.restore(root, archive)
            self.assertIn('app.py', restored)
            self.assertEqual((root / 'app.py').read_text(), 'x = 1')
    def test_diagnose_syntax_error(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'app.py').write_text('def oops(')
            result = resilience.diagnose(d)
            self.assertTrue(any(i['code'] == 'syntax_error' for i in result['issues']))
    def test_unapproved_source_rejected(self):
        with self.assertRaises(ValueError):
            resilience.fetch_metadata('evil', 'package')
    def test_bad_archive_cannot_escape(self):
        with tempfile.TemporaryDirectory() as d:
            import zipfile, json
            zpath = Path(d) / 'bad.zip'
            with zipfile.ZipFile(zpath, 'w') as z:
                z.writestr('manifest.json', json.dumps({'../outside': '0'}))
            with self.assertRaises(ValueError):
                resilience.restore(d, zpath)
