import json
import tempfile
import unittest
from pathlib import Path
from auto_repair import inspect_project, apply_safe_fixes

class AutoRepairTests(unittest.TestCase):
    def test_missing_requirements_detected(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'app.py').write_text('print(1)',encoding='utf-8')
            codes={x['code'] for x in inspect_project(Path(d))}
            self.assertIn('MISSING_REQUIREMENTS',codes)
    def test_safe_fixes_do_not_overwrite_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'app.py').write_text('original',encoding='utf-8')
            result=apply_safe_fixes(p)
            self.assertEqual((p/'app.py').read_text(),'original')
            self.assertTrue((p/'outputs').is_dir())
            self.assertTrue((p/'reports').is_dir())
            self.assertIn('actions',result)
    def test_invalid_python_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'app.py').write_text('def bad(:',encoding='utf-8'); (p/'requirements.txt').write_text('gradio\n')
            self.assertIn('SYNTAX_ERROR',{x['code'] for x in inspect_project(p)})
if __name__=='__main__': unittest.main()
