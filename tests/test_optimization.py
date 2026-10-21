import ast
import pathlib
import unittest

APP = pathlib.Path(__file__).resolve().parents[1] / 'app.py'
SOURCE = APP.read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)

class OptimizationTests(unittest.TestCase):
    def test_existing_features_preserved(self):
        for marker in ('positive_prompt', 'negative_prompt', 'revision_requests', 'background_mode', 'custom_background', 'save_generation', 'generation_backup.zip', 'TRIGGER_CAPTION', 'ADAPTERS'):
            self.assertIn(marker, SOURCE)
    def test_input_validation_before_model_load(self):
        generate = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'generate')
        code = ast.get_source_segment(SOURCE, generate)
        self.assertLess(code.index('if primary is None:'), code.index('pipe = get_pipeline()'))
        self.assertLess(code.index('progress(0.03'), code.index('pipe = get_pipeline()'))
    def test_reference_image_resize(self):
        self.assertIn('copy.thumbnail((MAX_REFERENCE_SIDE, MAX_REFERENCE_SIDE)', SOURCE)
    def test_inference_and_concurrency(self):
        self.assertIn('with torch.inference_mode():', SOURCE)
        self.assertIn('concurrency_limit=1', SOURCE)

if __name__ == '__main__':
    unittest.main()
