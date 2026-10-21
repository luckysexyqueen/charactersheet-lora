import tempfile, unittest, zipfile
from pathlib import Path
from operations import safe_extract, backup_outputs, parse_repo_url

class OpsTests(unittest.TestCase):
    def test_repo_url(self):
        self.assertEqual(parse_repo_url("https://huggingface.co/spaces/owner/demo"), ("space", "owner/demo"))
        self.assertEqual(parse_repo_url("https://huggingface.co/owner/model"), ("model", "owner/model"))
        with self.assertRaises(ValueError): parse_repo_url("https://evil.example/owner/model")
    def test_zip_slip_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            z=Path(t)/"bad.zip"
            with zipfile.ZipFile(z,"w") as f:f.writestr("../outside", "bad")
            with self.assertRaises(ValueError):safe_extract(z,Path(t)/"dest")
    def test_backup_restore(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/"outputs"; (root/"job1").mkdir(parents=True)
            (root/"job1"/"metadata.json").write_text("{}")
            z=backup_outputs(root,Path(t)/"backup.zip")
            safe_extract(z,Path(t)/"restored")
            self.assertTrue((Path(t)/"restored"/"job1"/"metadata.json").exists())
if __name__=="__main__":unittest.main()
