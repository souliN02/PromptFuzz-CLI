import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import TestCase

# Ensure src/ is on path for direct import
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cli import scan  # noqa: E402
from click.testing import CliRunner  # noqa: E402


class TestMockRun(TestCase):
    def test_mock_scan_creates_outputs(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts = ROOT / "prompts.txt"
            out_html = Path(tmpdir) / "report.html"
            out_json = Path(tmpdir) / "report.json"
            out_csv = Path(tmpdir) / "report.csv"

            result = runner.invoke(
                scan,
                [
                    "--target",
                    "gpt-mock",
                    "--prompts",
                    str(prompts),
                    "--mutations",
                    "prefix,base64",
                    "--out-html",
                    str(out_html),
                    "--out-json",
                    str(out_json),
                    "--out-csv",
                    str(out_csv),
                ],
            )

            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue(out_html.exists())
            self.assertTrue(out_json.exists())
            self.assertTrue(out_csv.exists())

            # Basic content checks
            data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertGreater(len(data), 0)
            self.assertIn("bypassed", data[0])
            self.assertIn("response_length", data[0])
