import os
import tempfile
import unittest
from unittest.mock import patch

from modules import model_downloader as md


class MirrorTest(unittest.TestCase):
    def test_facefusion_github_gets_hf_mirror_first(self):
        url = md._FF_GITHUB + "models-3.0.0/span_kendata_x4.onnx"
        self.assertEqual(md.mirrors(url), [
            "https://huggingface.co/facefusion/models-3.0.0/resolve/main/span_kendata_x4.onnx", url])
        self.assertEqual(md.mirrors("https://example.com/a.onnx"), ["https://example.com/a.onnx"])

    def test_fetch_tries_every_source_and_never_raises(self):
        calls = []
        with tempfile.TemporaryDirectory() as d, \
                patch.object(md, "_download", lambda n, u, t, s: calls.append(u) or False):
            out = md.fetch_url(md._FF_GITHUB + "models-3.0.0/x.onnx", os.path.join(d, "x.onnx"))
        self.assertIsNone(out)
        self.assertEqual(len(calls), 6)          # 2 sources x 3 attempts
        self.assertIn("huggingface.co", calls[0])


if __name__ == "__main__":
    unittest.main()
