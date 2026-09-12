import json
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sonnet_rewrite import api_key, digest, inventory, prompt_prefix, request_body, rewrite, save_json
from mario_data import record, read_jsonl


class SonnetRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path('derived/rewrites/train_all.jsonl').exists():
            from prepare_data import main
            main()

    def test_compacted_run_reconstructs_all_outputs(self):
        config = json.loads(Path('provenance/generation/config.json').read_text(encoding='utf-8'))
        pending, combined, counts = inventory(digest(config))
        self.assertEqual(pending, [])
        self.assertEqual(counts, {'gpt-6-astra': 475, 'claude-sonnet-5': 3025})
        for split in ('train', 'validation'):
            self.assertEqual(combined[split], read_jsonl(f'derived/rewrites/{split}_all.jsonl'))

    def test_stable_cached_prefix_and_effort(self):
        a = request_body("fixed examples", {"question": "a", "answer": "#### 1"})
        b = request_body("fixed examples", {"question": "b", "answer": "#### 2"}, "retry")
        self.assertEqual(a["system"], b["system"])
        self.assertEqual(a["output_config"], {"effort": "low"})
        self.assertEqual(a["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", a["messages"][0])
        self.assertNotEqual(a["messages"], b["messages"])

    def test_only_four_train_examples(self):
        prefix = prompt_prefix()
        examples = json.loads(prefix[prefix.index('[\n  {'):])
        self.assertEqual(len(examples), 4)
        self.assertIn("Stefan", examples[0]["question"])
        self.assertIn("zoo", examples[3]["question"])
        self.assertNotIn("Jack is ordering custom", prefix)

    def test_env_file_key_without_loading_other_values(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / ".env"
            path.write_text('OTHER=unused\nCLAUDE_API_KEY="test-key"\n', encoding="utf-8")
            self.assertEqual(api_key(path), "test-key")
            self.assertNotIn("OTHER", os.environ)

    def test_environment_precedence(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-only"}, clear=True):
            self.assertEqual(api_key("nonexistent.env"), "test-only")

    def test_atomic_json_and_stable_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sub/record.json"
            save_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})
            self.assertFalse(path.with_suffix(".json.tmp").exists())
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))

    def test_invalid_output_is_regenerated_not_repaired(self):
        source = record({"question": "One plus one?", "answer": "1+1=<<1+1=2>>2\n#### 2"}, 0, "train", False, 42)
        responses = [{"id": "test-request", "model": "claude-sonnet-5", "stop_reason": "end_turn",
                      "content": [{"type": "text", "text": answer}], "usage": {"output_tokens": 10}}
                     for answer in ("Wrong!\n#### 3", "I add the pair: <<1+1=2>>2!\n#### 2")]
        with tempfile.TemporaryDirectory() as directory, patch("sonnet_rewrite.ROOT", Path(directory)):
            with patch("sonnet_rewrite.urllib.request.urlopen", side_effect=[io.BytesIO(json.dumps(r).encode()) for r in responses]) as api:
                saved = rewrite(("train", source), "test-key", "fixed", "config")
            self.assertEqual(api.call_count, 2)
            self.assertEqual(saved["answer"], responses[1]["content"][0]["text"])
            attempts = read_jsonl(Path(directory) / "attempts.jsonl")
            self.assertFalse(attempts[0]["accepted"])
            self.assertTrue(attempts[1]["accepted"])
            first, second = [json.loads(call.args[0].data) for call in api.call_args_list]
            self.assertEqual(first["system"], second["system"])


if __name__ == "__main__":
    unittest.main()
