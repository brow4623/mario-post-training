"""Offline checks for the persona judge: blinding, schema validation, batching, resumability."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import judge  # noqa: E402

ROW = {"id": "train-1", "question": "Stefan buys 4 entrees at $20 each. Total?", "gold": "80",
       "output": "Mamma mia! 4 * 20 = 80.\n#### 80", "correct": True, "condition": "mario"}
GOOD = {"voice": 4, "consistency": 3, "naturalness": 4, "restraint": 5, "overall": 4,
        "catchphrase_only": False, "notes": "Reacts to the meal, light accent."}


def run(*extra):
    env = {**os.environ, "ANTHROPIC_API_KEY": "", "CLAUDE_API_KEY": ""}
    return subprocess.run([sys.executable, "judge.py", "score", *extra], capture_output=True, text=True, env=env, cwd=ROOT)


class Blinding(unittest.TestCase):
    def test_api_request_contains_only_question_and_answer(self):
        config = judge.judge_config("api", judge.MODEL, "low", 4096, 20)
        body = judge.request_body(config, ROW["question"], ROW["output"])
        user_text = body["messages"][0]["content"]
        self.assertIn(ROW["question"], user_text)
        self.assertIn(json.dumps(ROW["output"]), user_text)
        for leaked in ("gold", "condition", "mario_val", "correct", "train-1"):
            self.assertNotIn(leaked, user_text)
        self.assertEqual(body["output_config"]["format"]["schema"], judge.SCHEMA)
        self.assertEqual(body["temperature"], 0)

    def test_batch_prompt_uses_index_numbers_only(self):
        prompt = judge.batch_prompt([ROW, {**ROW, "id": "train-2"}], "output")
        self.assertIn('"index": 1', prompt)
        for leaked in ("gold", "condition", "correct", "train-1", "train-2"):
            self.assertNotIn(leaked, prompt)

    def test_config_hash_separates_backends_and_models(self):
        hashes = {judge.digest(judge.judge_config(b, m, "low", 4096, 20))
                  for b in ("api", "claude-code") for m in ("claude-haiku-4-5", "claude-opus-5")}
        self.assertEqual(len(hashes), 4)


class Parsing(unittest.TestCase):
    def test_single_rating(self):
        self.assertEqual(judge.parse_scores(json.dumps(GOOD)), GOOD)
        for bad in ({**GOOD, "voice": 6}, {**GOOD, "overall": 0}, {**GOOD, "voice": True},
                    {k: v for k, v in GOOD.items() if k != "notes"}, {**GOOD, "extra": 1},
                    {**GOOD, "catchphrase_only": "no"}):
            with self.assertRaises(ValueError):
                judge.parse_scores(json.dumps(bad))

    def test_batch_requires_one_rating_per_index(self):
        ok = {"ratings": [{"index": 1, **GOOD}, {"index": 0, **GOOD, "overall": 2}]}
        parsed = judge.parse_batch(ok, 2)
        self.assertEqual(parsed[0]["overall"], 2)
        self.assertNotIn("index", parsed[1])
        for bad in ({"ratings": [{"index": 0, **GOOD}]},                                  # missing index 1
                    {"ratings": [{"index": 0, **GOOD}, {"index": 0, **GOOD}]},           # duplicate
                    {"ratings": [{"index": 0, **GOOD}, {"index": 2, **GOOD}]},           # out of range
                    {"ratings": [{"index": 0, **GOOD}, {"index": 1, **GOOD, "voice": 9}]},
                    {"ratings": [{"index": 0, **GOOD}, {"index": 1, **GOOD}], "extra": 1}):
            with self.assertRaises(ValueError):
                judge.parse_batch(bad, 2)


class Resumability(unittest.TestCase):
    def test_summary_rejects_incomplete_or_mismatched_scores(self):
        row = {'id': 'one', **GOOD, 'config_sha256': 'same', 'model': 'same'}
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a.jsonl', Path(tmp) / 'b.jsonl'
            a.write_text(json.dumps(row) + '\n', encoding='utf-8')
            for rows in ([{**row, 'id': 'two'}], [row, row],
                         [{**row, 'model': 'different'}], [{**row, 'config_sha256': 'different'}]):
                b.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
                with self.assertRaises(ValueError):
                    judge.summarize(SimpleNamespace(files=[str(a), str(b)], resamples=10))

    def test_dry_run_and_skip_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "x_val.jsonl"
            inp.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
            for backend in ("api", "claude-code"):
                out = run(str(inp), "--backend", backend, "--dry-run")
                self.assertEqual(out.returncode, 0, out.stderr)
                self.assertIn(ROW["question"], out.stdout)
                self.assertNotIn('"gold"', out.stdout)
                # A pre-scored record with a matching config hash needs no call, so no key or CLI is needed.
                config_hash = judge.digest(judge.judge_config(backend, judge.MODEL, "low", 4096, 20))
                scored = Path(tmp) / f"scored_{backend}.jsonl"
                scored.write_text(json.dumps({"id": "train-1", **GOOD, "config_sha256": config_hash,
                                              "text_sha256": hashlib.sha256(ROW["output"].encode()).hexdigest(),
                                              "model": judge.MODEL, "request_id": "x", "usage": {}}) + "\n", encoding="utf-8")
                out = run(str(inp), "--backend", backend, "--output", str(scored))
                self.assertEqual(out.returncode, 0, out.stderr)
                self.assertIn("1 already scored, 0 to score", out.stdout)


if __name__ == "__main__":
    unittest.main()
