"""Offline checks for the three-condition comparison: pairing, matching, and the verdicts.

Synthetic outputs and ratings only, so the statistics and the guards against comparing
unmatched runs can be checked without the evaluation or the judge.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analyze_rlvr as analysis  # noqa: E402
from judge import CRITERIA, text_hash  # noqa: E402
from mario_data import write_jsonl  # noqa: E402

CONDITIONS = ("rlvr_baseline", "rlvr_math", "rlvr_combined")
REFS = [{"id": f"q{i}", "question": f"Question {i}?", "answer": str(i)} for i in range(30)]
CONFIG = {**{key: "same" for key in analysis.MATCHED_KEYS}, "adapter_hashes": {"adapter_config.json": "abc"}}


def answer(index, correct=True, words=10):
    return ("Wahoo! " * words) + f"\n#### {index if correct else index + 1000}"


def outputs(correct, words=10):
    return [{"id": ref["id"], "question": ref["question"], "gold": ref["answer"],
             "output": answer(i, correct[i], words), "format_valid": True, "hit_token_cap": False,
             "pred": str(i), "correct": correct[i]} for i, ref in enumerate(REFS)]


def ratings(rows, overall, config_hash="config", catchphrase=False):
    return [{"id": row["id"], "text_sha256": text_hash(row["output"]), "config_sha256": config_hash,
             "model": "claude-haiku-4-5", **{c: overall[i] for c in CRITERIA},
             "catchphrase_only": catchphrase, "notes": ""} for i, row in enumerate(rows)]


class Statistics(unittest.TestCase):
    def test_paired_difference_is_new_minus_old(self):
        result = analysis.paired([1, 1, 2], [2, 3, 2], resamples=200)
        self.assertAlmostEqual(result["mean"], 1.0)
        self.assertTrue(result["ci95"][0] <= result["mean"] <= result["ci95"][1])
        self.assertIsNone(analysis.paired([], []))
        with self.assertRaises(ValueError):
            analysis.paired([1], [1, 2])

    def test_the_predeclared_criteria(self):
        improved = {"ci95": [0.01, 0.08]}
        self.assertTrue(analysis.verdicts(improved, {"ci95": [-0.19, 0.2]})["accuracy_improved"])
        self.assertTrue(analysis.verdicts(improved, {"ci95": [-0.19, 0.2]})["persona_retained"])
        self.assertFalse(analysis.verdicts(improved, {"ci95": [-0.20, 0.2]})["persona_retained"])
        flat = analysis.verdicts({"ci95": [-0.01, 0.08]}, {"ci95": [-0.5, -0.3]})
        self.assertEqual((flat["accuracy_improved"], flat["accuracy_lost"], flat["persona_retained"]),
                         (False, False, False))
        self.assertTrue(analysis.verdicts({"ci95": [-0.08, -0.01]}, {"ci95": [0, 0]})["accuracy_lost"])

    def test_similar_length_subset(self):
        self.assertEqual(analysis.length_matched([100, 100, 0], [105, 200, 0]), [0])


def solved(count):
    return [i < count for i in range(len(REFS))]


class Report(unittest.TestCase):
    """Math-only buys accuracy and loses a persona point; the combined run keeps the voice."""

    def setUp(self):
        self.correct = {"rlvr_baseline": solved(15), "rlvr_math": solved(27), "rlvr_combined": solved(24)}
        self.overall = {name: [{"rlvr_math": 2}.get(name, 3)] * len(REFS) for name in CONDITIONS}
        self.outputs = {name: outputs(self.correct[name]) for name in CONDITIONS}
        self.scores = {name: ratings(self.outputs[name], self.overall[name]) for name in CONDITIONS}

    def report(self):
        return analysis.analyze(REFS, self.outputs, self.scores, resamples=200)

    def test_accuracy_and_persona_are_reported_separately(self):
        report = self.report()
        self.assertAlmostEqual(report["conditions"]["rlvr_math"]["accuracy"]["mean"], 27 / 30)
        self.assertAlmostEqual(report["conditions"]["rlvr_math"]["persona_overall"]["mean"], 2)
        self.assertEqual(report["conditions"]["rlvr_baseline"]["criteria"]["voice"], 3)
        contrast = report["contrasts"]["rlvr_math_minus_rlvr_baseline"]
        self.assertAlmostEqual(contrast["accuracy"]["mean"], 12 / 30)
        self.assertAlmostEqual(contrast["persona_overall"]["mean"], -1)
        self.assertEqual(contrast["criteria"]["restraint"], -1)

    def test_a_model_that_trades_persona_for_math_fails_the_retention_criterion(self):
        verdicts = self.report()["contrasts"]["rlvr_math_minus_rlvr_baseline"]["verdicts"]
        self.assertTrue(verdicts["accuracy_improved"])
        self.assertFalse(verdicts["persona_retained"])
        kept = self.report()["contrasts"]["rlvr_combined_minus_rlvr_baseline"]["verdicts"]
        self.assertTrue(kept["accuracy_improved"] and kept["persona_retained"])

    def test_the_similar_length_check_uses_the_pair_being_compared(self):
        self.outputs["rlvr_math"] = outputs(self.correct["rlvr_math"], words=40)
        self.scores["rlvr_math"] = ratings(self.outputs["rlvr_math"], self.overall["rlvr_math"])
        report = self.report()
        self.assertEqual(report["contrasts"]["rlvr_math_minus_rlvr_baseline"]["length_matched"]["n"], 0)
        self.assertEqual(report["contrasts"]["rlvr_combined_minus_rlvr_baseline"]["length_matched"]["n"], len(REFS))
        self.assertGreater(report["contrasts"]["rlvr_math_minus_rlvr_baseline"]["word_count"], 0)

    def test_tables_name_every_condition_and_comparison(self):
        text = analysis.table({"splits": {"test": self.report()}})
        for label in analysis.LABELS.values():
            self.assertIn(label, text)
        self.assertIn("persona not retained", text)
        self.assertIn("| Condition | Accuracy | Persona overall |", text)


class Loading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.results = Path(self.tmp.name) / "results"
        self.provenance = Path(self.tmp.name) / "provenance"
        self.data = Path(self.tmp.name) / "data"
        write_jsonl(self.data / "test.jsonl", REFS)
        self.rows = {name: outputs(solved(len(REFS))) for name in CONDITIONS}
        for name in CONDITIONS:
            write_jsonl(self.results / f"{name}_test.jsonl", self.rows[name])
            write_jsonl(self.results / "judge" / f"{name}_test.jsonl", self.ratings(name))
            self.write_config(name, CONFIG)

    def ratings(self, name, **kwargs):
        return ratings(self.rows[name], [4] * len(REFS), **kwargs)

    def write_config(self, name, config):
        path = self.provenance / f"{name}_test.config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config), encoding="utf-8")

    def load(self):
        return analysis.load_split("test", CONDITIONS, self.results, self.provenance, self.data)

    def test_matched_runs_load(self):
        refs, loaded, scores, provenance = self.load()
        self.assertEqual(len(refs), len(REFS))
        self.assertEqual(provenance["judge_model"], "claude-haiku-4-5")
        self.assertEqual(sorted(loaded), sorted(CONDITIONS))
        self.assertEqual(provenance["adapters"]["rlvr_math"], CONFIG["adapter_hashes"])

    def test_unmatched_decoding_or_judge_settings_are_refused(self):
        self.write_config("rlvr_math", {**CONFIG, "max_new_tokens": 512})
        with self.assertRaises(ValueError):
            self.load()
        self.write_config("rlvr_math", CONFIG)
        write_jsonl(self.results / "judge" / "rlvr_math_test.jsonl", self.ratings("rlvr_math", config_hash="other"))
        with self.assertRaises(ValueError):
            self.load()

    def test_ratings_of_a_stale_answer_are_refused(self):
        stale = self.ratings("rlvr_math")
        stale[0]["text_sha256"] = text_hash("an answer from an earlier run")
        write_jsonl(self.results / "judge" / "rlvr_math_test.jsonl", stale)
        with self.assertRaises(ValueError):
            self.load()


if __name__ == "__main__":
    unittest.main()
