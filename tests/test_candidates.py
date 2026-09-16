"""Offline checks for the candidate sampler: selection, answer records, resume safety."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sample_candidates as sc  # noqa: E402

ROWS = [{"id": f"train-{i}", "question": f"q{i}", "answer": f"steps\n#### {i}"} for i in range(50)]


class Selection(unittest.TestCase):
    def test_seeded_subset_is_deterministic_and_in_dataset_order(self):
        a = sc.select_questions(ROWS, 10, 42)
        b = sc.select_questions(ROWS, 10, 42)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 10)
        self.assertEqual([r["id"] for r in a], [r["id"] for r in ROWS if r in a])
        self.assertNotEqual(a, sc.select_questions(ROWS, 10, 7))
        self.assertEqual(sc.select_questions(ROWS, 500, 42), ROWS)
        self.assertEqual(sc.select_questions(ROWS, None, 42), ROWS)


class AnswerRecords(unittest.TestCase):
    def test_verifier_and_cap_flags(self):
        good = sc.answer_record("Wahoo! 2 + 2 = 4\n#### 4", 12, "x\n#### 4", 1024, True)
        self.assertTrue(good["correct"] and good["format_valid"])
        self.assertEqual(good["pred"], "4")
        wrong = sc.answer_record("Mamma mia, 5\n#### 5", 8, "x\n#### 4", 1024, True)
        self.assertFalse(wrong["correct"])
        self.assertTrue(wrong["format_valid"])
        capped = sc.answer_record("I keep going and going", 1024, "x\n#### 4", 1024, False)
        self.assertFalse(capped["correct"] or capped["format_valid"])
        self.assertTrue(capped["hit_token_cap"])


class Resume(unittest.TestCase):
    def test_mismatched_saved_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.jsonl"
            path.write_text(json.dumps({"id": "train-0", "question": "q0", "gold": "steps\n#### 0", "answers": []}) + "\n")
            self.assertEqual(len(sc.resume(path, ROWS)), 1)
            path.write_text(json.dumps({"id": "train-1", "question": "q1", "gold": "steps\n#### 1", "answers": []}) + "\n")
            with self.assertRaises(ValueError):
                sc.resume(path, ROWS)


if __name__ == "__main__":
    unittest.main()
