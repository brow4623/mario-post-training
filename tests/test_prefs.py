"""Offline checks for preference-pair construction."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_prefs as mp  # noqa: E402


def ans(text, correct=True):
    return {"text": text, "correct": correct, "format_valid": True, "pred": "1", "generated_tokens": 5, "hit_token_cap": False}


def score(overall, catchphrase_only=False, **crit):
    base = {"voice": overall, "consistency": overall, "naturalness": overall, "restraint": overall}
    base.update(crit)
    return {**base, "overall": overall, "catchphrase_only": catchphrase_only, "notes": ""}


Q = {"id": "train-1", "question": "q", "gold": "1",
     "answers": [ans("wahoo 1"), ans("mamma mia 1"), ans("plain 1"), ans("wrong 2", correct=False)]}


class Flatten(unittest.TestCase):
    def test_ids_and_fields(self):
        rows = mp.flatten([Q])
        self.assertEqual([r["id"] for r in rows], ["train-1#0", "train-1#1", "train-1#2", "train-1#3"])
        self.assertEqual(set(rows[0]), {"id", "question", "output"})
        self.assertEqual(rows[3]["output"], "wrong 2")


class Pairs(unittest.TestCase):
    def test_best_and_worst_correct_answers(self):
        scores = {"train-1#0": score(4), "train-1#1": score(5), "train-1#2": score(1), "train-1#3": score(5)}
        pairs, reasons = mp.build_pairs([Q], scores)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["chosen"], "mamma mia 1")   # the wrong answer scored 5 too, but is not eligible
        self.assertEqual(pairs[0]["rejected"], "plain 1")
        self.assertEqual((pairs[0]["chosen_score"], pairs[0]["rejected_score"]), (5, 1))
        self.assertEqual(set(pairs[0]) >= {"id", "prompt", "chosen", "rejected", "gold"}, True)
        self.assertEqual(reasons, {})

    def test_wrong_answer_can_be_rejected_only_when_allowed(self):
        scores = {"train-1#0": score(4), "train-1#1": score(4), "train-1#2": score(4), "train-1#3": score(1)}
        pairs, reasons = mp.build_pairs([Q], scores)
        self.assertEqual(pairs, [])
        self.assertIn("judge gap below 1", reasons)
        pairs, _ = mp.build_pairs([Q], scores, require_rejected_correct=False)
        self.assertEqual(pairs[0]["rejected"], "wrong 2")

    def test_ties_prefer_non_catchphrase_and_lower_index(self):
        scores = {"train-1#0": score(4, catchphrase_only=True), "train-1#1": score(4), "train-1#2": score(2), "train-1#3": score(2)}
        pairs, _ = mp.build_pairs([Q], scores)
        self.assertEqual(pairs[0]["chosen"], "mamma mia 1")
        scores["train-1#1"] = score(4, catchphrase_only=True)
        pairs, _ = mp.build_pairs([Q], scores)
        self.assertEqual(pairs[0]["chosen"], "wahoo 1")

    def test_drop_reasons(self):
        no_correct = {**Q, "answers": [ans("a", False), ans("b", False)]}
        identical = {**Q, "id": "train-2", "answers": [ans("same"), ans("same")]}
        unscored = {**Q, "id": "train-3"}
        scores = {"train-1#0": score(3), "train-1#1": score(1), "train-2#0": score(3), "train-2#1": score(1)}
        pairs, reasons = mp.build_pairs([no_correct, identical, unscored], scores)
        self.assertEqual(pairs, [])
        self.assertEqual(reasons["no correct answer to choose"], 1)
        self.assertEqual(reasons["all answers identical"], 1)
        self.assertEqual(reasons["fewer than 2 scored answers"], 1)
        self.assertEqual(reasons["answer missing judge score"], 4)

    def test_min_gap(self):
        scores = {"train-1#0": score(4), "train-1#1": score(3), "train-1#2": score(2), "train-1#3": score(1)}
        self.assertEqual(len(mp.build_pairs([Q], scores, min_gap=2)[0]), 1)
        self.assertEqual(len(mp.build_pairs([Q], scores, min_gap=3)[0]), 0)


if __name__ == "__main__":
    unittest.main()
