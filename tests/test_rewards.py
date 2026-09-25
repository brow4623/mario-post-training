"""Offline checks for the RLVR rewards: verifier rules, reward arithmetic, caching, failures.

These decide what training optimises, so every rule is pinned here rather than discovered
during a paid run. No network calls: the judge is replaced by a recording stub.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import judge  # noqa: E402
import rewards  # noqa: E402

GOLD = "80"
RIGHT = "Let's-a go! 4 * 20 = 80.\n#### 80"
WRONG = "Mamma mia! 4 * 20 = 90.\n#### 90"
RATING = {"voice": 4, "consistency": 3, "naturalness": 4, "restraint": 5, "overall": 4,
          "catchphrase_only": False, "notes": "Reacts to the problem."}


class Stub:
    """Stands in for judge.call_api: same signature, same record shape, counted calls."""

    def __init__(self, rating=RATING, error=None):
        self.rating, self.error, self.calls, self.rows = rating, error, [], []

    def __call__(self, key, config, row, text_field, budget=None):
        self.rows.append(row)
        self.calls.append((row["question"], row[text_field]))
        if self.error:
            raise self.error
        return [{**self.rating, "id": row["id"], "model": config["model"], "request_id": f"req-{len(self.calls)}",
                 "usage": {"input_tokens": 1000, "output_tokens": 100}}]


def rater(tmp, stub, budget=None):
    config = rewards.judge_config()
    cache = rewards.RatingCache(Path(tmp) / "cache.jsonl", judge.digest(config))
    return rewards.PersonaRater(config, "test-key", cache, budget, stub)


class Verifier(unittest.TestCase):
    """The strict rule, unchanged between conditions and recorded in full."""

    def test_correct_wrong_and_equivalent_values(self):
        self.assertEqual(rewards.correctness(RIGHT, GOLD), 1.0)
        self.assertEqual(rewards.correctness(WRONG, GOLD), 0.0)
        self.assertEqual(rewards.correctness("#### 1/2", "0.5"), 1.0)      # fraction equals decimal
        self.assertEqual(rewards.correctness("#### 0.50", "1/2"), 1.0)
        self.assertEqual(rewards.correctness("#### 1,080", "1080"), 1.0)   # thousands separators
        self.assertEqual(rewards.correctness("\\boxed{80}", GOLD), 1.0)

    def test_malformed_markers_and_prefixes_score_zero(self):
        for text in ("#### ", "\\boxed{80", "The answer is 80",           # no usable marker
                     "#### 80 dollars",                                    # trailing prose
                     "#### 80 + 5",                                        # right prefix, different value
                     "#### eighty"):
            self.assertEqual(rewards.correctness(text, GOLD), 0.0, text)

    def test_the_last_marker_wins_and_there_is_no_fallback(self):
        # Recorded explicitly: training uses exactly the evaluation rule, including this one.
        self.assertEqual(rewards.correctness("#### 80\n#### 90", GOLD), 0.0)
        self.assertEqual(rewards.correctness("#### 90\n#### 80", GOLD), 1.0)
        self.assertEqual(rewards.correctness("80 is the total", GOLD), 0.0)


class Arithmetic(unittest.TestCase):
    def test_persona_scale_maps_the_rubric_to_zero_one(self):
        self.assertEqual([rewards.persona_scale(v) for v in (1, 3, 5)], [0.0, 0.5, 1.0])
        for bad in (0, 6, 4.0, True, None, "4"):
            with self.assertRaises(ValueError):
                rewards.persona_scale(bad)

    def test_combined_reward_stays_between_one_and_one_and_a_quarter(self):
        self.assertEqual(rewards.combined_reward(1.0, 0.0), 1.0)
        self.assertEqual(rewards.combined_reward(1.0, 1.0), 1.25)
        self.assertEqual(rewards.combined_reward(1.0, 0.5), 1.125)
        self.assertEqual(rewards.combined_reward(0.0, 1.0), 0.0)  # style never rescues a wrong answer
        self.assertEqual(rewards.combined_reward(1.0, 1.0, weight=0.5), 1.5)

    def test_completion_formats(self):
        self.assertEqual(rewards.completion_text("text"), "text")
        self.assertEqual(rewards.completion_text([{"role": "assistant", "content": "text"}]), "text")
        with self.assertRaises(TypeError):
            rewards.completion_text({"content": "text"})

    def test_reward_kinds_require_the_matching_judge(self):
        with self.assertRaises(ValueError):
            rewards.Reward("combined")
        with self.assertRaises(ValueError):
            rewards.Reward("math", rater=object())
        with self.assertRaises(ValueError):
            rewards.Reward("style")


class Rewards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stub = Stub()

    def combined(self, **kwargs):
        return rewards.Reward("combined", rater(self.tmp.name, self.stub), workers=1, **kwargs)

    def test_math_reward_is_correctness_only_and_calls_no_judge(self):
        reward = rewards.Reward("math")
        values = reward(completions=[RIGHT, WRONG], question=["q", "q"], gold=[GOLD, GOLD])
        self.assertEqual(values, [1.0, 0.0])
        self.assertEqual(reward.summary()["judged_completions"], 0)

    def test_combined_reward_judges_correct_answers_only(self):
        reward = self.combined()
        values = reward(completions=[RIGHT, WRONG], question=["q", "q"], gold=[GOLD, GOLD], id=["train-1", "train-1"])
        self.assertEqual(values, [1.0 + 0.25 * 0.75, 0.0])
        self.assertEqual(self.stub.calls, [("q", RIGHT)])                 # the wrong answer costs nothing
        # The judge request carries the question and the answer only; nothing marks the gold value.
        self.assertEqual(set(self.stub.rows[0]), {"id", "question", "output"})

    def test_identical_answers_in_one_batch_cost_one_request(self):
        reward = self.combined()
        values = reward(completions=[RIGHT, RIGHT], question=["q", "q"], gold=[GOLD, GOLD])
        self.assertEqual(values[0], values[1])
        self.assertEqual(len(self.stub.calls), 1)

    def test_group_statistics_count_mixed_correctness(self):
        reward = rewards.Reward("math")
        reward(completions=[RIGHT, WRONG, RIGHT, RIGHT], question=["a", "a", "b", "b"],
               gold=[GOLD] * 4, id=["train-1"] * 2 + ["train-2"] * 2)
        report = reward.summary()
        self.assertEqual((report["groups"], report["mixed_correctness_groups"], report["all_correct_groups"]), (2, 1, 1))
        self.assertEqual(report["mean_correct"], 0.75)

    def test_reward_and_sample_logs_record_both_components(self):
        log, samples = Path(self.tmp.name) / "rewards.jsonl", Path(self.tmp.name) / "samples.jsonl"
        reward = self.combined(log_path=log, sample_path=samples, samples=1)
        reward(completions=[RIGHT, WRONG], question=["q", "q"], gold=[GOLD, GOLD])
        records = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual([(r["correct"], r["persona_overall"], r["persona_scale"]) for r in records],
                         [(1.0, 4, 0.75), (0.0, None, None)])
        saved = [json.loads(line) for line in samples.read_text().splitlines()]
        self.assertEqual([(r["answer"], r["reward"]) for r in saved], [(RIGHT, 1.1875)])

    def test_mismatched_batch_columns_are_rejected(self):
        with self.assertRaises(ValueError):
            rewards.Reward("math")(completions=[RIGHT, WRONG], question=["q"], gold=[GOLD])


class Caching(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stub = Stub()

    def test_repeated_answers_reuse_the_saved_rating_within_and_across_runs(self):
        first = rater(self.tmp.name, self.stub)
        self.assertEqual(first.rate("q", RIGHT)["overall"], 4)
        self.assertEqual(first.rate("q", RIGHT)["overall"], 4)
        later = rater(self.tmp.name, Stub({**RATING, "overall": 1}))   # a rerun must not rescore
        self.assertEqual(later.rate("q", RIGHT), first.rate("q", RIGHT))
        self.assertEqual(len(self.stub.calls), 1)
        self.assertEqual(later.requests, 0)

    def test_the_key_separates_questions_answers_and_judge_configurations(self):
        first = rater(self.tmp.name, self.stub)
        first.rate("q", RIGHT)
        first.rate("other question", RIGHT)
        first.rate("q", RIGHT + " Wahoo!")
        self.assertEqual(len(self.stub.calls), 3)
        cache = Path(self.tmp.name) / "cache.jsonl"
        self.assertEqual(len({json.loads(line)["key"] for line in cache.read_text().splitlines()}), 3)
        with self.assertRaises(ValueError):      # a different rubric or model never reuses these ratings
            rewards.RatingCache(cache, "another-config-hash")

    def test_a_failed_request_raises_instead_of_scoring_zero(self):
        stub = Stub(error=RuntimeError("Retries exhausted for train-1"))
        failing = rater(self.tmp.name, stub)
        with self.assertRaises(RuntimeError):
            failing.rate("q", RIGHT)
        self.assertFalse((Path(self.tmp.name) / "cache.jsonl").exists())
        reward = rewards.Reward("combined", failing, workers=1)
        with self.assertRaises(RuntimeError):    # the update stops; it does not continue with P = 0
            reward(completions=[RIGHT], question=["q"], gold=[GOLD])
        self.assertEqual(reward.records, [])

    def test_the_ledger_carries_spend_across_resumed_jobs(self):
        directory = Path(self.tmp.name)
        self.assertEqual(rewards.recorded_spend(directory), 0)
        (directory / "attempts.jsonl").write_text(
            json.dumps({"id": "a", "usage": {"input_tokens": 1000, "output_tokens": 100}}) + "\n"
            + json.dumps({"id": "b", "http_status": 429, "error": "overloaded"}) + "\n", encoding="utf-8")
        self.assertAlmostEqual(rewards.recorded_spend(directory), 0.0015)
        budget = judge.Budget(0.002, rewards.recorded_spend(directory))
        with self.assertRaises(RuntimeError):
            budget.reserve({"max_tokens": 512, "text": "x" * 4000})


if __name__ == "__main__":
    unittest.main()
