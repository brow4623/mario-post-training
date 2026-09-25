"""Offline checks for the GRPO runner and the frozen training questions.

Everything here runs without torch, TRL, a GPU, or an API key: argument validation, batch
arithmetic, the TRL-version probes, the prompt the policy actually sees, and the audit that
keeps both runs on one frozen 512-question subset.
"""
import contextlib
import dataclasses
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import make_rlvr_data  # noqa: E402
import rlvr  # noqa: E402
from eval import FORMAT_PROMPT  # noqa: E402
from mario_data import read_jsonl, write_jsonl  # noqa: E402

ROWS = [{"id": f"train-{i}", "question": f"Question {i}?", "response": f"work\n#### {i}", "answer": str(i)}
        for i in range(40)]


@dataclasses.dataclass
class NewerTRL:
    output_dir: str = "out"
    num_generations: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    max_completion_length: int = 256
    beta: float = 0.0
    scale_rewards: str = "group"


@dataclasses.dataclass
class OlderTRL:
    output_dir: str = "out"
    num_generations: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    max_completion_length: int = 256
    beta: float = 0.0
    scale_rewards: bool = True


@dataclasses.dataclass
class WrongTRL:
    output_dir: str = "out"


class Batching(unittest.TestCase):
    def test_updates_are_counted_in_completions(self):
        self.assertEqual(rlvr.check_batching(8, 8, 4),
                         {"completions_per_update": 64, "questions_per_update": 16})
        self.assertEqual(rlvr.check_batching(4, 1, 4)["questions_per_update"], 1)

    def test_groups_must_fit_the_batch(self):
        for batch, accum, generations in ((8, 8, 3), (2, 8, 4), (8, 8, 1), (0, 8, 4)):
            with self.assertRaises(ValueError):
                rlvr.check_batching(batch, accum, generations)


class TrlProbes(unittest.TestCase):
    def test_standard_deviation_scaling_is_turned_off_on_either_api(self):
        self.assertEqual(rlvr.scale_rewards_value(NewerTRL), "none")
        self.assertIs(rlvr.scale_rewards_value(OlderTRL), False)
        self.assertIsNone(rlvr.scale_rewards_value(WrongTRL))

    def test_unknown_settings_are_reported_and_essential_ones_are_required(self):
        settings, dropped = rlvr.config_kwargs(NewerTRL, {"beta": 0.02, "vllm_mode": "colocate"})
        self.assertEqual((settings, dropped), ({"beta": 0.02}, ["vllm_mode"]))
        with self.assertRaises(RuntimeError):
            rlvr.config_kwargs(WrongTRL, {"beta": 0.02})

    def test_initial_kl_comes_from_the_first_logged_update(self):
        self.assertEqual(rlvr.initial_kl([{"loss": 0.1}, {"kl": 1e-7}, {"kl": 0.3}]), 1e-7)
        self.assertIsNone(rlvr.initial_kl([{"loss": 0.1}]))


class Prompts(unittest.TestCase):
    def test_the_policy_sees_the_question_only(self):
        source = {"id": "train-9", "question": "How many coins?", "response": "work\n#### 137", "answer": "137"}
        rows = rlvr.build_rows([source])
        self.assertEqual(rows[0]["prompt"], [{"role": "system", "content": FORMAT_PROMPT},
                                             {"role": "user", "content": source["question"]}])
        self.assertEqual((rows[0]["id"], rows[0]["gold"]), ("train-9", "137"))
        self.assertNotIn("137", json.dumps(rows[0]["prompt"]))     # the gold answer stays with the verifier
        self.assertNotIn("response", rows[0])                      # and so does the reference solution


class Arguments(unittest.TestCase):
    def parse(self, *extra):
        return rlvr.parse_args(["--reward", "math", "--output", "/tmp/rlvr-test", *extra])

    def test_defaults_match_the_plan(self):
        args, batching = self.parse()
        self.assertEqual((args.num_generations, args.temperature, args.top_p), (4, 1.0, 0.95))
        self.assertEqual((args.lr, args.beta, args.seed, args.epochs), (1e-6, 0.02, 42, 1))
        self.assertEqual((args.max_completion_length, batching), (1024, {"completions_per_update": 64,
                                                                         "questions_per_update": 16}))
        self.assertEqual(args.revision, rlvr.QWEN_REVISION)
        self.assertEqual(args.run_name, "rlvr-test")

    def test_the_math_run_refuses_judge_flags_and_a_smaller_model_needs_its_own_revision(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse("--budget-usd", "5")
        args, _ = rlvr.parse_args(["--reward", "combined", "--output", "/tmp/x", "--budget-usd", "5",
                                   "--model", "Qwen/Qwen2.5-0.5B-Instruct"])
        self.assertEqual((args.revision, args.budget_usd), ("main", 5))


class FrozenQuestions(unittest.TestCase):
    def setUp(self):
        self.previous = os.getcwd()
        os.chdir(ROOT)
        self.addCleanup(os.chdir, self.previous)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_the_real_subset_is_512_seeded_questions_with_no_held_out_overlap(self):
        selected, record = make_rlvr_data.audit()
        self.assertEqual((len(selected), record["seed"]), (512, 42))
        self.assertEqual(record["ids"], [row["id"] for row in selected])
        self.assertEqual([leak["shared_questions"] for leak in record["overlap"]], [0, 0])
        self.assertEqual(record, make_rlvr_data.audit()[1])                      # deterministic
        self.assertNotEqual(record["ids"], make_rlvr_data.audit(seed=7)[1]["ids"])

    def test_the_committed_list_still_reproduces(self):
        selected, record = make_rlvr_data.load_frozen()
        self.assertEqual(len(selected), record["count"])

    def test_overlap_with_a_held_out_split_stops_the_run(self):
        source = Path(self.tmp.name) / "train.jsonl"
        held_out = Path(self.tmp.name) / "val.jsonl"
        write_jsonl(source, ROWS)
        write_jsonl(held_out, [{"id": "unrelated", "question": "Different?", "answer": "0"}])
        selected, _ = make_rlvr_data.audit(str(source), 4, 42, (str(held_out),))
        self.assertEqual(len(selected), 4)
        for leaked in ({"id": "other-1", "question": selected[0]["question"], "answer": "0"},
                       {"id": selected[1]["id"], "question": "Different?", "answer": "0"}):
            write_jsonl(held_out, [leaked])
            with self.assertRaises(ValueError):
                make_rlvr_data.audit(str(source), 4, 42, (str(held_out),))

    def test_a_changed_frozen_list_is_refused(self):
        frozen = Path(self.tmp.name) / "frozen.json"
        _, record = make_rlvr_data.audit()
        frozen.write_text(json.dumps({**record, "ids": record["ids"][::-1]}), encoding="utf-8")
        with self.assertRaises(ValueError):
            make_rlvr_data.load_frozen(frozen)
        frozen.write_text(json.dumps({**record, "source_sha256": "stale"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            make_rlvr_data.load_frozen(frozen)

    def test_the_resolved_questions_carry_no_reference_answer_text(self):
        resolved = read_jsonl(make_rlvr_data.RESOLVED) if make_rlvr_data.RESOLVED.exists() else []
        for row in resolved[:5]:
            self.assertEqual(set(row), {"id", "question", "answer"})


if __name__ == "__main__":
    unittest.main()
