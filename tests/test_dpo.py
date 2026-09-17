"""Offline checks for the DPO pairs loader. Also builds a smoke-test pairs file on demand:
  python -m tests.test_dpo --write /tmp/prefs.jsonl   # Mario response vs plain response, same questions
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dpo  # noqa: E402
from eval import FORMAT_PROMPT  # noqa: E402
from mario_data import read_jsonl  # noqa: E402

PAIR = {"id": "train-1", "prompt": "4 entrees at $20. Total?", "chosen": "Mamma mia! 4 * 20 = 80\n#### 80",
        "rejected": "4 * 20 = 80\n#### 80", "gold": "80"}


def smoke_pairs(limit=16):
    """Chosen = Mario training response, rejected = plain response for the same question."""
    plain = {r["id"]: r for r in read_jsonl(ROOT / "data/train_plain.jsonl")}
    pairs = []
    for row in read_jsonl(ROOT / "data/train_mario.jsonl")[:limit]:
        pairs.append({"id": row["id"], "prompt": row["question"], "chosen": row["response"],
                      "rejected": plain[row["id"]]["response"], "gold": row["answer"]})
    return pairs


class Loader(unittest.TestCase):
    def write(self, rows):
        path = Path(self.tmp.name) / "prefs.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return path

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_conversational_rows_carry_the_eval_system_prompt(self):
        rows = dpo.load_pairs(self.write([PAIR]))
        self.assertEqual(rows[0]["prompt"][0], {"role": "system", "content": FORMAT_PROMPT})
        self.assertEqual(rows[0]["prompt"][1]["content"], PAIR["prompt"])
        self.assertEqual(rows[0]["chosen"][0]["content"], PAIR["chosen"])
        self.assertEqual(rows[0]["rejected"][0]["content"], PAIR["rejected"])

    def test_rejects_bad_pairs(self):
        for bad in ({**PAIR, "chosen": PAIR["rejected"]},          # identical
                    {**PAIR, "rejected": ""},                      # empty
                    {k: v for k, v in PAIR.items() if k != "prompt"},
                    {**PAIR, "chosen": "Wahoo! 4 * 20 = 90\n#### 90"}):  # chosen wrong vs gold
            with self.assertRaises(ValueError):
                dpo.load_pairs(self.write([bad]))
        with self.assertRaises(ValueError):
            dpo.load_pairs(self.write([PAIR, PAIR]))                # duplicate
        with self.assertRaises(ValueError):
            dpo.load_pairs(self.write([]))

    def test_skip_verify_allows_wrong_chosen(self):
        wrong = {**PAIR, "chosen": "Wahoo! 4 * 20 = 90\n#### 90"}
        self.assertEqual(len(dpo.load_pairs(self.write([wrong]), verify=False)), 1)

    def test_smoke_pairs_are_valid(self):
        rows = dpo.load_pairs(self.write(smoke_pairs(8)))
        self.assertEqual(len(rows), 8)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--write":
        out = Path(sys.argv[2])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in smoke_pairs()), encoding="utf-8")
        print(f"wrote {out}")
    else:
        unittest.main()
