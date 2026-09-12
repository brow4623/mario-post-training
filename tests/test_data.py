import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import unittest

from check_answers import audit, extract_answer, restore_original, score
from mario_data import mario_answer, read_jsonl, record


class AnswerTests(unittest.TestCase):
    def test_equivalent_numbers(self):
        for text in ("#### 1,000", "work\n#### 1000.00", "#### +1000", "#### 2000/2"):
            self.assertEqual(extract_answer(text), Fraction(1000))
        self.assertEqual(extract_answer("#### -0.5"), Fraction(-1, 2))
        self.assertEqual(extract_answer("#### 1.5/3"), Fraction(1, 2))

    def test_reject_ambiguous_or_unsafe(self):
        for text in ("the answer is 42", "#### 2\n#### 3", "#### 1,00", "#### 1/0",
                     "#### 1 or 2", "#### 2 apples", "#### 2+2", "#### nan", "#### 2\nmore text", None):
            with self.subTest(text=text):
                self.assertIsNone(extract_answer(text))

    def test_score_missing_and_malformed(self):
        refs = [{"id": str(i), "answer": "#### 2"} for i in range(3)]
        summary, _ = score(refs, [{"id": "0", "output": "#### 2.0"}, {"id": "1", "output": "2"}])
        self.assertEqual(summary["accuracy"], 1 / 3)
        self.assertEqual(summary["missing"], 1)
        self.assertEqual(summary["valid_format"], 1)

    def test_duplicate_and_unknown_ids(self):
        refs = [{"id": "a", "answer": "#### 2"}]
        for preds in ([{"id": "z", "output": "#### 2"}], [{"id": "a", "output": "#### 2"}] * 2):
            with self.assertRaises(ValueError):
                score(refs, preds)

    def test_empty_and_invalid_gold(self):
        with self.assertRaises(ValueError):
            score([], [])
        with self.assertRaises(ValueError):
            score([{"id": "a", "answer": "bad"}], [])

    def test_empty_predictions_count_wrong(self):
        summary, _ = score([{"id": "a", "answer": "#### 2"}], [])
        self.assertEqual(summary["accuracy"], 0)


class TransformationTests(unittest.TestCase):
    row = {"question": "How many?", "answer": "There are 2.\nAdd 2 to get <<2+2=4>>4.\n#### 4"}

    def test_determinism_and_recovery(self):
        for i in range(100):
            transformed = mario_answer(self.row["answer"], str(i), 42)
            self.assertEqual(transformed, mario_answer(self.row["answer"], str(i), 42))
            self.assertEqual(restore_original(transformed), self.row["answer"])

    def test_audit_and_mutation(self):
        original = record(self.row, 1, "train", False, 42)
        modified = record(self.row, 1, "train", True, 42)
        self.assertEqual(audit([original], [modified])[0]["failed"], 0)
        for field, value in (("question", "different"), ("answer", modified["answer"].replace("#### 4", "#### 5")),
                             ("answer", modified["answer"].replace("There are 2", "There are 3"))):
            broken = copy.deepcopy(modified)
            broken[field] = value
            self.assertEqual(audit([original], [broken])[0]["failed"], 1)

    def test_missing_pair(self):
        with self.assertRaises(ValueError):
            audit([record(self.row, 1, "train", False, 42)], [record(self.row, 2, "train", True, 42)])


@unittest.skipUnless(Path("provenance/data_manifest.json").exists(), "Dataset provenance is unavailable")
class DownloadedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path('data/mario/train.jsonl').exists() or not Path('derived/rewrites/train_all.jsonl').exists():
            from prepare_data import main
            main()

    def test_training_exports(self):
        manifest = json.loads(Path('provenance/data_manifest.json').read_text(encoding='utf-8'))
        for name, expected in manifest['prepared_files'].items():
            self.assertEqual(hashlib.sha256((Path('data') / name).read_bytes()).hexdigest(), expected)

    def test_hashes(self):
        manifest = json.loads(Path("provenance/data_manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            self.assertEqual(hashlib.sha256((Path("data") / name).read_bytes()).hexdigest(), expected)
        for name, source in manifest["sources"].items():
            self.assertEqual(hashlib.sha256((Path("data/raw") / name).read_bytes()).hexdigest(), source["sha256"])

    def test_all_pairs_and_sizes(self):
        manifest = json.loads(Path("provenance/data_manifest.json").read_text(encoding="utf-8"))
        mode = "rewrite" if manifest["transformation"].startswith("mario-v2") else "legacy"
        for split, count in (("train", 3000), ("validation", 500)):
            original = read_jsonl(f"data/original/{split}.jsonl")
            modified = read_jsonl(f"data/mario/{split}.jsonl")
            self.assertEqual(len(original), count)
            self.assertEqual(audit(original, modified, mode=mode)[0]["failed"], 0)
        self.assertEqual(len(read_jsonl("data/original/test.jsonl")), 1319)

    def test_splits_disjoint(self):
        splits = [read_jsonl(f"data/original/{split}.jsonl") for split in ("train", "validation", "test")]
        for field in ("id", "question"):
            sets = [{row[field] for row in split} for split in splits]
            for i in range(len(sets)):
                self.assertEqual(len(sets[i]), len(splits[i]))
                for j in range(i):
                    self.assertFalse(sets[i] & sets[j])

    def test_source_fidelity_and_reproducibility(self):
        raw = read_jsonl("data/raw/train.jsonl")
        manifest = json.loads(Path("provenance/data_manifest.json").read_text(encoding="utf-8"))
        for split in ("train", "validation"):
            indices = manifest["source_train_indices"][split]
            original = [record(raw[i], i, "train", False, manifest["seed"]) for i in indices]
            self.assertEqual(original, read_jsonl(f"data/original/{split}.jsonl"))
            if manifest["transformation"].startswith("mario-v2"):
                from assemble_rewrites import assemble
                directory = manifest["generation"].get("source_shard_directory", "provenance/generation")
                shards = [row for path in sorted(Path(directory).glob(f"{split}_*.jsonl"))
                          for row in read_jsonl(path)]
                modified, _ = assemble(original, shards)
            else:
                modified = [record(raw[i], i, "train", True, manifest["seed"]) for i in indices]
            self.assertEqual(modified, read_jsonl(f"data/mario/{split}.jsonl"))

    def test_rewrite_provenance_hashes(self):
        manifest = json.loads(Path("provenance/data_manifest.json").read_text(encoding="utf-8"))
        for path, expected in manifest.get("generation", {}).get("shards", {}).items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
