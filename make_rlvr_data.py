"""Freeze the RLVR training questions and audit them against validation and test.

Step 1 of docs/RLVR_PLAN.md: 512 questions drawn from the 3,000-question training split
with seed 42, no filtering by correctness or difficulty. Both training runs use this exact
subset, so it is frozen once and checked by `rlvr.py` on every load.

  python make_rlvr_data.py            # write the frozen list and the audit
  python make_rlvr_data.py --check    # recompute and compare; changes nothing

The audit goes to provenance/rlvr/training_questions.json (IDs and hashes only). The
resolved questions are written to derived/rlvr/train_questions.jsonl for inspection; they
are a deterministic derivative of data/train_mario.jsonl and are not tracked.
"""

import argparse
import hashlib
import json
from pathlib import Path

from eval import sha256, validate_rows
from mario_data import read_jsonl, write_jsonl
from sample_candidates import select_questions

SOURCE = "data/train_mario.jsonl"
HELD_OUT = ("data/val.jsonl", "data/test.jsonl")
FROZEN = Path("provenance/rlvr/training_questions.json")
RESOLVED = Path("derived/rlvr/train_questions.jsonl")
COUNT = 512
SEED = 42


def digest_list(values):
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def overlap(selected, path):
    """IDs and exact question text shared with a held-out split."""
    rows = read_jsonl(path)
    ids = {r["id"] for r in selected} & {r["id"] for r in rows}
    questions = {r["question"].strip() for r in selected} & {r["question"].strip() for r in rows}
    return {"split": path, "shared_ids": sorted(ids), "shared_questions": len(questions)}


def audit(source=SOURCE, count=COUNT, seed=SEED, held_out=HELD_OUT):
    """Select the frozen subset and check it is clean. Raises on any leak or duplicate."""
    rows = validate_rows(read_jsonl(source))
    if count > len(rows):
        raise ValueError(f"{source} has only {len(rows)} questions")
    selected = select_questions(rows, count, seed)
    questions = [r["question"].strip() for r in selected]
    if len(set(questions)) != len(selected):
        raise ValueError("Duplicate questions inside the selected subset")
    overlaps = [overlap(selected, path) for path in held_out]
    for leak in overlaps:
        if leak["shared_ids"] or leak["shared_questions"]:
            raise ValueError(f"Selected training questions overlap {leak['split']}: {leak}")
    record = {"source": source, "source_sha256": sha256(source), "seed": seed, "count": len(selected),
              "selection": "random.Random(seed).sample over dataset indices, kept in dataset order",
              "script_sha256": sha256(__file__), "ids_sha256": digest_list([r["id"] for r in selected]),
              "questions_sha256": digest_list(questions), "overlap": overlaps,
              "note": "Gold answers stay with the verifier; the policy and the persona judge see the question only.",
              "ids": [r["id"] for r in selected]}
    return selected, record


def load_frozen(path=FROZEN, source=None):
    """Reselect the subset and confirm it still matches the frozen list."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    selected, current = audit(source or record["source"], record["count"], record["seed"])
    for key in ("ids", "ids_sha256", "questions_sha256", "count"):
        if current[key] != record[key]:
            raise ValueError(f"{path}: frozen training questions no longer reproduce ({key} differs)")
    if current["source_sha256"] != record["source_sha256"]:
        raise ValueError(f"{path}: {record['source']} has changed since the questions were frozen")
    return selected, record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--count", type=int, default=COUNT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--frozen", default=str(FROZEN))
    parser.add_argument("--check", action="store_true", help="verify the saved list without writing it")
    args = parser.parse_args()
    selected, record = audit(args.source, args.count, args.seed)
    frozen = Path(args.frozen)
    if args.check:
        load_frozen(frozen, args.source)
        print(f"{frozen}: {record['count']} frozen training questions still reproduce, no held-out overlap")
        return
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != record:
        raise SystemExit(f"{frozen} already holds a different frozen subset; training has to use one fixed list")
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    write_jsonl(RESOLVED, [{"id": r["id"], "question": r["question"], "answer": r["answer"]} for r in selected])
    print(f"wrote {frozen} ({record['count']} questions, seed {record['seed']}) and {RESOLVED}")


if __name__ == "__main__":
    main()
