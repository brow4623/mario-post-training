"""Turn sampled candidates plus judge scores into DPO preference pairs.

Step 4 of the RLAIF plan, in two commands:

  1. flatten: one judge-ready record per sampled answer, id "<question id>#<answer index>".
       python make_prefs.py flatten data/candidates/mario_train.jsonl
       -> derived/candidates/mario_train.flat.jsonl
     then score it (Claude Code backend shown; see docs/EVALUATION.md):
       python judge.py score derived/candidates/mario_train.flat.jsonl --backend claude-code --workers 3
       -> results/judge/mario_train.flat.jsonl

  2. pairs: per question, chosen = highest judge overall score, rejected = lowest.
       python make_prefs.py pairs data/candidates/mario_train.jsonl results/judge/mario_train.flat.jsonl
       -> data/prefs.jsonl and provenance/prefs/mario_train.summary.json

Default rules, all adjustable by flag: chosen must be correct, rejected must be correct
(so the only difference DPO learns from is persona, not math), and chosen must beat
rejected by at least 1 point overall. Questions that cannot satisfy the rules are dropped
and counted by reason.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from mario_data import read_jsonl, write_jsonl

CRITERIA = ("voice", "consistency", "naturalness", "restraint", "overall")


def flat_id(question_id, index):
    return f"{question_id}#{index}"


def flatten(candidates):
    rows = []
    for record in candidates:
        for i, answer in enumerate(record["answers"]):
            rows.append({"id": flat_id(record["id"], i), "question": record["question"], "output": answer["text"]})
    return rows


def rank_key(scored):
    """Higher is better: overall first, then the sum of the four criteria, then not catchphrase-only."""
    return (scored["overall"], sum(scored[c] for c in CRITERIA[:-1]), not scored["catchphrase_only"])


def build_pairs(candidates, scores, require_chosen_correct=True, require_rejected_correct=True, min_gap=1):
    """Return (pairs, reasons). Each pair: {id, prompt, chosen, rejected, gold, chosen_score, rejected_score}."""
    pairs, reasons = [], Counter()
    for record in candidates:
        answers = []
        for i, answer in enumerate(record["answers"]):
            score = scores.get(flat_id(record["id"], i))
            if score is None:
                reasons["answer missing judge score"] += 1
                continue
            answers.append((i, answer, score))
        if len(answers) < 2:
            reasons["fewer than 2 scored answers"] += 1
            continue
        if len({a["text"] for _, a, _ in answers}) < 2:
            reasons["all answers identical"] += 1
            continue
        chosen_pool = [x for x in answers if x[1]["correct"] or not require_chosen_correct]
        if not chosen_pool:
            reasons["no correct answer to choose"] += 1
            continue
        chosen = max(chosen_pool, key=lambda x: (rank_key(x[2]), -x[0]))
        rejected_pool = [x for x in answers if x[0] != chosen[0] and x[1]["text"] != chosen[1]["text"]
                         and (x[1]["correct"] or not require_rejected_correct)]
        if not rejected_pool:
            reasons["no eligible rejected answer"] += 1
            continue
        rejected = min(rejected_pool, key=lambda x: (rank_key(x[2]), -x[0]))
        if chosen[2]["overall"] - rejected[2]["overall"] < min_gap:
            reasons[f"judge gap below {min_gap}"] += 1
            continue
        pairs.append({"id": record["id"], "prompt": record["question"], "chosen": chosen[1]["text"],
                      "rejected": rejected[1]["text"], "gold": record["gold"],
                      "chosen_score": chosen[2]["overall"], "rejected_score": rejected[2]["overall"],
                      "chosen_index": chosen[0], "rejected_index": rejected[0]})
    return pairs, reasons


def summary(candidates, pairs, reasons, args):
    gaps = Counter(p["chosen_score"] - p["rejected_score"] for p in pairs)
    return {"candidates": str(args.candidates), "scores": str(args.scores), "questions_in": len(candidates),
            "pairs_out": len(pairs), "dropped": dict(reasons),
            "rules": {"require_chosen_correct": not args.allow_wrong_chosen,
                      "require_rejected_correct": not args.allow_wrong_rejected, "min_gap": args.min_gap},
            "chosen_overall_mean": sum(p["chosen_score"] for p in pairs) / len(pairs) if pairs else None,
            "rejected_overall_mean": sum(p["rejected_score"] for p in pairs) / len(pairs) if pairs else None,
            "gap_distribution": {str(k): v for k, v in sorted(gaps.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("flatten", help="one judge-ready record per sampled answer")
    f.add_argument("candidates")
    f.add_argument("--output", help="default: derived/candidates/<name>.flat.jsonl")
    p = sub.add_parser("pairs", help="build preference pairs from candidates and judge scores")
    p.add_argument("candidates")
    p.add_argument("scores", help="judge output for the flattened file")
    p.add_argument("--output", default="data/prefs.jsonl")
    p.add_argument("--summary-dir", default="provenance/prefs")
    p.add_argument("--min-gap", type=int, default=1, help="minimum chosen minus rejected overall score")
    p.add_argument("--allow-wrong-chosen", action="store_true")
    p.add_argument("--allow-wrong-rejected", action="store_true")
    args = parser.parse_args()

    candidates = read_jsonl(args.candidates)
    if args.command == "flatten":
        output = Path(args.output) if args.output else Path("derived/candidates") / (Path(args.candidates).stem + ".flat.jsonl")
        rows = flatten(candidates)
        write_jsonl(output, rows)
        print(f"wrote {output}: {len(rows)} answers from {len(candidates)} questions")
        return
    scores = {r["id"]: r for r in read_jsonl(args.scores)}
    pairs, reasons = build_pairs(candidates, scores, not args.allow_wrong_chosen, not args.allow_wrong_rejected, args.min_gap)
    if not pairs:
        raise SystemExit(f"No pairs built. Dropped: {dict(reasons)}")
    write_jsonl(args.output, pairs)
    report = summary(candidates, pairs, reasons, args)
    summary_path = Path(args.summary_dir) / (Path(args.candidates).stem + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.output} ({len(pairs)} pairs) and {summary_path}")


if __name__ == "__main__":
    main()
