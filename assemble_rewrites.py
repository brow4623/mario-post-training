"""Validate individually authored rewrite shards and assemble a complete paired dataset.

This does not generate prose or certify semantic fidelity/persona quality.
Partial batches produce a review file, never a partial canonical train split.
"""

import argparse
import hashlib
import json
from pathlib import Path

from check_answers import audit, index_rows, score
from mario_data import read_jsonl, write_jsonl


def assemble(references, rewrites, allow_partial=False):
    originals = index_rows(references)
    rewritten = index_rows(rewrites)
    unknown = rewritten.keys() - originals.keys()
    missing = originals.keys() - rewritten.keys()
    if unknown:
        raise ValueError(f"{len(unknown)} unknown rewrite IDs")
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)} missing rewrites; refusing an incomplete dataset")
    selected = [row for row in references if row["id"] in rewritten]
    candidates = []
    for row in selected:
        answer = rewritten[row["id"]].get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Invalid answer for {row['id']}")
        candidates.append({**row, "answer": answer, "messages": [
            *row["messages"][:-1], {"role": "assistant", "content": answer}]})
        if "provenance" in rewritten[row["id"]]:
            candidates[-1]["rewrite_provenance"] = rewritten[row["id"]]["provenance"]
    summary, details = audit(selected, candidates, mode="rewrite")
    if summary["failed"]:
        failed = [row["id"] for row in details if not row["passed"]]
        raise ValueError(f"Rewrite validation failed: {failed}")
    return candidates, {**summary, "expected_total": len(references),
                        "missing": len(missing), "complete": not missing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="all")
    parser.add_argument("--shards", default="derived/rewrites")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--publish", action="store_true",
                        help="Publish both complete, validated splits to data/mario and update the manifest")
    args = parser.parse_args()
    if args.publish and (args.allow_partial or args.split != "all"):
        parser.error("Publishing requires both complete splits: --split all without --allow-partial")
    splits = ("train", "validation") if args.split == "all" else (args.split,)
    prepared = {}
    try:
        metadata = json.loads((Path(args.shards) / "generation.json").read_text(encoding="utf-8"))
        if not all(metadata.get(key) for key in ("model", "reasoning_effort", "method")):
            raise ValueError("generation.json must identify the model, reasoning_effort, and method")
        for split in splits:
            paths = sorted(Path(args.shards).glob(f"{split}_*.jsonl"))
            rewrites = [row for path in paths for row in read_jsonl(path)]
            references = read_jsonl(f"data/original/{split}.jsonl")
            rows, report = assemble(references, rewrites, args.allow_partial)
            report["shards"] = {str(path).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
            report["generation"] = metadata
            prepared[split] = (references, rows, report)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    # Validate all requested splits before touching any output, especially canonical data.
    root = Path("data/mario" if args.publish else "data/mario_v2")
    root.mkdir(parents=True, exist_ok=True)
    suffix = ".preview" if args.allow_partial else ""
    for split, (references, rows, report) in prepared.items():
        output = root / f"{split}{suffix}.jsonl"
        write_jsonl(output, rows)
        report["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        audit_root = Path("derived/audits")
        audit_root.mkdir(parents=True, exist_ok=True)
        (audit_root / f"{split}.report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
        if args.publish:
            for name, result in ((f"{split}_integrity", audit(references, rows, mode="rewrite")),
                                 (f"mario_{split}_answers", score(references, rows, field="answer"))):
                summary, details = result
                path = audit_root / f"{name}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
                write_jsonl(path.with_suffix(".details.jsonl"), details)
        print(json.dumps({key: value for key, value in report.items() if key != "shards"}, indent=2))
    if args.publish:
        path = Path("provenance/data_manifest.json")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["transformation"] = metadata.get("transformation", "mario-v2-context-aware-astra-medium")
        manifest["generation"] = {**prepared["train"][2]["generation"],
                                  "prompt_sha256": hashlib.sha256(Path("prompts/mario_rewrite.md").read_bytes()).hexdigest(),
                                  "shards": {key: value for _, _, report in prepared.values() for key, value in report["shards"].items()}}
        manifest["files"] = {str(p.relative_to("data")).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                             for variant in ("original", "mario") for p in sorted((Path("data") / variant).glob("*.jsonl"))}
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
