"""Resumable, cached Sonnet rewrites. Standard library only; run from repo root."""

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import random
import re
import threading
import time
import urllib.error
import urllib.request

from assemble_rewrites import assemble
from mario_data import read_jsonl, write_jsonl

MODEL = "claude-sonnet-5"
ROOT = Path("provenance/generation")
LOG_LOCK = threading.Lock()


class RecordFailure(RuntimeError):
    """A bounded per-record failure; other independent examples can continue."""


@contextmanager
def run_lock():
    """OS-released lock prevents two runners from paying for the same missing IDs."""
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "run.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Another Sonnet runner is already active") from None
        yield


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def api_key(env_file=None):
    names = ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    if env_file:
        for line in Path(env_file).read_text(encoding="utf-8-sig").splitlines():
            match = re.match(r"\s*(?:export\s+)?(ANTHROPIC_API_KEY|CLAUDE_API_KEY)\s*=\s*(.*?)\s*$", line)
            if match:
                value = match[2].strip()
                if value[:1] in ("'", '"') and value[-1:] == value[:1]:
                    value = value[1:-1]
                else:
                    value = value.split(" #", 1)[0].strip()
                if value:
                    return value
    raise ValueError("Set ANTHROPIC_API_KEY or CLAUDE_API_KEY, or provide --env-file")


def prompt_prefix():
    instructions = Path("prompts/mario_rewrite.md").read_text(encoding="utf-8")
    originals = {row["id"]: row for row in read_jsonl("data/original/train.jsonl")}
    blocks = re.split(r"(?m)^## ", Path("prompts/mario_examples.md").read_text(encoding="utf-8"))[1:5]
    examples = []
    for block in blocks:
        key = re.search(r"gsm8k/train/\d+", block)[0]
        source = originals[key]  # Fail if an exemplar is not in our training split.
        answer = block.split("**Rewritten answer:**", 1)[1].strip()
        assemble([source], [{"id": key, "answer": answer}])
        examples.append({"question": source["question"], "reference": source["answer"], "rewrite": answer})
    return instructions + "\n\nThe following FOUR training-only examples define the desired voice. " \
        "Imitate their contextual performance, not their sentences. No prefatory explanation, JSON, or markdown fences. " \
        "Return just the rewritten answer. Keep every source calculation annotation and exact final line. " \
        "Do not add headings.\n" + json.dumps(examples, ensure_ascii=False, indent=2)


def request_body(prefix, row, feedback=""):
    return {"model": MODEL, "max_tokens": 3072,
            "thinking": {"type": "adaptive"}, "output_config": {"effort": "low"},
            "system": [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": json.dumps(
                {"question": row["question"], "reference": row["answer"], "validation_feedback": feedback}, ensure_ascii=False)}]}


def log_attempt(value):
    with LOG_LOCK:
        with (ROOT / "attempts.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            stream.flush()


def rewrite(job, key, prefix, config_hash):
    split, row = job
    feedback = ""
    for attempt in range(6):
        request = urllib.request.Request("https://api.anthropic.com/v1/messages",
            data=json.dumps(request_body(prefix, row, feedback)).encode(), method="POST",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace").replace(key, "[REDACTED]")[:600]
            log_attempt({"id": row["id"], "split": split, "http_status": error.code, "error": detail})
            if error.code not in (408, 409, 429, 500, 502, 503, 504, 529):
                raise RuntimeError(f"Claude HTTP {error.code}: {detail}") from None
            retry_after = error.headers.get("retry-after", "")
            delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else min(60, 2 ** (attempt + 1))
            time.sleep(min(delay, 180) + random.random())
            continue
        except (urllib.error.URLError, TimeoutError) as error:
            log_attempt({"id": row["id"], "split": split, "transport_error": type(error).__name__,
                         "billing_status": "unknown; retry may incur another request"})
            time.sleep(min(60, 2 ** (attempt + 1)))
            continue
        answer = "".join(block["text"] for block in result.get("content", []) if block["type"] == "text").strip()
        event = {"id": row["id"], "split": split, "request_id": result.get("id"),
                 "model": result.get("model"), "usage": result.get("usage", {}),
                 "stop_reason": result.get("stop_reason"), "seconds": round(time.monotonic() - started, 3)}
        try:
            if result.get("stop_reason") != "end_turn":
                raise ValueError("Completion did not finish normally")
            assemble([row], [{"id": row["id"], "answer": answer}])
        except ValueError as error:
            event.update({"accepted": False, "answer": answer, "error": str(error)})
            log_attempt(event)
            feedback = {"instruction": "Regenerate the complete answer. Correct the exact mismatch below. "
                        "Do not include an incorrect calculation followed by a self-correction. "
                        "Include each required annotation exactly once in order, including any repeated source annotations. "
                        "If required_annotations is empty, use no <<...>> annotations at all; write calculations as plain text. "
                        "Return only the final complete, concise Mario explanation.",
                        "required_annotations": re.findall(r"<<.*?>>", row["answer"]),
                        "previous_annotations": re.findall(r"<<.*?>>", answer),
                        "required_final_line": row["answer"].strip().splitlines()[-1],
                        "previous_final_line": answer.splitlines()[-1:]}
            continue
        event["accepted"] = True
        log_attempt(event)
        saved = {"id": row["id"], "answer": answer, "split": split, "source_hash": digest(row),
                 "config_hash": config_hash, "model": result.get("model", MODEL), "request_id": result.get("id"),
                 "usage": result.get("usage", {})}
        save_json(ROOT / "records" / split / (row["id"].replace("/", "_") + ".json"), saved)
        return saved
    raise RecordFailure(f"Retries exhausted for {row['id']}; deferred for a later retry")


def inventory(config_hash):
    pending, combined, provenance = [], {}, {}
    astra_records = read_jsonl(ROOT / "astra.jsonl")
    sonnet_records = read_jsonl(ROOT / "sonnet.jsonl") if (ROOT / "sonnet.jsonl").exists() else []
    sonnet_records += [json.loads(path.read_text(encoding="utf-8"))
                       for path in sorted((ROOT / "records").rglob("*.json"))]
    if any(row.get("split") not in ("train", "validation") for row in astra_records + sonnet_records):
        raise ValueError("Unknown split in generation records")
    for split in ("train", "validation"):
        references = read_jsonl(f"data/original/{split}.jsonl")
        original_by_id = {row["id"]: row for row in references}
        astra = [row for row in astra_records if row["split"] == split]
        assemble(references, astra, allow_partial=True)
        saved = {row["id"]: {"id": row["id"], "answer": row["answer"],
                            "provenance": {"model": "gpt-6-astra", "reasoning_effort": "medium"}} for row in astra}
        for row in (row for row in sonnet_records if row["split"] == split):
            if row["id"] in saved or row["id"] not in original_by_id:
                raise ValueError(f"Duplicate or unknown saved ID: {row['id']}")
            if row["config_hash"] != config_hash or row["source_hash"] != digest(original_by_id[row["id"]]):
                raise ValueError(f"Configuration/source changed for {row['id']}; refusing silent reuse")
            assemble([original_by_id[row["id"]]], [row])
            saved[row["id"]] = {"id": row["id"], "answer": row["answer"],
                                "provenance": {"model": row["model"], "reasoning_effort": "low", "request_id": row["request_id"]}}
        combined[split] = [saved[row["id"]] for row in references if row["id"] in saved]
        for row in combined[split]:
            model = row["provenance"]["model"]
            provenance[model] = provenance.get(model, 0) + 1
        pending.extend((split, row) for row in references if row["id"] not in saved)
    return pending, combined, provenance


def export_results(config_hash):
    pending, combined, provenance = inventory(config_hash)
    destination = Path("derived/rewrites")
    for split, rows in combined.items():
        write_jsonl(destination / f"{split}_all.jsonl", rows)
    save_json(destination / "generation.json", {"model": "mixed Astra and Sonnet", "reasoning_effort": "per-record",
        "method": "475 saved Astra subagent rewrites plus validated Sonnet API rewrites",
        "transformation": "mario-v2-context-aware-mixed", "models": provenance,
        "sonnet_config_hash": config_hash, "source_shard_directory": str(destination).replace("\\", "/")})
    usage = {key: 0 for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
    attempts = read_jsonl(ROOT / "attempts.jsonl") if (ROOT / "attempts.jsonl").exists() else []
    for attempt in attempts:
        for key in usage:
            usage[key] += attempt.get("usage", {}).get(key, 0)
    total_input = usage["input_tokens"] + usage["cache_creation_input_tokens"] + usage["cache_read_input_tokens"]
    report = {"remaining": len(pending), "models": provenance, "api_responses": sum("usage" in a for a in attempts),
        "rejected_responses": sum(a.get("accepted") is False for a in attempts), "usage": usage,
        "cached_input_fraction": usage["cache_read_input_tokens"] / total_input if total_input else 0,
        "estimated_usd": (usage["input_tokens"] * 2 + usage["output_tokens"] * 10 +
                          usage["cache_creation_input_tokens"] * 2.5 + usage["cache_read_input_tokens"] * .2) / 1e6,
        "cost_basis": "Sonnet5 standard rates checked2026-09-10; 5m cache. Reported usage only, not billing invoice."}
    save_json(destination / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file")
    parser.add_argument("--limit", type=int, default=0, help="New records to process; 0 means all remaining")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or not 1 <= args.workers <= 16:
        parser.error("limit must be nonnegative and workers must be1..16")
    ROOT.mkdir(parents=True, exist_ok=True)
    prefix = prompt_prefix()
    config = {"model": MODEL, "effort": "low", "thinking": "adaptive", "max_tokens": 3072,
              "prefix": prefix, "cache_ttl": "5m"}
    config_hash = digest(config)
    config_path = ROOT / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Saved prompt/settings differ; use a new run directory rather than mixing configurations")
    save_json(config_path, config)
    if args.export_only:
        export_results(config_hash)
        return
    key = api_key(args.env_file)
    pending, _, _ = inventory(config_hash)
    if args.limit:
        pending = pending[:args.limit]
    print(f"Processing {len(pending)} missing records with {MODEL}, low effort, workers={args.workers}", flush=True)
    try:
        if pending:
            # The first real request warms the static prefix before concurrent requests start.
            failed = 0
            try:
                first = rewrite(pending[0], key, prefix, config_hash)
                print("Warm-up usage: " + json.dumps(first["usage"]), flush=True)
            except RecordFailure as error:
                failed += 1
                print(str(error), flush=True)
            jobs = iter(pending[1:])
            completed = 1
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(rewrite, job, key, prefix, config_hash) for _, job in zip(range(args.workers), jobs)}
                while futures:
                    ready, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in ready:
                        try:
                            future.result()
                        except RecordFailure as error:
                            failed += 1
                            print(str(error), flush=True)
                        completed += 1
                        if completed % 25 == 0:
                            print(f"Processed {completed}/{len(pending)}; saved={completed-failed}, deferred={failed}", flush=True)
                        job = next(jobs, None)
                        if job is not None:
                            futures.add(pool.submit(rewrite, job, key, prefix, config_hash))
            if failed:
                raise RecordFailure(f"Run finished with {failed} deferred records; rerun to retry only missing IDs")
    finally:
        export_results(config_hash)


if __name__ == "__main__":
    with run_lock():
        main()
