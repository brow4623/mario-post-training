"""Blinded Mario persona judge. Standard library only; run from repo root.

Scores each answer 1-5 on voice, consistency, naturalness, restraint, and overall,
using a fixed rubric (prompts/mario_judge.md) and JSON schema. The judge sees only the
question and the answer text: no gold solution, no condition label.

Two backends produce the same output format:
  api          One answer per request to the Claude API, temperature 0. Needs an API key.
  claude-code  Batches of answers per `claude -p` call, billed to a Claude subscription.
               Inputs are pooled and shuffled across files so every batch mixes conditions.

Score result files (resumable; already-scored records are skipped):
  python judge.py score results/mario_val.jsonl --env-file .env
  python judge.py score results/*_val.jsonl --backend claude-code --workers 3

Summarize score files, with paired differences against the first:
  python judge.py summarize results/judge/mario_val.jsonl results/judge/base_val.jsonl

Print a request without calling anything:
  python judge.py score results/mario_val.jsonl --dry-run
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request

from mario_data import read_jsonl, write_jsonl
from sonnet_rewrite import api_key

MODEL = "claude-haiku-4-5"
CLI_ALIASES = {"claude-haiku-4-5": "haiku", "claude-sonnet-5": "sonnet", "claude-opus-5": "opus"}
PROMPT_PATH = Path("prompts/mario_judge.md")
OUTPUT_ROOT = Path("results/judge")
PROVENANCE = Path("provenance/judge")
CRITERIA = ("voice", "consistency", "naturalness", "restraint", "overall")
RATING_PROPERTIES = {
    **{name: {"type": "integer", "enum": [1, 2, 3, 4, 5]} for name in CRITERIA},
    "catchphrase_only": {"type": "boolean"},
    "notes": {"type": "string"},
}
SCHEMA = {"type": "object", "properties": RATING_PROPERTIES,
          "required": [*CRITERIA, "catchphrase_only", "notes"], "additionalProperties": False}
BATCH_SCHEMA = {"type": "object", "properties": {"ratings": {"type": "array", "items": {
    "type": "object", "properties": {"index": {"type": "integer"}, **RATING_PROPERTIES},
    "required": ["index", *CRITERIA, "catchphrase_only", "notes"], "additionalProperties": False}}},
    "required": ["ratings"], "additionalProperties": False}
BATCH_INSTRUCTION = ("\n\nYou will receive several answers, each with an index. Rate every answer independently "
                     "on its own merits, not relative to the others. Return exactly one rating object per index.")
LOG_LOCK = threading.Lock()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def judge_config(backend, model, effort, max_tokens, batch_size):
    rubric = PROMPT_PATH.read_text(encoding="utf-8")
    if backend == "claude-code":
        return {"backend": backend, "model": model, "batch_size": batch_size,
                "system": rubric + BATCH_INSTRUCTION, "schema": BATCH_SCHEMA}
    return {"backend": backend, "model": model, "effort": effort, "max_tokens": max_tokens,
            "system": rubric, "schema": SCHEMA}


def request_body(config, question, answer):
    """API backend. The judge sees the question and answer only. Nothing else may be added here."""
    user = json.dumps({"question": question, "answer": answer}, ensure_ascii=False)
    body = {"model": config["model"], "max_tokens": config["max_tokens"],
            "output_config": {"format": {"type": "json_schema", "schema": config["schema"]}},
            "system": [{"type": "text", "text": config["system"], "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": "Rate this answer.\n\n" + user}]}
    if config["model"].startswith("claude-haiku-4-5"):
        # Haiku 4.5 has no adaptive thinking or effort setting; run it deterministically instead.
        body["temperature"] = 0
    else:
        body["thinking"] = {"type": "adaptive"}
        body["output_config"]["effort"] = config["effort"]
    return body


def batch_prompt(batch, text_field):
    """Claude Code backend. Index numbers only; no ids, gold answers, or file names."""
    items = [{"index": i, "question": row["question"], "answer": row[text_field]} for i, row in enumerate(batch)]
    return "Rate these answers.\n\n" + json.dumps(items, ensure_ascii=False)


def check_rating(data):
    for name in CRITERIA:
        if not isinstance(data[name], int) or isinstance(data[name], bool) or not 1 <= data[name] <= 5:
            raise ValueError(f"{name} must be an integer 1-5, got {data[name]!r}")
    if not isinstance(data["catchphrase_only"], bool) or not isinstance(data["notes"], str):
        raise ValueError("catchphrase_only must be boolean and notes must be a string")


def parse_scores(text):
    """Validate one rating against SCHEMA; raise ValueError on anything else."""
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != set(SCHEMA["required"]):
        raise ValueError(f"Unexpected keys: {sorted(data) if isinstance(data, dict) else type(data)}")
    check_rating(data)
    return data


def parse_batch(data, size):
    """Validate a batch response: exactly one rating for each index 0..size-1. Returns {index: rating}."""
    if not isinstance(data, dict) or list(data) != ["ratings"] or not isinstance(data["ratings"], list):
        raise ValueError("Batch response must be {'ratings': [...]}")
    ratings = {}
    for item in data["ratings"]:
        if not isinstance(item, dict) or set(item) != set(BATCH_SCHEMA["properties"]["ratings"]["items"]["required"]):
            raise ValueError(f"Unexpected rating keys: {sorted(item) if isinstance(item, dict) else type(item)}")
        index = item["index"]
        if not isinstance(index, int) or isinstance(index, bool) or index in ratings or not 0 <= index < size:
            raise ValueError(f"Bad or duplicate index {index!r}")
        check_rating(item)
        ratings[index] = {k: v for k, v in item.items() if k != "index"}
    if len(ratings) != size:
        raise ValueError(f"Got {len(ratings)} ratings for {size} answers")
    return ratings


def log_attempt(value):
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK:
        with (PROVENANCE / "attempts.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def record(row, scores, config_hash, text_field, model, request_id, usage, batch=None):
    out = {"id": row["id"], **scores, "text_sha256": text_hash(row[text_field]), "config_sha256": config_hash,
           "model": model, "request_id": request_id, "usage": usage}
    if batch is not None:
        out["batch"] = batch
    return out


def call_api(key, config, row, text_field):
    body = request_body(config, row["question"], row[text_field])
    for attempt in range(6):
        request = urllib.request.Request("https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode(), method="POST",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace").replace(key, "[REDACTED]")[:600]
            log_attempt({"id": row["id"], "http_status": error.code, "error": detail})
            if error.code not in (408, 409, 429, 500, 502, 503, 504, 529):
                raise RuntimeError(f"Claude HTTP {error.code}: {detail}") from None
            retry_after = error.headers.get("retry-after", "")
            delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else min(60, 2 ** (attempt + 1))
            time.sleep(min(delay, 180) + random.random())
            continue
        except (urllib.error.URLError, TimeoutError) as error:
            log_attempt({"id": row["id"], "transport_error": type(error).__name__})
            time.sleep(min(60, 2 ** (attempt + 1)))
            continue
        text = "".join(block["text"] for block in result.get("content", []) if block["type"] == "text").strip()
        event = {"id": row["id"], "request_id": result.get("id"), "model": result.get("model"),
                 "usage": result.get("usage", {}), "stop_reason": result.get("stop_reason"),
                 "seconds": round(time.monotonic() - started, 3)}
        try:
            if result.get("stop_reason") != "end_turn":
                raise ValueError(f"stop_reason={result.get('stop_reason')}")
            scores = parse_scores(text)
        except ValueError as error:
            event.update({"accepted": False, "raw": text[:400], "error": str(error)})
            log_attempt(event)
            continue
        event["accepted"] = True
        log_attempt(event)
        return [record(row, scores, digest(config), text_field, result.get("model", config["model"]),
                       result.get("id"), result.get("usage", {}))]
    raise RuntimeError(f"Retries exhausted for {row['id']}")


def call_claude_code(config, batch, text_field, batch_number, timeout=600):
    """One `claude -p` call rating a whole batch; returns one record per row."""
    command = ["claude", "-p", batch_prompt(batch, text_field), "--model", CLI_ALIASES.get(config["model"], config["model"]),
               "--output-format", "json", "--tools", "", "--no-session-persistence",
               "--system-prompt", config["system"], "--json-schema", json.dumps(config["schema"])]
    ids = [row["id"] for row in batch]
    for attempt in range(3):
        started = time.monotonic()
        try:
            # Normal batches finish in 1-3 minutes; a hung CLI process must not block a worker forever.
            proc = subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout)
        except subprocess.TimeoutExpired:
            log_attempt({"batch": batch_number, "ids": ids, "attempt": attempt, "accepted": False,
                         "error": f"claude -p timed out after {timeout}s"})
            continue
        event = {"batch": batch_number, "ids": ids, "attempt": attempt, "seconds": round(time.monotonic() - started, 1)}
        try:
            result = json.loads(proc.stdout)
            event.update({"request_id": result.get("session_id"), "usage": result.get("usage", {}),
                          "model": ",".join(result.get("modelUsage", {})), "cost_usd": result.get("total_cost_usd")})
            if result.get("is_error"):
                raise ValueError(f"claude error: {str(result.get('result'))[:300]}")
            structured = result.get("structured_output")
            if structured is None:
                structured = json.loads(result["result"])
            ratings = parse_batch(structured, len(batch))
        except (ValueError, KeyError) as error:
            event.update({"accepted": False, "error": str(error), "stderr": proc.stderr[:300]})
            log_attempt(event)
            time.sleep(5 * (attempt + 1))
            continue
        event["accepted"] = True
        log_attempt(event)
        usage = result.get("usage", {})
        batch_info = {"number": batch_number, "size": len(batch),
                      "usage": {k: usage.get(k) for k in ("input_tokens", "cache_creation_input_tokens",
                                                            "cache_read_input_tokens", "output_tokens")}}
        model = ",".join(result.get("modelUsage", {})) or config["model"]
        return [record(row, ratings[i], digest(config), text_field, model, result.get("session_id"), {}, batch_info)
                for i, row in enumerate(batch)]
    raise RuntimeError(f"Retries exhausted for batch {batch_number}: {ids}")


def score(args):
    global PROVENANCE
    PROVENANCE = Path(args.provenance_dir)
    config = judge_config(args.backend, args.model, args.effort, args.max_tokens, args.batch_size)
    config_hash = digest(config)
    inputs = {}
    for path in args.inputs:
        rows = read_jsonl(path)
        if args.limit:
            rows = rows[:args.limit]
        if any(not row.get("question") or not isinstance(row.get(args.text_field), str) for row in rows):
            raise ValueError(f"{path}: every record needs a question and a string {args.text_field!r} field")
        inputs[Path(path)] = rows
    if args.dry_run:
        rows = next(iter(inputs.values()))
        if args.backend == "api":
            print(json.dumps(request_body(config, rows[0]["question"], rows[0][args.text_field]), indent=2, ensure_ascii=False))
        else:
            print(config["system"], "\n---\n", batch_prompt(rows[:2], args.text_field))
        return
    outputs = {path: Path(args.output) if args.output else OUTPUT_ROOT / path.name for path in inputs}
    if args.output and len(inputs) > 1:
        raise ValueError("--output only works with a single input file")
    done = {path: {} for path in inputs}
    for path, output in outputs.items():
        if output.exists():
            for old in read_jsonl(output):
                if old["config_sha256"] == config_hash:
                    done[path][old["id"]] = old
    pending = [(path, row) for path, rows in inputs.items() for row in rows
               if row["id"] not in done[path] or done[path][row["id"]]["text_sha256"] != text_hash(row[args.text_field])]
    total = sum(len(rows) for rows in inputs.values())
    print(f"{total} records in {len(inputs)} file(s), {total - len(pending)} already scored, {len(pending)} to score")

    def save():
        for path, rows in inputs.items():
            write_jsonl(outputs[path], [done[path][row["id"]] for row in rows if row["id"] in done[path]])

    if pending:
        PROVENANCE.mkdir(parents=True, exist_ok=True)
        (PROVENANCE / "config.json").write_text(json.dumps({**config, "sha256": config_hash}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # Each job is (items, callable); items are (path, row) pairs and the callable returns one record per item.
        if args.backend == "api":
            key = api_key(args.env_file)
            jobs = [([item], lambda row=item[1]: call_api(key, config, row, args.text_field)) for item in pending]
        else:
            # Pool and shuffle across files so each batch mixes conditions; the judge sees index numbers only.
            order = list(pending)
            random.Random(42).shuffle(order)
            batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
            jobs = [(b, lambda b=b, n=n: call_claude_code(config, [row for _, row in b], args.text_field, n, args.call_timeout))
                    for n, b in enumerate(batches)]
        scored = 0

        def finish(items, records):
            nonlocal scored
            for (path, row), rec in zip(items, records, strict=True):
                done[path][row["id"]] = rec
                scored += 1
            save()
            print(f"  {scored}/{len(pending)} scored", flush=True)

        if args.backend == "api":
            # First call alone warms the prompt cache; the rest run concurrently.
            first, jobs = jobs[0], jobs[1:]
            finish(first[0], first[1]())
        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(job): items for items, job in jobs}
            for future in as_completed(futures):
                try:
                    finish(futures[future], future.result())
                except Exception as error:  # noqa: BLE001 - keep every finished batch, then report
                    failures.append(str(error)[:200])
                    if len(failures) == 1:
                        # Stop starting new batches (usually a usage limit); let running ones finish and be saved.
                        cancelled = sum(f.cancel() for f in futures)
                        print(f"  batch failed, cancelled {cancelled} unstarted batches: {failures[0]}", flush=True)
        if failures:
            raise RuntimeError(f"{len(failures)} batch(es) failed; {scored}/{len(pending)} saved. First: {failures[0]}")
    save()
    for path, rows in inputs.items():
        print(f"wrote {outputs[path]} ({sum(row['id'] in done[path] for row in rows)}/{len(rows)} records)")


def bootstrap_mean(values, resamples=10000, seed=42):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return [means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]]


def summarize(args):
    files = [Path(path) for path in args.files]
    scored = {}
    configs, models = set(), set()
    for path in files:
        rows = read_jsonl(path)
        indexed = {r['id']: r for r in rows}
        if not rows or len(indexed) != len(rows):
            raise ValueError(f'{path}: empty scores or duplicate IDs')
        if path.stem in scored:
            raise ValueError('Score file names must be distinct')
        scored[path.stem] = indexed
        configs.update(r['config_sha256'] for r in rows)
        models.update(r['model'] for r in rows)
    ids = sorted(next(iter(scored.values())))
    if any(set(records) != set(ids) for records in scored.values()):
        raise ValueError('Score files must contain exactly the same question IDs')
    if len(configs) != 1 or len(models) != 1:
        raise ValueError('Score files use different judge configurations or models')
    print(f"{len(ids)} shared ids across {len(files)} files\n")
    header = f"{'file':<24}" + "".join(f"{name:>13}" for name in CRITERIA) + f"{'catchphr.':>11}"
    print(header)
    for name, records in scored.items():
        line = f"{name:<24}"
        for criterion in CRITERIA:
            line += f"{statistics.mean(records[i][criterion] for i in ids):>13.2f}"
        line += f"{sum(records[i]['catchphrase_only'] for i in ids) / len(ids):>11.1%}"
        print(line)
    print(f"\nOverall score distribution (1..5):")
    for name, records in scored.items():
        counts = [sum(records[i]["overall"] == k for i in ids) for k in range(1, 6)]
        print(f"  {name:<22} {counts}")
    reference = files[0].stem
    print(f"\nPaired overall difference vs {reference} (95% bootstrap CI, {args.resamples} resamples, seed 42):")
    for name in list(scored)[1:]:
        diffs = [scored[name][i]["overall"] - scored[reference][i]["overall"] for i in ids]
        low, high = bootstrap_mean(diffs, args.resamples)
        print(f"  {name:<22} {statistics.mean(diffs):+.2f}  [{low:+.2f}, {high:+.2f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="score jsonl files of {id, question, output}")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--backend", choices=("api", "claude-code"), default="api")
    s.add_argument("--output", help=f"single input only; default: {OUTPUT_ROOT}/<input name>")
    s.add_argument("--text-field", default="output")
    s.add_argument("--model", default=MODEL)
    s.add_argument("--effort", default="low", help="api backend, non-Haiku models only")
    s.add_argument("--max-tokens", type=int, default=4096, help="api backend only")
    s.add_argument("--batch-size", type=int, default=20, help="claude-code backend only")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--call-timeout", type=int, default=600, help="claude-code backend: seconds before a hung call is retried")
    s.add_argument("--limit", type=int, help="first N records of each input")
    s.add_argument("--env-file", help="api backend: dotenv file holding ANTHROPIC_API_KEY")
    s.add_argument("--provenance-dir", default=str(PROVENANCE), help="Separate run records from earlier judge experiments")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=score)
    m = sub.add_parser("summarize", help="per-file means and paired differences vs the first file")
    m.add_argument("files", nargs="+")
    m.add_argument("--resamples", type=int, default=10000)
    m.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
