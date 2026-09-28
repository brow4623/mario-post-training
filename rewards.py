"""Verifiable and persona rewards for stage 3 GRPO training. Standard library only.

Both rewards start from the strict verifier in `verifier.py`, which needs an explicit
`####` or `\\boxed{}` marker and never falls back to the last number in the text:

    C = 1 if the verifier accepts the final answer, else 0
    P = (persona overall score - 1) / 4          # the 1-5 rubric, mapped to 0-1

    math      R = C
    combined  R = C * (1 + w * P)                # w defaults to 0.25

A correct answer earns 1 to 1.25 in the combined run; a wrong answer earns 0 however it
sounds. Only correct completions are sent to the judge, because P is multiplied by C.
Nothing rewards catchphrases, length, or formatting on their own.

Persona ratings come from the frozen Haiku rubric through `judge.call_api`, cached by
question, answer, and judge configuration, and paid through one ledger shared by retries
and resumed jobs. A judge request that keeps failing raises: a missing rating must stop
the update rather than silently become a zero persona score.

Summarize a training reward log (the pilot check from docs/RLVR_PLAN.md):
  python rewards.py summarize provenance/rlvr/pilot/rewards.jsonl --require-mixed-groups 10
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading

import judge
from mario_data import read_jsonl
from verifier import is_correct

PERSONA_WEIGHT = 0.25
MAX_SCORE = 5
KINDS = ("math", "combined")
RATING_FIELDS = (*judge.CRITERIA, "catchphrase_only", "notes")
JUDGE_BACKEND = "api"
JUDGE_MAX_TOKENS = 512
PROVENANCE = Path("provenance/rlvr")


def completion_text(completion):
    """Accept TRL's plain-text and conversational completion formats."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return completion[-1].get("content", "")
    raise TypeError(f"Unsupported completion format: {type(completion).__name__}")


def correctness(text, gold):
    """C: the strict verifier only. Never change this rule between conditions."""
    return 1.0 if is_correct(text, gold) else 0.0


def persona_scale(overall):
    """P: the judge's 1-5 overall score mapped to 0-1."""
    if not isinstance(overall, int) or isinstance(overall, bool) or not 1 <= overall <= MAX_SCORE:
        raise ValueError(f"persona overall must be an integer 1-{MAX_SCORE}, got {overall!r}")
    return (overall - 1) / (MAX_SCORE - 1)


def combined_reward(correct, scale, weight=PERSONA_WEIGHT):
    """One scalar, computed before GRPO forms group advantages."""
    return correct * (1 + weight * scale)


def judge_config(model=judge.MODEL, max_tokens=JUDGE_MAX_TOKENS):
    """The same rubric, schema, and API settings the evaluation judge uses."""
    return judge.judge_config(JUDGE_BACKEND, model, "low", max_tokens, 0)


def recorded_spend(provenance_dir):
    """Judge spend already on the ledger, so retries and resumed jobs share one cap."""
    path = Path(provenance_dir) / "attempts.jsonl"
    return sum(judge.haiku_cost(r.get("usage", {})) for r in read_jsonl(path)) if path.exists() else 0.0


class RatingCache:
    """Judge ratings keyed by question, answer, and the judge configuration hash.

    The configuration hash covers the rubric text, model, and API settings, so a changed
    rubric can never collide with a stored rating; a cache written under a different
    configuration is refused rather than mixed in.
    """

    def __init__(self, path, config_hash):
        self.path, self.config_hash = Path(path), config_hash
        self.lock = threading.Lock()
        self.entries, self.hits, self.misses = {}, 0, 0
        if self.path.exists():
            for row in read_jsonl(self.path):
                if row["config_sha256"] != config_hash:
                    raise ValueError(f"{self.path}: cached ratings use a different judge configuration")
                self.entries[row["key"]] = row

    def key(self, question, answer):
        return judge.digest({"question": question, "answer": answer, "config_sha256": self.config_hash})

    def get(self, question, answer):
        with self.lock:
            entry = self.entries.get(self.key(question, answer))
            self.hits, self.misses = self.hits + (entry is not None), self.misses + (entry is None)
        return {name: entry[name] for name in RATING_FIELDS} if entry else None

    def put(self, question, answer, rating, model=None, request_id=None, usage=None):
        key = self.key(question, answer)
        row = {"key": key, "config_sha256": self.config_hash, "question_sha256": judge.text_hash(question),
               "text_sha256": judge.text_hash(answer), "model": model, "request_id": request_id,
               "usage": usage or {}, **{name: rating[name] for name in RATING_FIELDS}}
        with self.lock:
            if key in self.entries:  # a concurrent duplicate; keep the first rating
                return
            self.entries[key] = row
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def stats(self):
        return {"path": str(self.path), "entries": len(self.entries), "hits": self.hits, "misses": self.misses}


class PersonaRater:
    """Cached persona ratings for training. Raises when the judge cannot be reached."""

    def __init__(self, config, api_key, cache, budget=None, call=None):
        self.config, self.api_key, self.cache, self.budget = config, api_key, cache, budget
        self.call = call or judge.call_api
        self.requests = 0

    def rate(self, question, answer, record_id=None):
        cached = self.cache.get(question, answer)
        if cached is not None:
            return cached
        row = {"id": record_id or judge.text_hash(question)[:12], "question": question, "output": answer}
        self.requests += 1
        # call_api retries transient failures and then raises; it never returns a default rating.
        result = self.call(self.api_key, self.config, row, "output", self.budget)[0]
        rating = {name: result[name] for name in RATING_FIELDS}
        judge.check_rating(rating)
        self.cache.put(question, answer, rating, result.get("model"), result.get("request_id"), result.get("usage"))
        return rating


def build_rater(env_file=None, model=judge.MODEL, max_tokens=JUDGE_MAX_TOKENS, cache_path=None,
                provenance_dir=None, budget_usd=None):
    """Wire the judge for training: frozen rubric, shared ledger, spend ceiling, cache."""
    from sonnet_rewrite import api_key

    provenance_dir = Path(provenance_dir or PROVENANCE / "judge")
    config = judge_config(model, max_tokens)
    config_hash = judge.digest(config)
    judge.PROVENANCE = provenance_dir  # call_api appends every attempt to this ledger
    provenance_dir.mkdir(parents=True, exist_ok=True)
    (provenance_dir / "config.json").write_text(
        json.dumps({**config, "sha256": config_hash}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    budget, spent = None, recorded_spend(provenance_dir)
    if budget_usd is not None:
        if not config["model"].startswith("claude-haiku-4-5") or budget_usd <= 0:
            raise ValueError("Budget accounting currently supports Haiku 4.5 with a positive USD limit")
        budget = judge.Budget(budget_usd, spent)
    cache = RatingCache(cache_path or PROVENANCE / "persona_cache.jsonl", config_hash)
    rater = PersonaRater(config, api_key(env_file), cache, budget)
    info = {"judge_config_sha256": config_hash, "judge_model": config["model"], "judge_max_tokens": max_tokens,
            "ledger": str(provenance_dir / "attempts.jsonl"), "recorded_spend_usd": spent,
            "budget_usd": budget_usd, "cache": str(cache.path), "cached_ratings": len(cache.entries)}
    return rater, info


class Reward:
    """A TRL reward function: one scalar per completion, logged with its components.

    `math` never calls the judge. `combined` calls it only for completions the verifier
    accepts, because the persona term is multiplied by correctness. Judge calls inside a
    batch run concurrently and are de-duplicated, so identical answers cost one request.
    """

    def __init__(self, kind, rater=None, weight=PERSONA_WEIGHT, log_path=None, workers=8,
                 sample_path=None, samples=0):
        if kind not in KINDS:
            raise ValueError(f"reward must be one of {KINDS}, got {kind!r}")
        if (rater is None) != (kind == "math"):
            raise ValueError("the combined reward needs a persona rater; the math reward must not have one")
        if not 0 <= weight <= 1:
            raise ValueError("persona weight must be between 0 and 1")
        self.__name__ = f"{kind}_reward"
        self.kind, self.rater, self.weight, self.workers = kind, rater, weight, max(1, workers)
        self.log_path = Path(log_path) if log_path else None
        self.sample_path = Path(sample_path) if sample_path else None
        self.samples, self.sampled = samples, 0
        self.calls, self.records = 0, []
        self.lock = threading.Lock()

    def score(self, question, text, gold, record_id=None):
        """One completion: (reward, components). The judge never sees the gold answer."""
        correct = correctness(text, gold)
        overall = scale = None
        if self.kind == "combined" and correct:
            overall = self.rater.rate(question, text, record_id)["overall"]
            scale = persona_scale(overall)
        reward = correct if self.kind == "math" else combined_reward(correct, 0.0 if scale is None else scale, self.weight)
        return reward, {"correct": correct, "persona_overall": overall, "persona_scale": scale, "reward": reward}

    def __call__(self, prompts=None, completions=None, *, question, gold, **kwargs):
        # TRL passes its columns by keyword; the leading positional parameters match older versions.
        texts = [completion_text(c) for c in completions or []]
        if not len(texts) == len(question) == len(gold):
            raise ValueError("completions, question, and gold must line up")
        ids = kwargs.get("id") or [None] * len(texts)
        with self.lock:
            call = self.calls
            self.calls += 1
        # One judge request per distinct (question, answer); repeats in a batch reuse it.
        unique = {}
        for i, (q, text) in enumerate(zip(question, texts)):
            unique.setdefault((q, text, gold[i]), []).append(i)
        keys = list(unique)

        def run(key):
            return self.score(key[0], key[1], key[2], ids[unique[key][0]])

        if self.kind == "combined" and self.workers > 1 and len(keys) > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                scored = list(pool.map(run, keys))
        else:
            scored = [run(key) for key in keys]
        rewards, records = [None] * len(texts), [None] * len(texts)
        for key, (reward, parts) in zip(keys, scored):
            for i in unique[key]:
                rewards[i] = reward
                records[i] = {"call": call, "reward_kind": self.kind, "id": ids[i],
                              "question_sha256": judge.text_hash(question[i]),
                              "text_sha256": judge.text_hash(texts[i]), **parts}
        self.log(records)
        self.sample(records, question, texts)
        return rewards

    def sample(self, records, questions, texts):
        """Full text for the first few completions, so a smoke test can hand-check rewards."""
        with self.lock:
            if not self.sample_path or self.sampled >= self.samples:
                return
            rows = [{**record, "question": question, "answer": text} for record, question, text
                    in list(zip(records, questions, texts))[:self.samples - self.sampled]]
            self.sampled += len(rows)
            self.sample_path.parent.mkdir(parents=True, exist_ok=True)
            with self.sample_path.open("a", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log(self, records):
        with self.lock:
            self.records.extend(records)
            if not self.log_path:
                return
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8", newline="\n") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def summary(self):
        report = summarize(self.records)
        if self.rater is not None:
            report["judge_requests"] = self.rater.requests
            report["cache"] = self.rater.cache.stats()
            if self.rater.budget is not None:
                report["judge_spend_usd"] = self.rater.budget.spent
        return report


def summarize(records):
    """Reward components and group diversity. A group is one question inside one call."""
    groups = defaultdict(list)
    for record in records:
        groups[(record["call"], record.get("id") or record["question_sha256"])].append(record)
    sizes = [sum(r["correct"] for r in rows) for rows in groups.values()]
    judged = [r["persona_overall"] for r in records if r["persona_overall"] is not None]
    correct = [r["correct"] for r in records]
    scales = [r["persona_scale"] for r in records if r["persona_scale"] is not None]
    mean = lambda values: sum(values) / len(values) if values else None  # noqa: E731
    return {"completions": len(records), "groups": len(groups),
            "mixed_correctness_groups": sum(0 < n < len(rows) for n, rows in zip(sizes, groups.values())),
            "all_correct_groups": sum(n == len(rows) for n, rows in zip(sizes, groups.values())),
            "all_wrong_groups": sum(n == 0 for n in sizes),
            "mean_correct": mean(correct), "judged_completions": len(judged),
            "mean_persona_overall": mean(judged), "mean_persona_scale": mean(scales),
            "mean_reward": mean([r["reward"] for r in records])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summarize", help="reward components and mixed-correctness groups from a reward log")
    s.add_argument("logs", nargs="+")
    s.add_argument("--require-mixed-groups", type=int,
                   help="exit non-zero below this many mixed-correctness groups (the pilot check)")
    args = parser.parse_args()
    records = [record for path in args.logs for record in read_jsonl(path)]
    if not records:
        raise SystemExit("No reward records found")
    report = summarize(records)
    print(json.dumps(report, indent=2))
    if args.require_mixed_groups is not None and report["mixed_correctness_groups"] < args.require_mixed_groups:
        print(f"Only {report['mixed_correctness_groups']} of {report['groups']} groups mix correct and wrong answers; "
              f"GRPO learns nothing from the rest. Reconsider the answers per question, symmetrically in both "
              f"conditions, and redo the cost estimate before the full run.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
