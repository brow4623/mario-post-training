"""Compare the three stage-3 conditions: math accuracy and persona as separate outcomes.

Step 5 of docs/RLVR_PLAN.md. Reads the matched evaluation outputs and their persona
ratings and reports paired differences with 95% question-bootstrap intervals:

  rlvr_baseline   the released RLAIF adapter, re-evaluated in the same environment
  rlvr_math       GRPO with R = C
  rlvr_combined   GRPO with R = C * (1 + 0.25 * P)

Accuracy and persona stay separate outcomes; the training reward is never a headline
metric. Predeclared criteria: accuracy improved when the paired interval excludes zero on
the positive side, persona retained when its lower bound is above -0.20 points.

  python analyze_rlvr.py summarize            # tables and derived/rlvr/summary.json
  python analyze_rlvr.py review               # blinded 50-question, 3-condition packet
  python analyze_rlvr.py plot                 # reports/figures/rlvr_accuracy_persona.png

Standard library only, except `plot`, which needs Pillow. The bootstrap is pure Python and
takes a few minutes at the default resample count; use --resamples for a quick look.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics

from compare import paired_outputs
from evaluate_persona import aligned_scores
from judge import CRITERIA, bootstrap_mean, text_hash
from mario_data import read_jsonl
from verifier import is_correct

CONDITIONS = ("rlvr_baseline", "rlvr_math", "rlvr_combined")
LABELS = {"rlvr_baseline": "RLAIF baseline", "rlvr_math": "Math-only RLVR", "rlvr_combined": "Combined reward"}
COMPARISONS = (("rlvr_math", "rlvr_baseline"), ("rlvr_combined", "rlvr_baseline"), ("rlvr_combined", "rlvr_math"))
SPLITS = ("val", "test")
# The predeclared retention margin: a practical project choice, not a validated standard.
PERSONA_LOSS_LIMIT = 0.20
LENGTH_RATIO = 1.1
MATCHED_KEYS = ("model", "revision", "data_sha256", "n", "system_prompt", "max_new_tokens",
                "batch_size", "do_sample", "dtype", "packages", "eval_sha256", "verifier_sha256")
RESULTS = Path("results/rlvr")
PROVENANCE = Path("provenance/rlvr")
SUMMARY = Path("derived/rlvr/summary.json")

# `stages` spans all four post-training stages. They sit in different directories because
# earlier checkpoints of this project wrote them, so each needs its own source paths.
STAGES = ("mario_sft", "rlaif", "rlvr_math", "rlvr_combined")
STAGE_LABELS = {"mario_sft": "Mario SFT", "rlaif": "RLAIF", "rlvr_math": "RLVR math-only",
                "rlvr_combined": "RLVR combined"}
STAGE_SOURCES = {  # stage -> (results dir, evaluation provenance dir, file stem)
    "mario_sft": (Path("results"), Path("provenance/evaluation"), "mario_rlaif_baseline"),
    "rlaif": (RESULTS, Path("provenance/evaluation/rlvr"), "rlvr_baseline"),
    "rlvr_math": (RESULTS, Path("provenance/evaluation/rlvr"), "rlvr_math"),
    "rlvr_combined": (RESULTS, Path("provenance/evaluation/rlvr"), "rlvr_combined"),
}
STAGE_COMPARISONS = (("rlaif", "mario_sft"), ("rlvr_math", "rlaif"),
                     ("rlvr_combined", "rlaif"), ("rlvr_combined", "mario_sft"))
STAGE_SUMMARY = Path("derived/rlvr/stages.json")


def paired(a, b, resamples=10000):
    """Mean paired difference (b minus a) with a 95% question-bootstrap interval."""
    values = [float(y) - float(x) for x, y in zip(a, b, strict=True)]
    if not values:
        return None
    return {"mean": statistics.mean(values), "ci95": bootstrap_mean(values, resamples)}


def mean_with_interval(values, resamples=10000):
    values = [float(v) for v in values]
    return {"mean": statistics.mean(values), "ci95": bootstrap_mean(values, resamples)}


def verdicts(accuracy, persona):
    """The criteria frozen before the runs: improvement excludes zero, retention beats -0.20."""
    return {"accuracy_improved": accuracy["ci95"][0] > 0,
            "accuracy_lost": accuracy["ci95"][1] < 0,
            "persona_retained": persona["ci95"][0] > -PERSONA_LOSS_LIMIT,
            "persona_loss_limit": -PERSONA_LOSS_LIMIT}


def length_matched(a, b):
    """Indices where two answers are within 10% of each other in word count."""
    return [i for i, (x, y) in enumerate(zip(a, b)) if min(x, y) > 0 and max(x, y) / min(x, y) <= LENGTH_RATIO]


def analyze(refs, outputs, scores, resamples=10000, conditions=CONDITIONS, comparisons=COMPARISONS):
    """Per-condition metrics and paired contrasts for one split, from already-loaded rows."""
    correct = {name: [1.0 if is_correct(row["output"], ref["answer"]) else 0.0
                      for row, ref in zip(outputs[name], refs, strict=True)] for name in conditions}
    words = {name: [len(row["output"].split()) for row in outputs[name]] for name in conditions}
    overall = {name: [row["overall"] for row in scores[name]] for name in conditions}
    catchphrase = {name: [float(row["catchphrase_only"]) for row in scores[name]] for name in conditions}
    metrics = {}
    for name in conditions:
        metrics[name] = {
            "n": len(refs), "accuracy": mean_with_interval(correct[name], resamples),
            "persona_overall": mean_with_interval(overall[name], resamples),
            "criteria": {c: statistics.mean(row[c] for row in scores[name]) for c in CRITERIA},
            "catchphrase_only_rate": statistics.mean(catchphrase[name]),
            "word_count": statistics.mean(words[name]),
            "format_success": statistics.mean(float(row["format_valid"]) for row in outputs[name]),
            "hit_token_cap": sum(row["hit_token_cap"] for row in outputs[name])}
    contrasts = {}
    for new, old in [pair for pair in comparisons if set(pair) <= set(conditions)]:
        matched = length_matched(words[old], words[new])
        accuracy = paired(correct[old], correct[new], resamples)
        persona = paired(overall[old], overall[new], resamples)
        contrasts[f"{new}_minus_{old}"] = {
            "accuracy": accuracy, "persona_overall": persona,
            "criteria": {c: statistics.mean(y[c] - x[c] for x, y in zip(scores[old], scores[new]))
                         for c in CRITERIA},
            "catchphrase_only": paired(catchphrase[old], catchphrase[new], resamples),
            "word_count": statistics.mean(words[new]) - statistics.mean(words[old]),
            "length_matched": {"n": len(matched), "max_word_count_ratio": LENGTH_RATIO,
                               "persona_overall": paired([overall[old][i] for i in matched],
                                                         [overall[new][i] for i in matched], resamples)},
            "verdicts": verdicts(accuracy, persona)}
    return {"n": len(refs), "conditions": metrics, "contrasts": contrasts}


def load_split(split, conditions=CONDITIONS, results_dir=RESULTS, provenance_dir=Path("provenance/evaluation/rlvr"),
               data_dir=Path("data")):
    """Outputs and persona ratings for one split, refusing anything that is not matched."""
    refs = read_jsonl(Path(data_dir) / f"{split}.jsonl")
    outputs, scores, configs = {}, {}, {}
    for name in conditions:
        outputs[name] = paired_outputs(refs, read_jsonl(results_dir / f"{name}_{split}.jsonl"))
        configs[name] = json.loads((provenance_dir / f"{name}_{split}.config.json").read_text(encoding="utf-8"))
    for key in MATCHED_KEYS:
        if len({json.dumps(configs[name][key], sort_keys=True) for name in conditions}) != 1:
            raise ValueError(f"Unmatched evaluation settings across conditions: {key} ({split})")
    rated = {name: read_jsonl(results_dir / "judge" / f"{name}_{split}.jsonl") for name in conditions}
    identities = {(row["config_sha256"], row["model"]) for rows in rated.values() for row in rows}
    if len(identities) != 1:
        raise ValueError(f"Persona ratings use more than one judge configuration or model ({split})")
    config_hash, model = identities.pop()
    for name in conditions:
        scores[name] = aligned_scores(outputs[name], rated[name], config_hash)
    return refs, outputs, scores, {"judge_config_sha256": config_hash, "judge_model": model,
                                   "evaluation": {k: configs[conditions[0]][k] for k in
                                                  ("model", "revision", "system_prompt", "max_new_tokens", "do_sample")},
                                   "adapters": {name: configs[name]["adapter_hashes"] for name in conditions}}


def table(report, labels=LABELS):
    """Markdown tables, ready for reports/RLVR_RESULTS.md."""
    lines = []
    for split in report["splits"]:
        section = report["splits"][split]
        lines += [f"### {split.capitalize()} ({section['n']} questions)", "",
                  "| Condition | Accuracy | Persona overall | Catchphrase-only | Words |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for name, metrics in section["conditions"].items():
            accuracy, persona = metrics["accuracy"], metrics["persona_overall"]
            lines.append(f"| {labels.get(name, name)} | {accuracy['mean']:.1%} "
                         f"[{accuracy['ci95'][0]:.1%}, {accuracy['ci95'][1]:.1%}] | {persona['mean']:.2f} "
                         f"[{persona['ci95'][0]:.2f}, {persona['ci95'][1]:.2f}] | "
                         f"{metrics['catchphrase_only_rate']:.1%} | {metrics['word_count']:.0f} |")
        lines += ["", "| Paired difference | Accuracy | Persona overall | Persona, similar length | Verdict |",
                  "| --- | ---: | ---: | ---: | --- |"]
        for key, contrast in section["contrasts"].items():
            new, old = key.split("_minus_")
            accuracy, persona = contrast["accuracy"], contrast["persona_overall"]
            matched = contrast["length_matched"]
            verdict = contrast["verdicts"]
            calls = ("accuracy improved" if verdict["accuracy_improved"] else
                     "accuracy lost" if verdict["accuracy_lost"] else "accuracy unchanged")
            calls += ", persona retained" if verdict["persona_retained"] else ", persona not retained"
            matched_text = (f"{matched['persona_overall']['mean']:+.2f} "
                            f"[{matched['persona_overall']['ci95'][0]:+.2f}, {matched['persona_overall']['ci95'][1]:+.2f}] "
                            f"(n={matched['n']})" if matched["persona_overall"] else "n/a")
            lines.append(f"| {labels.get(new, new)} minus {labels.get(old, old)} | "
                         f"{accuracy['mean']:+.1%} [{accuracy['ci95'][0]:+.1%}, {accuracy['ci95'][1]:+.1%}] | "
                         f"{persona['mean']:+.2f} [{persona['ci95'][0]:+.2f}, {persona['ci95'][1]:+.2f}] | "
                         f"{matched_text} | {calls} |")
        lines.append("")
    return "\n".join(lines)


def summarize(args):
    report = {"conditions": list(args.conditions), "resamples": args.resamples, "splits": {},
              "criteria": {"accuracy_improved": "paired 95% interval excludes zero on the positive side",
                           "persona_retained": f"paired 95% lower bound above -{PERSONA_LOSS_LIMIT:.2f} points",
                           "note": "Predeclared before the runs; report the effects and intervals either way."},
              "method": ("Paired question bootstrap, 10,000 resamples, seed 42, over the same questions for every "
                         "condition. Accuracy is the strict verifier; persona is the frozen Haiku rubric's overall "
                         "score. The similar-length subset is a sensitivity check, not a causal adjustment.")}
    for split in args.splits:
        refs, outputs, scores, provenance = load_split(split, args.conditions, Path(args.results_dir),
                                                       Path(args.provenance_dir), Path(args.data_dir))
        report["splits"][split] = {**analyze(refs, outputs, scores, args.resamples, args.conditions),
                                   "provenance": provenance}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(table(report))
    print(f"wrote {path}")


def load_stages(split, stages=STAGES, data_dir=Path("data")):
    """The same loader across all four stages, which do not share a directory.

    `load_split` refuses any mismatch outright, which is right for the three stage-3
    conditions that were produced in one session. Across stages a mismatch is expected
    rather than disqualifying, so this records what differs and lets the caller judge it.
    """
    refs = read_jsonl(Path(data_dir) / f"{split}.jsonl")
    outputs, scores, configs = {}, {}, {}
    for name in stages:
        results_dir, provenance_dir, stem = STAGE_SOURCES[name]
        outputs[name] = paired_outputs(refs, read_jsonl(results_dir / f"{stem}_{split}.jsonl"))
        configs[name] = json.loads((provenance_dir / f"{stem}_{split}.config.json").read_text(encoding="utf-8"))
    differences = {key: {name: configs[name][key] for name in stages}
                   for key in MATCHED_KEYS
                   if len({json.dumps(configs[name][key], sort_keys=True) for name in stages}) != 1}
    # Comparing across stages means tolerating environment drift, but not a different
    # question set or a different base model: those make the comparison meaningless.
    for key in ("data_sha256", "model", "revision", "system_prompt", "do_sample"):
        if key in differences:
            raise ValueError(f"Stages differ on {key}, which makes them incomparable ({split}): "
                             f"{differences[key]}")
    rated = {}
    for name in stages:
        results_dir, _, stem = STAGE_SOURCES[name]
        rated[name] = read_jsonl(results_dir / "judge" / f"{stem}_{split}.jsonl")
    identities = {(row["config_sha256"], row["model"]) for rows in rated.values() for row in rows}
    if len(identities) != 1:
        raise ValueError(f"Persona ratings span more than one judge configuration ({split}): {identities}")
    config_hash, model = identities.pop()
    for name in stages:
        scores[name] = aligned_scores(outputs[name], rated[name], config_hash)
    # The RLAIF adapter was evaluated twice, once per stage. Identical greedy outputs are a
    # stronger comparability check than matching configs, so record whether they reproduced.
    reproduced = None
    if {"rlaif"} <= set(stages):
        previous = Path("results") / f"rlaif_{split}.jsonl"
        if previous.exists():
            before = {row["id"]: row["output"] for row in read_jsonl(previous)}
            reproduced = sum(before.get(row["id"]) == row["output"] for row in outputs["rlaif"])
    return refs, outputs, scores, {
        "judge_config_sha256": config_hash, "judge_model": model,
        "config_differences_across_stages": differences,
        "rlaif_outputs_reproduced": reproduced, "n": len(refs)}


def stages(args):
    """Both metrics across every stage, the deliverable that spans SFT, RLAIF, and RLVR."""
    report = {"stages": list(args.stages), "resamples": args.resamples, "splits": {},
              "method": ("Paired question bootstrap, 10,000 resamples, seed 42, over the same questions. "
                         "Stages were evaluated in separate sessions; config differences between them are "
                         "recorded per split rather than assumed away.")}
    for split in args.splits:
        refs, outputs, scores, provenance = load_stages(split, tuple(args.stages), Path(args.data_dir))
        report["splits"][split] = {**analyze(refs, outputs, scores, args.resamples,
                                             tuple(args.stages), STAGE_COMPARISONS),
                                   "provenance": provenance}
    path = Path(args.stage_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(table(report, STAGE_LABELS))
    for split, section in report["splits"].items():
        differences = section["provenance"]["config_differences_across_stages"]
        reproduced = section["provenance"]["rlaif_outputs_reproduced"]
        if reproduced is not None:
            print(f"{split}: RLAIF outputs reproduced across stages for {reproduced}/{section['n']} questions")
        print(f"{split}: evaluation settings differing across stages: "
              f"{', '.join(sorted(differences)) if differences else 'none'}")
    print(f"wrote {path}")


def review(args):
    """One blinded packet: 50 test questions, all three answers, labels hidden and order shuffled."""
    rng = random.Random(args.seed)
    refs = read_jsonl(Path(args.data_dir) / "test.jsonl")
    outputs = {name: paired_outputs(refs, read_jsonl(Path(args.results_dir) / f"{name}_test.jsonl"))
               for name in args.conditions}
    cases = sorted(rng.sample(range(len(refs)), args.cases))
    rng.shuffle(cases)
    root, sides = Path(args.review_dir), "ABC"
    lines = ["# Blinded Mario persona review: stage 3", "",
             "Rate the voice, not the math. Read the [frozen rubric](../../prompts/mario_judge.md), then rate each "
             "answer independently from 1 to 5 on all five criteria. Mark whether it is mainly catchphrases. Do not "
             "look at model labels or automated scores before finishing.", "",
             f"Record scores in [ratings.csv](ratings.csv). Leave unreviewed answers blank. This packet holds "
             f"{args.cases} test questions with {len(args.conditions)} answers each "
             f"({args.cases * len(args.conditions)} outputs). The order of the answers is randomised per question and "
             "the key is stored separately under provenance.", "", "## Questions", ""]
    lines += [f"- [Question {i:02d}](#question-{i:02d})" for i in range(1, args.cases + 1)]
    key, ratings = [], []
    for number, index in enumerate(cases, 1):
        order = list(args.conditions)
        rng.shuffle(order)
        lines += ["", f"## Question {number:02d}", "", refs[index]["question"], ""]
        for side, name in zip(sides, order):
            answer = outputs[name][index]["output"]
            lines += [f"### Answer {side}", "", *[("> " + line).rstrip() for line in answer.splitlines()], ""]
            key.append({"case": number, "side": side, "split": "test", "id": refs[index]["id"],
                        "condition": name, "text_sha256": text_hash(answer)})
            ratings.append({"case": number, "side": side, **{k: "" for k in (*CRITERIA, "catchphrase_only", "notes")}})
    targets = [root / "review.md", root / "ratings.csv", Path(args.key)]
    if any(path.exists() for path in targets):
        raise SystemExit("Review files already exist; refusing to overwrite possible human ratings")
    root.mkdir(parents=True, exist_ok=True)
    targets[0].write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    with targets[1].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ratings[0]))
        writer.writeheader()
        writer.writerows(ratings)
    targets[2].parent.mkdir(parents=True, exist_ok=True)
    targets[2].write_text(json.dumps({"seed": args.seed, "conditions": list(args.conditions),
                                      "rubric_sha256": hashlib.sha256(Path("prompts/mario_judge.md").read_bytes()).hexdigest(),
                                      "key": key}, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {args.cases} blinded questions and {len(ratings)} blank rating rows under {root}")


def plot(args):
    """Left: where each condition sits on accuracy and persona. Right: the paired differences."""
    from PIL import Image, ImageDraw
    from plot_results import font, INK, MUTED, GRID

    report = json.loads(Path(args.output).read_text(encoding="utf-8"))
    section = report["splits"][args.split]
    colors = {"rlvr_baseline": "#64748B", "rlvr_math": "#167D9A", "rlvr_combined": "#C94740"}
    image = Image.new("RGB", (1800, 1000), "#FAFAF7")
    draw = ImageDraw.Draw(image)
    draw.text((65, 70), "RLVR: math accuracy against Mario persona", font=font(49, True), fill=INK)
    draw.text((65, 148), f"{args.split.capitalize()} split, {section['n']} questions per condition. "
                         "Separate outcomes; the training reward is not an evaluation metric.", font=font(27), fill=MUTED)
    left, right, top, bottom = 190, 850, 300, 700
    accuracies = [v for m in section["conditions"].values() for v in m["accuracy"]["ci95"]]
    personas = [v for m in section["conditions"].values() for v in m["persona_overall"]["ci95"]]
    low_x, high_x = min(personas) - .1, max(personas) + .1
    low_y, high_y = min(accuracies) - .02, max(accuracies) + .02
    px = lambda v: left + (v - low_x) / (high_x - low_x) * (right - left)  # noqa: E731
    py = lambda v: bottom - (v - low_y) / (high_y - low_y) * (bottom - top)  # noqa: E731
    draw.text((65, 208), "Where each condition lands", font=font(31, True), fill=INK)
    draw.text((65, 252), "Accuracy, share of questions solved", font=font(24), fill=MUTED)
    for i in range(5):
        value = low_x + i * (high_x - low_x) / 4
        draw.line((px(value), top, px(value), bottom), fill=GRID, width=2)
        draw.text((px(value), bottom + 18), f"{value:.2f}", anchor="mt", font=font(22), fill=MUTED)
        accuracy = low_y + i * (high_y - low_y) / 4
        draw.line((left, py(accuracy), right, py(accuracy)), fill=GRID, width=2)
        draw.text((left - 18, py(accuracy)), f"{accuracy:.0%}", anchor="rm", font=font(22), fill=MUTED)
    for name, metrics in section["conditions"].items():
        x, y = px(metrics["persona_overall"]["mean"]), py(metrics["accuracy"]["mean"])
        color = colors.get(name, INK)
        label = LABELS.get(name, name)
        draw.line((px(metrics["persona_overall"]["ci95"][0]), y, px(metrics["persona_overall"]["ci95"][1]), y), fill=color, width=4)
        draw.line((x, py(metrics["accuracy"]["ci95"][0]), x, py(metrics["accuracy"]["ci95"][1])), fill=color, width=4)
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=color)
        # Keep the name clear of both whiskers, and inside the panel near the right edge.
        flip = x > right - 240
        draw.text((x - 22 if flip else x + 22, y - 28), label, anchor="rm" if flip else "lm", font=font(26, True), fill=color)
    draw.text((520, bottom + 58), "Persona overall (1-5)", anchor="mt", font=font(25), fill=INK)

    panel_left, panel_right = 1010, 1620
    draw.text((panel_left, 208), "Paired differences from the RLAIF baseline", font=font(31, True), fill=INK)
    bars = [(key, contrast) for key, contrast in section["contrasts"].items() if key.endswith("_minus_rlvr_baseline")]
    for panel, (metric, caption, factor, digits) in enumerate((("accuracy", "Accuracy, percentage points", 100, 1),
                                                               ("persona_overall", "Persona overall, score points", 1, 2))):
        base = 330 + panel * 225
        # Each metric gets its own scale: percentage points and score points are not comparable.
        span = max(abs(v * factor) for _, contrast in bars for v in contrast[metric]["ci95"]) * 1.4 or 1
        bx = lambda v, span=span: (panel_left + panel_right) / 2 + v / span * (panel_right - panel_left) / 2  # noqa: E731
        draw.text((panel_left, base - 58), caption, font=font(27, True), fill=INK)
        draw.line((bx(0), base - 10, bx(0), base + 150), fill=INK, width=3)
        if metric == "persona_overall" and PERSONA_LOSS_LIMIT * factor < span:
            for offset in range(base - 10, base + 150, 14):  # the predeclared retention margin
                draw.line((bx(-PERSONA_LOSS_LIMIT), offset, bx(-PERSONA_LOSS_LIMIT), offset + 7), fill=MUTED, width=2)
        for i, (key, contrast) in enumerate(bars):
            name = key.split("_minus_")[0]
            y, color = base + 35 + i * 85, colors.get(name, INK)
            mean, lo, hi = (v * factor for v in (contrast[metric]["mean"], *contrast[metric]["ci95"]))
            draw.text((panel_left, y - 34), LABELS.get(name, name), font=font(23), fill=MUTED)
            draw.line((bx(lo), y, bx(hi), y), fill=color, width=6)
            for end in (lo, hi):
                draw.line((bx(end), y - 9, bx(end), y + 9), fill=color, width=3)
            draw.ellipse((bx(mean) - 9, y - 9, bx(mean) + 9, y + 9), fill=color)
            draw.text((panel_right + 25, y), f"{mean:+.{digits}f}", anchor="lm", font=font(26, True), fill=color)
    draw.text((panel_left, 775), f"Dotted line: the predeclared -{PERSONA_LOSS_LIMIT:.2f} retention margin.",
              font=font(23), fill=MUTED)
    draw.text((65, 880), "Bars: paired 95% question-bootstrap intervals over the same questions, seed 42. "
                         "One training run per condition.", font=font(24), fill=MUTED)
    draw.text((65, 926), "The persona reward and the persona score come from the same judge family, so agreement "
                         "with the blinded human ratings matters.", font=font(24), fill=MUTED)
    target = Path(args.figure)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, optimize=True)
    print(f"wrote {target}")


def stage_plot(args):
    """Both metrics across every stage, side by side on their own scales."""
    from PIL import Image, ImageDraw
    from plot_results import font, INK, MUTED, GRID

    report = json.loads(Path(args.stage_output).read_text(encoding="utf-8"))
    section = report["splits"][args.split]
    order = [name for name in report["stages"] if name in section["conditions"]]
    colors = {"mario_sft": "#B77927", "rlaif": "#64748B",
              "rlvr_math": "#167D9A", "rlvr_combined": "#C94740"}
    image = Image.new("RGB", (1800, 1000), "#FAFAF7")
    draw = ImageDraw.Draw(image)
    draw.rectangle((65, 58, 120, 65), fill="#C94740")
    draw.text((65, 82), "Persona and math capability across every stage", font=font(49, True), fill=INK)
    draw.text((65, 160), f"{args.split.capitalize()} split, {section['n']} questions per stage. "
                         "Separate outcomes, each on its own scale; bars are 95% question-bootstrap intervals.",
              font=font(27), fill=MUTED)

    panels = (("persona_overall", "Persona adherence, judge overall (1-5)", 65, lambda v: f"{v:.2f}"),
              ("accuracy", "Math capability, GSM8K accuracy", 960, lambda v: f"{v:.0%}"))
    for metric, caption, x0, fmt in panels:
        width, top, bottom = 760, 330, 760
        bounds = [v for name in order for v in section["conditions"][name][metric]["ci95"]]
        span = max(bounds) - min(bounds)
        low, high = min(bounds) - span * 0.35, max(bounds) + span * 0.35
        py = lambda v: bottom - (v - low) / (high - low) * (bottom - top)  # noqa: E731
        step = width / (len(order) + 1)
        px = lambda i: x0 + 120 + step * i * 0.86  # noqa: E731
        draw.text((x0, 250), caption, font=font(31, True), fill=INK)
        for i in range(5):
            value = low + i * (high - low) / 4
            draw.line((x0 + 95, py(value), x0 + width - 20, py(value)), fill=GRID, width=2)
            draw.text((x0 + 80, py(value)), fmt(value), anchor="rm", font=font(22), fill=MUTED)
        points = [(px(i), py(section["conditions"][name][metric]["mean"])) for i, name in enumerate(order)]
        for start, end in zip(points, points[1:]):  # the progression from one stage to the next
            draw.line((*start, *end), fill="#C9CFD3", width=3)
        for i, name in enumerate(order):
            metrics = section["conditions"][name][metric]
            x, y = px(i), py(metrics["mean"])
            color = colors.get(name, INK)
            draw.line((x, py(metrics["ci95"][0]), x, py(metrics["ci95"][1])), fill=color, width=5)
            for end in metrics["ci95"]:
                draw.line((x - 9, py(end), x + 9, py(end)), fill=color, width=3)
            draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=color)
            # Above the whisker cap, not the mean, so the label never sits on the interval.
            draw.text((x, py(metrics["ci95"][1]) - 14), fmt(metrics["mean"]), anchor="mb",
                      font=font(25, True), fill=color)
            for line, part in enumerate(STAGE_LABELS.get(name, name).split(" ", 1)):
                draw.text((x, bottom + 22 + line * 30), part, anchor="mt", font=font(23), fill=INK)

    reproduced = section["provenance"].get("rlaif_outputs_reproduced")
    note = ("Persona rose once, at the RLAIF stage, and neither RLVR reward moved either metric "
            "beyond its interval.")
    draw.text((65, 862), note, font=font(25), fill=INK)
    if reproduced:
        draw.text((65, 906), f"All stages share one frozen judge configuration, and re-evaluating the RLAIF "
                             f"adapter reproduced {reproduced}/{section['n']} answers exactly.",
                  font=font(24), fill=MUTED)
    draw.text((65, 946), "One training run per stage, so these intervals cover variation across questions, "
                         "not across runs.", font=font(24), fill=MUTED)
    target = Path(args.stage_figure)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, optimize=True)
    print(f"wrote {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("summarize", "review", "plot", "stages", "stage-plot"))
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--stages", nargs="+", default=list(STAGES))
    parser.add_argument("--stage-output", default=str(STAGE_SUMMARY))
    parser.add_argument("--stage-figure", default="reports/figures/rlvr_stages.png")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--split", default="test", help="plot: which split to draw")
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--provenance-dir", default="provenance/evaluation/rlvr")
    parser.add_argument("--review-dir", default="reports/rlvr_review")
    parser.add_argument("--key", default=str(PROVENANCE / "human_review_key.json"),
                        help="review: where the hidden condition labels are stored")
    parser.add_argument("--output", default=str(SUMMARY))
    parser.add_argument("--figure", default="reports/figures/rlvr_accuracy_persona.png")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--cases", type=int, default=50, help="review: questions in the blinded packet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    {"summarize": summarize, "review": review, "plot": plot,
     "stages": stages, "stage-plot": stage_plot}[args.command](args)


if __name__ == "__main__":
    main()
