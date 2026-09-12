"""Render the two report figures from compare.py summaries (requires Pillow)."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CONDITIONS = ("base", "prompted", "control", "mario")
LABELS = ("Original", "Original + Mario prompt", "Plain-GSM8K SFT", "Mario SFT")
COLORS = ("#64748B", "#167D9A", "#B77927", "#C94740")
INK, MUTED, GRID = "#182B39", "#596873", "#DCE2E4"


def font(size, bold=False):
    candidates = (
        "C:/Windows/Fonts/" + ("arialbd.ttf" if bold else "arial.ttf"),
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    raise RuntimeError("Install Arial or DejaVu Sans to render the figures.")


def figure(title, subtitle, panels, maximum, ticks, percent, takeaway, note, path):
    image = Image.new("RGB", (1800, 1000), "#FAFAF7")
    draw = ImageDraw.Draw(image)
    draw.rectangle((65, 58, 120, 65), fill=COLORS[-1])
    draw.text((65, 87), title, font=font(49, True), fill=INK)
    draw.text((65, 157), subtitle, font=font(26), fill=MUTED)
    rows = (340, 440, 540, 640)
    for label, color, y in zip(LABELS, COLORS, rows):
        draw.ellipse((65, y - 7, 79, y + 7), fill=color)
        draw.text((95, y), label, anchor="lm", font=font(27, label == "Mario SFT"), fill=INK)

    for (heading, metrics, key), left in zip(panels, (480, 1150)):
        right = left + 555
        draw.text((left, 242), heading, font=font(31, True), fill=INK)
        for tick in ticks:
            x = left + tick / maximum * (right - left)
            draw.line((x, 300, x, 690), fill=GRID, width=2)
            label = f"{tick * 100:.0f}%" if percent else f"{tick:.1f}"
            draw.text((x, 713), label, anchor="mt", font=font(23), fill=MUTED)
        for name, color, y in zip(CONDITIONS, COLORS, rows):
            value = metrics[name][key]
            lo, hi = metrics[name][key + "_ci95"]
            if not 0 <= lo <= value <= hi <= maximum:
                raise ValueError(f"Invalid interval or axis bounds: {name}, {key}")
            x, low, high = [left + v / maximum * (right - left) for v in (value, lo, hi)]
            draw.line((low, y, high, y), fill=color, width=6)
            for end in (low, high):
                draw.line((end, y - 10, end, y + 10), fill=color, width=3)
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color)
            label = f"{100 * value:.1f}%" if percent else f"{value:.4f}"
            draw.text((x, y - 22), label, anchor="mb", font=font(27, True), fill=color)

    draw.rounded_rectangle((65, 793, 1735, 882), radius=14, fill="#EDF0EB")
    draw.text((90, 821), takeaway, font=font(27, True), fill=INK)
    draw.text((65, 916), note, font=font(23), fill=MUTED)
    draw.text((65, 950), "Bars: 95% question-bootstrap intervals; 10,000 resamples, seed 42. One training run per adapter.", font=font(23), fill=MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("derived/comparisons"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    summaries = {split: json.loads((args.input_dir / f"comparison_{split}.json").read_text())
                 for split in ("val", "test")}
    val, test = (summaries[s]["conditions"] for s in ("val", "test"))
    delta = summaries["test"]["contrasts"]["mario_minus_base"]["accuracy_delta"] * 100
    figure(
        "Both SFT models lost math accuracy",
        "Exact final-answer accuracy on GSM8K. Each panel contains 500 questions shared by all four models.",
        (("Validation", val, "accuracy"), ("Held-out test", test, "accuracy")),
        1, (0, .2, .4, .6, .8, 1), True,
        f"Mario SFT vs. original on test: {delta:+.1f} percentage points. Plain SFT also performed worse than the original.",
        "Validation and test contain different questions; compare models within each panel, not across panels.",
        args.output_dir / "math_accuracy.png",
    )
    contrast = summaries["val"]["contrasts"]["mario_minus_prompted"]
    delta = contrast["cosine_raw_delta"]
    lo, hi = contrast["cosine_raw_delta_ci95"]
    figure(
        "Mario SFT resembled the reference voice more closely",
        "Mean character n-gram TF-IDF cosine similarity to same-question Mario references; 500 validation answers.",
        (("Full answer", val, "cosine_raw"), ("Math notation masked", val, "cosine_math_masked")),
        .5, (0, .1, .2, .3, .4, .5), False,
        f"Mario SFT vs. prompting: +{delta:.4f} similarity (paired 95% CI: {lo:.4f} to {hi:.4f}).",
        "Higher means more textual resemblance, not a direct persona rating. Cosine range: 0 to 1; axes show 0 to 0.5.",
        args.output_dir / "persona_similarity.png",
    )


if __name__ == "__main__":
    main()
