"""Audit saved judge scores and prepare a reproducible blinded human review."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics

from compare import interval, paired_outputs
from judge import CRITERIA, digest, haiku_cost, text_hash
from mario_data import read_jsonl

CONDITIONS = ('mario_rlaif_baseline', 'rlaif')


def paired_delta(a, b):
    values = [y - x for x, y in zip(a, b, strict=True)]
    return {'mean': statistics.mean(values), 'ci95': interval(values)} if values else None


def aligned_scores(outputs, scores, config_hash):
    indexed = {r['id']: r for r in scores}
    if len(indexed) != len(scores) or set(indexed) != {r['id'] for r in outputs}:
        raise ValueError('Judge scores must cover every output exactly once')
    result = [indexed[r['id']] for r in outputs]
    for output, score in zip(outputs, result):
        if score['text_sha256'] != text_hash(output['output']) or score['config_sha256'] != config_hash:
            raise ValueError('Stale answer hash or mismatched judge configuration')
    return result


def summarize():
    config = json.loads(Path('provenance/judge/rlaif-api/config.json').read_text())
    saved_hash = config.pop('sha256')
    if saved_hash != digest(config):
        raise ValueError('Judge config hash mismatch')
    report = {'judge_config_sha256': saved_hash, 'splits': {}}
    models = set()
    for split in ('val', 'test'):
        refs = read_jsonl(f'data/{split}.jsonl')
        outputs = [paired_outputs(refs, read_jsonl(f'results/{name}_{split}.jsonl')) for name in CONDITIONS]
        scores = [aligned_scores(rows, read_jsonl(f'results/judge/{name}_{split}.jsonl'), saved_hash)
                  for name, rows in zip(CONDITIONS, outputs)]
        models.update(r['model'] for rows in scores for r in rows)
        lengths = [[len(r['output'].split()) for r in rows] for rows in outputs]
        matched = [i for i, (a, b) in enumerate(zip(*lengths)) if min(a, b) > 0 and max(a, b) / min(a, b) <= 1.1]
        section = {'n': len(refs), 'criteria': {}}
        for criterion in CRITERIA:
            values = [[r[criterion] for r in rows] for rows in scores]
            section['criteria'][criterion] = {
                'sft_mean': statistics.mean(values[0]), 'rlaif_mean': statistics.mean(values[1]),
                'difference': paired_delta(*values)}
        overall = [[r['overall'] for r in rows] for rows in scores]
        section['rlaif_wins_ties_losses'] = {
            'wins': sum(b > a for a, b in zip(*overall)), 'ties': sum(b == a for a, b in zip(*overall)),
            'losses': sum(b < a for a, b in zip(*overall))}
        flags = [[int(r['catchphrase_only']) for r in rows] for rows in scores]
        section['catchphrase_only'] = {'sft_rate': statistics.mean(flags[0]), 'rlaif_rate': statistics.mean(flags[1]),
                                      'difference': paired_delta(*flags)}
        section['word_count'] = {'sft_mean': statistics.mean(lengths[0]), 'rlaif_mean': statistics.mean(lengths[1])}
        section['length_matched'] = {'n': len(matched), 'max_word_count_ratio': 1.1,
            'overall_difference': paired_delta(*[[v[i] for i in matched] for v in overall])}
        report['splits'][split] = section
    if len(models) != 1:
        raise ValueError('Resolved judge models differ across scores')
    attempts = read_jsonl('provenance/judge/rlaif-api/attempts.jsonl')
    report.update({'resolved_model': models.pop(), 'estimated_api_cost_usd': sum(haiku_cost(r.get('usage', {})) for r in attempts),
                   'calls_with_usage': sum('usage' in r for r in attempts),
                   'method': 'RLAIF minus matched SFT; paired bootstrap, 10000 resamples, seed 42. Length-matched subset is a sensitivity check, not a causal adjustment. Human review pending.'})
    target = Path('derived/persona-evaluation/summary.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def review():
    rng = random.Random(42)
    cases = []
    for split in ('val', 'test'):
        refs = read_jsonl(f'data/{split}.jsonl')
        outputs = [paired_outputs(refs, read_jsonl(f'results/{name}_{split}.jsonl')) for name in CONDITIONS]
        for i in rng.sample(range(len(refs)), 25):
            cases.append((split, refs[i], [rows[i]['output'] for rows in outputs]))
    rng.shuffle(cases)
    root = Path('reports/persona_review')
    root.mkdir(parents=True, exist_ok=True)
    lines = ['# Blinded Mario persona review', '',
             'Rate the voice, not the math. Read the [frozen rubric](../../prompts/mario_judge.md), then rate each answer independently from 1 to 5 on all five criteria. Mark whether it consists mainly of catchphrases. Do not look at model labels or automated scores before finishing.', '',
             'Record scores in [ratings.csv](ratings.csv). Leave unreviewed answers blank. Keep your own completed copy; rerunning the preparation script refuses to overwrite these files.', '',
             'This sample contains 50 question pairs (100 answers). A/B order is randomized. The key is stored separately; reviewers should not open provenance until ratings are complete.', '', '## Questions', '']
    lines += [f'- [Question {i:02d}](#question-{i:02d})' for i in range(1, 51)]
    key, ratings = [], []
    for number, (split, ref, answers) in enumerate(cases, 1):
        order = [0, 1]
        rng.shuffle(order)
        lines += ['', f'## Question {number:02d}', '', ref['question'], '']
        for side, index in zip(('A', 'B'), order):
            lines += [f'### Answer {side}', '', *[('> ' + line).rstrip() for line in answers[index].splitlines()], '']
            key.append({'case': number, 'side': side, 'split': split, 'id': ref['id'],
                        'condition': CONDITIONS[index], 'text_sha256': text_hash(answers[index])})
            ratings.append({'case': number, 'side': side, **{k: '' for k in (*CRITERIA, 'catchphrase_only', 'notes')}})
    targets = [root / 'review.md', root / 'ratings.csv', Path('provenance/judge/rlaif-api/human_review_key.json')]
    if any(path.exists() for path in targets):
        raise ValueError('Review files already exist; refusing to overwrite possible human ratings')
    targets[0].write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')
    with targets[1].open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ratings[0]))
        writer.writeheader()
        writer.writerows(ratings)
    targets[2].parent.mkdir(parents=True, exist_ok=True)
    targets[2].write_text(json.dumps({'seed': 42, 'rubric_sha256': hashlib.sha256(Path('prompts/mario_judge.md').read_bytes()).hexdigest(), 'key': key}, indent=2) + '\n')
    print('Prepared 50 blinded question pairs and 100 blank rating rows.')


def plot():
    from PIL import Image, ImageDraw
    from plot_results import font, INK, MUTED, GRID
    summary = json.loads(Path('derived/persona-evaluation/summary.json').read_text())
    image = Image.new('RGB', (1800, 1000), '#FAFAF7')
    draw = ImageDraw.Draw(image)
    draw.text((65, 75), "DPO improved the judge's Mario persona scores", font=font(49, True), fill=INK)
    draw.text((65, 153), 'Overall persona rating, 1-5 scale. The advantage also appears among similar-length answers.', font=font(27), fill=MUTED)
    left, right, low, high = 650, 1490, -.1, .7
    def x(value):
        return left + (value - low) / (high - low) * (right - left)
    for tick in (-.1, 0, .1, .2, .3, .4, .5, .6, .7):
        draw.line((x(tick), 270, x(tick), 710), fill=INK if tick == 0 else GRID, width=3 if tick == 0 else 2)
        draw.text((x(tick), 730), f'{tick:+.1f}' if tick else '0', anchor='mt', font=font(23), fill=MUTED)
    rows = [('val', False, 'Validation: all questions'), ('test', False, 'Test: all questions'),
            ('val', True, 'Validation: similar length'), ('test', True, 'Test: similar length')]
    for i, (split, matched, label) in enumerate(rows):
        section = summary['splits'][split]
        result = section['length_matched']['overall_difference'] if matched else section['criteria']['overall']['difference']
        n = section['length_matched']['n'] if matched else section['n']
        mean, (lo, hi) = result['mean'], result['ci95']
        if not low <= lo <= mean <= hi <= high:
            raise ValueError('Figure axis does not contain all intervals')
        y, color = 325 + i * 110, '#167D9A' if matched else '#C94740'
        draw.text((65, y - 16), label, font=font(29, True), fill=INK)
        draw.text((65, y + 23), f'{n} paired questions', font=font(23), fill=MUTED)
        draw.line((x(lo), y, x(hi), y), fill=color, width=6)
        for end in (lo, hi):
            draw.line((x(end), y - 10, x(end), y + 10), fill=color, width=3)
        draw.ellipse((x(mean) - 10, y - 10, x(mean) + 10, y + 10), fill=color)
        draw.text((1560, y), f'{mean:+.2f}', anchor='lm', font=font(32, True), fill=color)
    draw.text((930, 790), 'RLAIF minus Mario SFT (score points)', anchor='mt', font=font(26), fill=INK)
    draw.text((65, 877), 'Bars: paired 95% bootstrap intervals. Similar length: word counts within 10% of each other.', font=font(24), fill=MUTED)
    draw.text((65, 923), 'The same judge family supplied training preferences. Independent human review is still pending.', font=font(24), fill=MUTED)
    target = Path('reports/figures/rlaif_persona_judge.png')
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, optimize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('summarize', 'review', 'plot'))
    args = parser.parse_args()
    {'summarize': summarize, 'review': review, 'plot': plot}[args.command]()
