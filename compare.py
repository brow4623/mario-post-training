"""Compare four matched runs: exact math accuracy and held-out persona-reference similarity."""
import argparse
import json
from pathlib import Path
import re

from mario_data import read_jsonl
from verifier import is_correct

CONDITIONS = ('base', 'prompted', 'control', 'mario')


def mask_math(text):
    text = re.sub(r'<<.*?>>|####[^\n]*', ' ', text)
    return re.sub(r'[\d+*/=<>$.,%-]+', ' ', text)


def interval(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(42)
    means = [values[rng.integers(0, len(values), (250, len(values)))].mean(axis=1) for _ in range(40)]
    return [float(v) for v in np.quantile(np.concatenate(means), [.025, .975])]


def paired_outputs(rows, predictions):
    indexed = {r['id']: r for r in predictions}
    if len(indexed) != len(predictions) or set(indexed) != {r['id'] for r in rows}:
        raise ValueError('Missing, duplicate, or unknown predictions')
    ordered = [indexed[r['id']] for r in rows]
    if any(p['question'] != r['question'] or p['gold'] != r['answer'] for p, r in zip(ordered, rows)):
        raise ValueError('Prediction/reference mismatch')
    return ordered


def main():
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=('val', 'test'), required=True)
    p.add_argument('--results-dir', default='results')
    p.add_argument('--provenance-dir', default='provenance/evaluation')
    p.add_argument('--output-dir', default='derived/comparisons')
    p.add_argument('--rlaif', action='store_true', help='Compare RLAIF with a freshly evaluated Mario SFT baseline')
    args = p.parse_args()
    conditions = ('mario_rlaif_baseline', 'rlaif') if args.rlaif else CONDITIONS
    root = Path(args.results_dir)
    refs = read_jsonl(f'data/{args.split}.jsonl')
    runs = {name: paired_outputs(refs, read_jsonl(root / f'{name}_{args.split}.jsonl')) for name in conditions}
    configs = {name: json.loads((Path(args.provenance_dir) / f'{name}_{args.split}.config.json').read_text(encoding='utf-8')) for name in conditions}
    for key in ('model', 'revision', 'data_sha256', 'n', 'max_new_tokens', 'batch_size', 'do_sample', 'dtype', 'packages', 'eval_sha256', 'verifier_sha256'):
        if any(configs[n][key] != configs[conditions[0]][key] for n in conditions):
            raise ValueError(f'Unmatched evaluation settings: {key}')
    if args.rlaif and configs['rlaif']['system_prompt'] != configs['mario_rlaif_baseline']['system_prompt']:
        raise ValueError('Unmatched system prompts')
    metrics, accuracy, similarities = {}, {}, {}
    for name, rows in runs.items():
        accuracy[name] = np.array([is_correct(r['output'], ref['answer']) for r, ref in zip(rows, refs)], dtype=float)
        metrics[name] = {'n': len(rows), 'accuracy': float(accuracy[name].mean()),
                         'accuracy_ci95': interval(accuracy[name]),
                         'format_success': sum(r['format_valid'] for r in rows) / len(rows),
                         'hit_token_cap': sum(r['hit_token_cap'] for r in rows)}
    if args.split == 'val':
        persona = {r['question']: r['answer'] for r in read_jsonl('data/mario/validation.jsonl')}
        targets = [persona[r['question']] for r in refs]
        training = [r['response'] for r in read_jsonl('data/train_mario.jsonl')]
        for variant, transform in (('raw', lambda x: x), ('math_masked', mask_math)):
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2, sublinear_tf=True)
            vectorizer.fit([transform(t) for t in training])
            gold = vectorizer.transform([transform(t) for t in targets])
            for name, rows in runs.items():
                generated = vectorizer.transform([transform(r['output']) for r in rows])
                scores = np.asarray(gold.multiply(generated).sum(axis=1)).ravel()
                similarities[(name, variant)] = scores
                metrics[name][f'cosine_{variant}'] = float(scores.mean())
                metrics[name][f'cosine_{variant}_ci95'] = interval(scores)
    contrasts = {}
    comparisons = (('rlaif', 'mario_rlaif_baseline'),) if args.rlaif else (('mario', 'base'), ('mario', 'control'), ('mario', 'prompted'))
    for a, b in comparisons:
        delta = accuracy[a] - accuracy[b]
        result = {'accuracy_delta': float(delta.mean()), 'accuracy_delta_ci95': interval(delta),
                  'a_only_correct': int((delta > 0).sum()), 'b_only_correct': int((delta < 0).sum())}
        for variant in ('raw', 'math_masked'):
            if (a, variant) in similarities:
                diff = similarities[(a, variant)] - similarities[(b, variant)]
                result[f'cosine_{variant}_delta'] = float(diff.mean())
                result[f'cosine_{variant}_delta_ci95'] = interval(diff)
        contrasts[f'{a}_minus_{b}'] = result
    report = {'split': args.split, 'conditions': metrics, 'contrasts': contrasts,
              'method': 'Paired question bootstrap: 10000 resamples, seed 42. Character 3-5 gram TF-IDF fitted only on Mario training responses; cosine against same-question held-out Mario references. Math-masked sensitivity removes annotations and numeric/operator characters.',
              'limitations': 'Cosine is textual resemblance, not an independent persona judgment. Single training seed. No persona-reference similarity on test because no test persona references were generated.'}
    path = Path(args.output_dir) / f'comparison_{args.split}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
