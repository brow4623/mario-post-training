"""Rebuild the exact experiment datasets locally, without downloads or model calls."""
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys

from mario_data import read_jsonl, record, write_jsonl
from verifier import gold_answer, is_correct


def training_record(row):
    """Convert archived chat data to the schema consumed by sft.py and eval.py."""
    value = gold_answer(row['answer'])
    answer = str(int(value)) if value.denominator == 1 else str(float(value))
    return {'id': row['id'].removeprefix('gsm8k/').replace('/', '-'),
            'question': row['question'], 'response': re.sub(r'<<[^>]*>>', '', row['answer']),
            'answer': answer}


def main():
    manifest = json.loads(Path('provenance/data_manifest.json').read_text(encoding='utf-8'))
    for name, source in manifest['sources'].items():
        if hashlib.sha256((Path('data/raw') / name).read_bytes()).hexdigest() != source['sha256']:
            raise ValueError(f'Source checksum mismatch: {name}')
    train, test = (read_jsonl(f'data/raw/{split}.jsonl') for split in ('train', 'test'))
    for split, indices in manifest['source_train_indices'].items():
        write_jsonl(f'data/original/{split}.jsonl',
                    [record(train[i], i, 'train', False, manifest['seed']) for i in indices])
    original_test = [record(row, i, 'test', False, manifest['seed']) for i, row in enumerate(test)]
    write_jsonl('data/original/test.jsonl', original_test)
    subprocess.run([sys.executable, 'sonnet_rewrite.py', '--export-only'], check=True)
    subprocess.run([sys.executable, 'assemble_rewrites.py', '--publish'], check=True)

    # Match the original preparation: the same RNG shuffles train, then test.
    rng = random.Random(manifest['seed'])
    indices = list(range(len(train)))
    rng.shuffle(indices)
    if indices[:3000] != manifest['source_train_indices']['train'] or indices[3000:3500] != manifest['source_train_indices']['validation']:
        raise ValueError('Recorded split does not match the original seeded shuffle')
    rng.shuffle(original_test)
    datasets = {
        'train_plain': read_jsonl('data/original/train.jsonl'),
        'train_mario': read_jsonl('data/mario/train.jsonl'),
        'val': read_jsonl('data/original/validation.jsonl'),
        'test': original_test[:500],
    }
    prepared = {name: [training_record(row) for row in rows] for name, rows in datasets.items()}
    for name, rows in prepared.items():
        if not all(is_correct(row['response'], row['answer']) for row in rows):
            raise ValueError(f'Invalid final answer in {name}')
        payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf-8')
        expected = manifest['prepared_files'][f'{name}.jsonl']
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f'Regenerated dataset differs from the experiment: {name}')
    for name, rows in prepared.items():
        write_jsonl(f'data/{name}.jsonl', rows)
    print('Rebuilt and hash-verified all four experiment datasets. No API calls were made.')


if __name__ == '__main__':
    main()
