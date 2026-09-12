"""Resumable greedy Qwen evaluation with identical formatting across conditions."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from mario_data import read_jsonl
from verifier import extract_answer, gold_answer, is_correct

FORMAT_PROMPT = 'Solve the math problem step by step. End with a line containing #### followed by only the numerical answer.'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_rows(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty dataset or duplicate IDs')
    for row in rows:
        gold_answer(row['answer'])
    return rows


def resume_rows(path, rows):
    saved = read_jsonl(path) if path.exists() else []
    if len(saved) > len(rows):
        raise ValueError('Too many saved predictions')
    for old, source in zip(saved, rows):
        if any(old[k] != source[k] for k in ('id', 'question')) or old['gold'] != source['answer']:
            raise ValueError('Saved predictions do not match dataset order/content')
    return saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct')
    p.add_argument('--revision', default='main')
    p.add_argument('--adapter')
    p.add_argument('--system-prompt-file')
    p.add_argument('--max-examples', type=int)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--results-dir', default='results')
    args = p.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1 or (args.max_examples is not None and args.max_examples < 1):
        p.error('Counts must be positive')
    if Path(args.name).name != args.name or '/' in args.name or '\\' in args.name:
        p.error('name must be a filename')
    rows = validate_rows(read_jsonl(args.data)[:args.max_examples])
    system = FORMAT_PROMPT
    if args.system_prompt_file:
        system = Path(args.system_prompt_file).read_text(encoding='utf-8').strip() + '\n\n' + FORMAT_PROMPT
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import model_info

    revision = model_info(args.model, revision=args.revision).sha
    root = Path(args.results_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{args.name}.jsonl'
    metadata_path = root / f'{args.name}.config.json'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    config = {'model': args.model, 'revision': revision, 'data_sha256': sha256(args.data),
              'n': len(rows), 'system_prompt': system, 'max_new_tokens': args.max_new_tokens,
              'batch_size': args.batch_size, 'do_sample': False, 'dtype': str(dtype),
              'adapter_hashes': {n: sha256(Path(args.adapter) / n) for n in
                                 ('adapter_config.json', 'adapter_model.safetensors')} if args.adapter else {},
              'packages': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft', 'accelerate')},
              'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
              'eval_sha256': sha256(__file__), 'verifier_sha256': sha256('verifier.py')}
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding='utf-8')) != config:
            raise ValueError('Run configuration changed; choose a new name')
    elif path.exists():
        raise ValueError('Predictions exist without run metadata')
    else:
        metadata_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    saved = resume_rows(path, rows)
    if len(saved) < len(rows):
        tok = AutoTokenizer.from_pretrained(args.model, revision=revision)
        tok.padding_side = 'left'
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, dtype=dtype).to(device)
        if args.adapter:
            model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        model.eval()
        with path.open('a', encoding='utf-8', newline='\n') as stream:
            for start in range(len(saved), len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                prompts = [tok.apply_chat_template([{'role': 'system', 'content': system},
                           {'role': 'user', 'content': r['question']}], tokenize=False,
                           add_generation_prompt=True) for r in batch]
                enc = tok(prompts, return_tensors='pt', padding=True).to(device)
                t0 = time.monotonic()
                with torch.inference_mode():
                    gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                elapsed = time.monotonic() - t0
                tokens = gen[:, enc['input_ids'].shape[1]:]
                eos = model.generation_config.eos_token_id
                eos = set(eos if isinstance(eos, list) else [eos])
                for source, sequence in zip(batch, tokens.tolist()):
                    stop = next((i for i, token in enumerate(sequence) if token in eos), None)
                    length = stop + 1 if stop is not None else len(sequence)
                    output = tok.decode(sequence[:length], skip_special_tokens=True)
                    pred = extract_answer(output)
                    result = {'id': source['id'], 'question': source['question'], 'gold': source['answer'],
                              'output': output, 'pred': str(pred) if pred is not None else None,
                              'correct': is_correct(output, source['answer']), 'format_valid': pred is not None,
                              'generated_tokens': length, 'hit_token_cap': stop is None and length >= args.max_new_tokens,
                              'batch_seconds_per_example': elapsed / len(batch)}
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                    saved.append(result)
                stream.flush()
                print(f'{args.name}: {len(saved)}/{len(rows)}', flush=True)
    summary = {'name': args.name, 'n': len(saved),
               'accuracy': sum(r['correct'] for r in saved) / len(saved),
               'format_success': sum(r['format_valid'] for r in saved) / len(saved),
               'hit_token_cap': sum(r['hit_token_cap'] for r in saved),
               'generation_seconds': sum(r['batch_seconds_per_example'] for r in saved)}
    (root / f'{args.name}.metrics.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
