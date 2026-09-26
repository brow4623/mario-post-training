"""Stage 3 RLVR on Modal: smoke test, pilot gate, training runs, and matched evaluation.

Modal replaces the RunPod pod that run_rlvr.sh assumes. The repository code and the data it
needs are baked into the image; every artifact is written to the `rlvr-outputs` volume, which
outlives the container. Nothing lands on your laptop automatically -- see "Retrieving results".

    modal run modal_rlvr.py::fetch_adapter          # once: private release asset -> volume
    modal run modal_rlvr.py                         # smoke -> pilot -> two runs -> evaluation

Retrieving results (the volume is not synced to your checkout):

    modal volume get rlvr-outputs /checkpoints ./
    modal volume get rlvr-outputs /results ./
    modal volume get rlvr-outputs /provenance ./

Costs: the combined run pays Anthropic for persona ratings and every stage rents an A100.
Read docs/RLVR.md before starting and keep --judge-budget-usd below your Anthropic balance.
"""

import os
import subprocess
import sys
from pathlib import Path

import modal

REPO = "/repo"
OUT = "/outputs"
ADAPTER_SHA256 = "278224dcc00ea44511fcce4a760e6f50e5a99813dacd73baaae1837e6f158e58"
ADAPTER_ASSET = "qwen3b-gsm8k-mario-rlaif-lora.safetensors"
ADAPTER_TAG = "rlaif-v1"
GITHUB_REPO = "harvard-cs2881f26/brown-ogbogu-tantia"
GPU = "A100"

# torch 2.8.0 (requirements-rlvr.txt, from the RunPod image) is gone from PyPI; 2.9.0 is the
# oldest still published. Everything else stays pinned, and packages.txt records what ran.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.9.0",
        "transformers==5.17.0",
        "datasets==5.0.1",
        "accelerate==1.15.0",
        "peft==0.21.0",
        "trl==1.13.0",
        "scikit-learn==1.7.2",
        "requests",
    )
    # The scripts resolve every path from the working directory, so the tree must be real
    # files in the image, not an empty volume. copy=True keeps them writable at runtime.
    .add_local_dir(".", f"{REPO}", copy=True, ignore=[
        "venv/**", ".venv/**", ".git/**", "__pycache__/**", "**/__pycache__/**",
        "checkpoints/**", "derived/**", "results/**", "reports/**", "docs/**",
        "data/original/**", "data/mario/**", "data/raw/**", "data/candidates/**",
        "provenance/generation/**", "provenance/evaluation/**", "*.safetensors",
        ".env", ".env.*", "modal_rlvr.py",
    ])
)

app = modal.App("rlvr-mario", image=image)
outputs = modal.Volume.from_name("rlvr-outputs", create_if_missing=True)
hf_cache = modal.Volume.from_name("rlvr-hf-cache", create_if_missing=True)

def dotenv(path=".env"):
    """Read the ignored .env into a Modal secret without depending on python-dotenv.

    judge.api_key() checks the environment before any --env-file, so exporting the keys this
    way is enough. A missing .env is fine until a combined run actually needs the Anthropic
    key, or fetch_adapter needs the GitHub one.
    """
    import re

    values = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
            if not match:
                continue
            name, value = match[1], match[2].strip()
            if value[:1] in ("'", '"') and value[-1:] == value[:1]:
                value = value[1:-1]
            else:
                value = value.split(" #", 1)[0].strip()
            if value:
                values[name] = value
    return [modal.Secret.from_dict(values)] if values else []


env_secret = dotenv()


def workspace():
    """Point the repository's relative output paths at the persistent volume."""
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    os.environ.update(HF_HOME="/hf-cache", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="8")

    # rewards.PROVENANCE is the literal path "provenance/rlvr": the judge ledger, the rating
    # cache, and each run's reward log all land there. Redirect it to the volume so a crashed
    # container does not lose ratings that were already paid for, keeping the committed
    # training_questions.json readable at its usual place.
    live = Path(f"{OUT}/provenance/rlvr")
    live.mkdir(parents=True, exist_ok=True)
    frozen = Path(REPO) / "provenance/rlvr/training_questions.json"
    if frozen.exists() and not (live / "training_questions.json").exists():
        (live / "training_questions.json").write_bytes(frozen.read_bytes())
    target = Path(REPO) / "provenance/rlvr"
    if not target.is_symlink():
        if target.exists():
            import shutil
            shutil.rmtree(target)
        target.symlink_to(live)

    # peft needs adapter_config.json (image) and adapter_model.safetensors (volume) together.
    weights = Path(f"{OUT}/adapters/rlaif/adapter_model.safetensors")
    if not weights.exists():
        raise RuntimeError("RLAIF adapter missing from the volume; "
                           "run `modal run modal_rlvr.py::fetch_adapter` first")
    link = Path(REPO) / "adapters/rlaif/adapter_model.safetensors"
    if not link.exists():
        link.symlink_to(weights)


def run(command, label):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
    result = subprocess.run(command, shell=True, cwd=REPO)
    outputs.commit()
    if result.returncode:
        raise RuntimeError(f"{label} failed (exit {result.returncode})")


@app.function(volumes={OUT: outputs}, timeout=1800, secrets=env_secret)
def fetch_adapter(force: bool = False):
    """Download the RLAIF weights from the private release into the volume, hash-checked.

    The repository is private, so an unauthenticated download returns 404. Put a GitHub token
    with `repo` scope in .env as GITHUB_TOKEN, or upload the file yourself:
        modal volume put rlvr-outputs adapter_model.safetensors /adapters/rlaif/
    """
    import hashlib
    import requests

    dest = Path(f"{OUT}/adapters/rlaif/adapter_model.safetensors")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        print(f"already present, sha256 {digest}")
        if digest != ADAPTER_SHA256:
            raise RuntimeError(f"expected {ADAPTER_SHA256}; re-run with --force")
        return digest

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError("Set GITHUB_TOKEN in .env: the release is private and 404s without it")
    headers = {"Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"}

    release = requests.get(f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{ADAPTER_TAG}",
                           headers={**headers, "Accept": "application/vnd.github+json"}, timeout=60)
    release.raise_for_status()
    asset = next((a for a in release.json()["assets"] if a["name"] == ADAPTER_ASSET), None)
    if asset is None:
        raise RuntimeError(f"{ADAPTER_ASSET} is not in release {ADAPTER_TAG}")

    print(f"downloading {ADAPTER_ASSET} ({asset['size'] / 1e6:.0f} MB)", flush=True)
    stream = requests.get(f"https://api.github.com/repos/{GITHUB_REPO}/releases/assets/{asset['id']}",
                          headers={**headers, "Accept": "application/octet-stream"}, stream=True, timeout=300)
    stream.raise_for_status()
    sha, size = hashlib.sha256(), 0
    with dest.open("wb") as handle:
        for chunk in stream.iter_content(chunk_size=1 << 20):
            handle.write(chunk)
            sha.update(chunk)
            size += len(chunk)
    outputs.commit()

    digest = sha.hexdigest()
    print(f"{size / 1e6:.1f} MB, sha256 {digest}")
    if digest != ADAPTER_SHA256:
        dest.unlink()
        outputs.commit()
        raise RuntimeError(f"hash mismatch: expected {ADAPTER_SHA256}, got {digest}")
    print("hash matches adapters/README.md")
    return digest


@app.function(gpu=GPU, volumes={OUT: outputs, "/hf-cache": hf_cache}, timeout=3600, secrets=env_secret)
def preflight():
    """Package record, RLVR unit tests, and the frozen-question check, before any GPU spend."""
    import torch

    workspace()
    run(f"pip freeze > {OUT}/provenance/rlvr/packages.txt", "record packages")
    print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}, "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu'}")
    # The full suite reads files this image leaves out; these two cover the RLVR paths.
    run("python -m unittest tests.test_rewards tests.test_rlvr -q", "RLVR unit tests")
    run("python make_rlvr_data.py --check", "verify frozen training questions")


@app.function(gpu=GPU, volumes={OUT: outputs, "/hf-cache": hf_cache}, timeout=86400, secrets=env_secret)
def train(run_name: str, reward: str, persona_weight: float = 0.25,
          budget_usd: float = 0.0, flags: str = ""):
    """One GRPO run. `flags` carries the smoke/pilot overrides; everything else is rlvr.py's default."""
    workspace()
    if reward == "combined" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set; put it in .env so the secret picks it up")
    budget = f"--budget-usd {budget_usd}" if reward == "combined" else ""
    weight = f"--persona-weight {persona_weight}" if reward == "combined" else ""
    run(f"python -u rlvr.py --reward {reward} --adapter adapters/rlaif "
        f"--output {OUT}/checkpoints/{run_name} --run-name {run_name} {weight} {budget} {flags}",
        f"train {run_name} ({reward}"
        f"{f', w={persona_weight}' if reward == 'combined' else ''})")


@app.function(gpu=GPU, volumes={OUT: outputs, "/hf-cache": hf_cache}, timeout=86400)
def evaluate(condition: str, adapter: str, split: str):
    """Greedy decoding, identical settings for every condition, including the RLAIF baseline."""
    workspace()
    run(f"python -u eval.py --data data/{split}.jsonl --name {condition}_{split} "
        f"--adapter {adapter} --batch-size 16 --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 "
        f"--results-dir {OUT}/results/rlvr --provenance-dir {OUT}/provenance/evaluation/rlvr "
        f"--metrics-dir {OUT}/derived/rlvr/metrics", f"eval {condition} / {split}")


@app.function(volumes={OUT: outputs}, timeout=900)
def pilot_gate(minimum: int = 10):
    """Stop before the paid runs unless enough groups mix correct and wrong answers."""
    workspace()
    run(f"python rewards.py summarize {OUT}/provenance/rlvr/pilot/rewards.jsonl "
        f"--require-mixed-groups {minimum}", "pilot gate")


@app.local_entrypoint()
def main(stage: str = "all", judge_budget_usd: float = 6.0, sensitivity: bool = False,
         persona_weight: float = 0.25):
    """Run the stage 3 pipeline on Modal.

    stage:            all | preflight | smoke | pilot | train | eval
    judge_budget_usd: hard ceiling for persona ratings, shared across retries via the ledger
    sensitivity:      also train w=0.5, 0.75, 1.0 (three more GPU runs and ~3x the judge spend)
    persona_weight:   w in C * (1 + w * P) for the main combined run
    """
    stages = ("preflight", "smoke", "pilot", "train", "eval")
    if stage not in stages + ("all",):
        raise SystemExit(f"stage must be one of {', '.join(stages + ('all',))}")
    todo = stages if stage == "all" else (stage,)

    combined = [("rlvr-combined", persona_weight)]
    if sensitivity:
        combined += [(f"rlvr-combined-w{w}".replace(".", "_"), w) for w in (0.5, 0.75, 1.0)]

    if "preflight" in todo:
        preflight.remote()

    if "smoke" in todo:
        # Two updates on the real adapter, no judge calls, throwaway checkpoint.
        train.remote("smoke", "math",
                     flags="--max-examples 8 --max-steps 2 --batch-size 4 --grad-accum 1 "
                           "--max-completion-length 256")
        print("\nHand-check provenance/rlvr/smoke/samples.jsonl:\n"
              "  modal volume get rlvr-outputs /provenance/rlvr/smoke ./\n")

    if "pilot" in todo:
        # Real sampling settings at lr 0: describes the starting policy without training it.
        train.remote("pilot", "math", flags="--max-examples 64 --lr 0")
        pilot_gate.remote()

    if "train" in todo:
        train.remote("rlvr-math", "math")
        for name, weight in combined:
            # Each run starts from the RLAIF adapter, never from the math-only result.
            train.remote(name, "combined", persona_weight=weight, budget_usd=judge_budget_usd)

    if "eval" in todo:
        conditions = [("rlvr_baseline", "adapters/rlaif"),
                      ("rlvr_math", f"{OUT}/checkpoints/rlvr-math")]
        conditions += [(name.replace("-", "_"), f"{OUT}/checkpoints/{name}") for name, _ in combined]
        for condition, adapter in conditions:
            for split in ("val", "test"):
                evaluate.remote(condition, adapter, split)

    print("\nArtifacts stay on the volume. Pull them down with:\n"
          "  modal volume get rlvr-outputs /checkpoints ./\n"
          "  modal volume get rlvr-outputs /results ./\n"
          "  modal volume get rlvr-outputs /provenance ./\n"
          "Then score the persona ratings and analyse per docs/RLVR.md steps 7-8.")
