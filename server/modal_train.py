"""Run the full probe pipeline (scripts/train_probes.sh + eval_prose_risk.py) on a Modal GPU.

Run:  PROBE_MODEL=Qwen/Qwen2.5-7B-Instruct modal run --detach server/modal_train.py
Outputs land in the "probed-data" volume under <model slug>/; fetch them with scripts/train_modal.sh.
Style labels need the Modal secret "anthropic-key" (ANTHROPIC_API_KEY).
"""
import os

import modal

MODEL = os.environ.get("PROBE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
GPU = os.environ.get("MODAL_GPU", "L40S")  # use H100 for 20B+ models
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.10.0", "transformers==5.3.0", "accelerate==1.10.1", "numpy", "datasets==4.8.4",
                    "scikit-learn==1.9.1", "anthropic==2.3.0", "pydantic==2.14.1",
                    "flash-linear-attention")  # Gated DeltaNet kernels for hybrid models (Qwen3.5/3.6)
    .env({"HF_HOME": "/cache/hf", "PROBE_MODEL": MODEL, "PYTHONUNBUFFERED": "1", "PYTORCH_ALLOC_CONF": "expandable_segments:True",
          "LABEL_MODEL": os.environ.get("LABEL_MODEL", "claude-sonnet-5-5"),
          "GEN_BATCH": os.environ.get("GEN_BATCH", "16")})
    .add_local_dir(os.path.join(ROOT, "probes"), "/root/app/probes", ignore=["__pycache__"])
)

app = modal.App("probe-train")
cache = modal.Volume.from_name("probed-hf-cache", create_if_missing=True)
data = modal.Volume.from_name("probed-data", create_if_missing=True)


@app.function(image=image, gpu=GPU, volumes={"/cache": cache, "/root/app/data": data},
              secrets=[modal.Secret.from_name("anthropic-key")], timeout=10 * 3600, memory=131072)
def train(steps: str = "gen_trivia,train_correctness,gen_responses,label_style,train_style,eval_prose_risk"):
    import subprocess
    for step in steps.split(","):
        print(f"=== {step} ({MODEL})", flush=True)
        extra = ["--n", "8000"] if step == "gen_trivia" else []  # match train_probes.sh
        if step in ("gen_trivia", "gen_responses"):
            extra += ["--batch", os.environ.get("GEN_BATCH", "16")]  # 16 keeps a 27B model within 80 GB
        subprocess.run(["python", f"{step}.py", *extra], cwd="/root/app/probes", check=True)
        data.commit()  # keep finished stages if a later one fails
        cache.commit()


@app.local_entrypoint()
def main(steps: str = ""):
    train.remote(steps) if steps else train.remote()


@app.function(image=image, volumes={"/root/app/data": data}, memory=16384, timeout=1800)
def export_text(slug: str):
    """Write trivia.jsonl (question, model answer, correctness label, logprob baselines) next to trivia.pt; no GPU."""
    import json

    import torch
    d = torch.load(f"/root/app/data/{slug}/trivia.pt", map_location="cpu", mmap=True, weights_only=False)
    with open(f"/root/app/data/{slug}/trivia.jsonl", "w") as f:
        for i, (q, a) in enumerate(zip(d["questions"], d["preds"])):
            f.write(json.dumps({"question": q, "answer": a, "correct": int(d["labels"][i]),
                                **{k: round(float(d[k][i]), 4) for k in ("lp_first", "lp_mean", "lp_min")}}) + "\n")
    data.commit()
    return len(d["questions"])
