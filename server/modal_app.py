"""Serve probed_server.py on a Modal GPU. Open WebUI stays local and points at the printed URL.

Deploy:  PROBE_MODEL=Qwen/Qwen2.5-1.5B-Instruct modal deploy server/modal_app.py
Requires the Modal secret "probed-server-key" holding PROBE_API_KEY (see scripts/deploy_modal.sh).
The container stays warm for SCALEDOWN seconds after the last request, then scales to zero.
"""
import os

import modal

MODEL = os.environ.get("PROBE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
GPU = os.environ.get("MODAL_GPU", "L4,A10G,L40S").split(",")  # fallbacks when L4 capacity is short
SCALEDOWN = 20 * 60
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLUG = MODEL.split("/")[-1]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.10.0", "transformers==5.3.0", "accelerate==1.10.1", "numpy", "fastapi", "uvicorn",
                    "flash-linear-attention")  # Gated DeltaNet kernels for Qwen3.5/3.6
    .env({"HF_HOME": "/cache/hf", "PROBE_MODEL": MODEL})
    .add_local_dir(os.path.join(ROOT, "probes"), "/root/app/probes", ignore=["__pycache__"])
    .add_local_dir(os.path.join(ROOT, "server"), "/root/app/server", ignore=["__pycache__"])
)
for f in ["probe_correctness.npz", "probe_boilerplate.npz", "probe_filler.npz", "probe_style.json"]:
    image = image.add_local_file(os.path.join(ROOT, "data", SLUG, f), f"/root/app/data/{SLUG}/{f}")

app = modal.App("probed-chat-server")
cache = modal.Volume.from_name("probed-hf-cache", create_if_missing=True)


@app.cls(image=image, gpu=GPU, volumes={"/cache": cache}, secrets=[modal.Secret.from_name("probed-server-key")],
         scaledown_window=SCALEDOWN, max_containers=1, timeout=900)
@modal.concurrent(max_inputs=8)  # generation is serialized by Engine.lock; this lets /v1/models etc. through
class Server:
    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, "/root/app/server")
        from probed_server import Engine
        self.engine = Engine(MODEL)
        cache.commit()  # keep downloaded weights for the next cold start

    @modal.asgi_app()
    def web(self):
        from probed_server import build_app
        return build_app(self.engine)
