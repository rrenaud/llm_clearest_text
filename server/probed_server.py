"""OpenAI-compatible chat server that annotates generated text with internal-probe signals.

Annotations are inline tags understood by the patched Open WebUI renderer:
  <probe d="...">span</probe>        factual span: correctness risk, provenance, top-k alternatives
  <probe-s d="...">sentence</probe-s>  sentence: boilerplate / filler probabilities
  <probe-t d="">piece</probe-t>        inside a scored span: the token the alternatives were taken at
`d` is URL-encoded JSON. Only flagged items are tagged unless the "-debug" model is used.
Model ids: "<name>-probed", "<name>-probed-debug" (tag every span), "<name>-plain" (no tags).

Run:  python probed_server.py --port 8100
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import uuid
from urllib.parse import quote

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probes"))
from common import DATA_DIR, DEFAULT_MODEL, chat_prompt, load_model, model_slug, normalize, reset_position_state  # noqa: E402
from features import pool  # noqa: E402
from segment import factual_spans, split_sentences  # noqa: E402

TOPK = 5
MIN_CONTEXT_CHARS = 400  # below this, nearly everything is parametric and tinting it is noise


class Probe:
    def __init__(self, path):
        z = np.load(path)
        self.w, self.b = torch.from_numpy(z["w"]), float(z["b"])
        self.mu, self.sd = torch.from_numpy(z["mu"]), torch.from_numpy(z["sd"])
        self.layer, self.pool = int(z["layer"]), str(z["pool"])
        self.threshold = float(z["threshold"]) if "threshold" in z else None

    def __call__(self, hs_by_layer, idx):
        x = pool(hs_by_layer[self.layer], idx, self.pool)
        return torch.sigmoid(((x - self.mu) / self.sd) @ self.w + self.b).item()


class Engine:
    def __init__(self, model_name):
        self.name = model_name
        self.tok, self.model, self.device = load_model(model_name)
        path = os.path.join(DATA_DIR, model_slug(model_name))
        self.probes = {}
        for k in ["correctness", "boilerplate", "filler"]:
            p = os.path.join(path, f"probe_{k}.npz")
            if os.path.exists(p):
                self.probes[k] = Probe(p)
        style = os.path.join(path, "probe_style.json")
        self.risk_threshold = json.load(open(style))["risk"]["threshold"] if os.path.exists(style) else 0.8
        self.layers = sorted({p.layer for p in self.probes.values()})
        self.lock = threading.Lock()
        self.stop_ids = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.convert_tokens_to_ids("<|endoftext|>")}
        print(f"loaded {model_name} on {self.device}; probes={list(self.probes)} risk_threshold={self.risk_threshold:.3f}")

    @torch.no_grad()
    def generate(self, messages, max_tokens=1024, temperature=0.7, top_p=0.9):
        """Yield one dict per generated token: id, alts, entropy; hidden states arrive one step late."""
        ids = chat_prompt(self.tok, messages, return_tensors="pt", return_dict=True)["input_ids"].to(self.device)
        past = None
        step_input = ids
        reset_position_state(self.model)  # Qwen3.5/3.6 cache rope offsets across calls
        pending = None  # token whose hidden state is computed by the next forward
        for _ in range(max_tokens + 1):
            out = self.model(input_ids=step_input, past_key_values=past, use_cache=True, output_hidden_states=True,
                             logits_to_keep=1)
            past = out.past_key_values
            if pending is not None:
                pending["hs"] = {l: out.hidden_states[l][0, -1].float().cpu() for l in self.layers}
                yield pending
                if pending["id"] in self.stop_ids:
                    return
            logits = out.logits[0, -1].float()
            probs = torch.softmax(logits, -1)
            ent = float(-(probs * torch.log(probs + 1e-12)).sum())
            top = torch.topk(probs, TOPK)
            if temperature <= 0:
                nxt = int(top.indices[0])
            else:
                p = torch.softmax(logits / temperature, -1)
                sp, si = torch.sort(p, descending=True)
                keep = torch.cumsum(sp, 0) - sp < top_p
                sp = sp * keep
                nxt = int(si[torch.multinomial(sp / sp.sum(), 1)])
            # Merge tokenizer variants of one word ("Personal" / " Personal") so alternatives are distinct
            # words and p is not split across spellings.
            alts = {}
            for i, v in zip(top.indices, top.values):
                w = self.tok.decode([int(i)]).strip()
                alts[w] = alts.get(w, 0.0) + float(v)
            chosen = self.tok.decode([nxt]).strip()
            pending = {"id": nxt, "p": alts.get(chosen, float(probs[nxt])), "ent": ent,
                       "alts": [(w, round(v, 3)) for w, v in sorted(alts.items(), key=lambda x: -x[1])]}
            step_input = torch.tensor([[nxt]], device=self.device)


def esc(d):
    return quote(json.dumps(d, separators=(",", ":")), safe="")


def context_text(messages):
    """Everything the model was given (excluding its own earlier answers)."""
    return "\n".join(m["content"] for m in messages if m["role"] in ("system", "user", "tool") and isinstance(m["content"], str))


def provenance(span, ctx_norm, ctx_has_numbers):
    s = normalize(span)
    if s and s in ctx_norm:
        return "ctx"
    if re.search(r"\d", span) and ctx_has_numbers:
        return "computed"
    return "param"


class Annotator:
    """Incrementally turns generated tokens into annotated markdown."""

    def __init__(self, engine, messages, debug=False):
        self.e = engine
        self.debug = debug
        ctx = context_text(messages)
        self.ctx_norm = normalize(ctx)
        self.ctx_has_numbers = len(re.findall(r"\d+(?:\.\d+)?", ctx)) >= 2
        self.show_prov = len(ctx) >= MIN_CONTEXT_CHARS
        self.toks = []  # per token: dict with id, alts, hs, start, end
        self.ids = []
        self.text = ""
        self.emitted = 0

    def add(self, t):
        if t["id"] in self.e.stop_ids:
            return
        self.ids.append(t["id"])
        new_text = self.e.tok.decode(self.ids, skip_special_tokens=True)
        t["start"], t["end"] = len(self.text), len(new_text)
        self.text = new_text
        self.toks.append(t)

    def _tok_idx(self, a, b):
        return [i for i, t in enumerate(self.toks) if t["start"] < b and t["end"] > a and t["end"] > t["start"]]

    def _hs(self, idx):
        return {l: torch.stack([self.toks[i]["hs"][l] for i in idx]) for l in self.e.layers}

    def ready(self, final=False):
        """Annotated markdown for newly completed text."""
        if final:
            upto = len(self.text)
        else:
            segs = split_sentences(self.text)
            last_nl = self.text.rfind("\n") + 1
            upto = max(last_nl, segs[-1].start if segs else 0)
        if upto <= self.emitted:
            return ""
        out = self._annotate(self.emitted, upto)
        self.emitted = upto
        return out

    def _annotate(self, a, b):
        text = self.text
        inserts = []  # (pos, order, string); at one pos: close token, span, sentence; then open sentence, span, token
        for s in split_sentences(text):
            if s.start < a or s.end > b:
                continue
            idx = self._tok_idx(s.start, s.end)
            if not idx:
                continue
            hs = self._hs(idx)
            d, flags = {}, []
            for k in ["boilerplate", "filler"]:
                if k in self.e.probes:
                    p = self.e.probes[k](hs, list(range(len(idx))))
                    if p >= self.e.probes[k].threshold:
                        flags.append(k)
                    if self.debug or p >= self.e.probes[k].threshold:
                        d[k] = round(p, 3)
            if d:
                d["flags"] = flags
                inserts.append((s.start, 2, f'<probe-s d="{esc(d)}">'))
                inserts.append((s.end, 1, "</probe-s>"))
        for sp in factual_spans(text):
            if sp.start < a or sp.end > b:
                continue
            idx = self._tok_idx(sp.start, sp.end)
            if not idx:
                continue
            d = {}
            prov = provenance(sp.text, self.ctx_norm, self.ctx_has_numbers)
            # Spans copied from the user's own messages are not the model's claims; skip risk there.
            if "correctness" in self.e.probes and prov != "ctx":
                r = self.e.probes["correctness"](self._hs(idx), list(range(len(idx))))
                if r >= self.e.risk_threshold:
                    d["flags"] = ["risk"]
                if self.debug or r >= self.e.risk_threshold:
                    d["risk"] = round(r, 3)
                    # Alternatives at the span's least certain token (the decision point).
                    j = min(idx, key=lambda i: self.toks[i]["p"])
                    d["alts"] = self.toks[j]["alts"]
                    d["tok"] = self.e.tok.decode([self.toks[j]["id"]]).strip()
                    d["p"] = round(self.toks[j]["p"], 3)
                    # Mark the contested token's characters inside the span (minus leading space).
                    t0, t1 = max(self.toks[j]["start"], sp.start), min(self.toks[j]["end"], sp.end)
                    while t0 < t1 and text[t0].isspace():
                        t0 += 1
                    if t0 < t1:
                        inserts.append((t0, 4, '<probe-t d="">'))
                        inserts.append((t1, -1, "</probe-t>"))
            if self.show_prov or self.debug:
                d["prov"] = prov
            if d:
                inserts.append((sp.start, 3, f'<probe d="{esc(d)}">'))
                inserts.append((sp.end, 0, "</probe>"))
        inserts.sort(key=lambda x: (x[0], x[1]))
        out, pos = [], a
        for p, _, s in inserts:
            out.append(text[pos:p])
            out.append(s)
            pos = p
        out.append(text[pos:b])
        return "".join(out)


def build_app(engine):
    app = FastAPI()
    api_key = os.environ.get("PROBE_API_KEY")  # set when the server is reachable from the internet

    @app.middleware("http")
    async def check_key(request: Request, call_next):
        if api_key and request.headers.get("authorization") != f"Bearer {api_key}":
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    base = model_slug(engine.name).lower()
    variants = {f"{base}-probed": "probed", f"{base}-probed-debug": "debug", f"{base}-plain": "plain"}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "local", "created": 0} for m in variants]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        mode = variants.get(body.get("model"), "probed")
        messages = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str)
                     else "".join(c.get("text", "") for c in m["content"] if c.get("type") == "text")}
                    for m in body["messages"]]
        # Strip tags from earlier assistant turns before they go back into the model.
        for m in messages:
            if m["role"] == "assistant":
                m["content"] = re.sub(r"</?probe(?:-s)?(?: d=\"[^\"]*\")?>", "", m["content"])
        kw = dict(max_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024),
                  temperature=float(body.get("temperature", 0.7)), top_p=float(body.get("top_p", 0.9)))
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())

        def pieces():
            with engine.lock:
                ann = Annotator(engine, messages, debug=(mode == "debug"))
                for t in engine.generate(messages, **kw):
                    ann.add(t)
                    if mode == "plain":
                        chunk = ann.text[ann.emitted:]
                        ann.emitted = len(ann.text)
                    else:
                        chunk = ann.ready()
                    if chunk:
                        yield chunk
                if mode != "plain":
                    tail = ann.ready(final=True)
                    if tail:
                        yield tail

        def chunk_json(delta, finish=None):
            return json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": body.get("model"),
                               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

        if body.get("stream"):
            def sse():
                yield f"data: {chunk_json({'role': 'assistant'})}\n\n"
                for p in pieces():
                    yield f"data: {chunk_json({'content': p})}\n\n"
                yield f"data: {chunk_json({}, 'stop')}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")
        # Generation blocks; keep it off the event loop so other requests (e.g. /v1/models) stay responsive.
        content = await run_in_threadpool(lambda: "".join(pieces()))
        return JSONResponse({"id": cid, "object": "chat.completion", "created": created, "model": body.get("model"),
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()
    engine = Engine(args.model)
    uvicorn.run(build_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
