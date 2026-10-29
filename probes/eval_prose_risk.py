"""Does the short-answer correctness probe transfer to prose? Ask held-out TriviaQA questions for a
full-sentence answer, score factual spans not copied from the question, and compare max span risk
against correctness (alias appears in the sentence). Uses questions after the probe's training range.
"""
import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

from common import DATA_DIR, DEFAULT_MODEL, chat_prompt, is_correct, load_model, model_slug, normalize
from features import pool, response_hidden, token_range
from segment import factual_spans

PROMPT = "Answer in one complete sentence.\n\nQuestion: {q}"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()
    path = os.path.join(DATA_DIR, model_slug(args.model))
    pc = np.load(os.path.join(path, "probe_correctness.npz"))
    thr = json.load(open(os.path.join(path, "probe_style.json")))["risk"]["threshold"]
    tok, model, device = load_model(args.model)
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").shuffle(seed=0)
    ds = ds.select(range(10000, 10000 + args.n))  # disjoint from gen_trivia's first 8000

    rows = []
    for b in range(0, args.n, 16):
        batch = ds[b : b + 16]
        msgs = [[{"role": "user", "content": PROMPT.format(q=q)}] for q in batch["question"]]
        enc = tok([chat_prompt(tok, m, tokenize=False) for m in msgs], return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, max_new_tokens=60, do_sample=False)
        for i, ids in enumerate(out[:, enc.input_ids.shape[1] :]):
            resp = tok.decode(ids, skip_special_tokens=True).strip()
            if not resp:  # an empty answer has no tokens to probe
                continue
            ans = batch["answer"][i]
            correct = is_correct(resp, ans["normalized_aliases"] + ans["aliases"])
            hs, offs = response_hidden(tok, model, device, msgs[i], resp, [int(pc["layer"])])
            qn = normalize(batch["question"][i])
            risks = []
            for sp in factual_spans(resp):
                if normalize(sp.text) in qn:
                    continue
                idx = token_range(offs, sp.start, sp.end)
                if idx:
                    x = pool(hs[0], idx, str(pc["pool"])).numpy()
                    risks.append(float(1 / (1 + np.exp(-(((x - pc["mu"]) / pc["sd"]) @ pc["w"] + pc["b"])))))
            if risks:
                rows.append((correct, max(risks)))
        print(f"{b + 16}/{args.n}", flush=True)

    y = np.array([1 - c for c, _ in rows])
    r = np.array([x for _, x in rows])
    flagged = r >= thr
    res = {"n": len(rows), "wrong_rate": float(y.mean()), "auroc_max_span_risk": float(roc_auc_score(y, r)),
           "threshold": thr, "flag_rate": float(flagged.mean()),
           "precision_at_threshold": float(y[flagged].mean()) if flagged.any() else None,
           "recall_at_threshold": float(flagged[y == 1].mean())}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(path, "eval_prose_risk.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
