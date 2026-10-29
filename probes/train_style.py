"""Train sentence-level boilerplate/filler probes, and calibrate the correctness-risk
threshold on factual spans in everyday chat text (not TriviaQA), so "flag the top 5%"
means 5% of spans in the deployment distribution.
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from common import DATA_DIR, DEFAULT_MODEL, load_model, model_slug, num_layers
from features import pool, response_hidden, token_range
from segment import factual_spans, split_sentences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--risk-quantile", type=float, default=0.95)
    ap.add_argument("--min-precision", type=float, default=0.8)
    args = ap.parse_args()
    path = os.path.join(DATA_DIR, model_slug(args.model))
    rows = [json.loads(l) for l in open(os.path.join(path, "style_labels.jsonl"))]
    pc = np.load(os.path.join(path, "probe_correctness.npz"))

    tok, model, device = load_model(args.model)
    n_layers = num_layers(model)
    layers = sorted(set(range(4, n_layers + 1, 4)) | {int(pc["layer"])})
    cache = os.path.join(path, "style_feats.pt")
    if os.path.exists(cache):
        F = torch.load(cache, weights_only=False)
    else:
        X, Yb, Yf, groups, risk_scores = [], [], [], [], []
        for r in rows:
            hs, offs = response_hidden(tok, model, device, [{"role": "user", "content": r["prompt"]}], r["response"], layers)
            segs = split_sentences(r["response"])
            if [s.text for s in segs] != [s["text"] for s in r["sentences"]]:
                continue
            for s, lab in zip(segs, r["sentences"]):
                idx = token_range(offs, s.start, s.end)
                if not idx:
                    continue
                X.append(pool(hs, idx, "mean").half())
                Yb.append(lab["boilerplate"])
                Yf.append(lab["filler"])
                groups.append(r["i"])
            hl = hs[layers.index(int(pc["layer"]))].numpy()
            for sp in factual_spans(r["response"]):
                idx = token_range(offs, sp.start, sp.end)
                if idx:
                    x = pool(torch.from_numpy(hl), idx, str(pc["pool"])).numpy()
                    risk_scores.append(float(1 / (1 + np.exp(-(((x - pc["mu"]) / pc["sd"]) @ pc["w"] + pc["b"])))))
        F = {"X": torch.stack(X), "boilerplate": np.array(Yb), "filler": np.array(Yf), "groups": np.array(groups),
             "risk_scores": np.array(risk_scores), "layers": layers}
        torch.save(F, cache)

    X = F["X"].float().numpy()
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(X, groups=F["groups"]))
    report = {"n_sentences": int(len(X)), "layers": F["layers"]}
    for name in ["boilerplate", "filler"]:
        y = F[name].astype(int)
        print(f"{name}: positive rate {y.mean():.3f}")
        best = None
        for li, layer in enumerate(F["layers"]):
            Xl = X[:, li]
            mu, sd = Xl[tr].mean(0), Xl[tr].std(0) + 1e-6
            clf = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit((Xl[tr] - mu) / sd, y[tr])
            p = clf.predict_proba((Xl[te] - mu) / sd)[:, 1]
            ap_ = average_precision_score(y[te], p)
            print(f"  layer {layer}: AUROC={roc_auc_score(y[te], p):.3f} AP={ap_:.3f}")
            if best is None or ap_ > best[0]:
                best = (ap_, li, layer, clf, mu, sd, p)
        ap_, li, layer, clf, mu, sd, p = best
        prec, rec, thr = precision_recall_curve(y[te], p)
        ok = np.where(prec[:-1] >= args.min_precision)[0]
        t = float(thr[ok[0]]) if len(ok) else 0.9
        r_at = float(rec[ok[0]]) if len(ok) else 0.0
        print(f"  -> layer {layer}, threshold {t:.3f} gives precision>={args.min_precision} recall={r_at:.2f}")
        np.savez(os.path.join(path, f"probe_{name}.npz"), w=clf.coef_[0].astype(np.float32), b=np.float32(clf.intercept_[0]),
                 mu=mu.astype(np.float32), sd=sd.astype(np.float32), layer=layer, pool="mean", threshold=t)
        report[name] = {"layer": int(layer), "test_auroc": float(roc_auc_score(y[te], p)), "test_ap": float(ap_),
                        "threshold": t, "recall_at_threshold": r_at, "positive_rate": float(y.mean())}

    rs = F["risk_scores"]
    thr = float(np.quantile(rs, args.risk_quantile))
    print(f"risk: {len(rs)} chat spans, q{args.risk_quantile} threshold = {thr:.3f}")
    report["risk"] = {"n_spans": int(len(rs)), "quantile": args.risk_quantile, "threshold": thr}
    with open(os.path.join(path, "probe_style.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
