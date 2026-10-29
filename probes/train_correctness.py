"""Train linear correctness-risk probes on TriviaQA answer-token hidden states.

Sweeps layer x pooling, reports held-out AUROC against logprob baselines, and saves the best probe.
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from common import DATA_DIR, DEFAULT_MODEL, model_slug


def fit(X, y, C):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=2000).fit((X - mu) / sd, y)
    return clf, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    path = os.path.join(DATA_DIR, model_slug(args.model))
    d = torch.load(os.path.join(path, "trivia.pt"), weights_only=False)
    y = d["labels"].numpy()
    # Risk = P(wrong); train on wrong=1.
    risk = 1 - y
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=0, stratify=y)
    idx_fit, idx_val = train_test_split(idx_tr, test_size=0.2, random_state=0, stratify=y[idx_tr])

    print(f"n={len(y)} accuracy={y.mean():.3f}")
    for k in ["lp_first", "lp_mean", "lp_min"]:
        print(f"baseline {k:8s} test AUROC={roc_auc_score(risk[idx_te], -d[k].numpy()[idx_te]):.3f}")

    best = None
    for pool in ["first", "last", "mean"]:
        H = d[pool].float().numpy()
        row = []
        for li, layer in enumerate(d["layers"]):
            X = H[:, li]
            for C in [0.001, 0.01]:
                clf, mu, sd = fit(X[idx_fit], risk[idx_fit], C)
                auc = roc_auc_score(risk[idx_val], clf.predict_proba((X[idx_val] - mu) / sd)[:, 1])
                if best is None or auc > best[0]:
                    best = (auc, pool, li, layer, C)
            row.append(f"{layer}:{auc:.3f}")
        print(pool, " ".join(row))

    auc_val, pool, li, layer, C = best
    X = d[pool].float().numpy()[:, li]
    clf, mu, sd = fit(X[idx_tr], risk[idx_tr], C)
    p_te = clf.predict_proba((X[idx_te] - mu) / sd)[:, 1]
    auc_te = roc_auc_score(risk[idx_te], p_te)
    print(f"best: pool={pool} layer={layer} C={C} val AUROC={auc_val:.3f} test AUROC={auc_te:.3f}")

    np.savez(
        os.path.join(path, "probe_correctness.npz"),
        w=clf.coef_[0].astype(np.float32), b=np.float32(clf.intercept_[0]),
        mu=mu.astype(np.float32), sd=sd.astype(np.float32), layer=layer, pool=pool,
    )
    with open(os.path.join(path, "probe_correctness.json"), "w") as f:
        json.dump({"pool": pool, "layer": int(layer), "C": C, "val_auroc": auc_val, "test_auroc": auc_te,
                   "baseline_test_auroc": {k: roc_auc_score(risk[idx_te], -d[k].numpy()[idx_te]) for k in ["lp_first", "lp_mean", "lp_min"]},
                   "n": int(len(y)), "accuracy": float(y.mean())}, f, indent=1)


if __name__ == "__main__":
    main()
