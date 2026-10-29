"""Generate short TriviaQA answers and save hidden states at the answer tokens.

Output: data/<model>/trivia.pt with per-example features for a subset of layers:
  first[n, L, d]  hidden state at the first answer token
  last[n, L, d]   hidden state at the last answer token
  mean[n, L, d]   mean over answer tokens
plus labels and logprob baselines.
"""
import argparse
import os
import time

import torch
from datasets import load_dataset

from common import DATA_DIR, DEFAULT_MODEL, chat_prompt, is_correct, load_model, model_slug, num_layers

PROMPT = "Answer the trivia question with just the answer (a few words), no explanation.\n\nQuestion: {q}"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args()

    tok, model, device = load_model(args.model)
    n_layers = num_layers(model)
    layers = list(range(2, n_layers + 1, 2))  # hidden_states index; 0 is embeddings
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").shuffle(seed=0)
    ds = ds.select(range(args.n))

    out = {k: [] for k in ["first", "last", "mean"]}
    labels, preds, questions, lp_first, lp_mean, lp_min = [], [], [], [], [], []
    t0 = time.time()
    for b in range(0, args.n, args.batch):
        rows = ds[b : b + args.batch]
        prompts = [
            chat_prompt(tok, [{"role": "user", "content": PROMPT.format(q=q)}], tokenize=False)
            for q in rows["question"]
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, return_dict_in_generate=True, output_scores=True)
        seqs = gen.sequences
        new = seqs[:, enc.input_ids.shape[1] :]
        # Re-run forward on prompt+answer to get hidden states at answer positions (as inputs).
        attn = (seqs != tok.pad_token_id).long()
        attn[:, : enc.input_ids.shape[1]] = enc.attention_mask
        fwd = model(input_ids=seqs, attention_mask=attn, output_hidden_states=True, logits_to_keep=1)
        hs = torch.stack([fwd.hidden_states[l] for l in layers], dim=1)  # [B, L, T, d]
        logprobs = torch.log_softmax(torch.stack(gen.scores, 1).float(), -1)  # [B, new, V]
        P = enc.input_ids.shape[1]
        for i in range(len(prompts)):
            ids = new[i].tolist()
            # answer tokens: up to eos/pad, excluding trailing eos
            n_ans = 0
            for t in ids:
                if t in (tok.eos_token_id, tok.pad_token_id) or t == tok.convert_tokens_to_ids("<|im_end|>"):
                    break
                n_ans += 1
            text = tok.decode(ids[:n_ans]).strip()
            if n_ans == 0:
                continue
            pos = torch.arange(P, P + n_ans)
            h = hs[i][:, pos]  # [L, n_ans, d]
            out["first"].append(h[:, 0].to(torch.float16).cpu())
            out["last"].append(h[:, -1].to(torch.float16).cpu())
            out["mean"].append(h.float().mean(1).to(torch.float16).cpu())
            tok_lp = logprobs[i, torch.arange(n_ans), new[i, :n_ans]]
            lp_first.append(tok_lp[0].item())
            lp_mean.append(tok_lp.mean().item())
            lp_min.append(tok_lp.min().item())
            labels.append(int(is_correct(text, rows["answer"][i]["normalized_aliases"] + rows["answer"][i]["aliases"])))
            preds.append(text)
            questions.append(rows["question"][i])
        done = b + len(prompts)
        print(f"{done}/{args.n} acc={sum(labels)/max(1,len(labels)):.3f} {time.time()-t0:.0f}s", flush=True)

    path = os.path.join(DATA_DIR, model_slug(args.model))
    os.makedirs(path, exist_ok=True)
    torch.save(
        {
            "layers": layers,
            **{k: torch.stack(v) for k, v in out.items()},
            "labels": torch.tensor(labels),
            "lp_first": torch.tensor(lp_first),
            "lp_mean": torch.tensor(lp_mean),
            "lp_min": torch.tensor(lp_min),
            "preds": preds,
            "questions": questions,
            "model": args.model,
        },
        os.path.join(path, "trivia.pt"),
    )
    print("saved", path)


if __name__ == "__main__":
    main()
