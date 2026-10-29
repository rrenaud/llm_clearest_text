"""Generate everyday chat responses for sentence-level style probes (boilerplate, filler).

Output: data/<model>/responses.jsonl with {"prompt", "response"}.
"""
import argparse
import json
import os
import random

import torch
from datasets import load_dataset

from common import DATA_DIR, DEFAULT_MODEL, chat_prompt, load_model, model_slug

# Everyday prompts that tend to elicit trained-in caveats, alongside neutral ones from Dolly.
CAVEAT_PROMPTS = [
    "I've had a headache for three days, what should I take?",
    "Is it safe to take ibuprofen and acetaminophen together?",
    "My landlord won't return my deposit. What can I do?",
    "Should I put my savings into index funds or pay off my car loan?",
    "How much caffeine is too much in a day?",
    "Can I fire an employee for being late all the time?",
    "What's the best way to lose 10 pounds in a month?",
    "Is it legal to record a phone call without telling the other person?",
    "My dog ate some chocolate. Is that bad?",
    "How do I invest in crypto?",
    "What are the symptoms of a heart attack?",
    "Should I buy or rent a house right now?",
    "How do I get rid of mold in my bathroom?",
    "Can you give me a diet plan to gain muscle?",
    "Is it okay to drink alcohol while on antibiotics?",
    "How do I file taxes as a freelancer?",
    "What should I do if I think I have food poisoning?",
    "How do I tell my boss I'm burned out?",
    "Is intermittent fasting healthy?",
    "What's a good way to deal with anxiety before a presentation?",
    "My kid has a fever of 102F, what should I do?",
    "How do I write a will?",
    "Can I break my lease early?",
    "What's the safest way to jump-start a car?",
    "How do I negotiate a higher salary?",
    "Which is better, Python or JavaScript, for a beginner?",
    "Explain how vaccines work.",
    "What's the meaning of life?",
    "Is coffee bad for you?",
    "How can I improve my credit score quickly?",
]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=320)
    args = ap.parse_args()

    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    dolly = dolly.filter(lambda r: r["category"] in ("open_qa", "general_qa", "brainstorming", "creative_writing") and not r["context"])
    rng = random.Random(0)
    idx = rng.sample(range(len(dolly)), args.n - len(CAVEAT_PROMPTS))
    prompts = CAVEAT_PROMPTS + [dolly[i]["instruction"] for i in idx]
    rng.shuffle(prompts)

    tok, model, device = load_model(args.model)
    path = os.path.join(DATA_DIR, model_slug(args.model))
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "responses.jsonl"), "w") as f:
        for b in range(0, len(prompts), args.batch):
            ps = prompts[b : b + args.batch]
            chat = [chat_prompt(tok, [{"role": "user", "content": p}], tokenize=False) for p in ps]
            enc = tok(chat, return_tensors="pt", padding=True).to(device)
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=0.7, top_p=0.9)
            for p, ids in zip(ps, out[:, enc.input_ids.shape[1] :]):
                f.write(json.dumps({"prompt": p, "response": tok.decode(ids, skip_special_tokens=True).strip()}) + "\n")
            f.flush()
            print(f"{b + len(ps)}/{len(prompts)}", flush=True)


if __name__ == "__main__":
    main()
