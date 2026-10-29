"""Label each sentence of generated responses as boilerplate / filler / substantive, using Claude.

Input:  data/<model>/responses.jsonl
Output: data/<model>/style_labels.jsonl  {"i", "prompt", "response", "sentences": [{"text", "boilerplate", "filler"}]}
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

import anthropic
from pydantic import BaseModel

from common import DATA_DIR, DEFAULT_MODEL, model_slug
from segment import split_sentences

INSTRUCTIONS = """You are labeling sentences from an AI assistant's reply, to train a detector that dims low-value text.

For each numbered sentence decide two independent flags:
- boilerplate: a trained-in caveat, disclaimer, safety warning, or pleasantry that is generic rather than prompted by the specifics of the question. Examples: "Consult a healthcare professional before making changes.", "As an AI, I don't have personal opinions.", "I hope this helps!", "It's important to note that everyone is different." A specific, actionable warning that matters for this exact question ("Chocolate's theobromine is toxic to dogs; call your vet if they ate dark chocolate") is NOT boilerplate.
- filler: text that adds no information: restating the question, generic throat-clearing ("There are many factors to consider."), or a closing summary that only repeats earlier points.

Most sentences are neither. Return one label per sentence, in order, with the same index."""


class SentenceLabel(BaseModel):
    index: int
    boilerplate: bool
    filler: bool


class Labels(BaseModel):
    labels: List[SentenceLabel]


LABEL_MODEL = os.environ.get("LABEL_MODEL", "claude-sonnet-5-5")


def label_one(client, prompt, sentences):
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    resp = client.messages.parse(
        model=LABEL_MODEL,
        max_tokens=8000,
        output_config={"effort": "low"},
        system=INSTRUCTIONS,
        messages=[{"role": "user", "content": f"User question:\n{prompt}\n\nAssistant reply, split into sentences:\n{numbered}"}],
        output_format=Labels,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        return None
    by_idx = {l.index: l for l in resp.parsed_output.labels}
    if set(by_idx) != set(range(len(sentences))):
        return None
    return [{"text": s, "boilerplate": by_idx[i].boilerplate, "filler": by_idx[i].filler} for i, s in enumerate(sentences)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    key_file = os.path.expanduser("~/anthropic_api_key.txt")
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY") or open(key_file).read().strip())
    path = os.path.join(DATA_DIR, model_slug(args.model))
    rows = [json.loads(l) for l in open(os.path.join(path, "responses.jsonl"))]

    def work(i):
        r = rows[i]
        sents = [s.text for s in split_sentences(r["response"])]
        if not sents:
            return None
        try:
            labels = label_one(client, r["prompt"], sents)
        except anthropic.APIError as e:
            print("error", i, e)
            return None
        return labels and {"i": i, "prompt": r["prompt"], "response": r["response"], "sentences": labels}

    n_ok = 0
    with ThreadPoolExecutor(args.workers) as ex, open(os.path.join(path, "style_labels.jsonl"), "w") as f:
        for res in ex.map(work, range(len(rows))):
            if res:
                n_ok += 1
                f.write(json.dumps(res) + "\n")
    print(f"labeled {n_ok}/{len(rows)}")


if __name__ == "__main__":
    main()
