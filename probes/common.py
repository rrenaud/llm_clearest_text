"""Shared helpers for probe data generation and training."""
import os
import re
import string

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

DEFAULT_MODEL = os.environ.get("PROBE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def model_slug(name):
    return name.split("/")[-1]


def load_model(name=DEFAULT_MODEL):
    device = "cuda" if torch.cuda.is_available() and os.environ.get("PROBE_DEVICE", "auto") != "cpu" else "cpu"
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    # Vision-language checkpoints (e.g. Qwen3.6) keep the LM under model.language_model; load them whole
    # and feed text only.
    archs = AutoConfig.from_pretrained(name).architectures or []
    cls = AutoModelForImageTextToText if any(a.endswith("ForConditionalGeneration") for a in archs) else AutoModelForCausalLM
    model = cls.from_pretrained(name, dtype=dtype).to(device).eval()
    return tok, model, device


def reset_position_state(model):
    """Qwen-VL-style models cache rope offsets from the last generate() call; a later forward with a
    different batch size reuses them and breaks. Clear them before standalone forwards."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None


def num_layers(model):
    return model.config.get_text_config().num_hidden_layers


def chat_prompt(tok, messages, **kw):
    """Chat template with any reasoning mode off: probes read the direct answer, not a hidden scratchpad."""
    return tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, **kw)


def normalize(s):
    """TriviaQA-style normalization."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_correct(prediction, aliases):
    p = normalize(prediction)
    if not p:
        return False
    for a in aliases:
        a = normalize(a)
        if a and (a == p or re.search(r"\b" + re.escape(a) + r"\b", p)):
            return True
    return False
