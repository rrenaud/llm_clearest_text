"""Hidden-state features for character ranges of an assistant response."""
import torch

from common import chat_prompt, reset_position_state


@torch.no_grad()
def response_hidden(tok, model, device, messages, response, layers):
    """Teacher-force `response` after the chat prompt.

    Returns (hs[L, T, d] float32 on cpu for response tokens, offsets[T] of (start, end) chars in response).
    """
    prefix = chat_prompt(tok, messages, tokenize=False)
    enc = tok(prefix + response, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    offs = enc.pop("offset_mapping")[0].tolist()
    reset_position_state(model)
    out = model(input_ids=enc.input_ids.to(device), output_hidden_states=True, logits_to_keep=1)
    keep = [i for i, (a, b) in enumerate(offs) if a >= len(prefix)]
    hs = torch.stack([out.hidden_states[l][0, keep] for l in layers]).float().cpu()
    return hs, [(offs[i][0] - len(prefix), offs[i][1] - len(prefix)) for i in keep]


def token_range(offsets, start, end):
    """Indices of tokens overlapping chars [start, end)."""
    return [i for i, (a, b) in enumerate(offsets) if a < end and b > start]


def pool(hs, idx, how):
    """hs[..., T, d] -> [..., d] pooled over token indices."""
    if how == "first":
        return hs[..., idx[0], :]
    if how == "last":
        return hs[..., idx[-1], :]
    return hs[..., idx, :].mean(-2)
