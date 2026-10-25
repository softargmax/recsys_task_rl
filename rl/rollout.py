import torch

from . import config as C
from .data import append_items, repeat
from .model import NEG


@torch.no_grad()
def sample_slates(policy, batch, data, n, k, greedy=False):
    b = repeat(batch, n)
    chosen = []
    for _ in range(k):
        logits = policy.next_logits(b).float()
        if chosen:
            logits = logits.scatter(1, torch.stack(chosen, 1), NEG)
        if greedy:
            items = logits.argmax(-1)
        else:
            items = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1)
        chosen.append(items)
        b = append_items(b, items, data)
    return torch.stack(chosen, 1).view(-1, n, k)


def slate_logprobs(policy, batch, slates, data):
    n, k = slates.shape[1], slates.shape[2]
    flat = slates.reshape(-1, k)
    b = repeat(batch, n)
    for j in range(k - 1):
        b = append_items(b, flat[:, j], data)
    logits = policy.logits(policy(b)[:, -k:]).float()
    earlier = torch.arange(k, device=flat.device)[None, :] < torch.arange(k, device=flat.device)[:, None]
    seen = torch.where(earlier[None], flat[:, None, :], torch.full_like(flat[:, None, :], C.PAD))
    return torch.log_softmax(logits.scatter(-1, seen, NEG), -1)


def history_for_reward(batch, slates, step, protocol, data, predicted=None):
    if protocol == "real":
        return batch
    n = slates.shape[1]
    b = repeat(batch, n)
    for j in range(step):
        b = append_items(b, slates[:, :, j].reshape(-1), data)
        if protocol == "listened":
            ratio = torch.full_like(b["ratio"][:, -1], 100)
        else:
            ratio = (predicted[:, :, j].reshape(-1) * 100).round().long()
        b["ratio"][:, -1] = ratio
        b["ratio_b"][:, -1] = 1 + torch.bucketize(ratio, torch.tensor(C.RATIO_EDGES, device=ratio.device), right=True)
    return b
