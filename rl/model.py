from dataclasses import asdict

import torch
from torch import nn
from transformers import GPT2Config, GPT2Model

from . import config as C

NEG = -1e4
N_MATCH = 4 * len(C.MATCH_TAUS) + 4


class EventEmbedding(nn.Module):
    def __init__(self, n_items, n_artists, d):
        super().__init__()
        self.item = nn.Embedding(n_items, d, padding_idx=C.PAD)
        self.artist = nn.Embedding(n_artists, d, padding_idx=C.PAD)
        self.ratio = nn.Embedding(C.N_RATIO, d)
        self.organic = nn.Embedding(2, d)
        self.dur = nn.Embedding(C.N_DUR, d)
        self.gap = nn.Embedding(C.N_GAP, d)
        for emb in (self.item, self.artist, self.ratio, self.organic, self.dur, self.gap):
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, b):
        return (self.item(b["item"]) + self.artist(b["artist"]) + self.ratio(b["ratio_b"])
                + self.organic(b["organic"]) + self.dur(b["dur_b"]) + self.gap(b["gap_b"]))


class Encoder(nn.Module):
    def __init__(self, n_items, n_artists, cfg):
        super().__init__()
        self.emb = EventEmbedding(n_items, n_artists, cfg.d)
        self.gpt = GPT2Model(GPT2Config(
            vocab_size=1, n_positions=cfg.ctx + C.MAX_SLATE, n_embd=cfg.d, n_layer=cfg.layers, n_head=cfg.heads,
            resid_pdrop=cfg.dropout, embd_pdrop=cfg.dropout, attn_pdrop=cfg.dropout, bos_token_id=None,
            eos_token_id=None, attn_implementation="sdpa",
        ))

    def forward(self, b):
        out = self.gpt(inputs_embeds=self.emb(b), attention_mask=b["mask"].long(), position_ids=b["pos"],
                       use_cache=False)
        return out.last_hidden_state


class Policy(nn.Module):
    def __init__(self, n_items, n_artists, cfg):
        super().__init__()
        self.args = {"n_items": n_items, "n_artists": n_artists, "cfg": asdict(cfg)}
        self.cfg = cfg
        self.encoder = Encoder(n_items, n_artists, cfg)
        self.bias = nn.Parameter(torch.zeros(n_items))
        blocked = torch.zeros(n_items, dtype=torch.bool)
        blocked[:C.N_SPECIAL] = True
        self.register_buffer("blocked", blocked, persistent=False)

    def logits(self, h):
        return (h @ self.encoder.emb.item.weight.T + self.bias).masked_fill(self.blocked, NEG)

    def forward(self, b):
        return self.encoder(b)

    def next_logits(self, b):
        return self.logits(self.encoder(b)[:, -1])


def match_features(hist, cand_item, cand_artist, t0, valid, chunk=16):
    if cand_item.shape[1] > chunk:
        return torch.cat([match_features(hist, cand_item[:, s:s + chunk], cand_artist[:, s:s + chunk],
                                         t0[:, s:s + chunk], valid[:, s:s + chunk], chunk)
                          for s in range(0, cand_item.shape[1], chunk)], 1)
    same_track = valid & (hist["item"][:, None, :] == cand_item[..., None]) & (cand_item[..., None] >= C.N_SPECIAL)
    same_artist = valid & (hist["artist"][:, None, :] == cand_artist[..., None]) & (cand_artist[..., None] >= C.N_SPECIAL)
    age = (t0[..., None] - hist["ts"][:, None, :]).clamp(min=0).float()
    listen_plus = (hist["ratio"][:, None, :] >= C.LISTEN_PLUS).float()
    counts = []
    for tau in C.MATCH_TAUS:
        w = torch.exp(-age / tau)
        for same in (same_track, same_artist):
            sw = same * w
            counts += [sw.sum(-1), (sw * listen_plus).sum(-1)]
    recency = []
    for same in (same_track, same_artist):
        has = same.any(-1)
        hours = torch.where(same, age, torch.full_like(age, float("inf"))).amin(-1) / 3600
        recency += [has.float(), torch.where(has, torch.log1p(hours), torch.zeros_like(hours))]
    return torch.cat([torch.log1p(torch.stack(counts, -1)), torch.stack(recency, -1)], -1)


class RewardModel(nn.Module):
    def __init__(self, n_items, n_artists, cfg, duration_only=False, match_window=None):
        super().__init__()
        self.args = {"n_items": n_items, "n_artists": n_artists, "cfg": asdict(cfg), "duration_only": duration_only,
                     "match_window": match_window}
        self.cfg = cfg
        self.duration_only = duration_only
        self.match_window = match_window or cfg.ctx
        self.far_len = 0 if duration_only else self.match_window - cfg.ctx
        self.encoder = Encoder(n_items, n_artists, cfg)
        self.cand_dur = nn.Embedding(C.N_DUR, cfg.d)
        self.cand_organic = nn.Embedding(2, cfg.d)
        self.match = nn.Linear(N_MATCH, cfg.d)
        self.head = nn.Sequential(nn.Linear(4 * cfg.d, cfg.d), nn.GELU(), nn.Linear(cfg.d, 2))

    def candidate(self, c):
        e = self.cand_dur(c["dur_b"]) + self.cand_organic(c["organic"])
        if not self.duration_only:
            e = e + self.encoder.emb.item(c["item"]) + self.encoder.emb.artist(c["artist"])
        return e

    def head_logits(self, state, cand, hist, t0, valid):
        e = self.candidate(cand)
        if self.duration_only:
            m = torch.zeros_like(e)
        else:
            m = self.match(match_features(hist, cand["item"], cand["artist"], t0, valid).to(e.dtype))
        return self.head(torch.cat([state, e, state * e, m], -1))

    def forward_train(self, inputs, nxt, far=None):
        hist = long_history(inputs, far)
        n, length = inputs["mask"].shape[1], hist["mask"].shape[1]
        i = torch.arange(n, device=hist["mask"].device)[:, None] + length - n
        j = torch.arange(length, device=hist["mask"].device)[None, :]
        valid = hist["mask"][:, None, :] & (j <= i) & (j > i - self.match_window)
        return self.head_logits(self.encoder(inputs), nxt, hist, inputs["ts"], valid)

    def score(self, b, cand, far=None):
        n = cand["item"].shape[1]
        state = self.encoder(b)[:, -1:].expand(-1, n, -1)
        hist = long_history(b, far)
        valid = hist["mask"][:, None, :].expand(-1, n, -1)
        return self.head_logits(state, cand, hist, b["ts"][:, -1:].expand(-1, n), valid)


def long_history(b, far):
    keys = ("item", "artist", "ts", "ratio", "mask")
    if far is None:
        return {k: b[k] for k in keys}
    return {k: torch.cat([far[k], b[k]], 1) for k in keys}


KINDS = {"Policy": Policy, "RewardModel": RewardModel}


def save(model, path, **meta):
    torch.save({"kind": type(model).__name__, "args": model.args, "state": model.state_dict(), "meta": meta}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    args = {**ck["args"], "cfg": C.ModelConfig(**ck["args"]["cfg"])}
    model = KINDS[ck["kind"]](**args)
    model.load_state_dict(ck["state"])
    return model.to(device)
