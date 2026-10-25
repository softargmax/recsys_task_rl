import json
from pathlib import Path

import numpy as np
import torch

from . import config as C

SESSION_GAP_B = 2 + C.GAP_EDGES.index(C.SESSION_IDLE)
FAR_FIELDS = ("item", "artist", "ts", "ratio")


class Data:
    def __init__(self, root):
        root = Path(root)
        self.ev = {f: np.load(root / f"{f}.npy") for f in C.EVENT_FIELDS}
        self.offsets = np.load(root / "offsets.npy")
        with np.load(root / "catalog.npz") as cat:
            self.catalog = {k: cat[k] for k in cat.files}
        with np.load(root / "queries.npz") as q:
            self.queries = {k: q[k] for k in q.files}
        self.stats = json.loads((root / "stats.json").read_text())
        self.n_items = len(self.catalog["artist"])
        self.n_artists = self.stats["n_artists"]
        self.n_users = len(self.offsets) - 1
        days = (self.stats["t_end"] - self.ev["ts"].astype(np.int64)) // C.DAY
        is_train = days >= C.TEST_DAYS + C.VAL_DAYS
        self.train_end = self.offsets[:-1] + np.add.reduceat(is_train, self.offsets[:-1])
        new_session = (self.ev["gap_b"] == 0) | (self.ev["gap_b"] >= SESSION_GAP_B)
        self.session = np.cumsum(new_session)
        self._catalog_cache = {}

    def user_of(self, pos):
        return np.searchsorted(self.offsets, pos, side="right") - 1

    def gather(self, ends, length, fields, skip=0):
        ends = np.asarray(ends, dtype=np.int64)
        starts = self.offsets[self.user_of(ends - 1)]
        idx = ends[:, None] - skip - length + np.arange(length)
        mask = idx >= starts[:, None]
        idx = np.where(mask, idx, 0)
        batch = {f: torch.from_numpy(np.where(mask, self.ev[f][idx], 0).astype(np.int64)) for f in fields}
        batch["mask"] = torch.from_numpy(mask)
        return batch

    def window(self, ends, length):
        batch = self.gather(ends, length, C.EVENT_FIELDS)
        batch["pos"] = (batch["mask"].long().cumsum(1) - 1).clamp(min=0)
        return batch

    def far(self, ends, near, length):
        return self.gather(ends, length, FAR_FIELDS, skip=near)

    def targets(self, pos):
        pos = np.asarray(pos, dtype=np.int64)
        return {f: torch.from_numpy(self.ev[f][pos].astype(np.int64)) for f in C.EVENT_FIELDS}

    def future(self, pos, horizon=10, within_train=False):
        pos = np.asarray(pos, dtype=np.int64)
        idx = np.minimum(pos[:, None] + np.arange(horizon), len(self.session) - 1)
        same = self.session[idx] == self.session[pos][:, None]
        if within_train:
            same &= idx < self.train_end[self.user_of(pos)][:, None]
        return np.where(same, self.ev["item"][idx], C.PAD), np.where(same, self.ev["ratio"][idx], 0)

    def future_items(self, pos, horizon=10):
        return self.future(pos, horizon)[0]

    def catalog_on(self, device):
        if device not in self._catalog_cache:
            self._catalog_cache[device] = {
                k: torch.from_numpy(self.catalog[k].astype(np.int64)).to(device)
                for k in ("artist", "dur", "dur_b", "train_count")
            }
        return self._catalog_cache[device]

    def item_fields(self, items):
        cat = self.catalog_on(items.device)
        return {"item": items, "artist": cat["artist"][items], "dur_b": cat["dur_b"][items],
                "organic": torch.zeros_like(items)}

    def train_windows(self, length, rng, users=None):
        a, b = self.offsets[:-1], self.train_end
        top = b - rng.integers(0, length, size=self.n_users)
        count = np.maximum(0, -(-(top - a - 1) // length))
        if users is not None:
            count = np.where(np.isin(np.arange(self.n_users), users), count, 0)
        k = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        ends = np.repeat(top, count) - k * length
        return ends[rng.permutation(len(ends))]


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def shift(batch):
    inputs = {k: v[:, :-1] for k, v in batch.items()}
    nxt = {k: v[:, 1:] for k, v in batch.items()}
    nxt["mask"] = batch["mask"][:, 1:] & batch["mask"][:, :-1]
    return inputs, nxt


def drop_feedback(batch, p):
    keep = torch.rand(batch["ratio_b"].shape, device=batch["ratio_b"].device) >= p
    return {**batch, "ratio_b": batch["ratio_b"] * keep}


def append_items(batch, items, data):
    zeros = torch.zeros_like(items)
    step = {
        **data.item_fields(items),
        "ratio_b": zeros, "ratio": zeros,
        "gap_b": torch.full_like(items, C.GAP_GENERATED), "cont": torch.full_like(items, C.CONT_UNKNOWN),
        "ts": batch["ts"][:, -1], "mask": torch.ones_like(items, dtype=torch.bool), "pos": batch["pos"][:, -1] + 1,
    }
    return {k: torch.cat([batch[k], step[k][:, None]], 1) for k in batch}


def repeat(batch, n):
    return {k: v.repeat_interleave(n, 0) for k, v in batch.items()}
