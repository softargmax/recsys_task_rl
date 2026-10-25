import numpy as np
from sklearn.metrics import roc_auc_score

from . import config as C


def unique_targets(targets):
    t = np.sort(targets, axis=1)
    dup = np.zeros_like(t, dtype=bool)
    dup[:, 1:] = t[:, 1:] == t[:, :-1]
    return np.where(dup | (t < C.N_SPECIAL), C.PAD, t)


def ranking(topk, targets, ks=(10, 20)):
    targets = unique_targets(targets)
    rel = ((topk[:, :, None] == targets[:, None, :]) & (targets[:, None, :] != C.PAD)).any(-1)
    n_rel = (targets != C.PAD).sum(1)
    discount = 1 / np.log2(np.arange(topk.shape[1]) + 2)
    out = {}
    empty = n_rel == 0
    for k in ks:
        ideal = np.cumsum(discount)[np.clip(np.minimum(n_rel, k) - 1, 0, None)]
        out[f"recall@{k}"] = np.where(empty, np.nan, rel[:, :k].sum(1) / np.maximum(n_rel, 1))
        out[f"ndcg@{k}"] = np.where(empty, np.nan, (rel[:, :k] * discount[:k]).sum(1) / ideal)
    return out


def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else float("nan")


def group_aucs(y, s, groups):
    order = np.argsort(groups, kind="stable")
    y, s, groups = y[order], s[order], groups[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(groups)) + 1, len(groups)]
    aucs, weights = [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        yy = y[lo:hi]
        if 0 < yy.mean() < 1:
            aucs.append(roc_auc_score(yy, s[lo:hi]))
            weights.append(hi - lo)
    return np.array(aucs), np.array(weights, dtype=float)


def gauc(y, s, groups):
    aucs, weights = group_aucs(y, s, groups)
    return float(aucs @ weights / weights.sum()) if len(aucs) else float("nan")


def gini(counts):
    c = np.sort(counts)
    n = len(c)
    return float((2 * np.arange(1, n + 1) - n - 1) @ c / (n * c.sum()))


def concentration(topk, n_catalog, rng, pairs=5000):
    counts = np.bincount(topk.ravel(), minlength=n_catalog + C.N_SPECIAL)[C.N_SPECIAL:]
    a, b = rng.integers(0, len(topk), size=(2, pairs))
    inter = (topk[a][:, :, None] == topk[b][:, None, :]).any(-1).sum(1)
    jaccard = inter / (2 * topk.shape[1] - inter)
    return {
        "coverage": float((counts > 0).mean()),
        "gini": gini(counts),
        "top100_share": float(np.sort(counts)[::-1][:100].sum() / counts.sum()),
        "personalization": float(1 - jaccard[a != b].mean()),
    }


def bootstrap_weighted(values, weights, n_boot=200, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(values), size=(n_boot, len(values)))
    means = (values[idx] * weights[idx]).sum(1) / weights[idx].sum(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values @ weights / weights.sum()), "lo": float(lo), "hi": float(hi)}


def bootstrap(values, users, n_boot=200, seed=0):
    keep = np.isfinite(values)
    values, users = values[keep], users[keep]
    uniq, inv = np.unique(users, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi)}
