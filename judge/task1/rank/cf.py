import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd

from . import config

RECENT = 10
BATCH = 20000


class ItemFactors:
    def __init__(self, item_ids, item_vecs, user_ids, user_vecs, cutoff_ts):
        self.item_ids = item_ids
        self.item_vecs = item_vecs
        self.item_unit = item_vecs / np.maximum(np.linalg.norm(item_vecs, axis=1, keepdims=True), 1e-9)
        self.user_ids = user_ids
        self.user_vecs = user_vecs
        self.cutoff_ts = cutoff_ts

    @staticmethod
    def lookup(ids, keys):
        keys = np.asarray(keys, dtype=np.int64)
        pos = np.clip(np.searchsorted(ids, keys), 0, len(ids) - 1)
        return np.where(ids[pos] == keys, pos, -1)

    def item_rows(self, items):
        return self.lookup(self.item_ids, items)

    def user_rows(self, users):
        return self.lookup(self.user_ids, users)


def fit_item_factors(listens, cutoff_ts, dim=64, min_users=3):
    pairs = (
        listens.filter(pl.col("timestamp") < cutoff_ts, pl.col("listen_plus"))
        .group_by("uid", "item_id")
        .agg(pl.len().alias("n"))
        .collect(engine="streaming")
    )
    items = pairs.group_by("item_id").agg(pl.len().alias("users")).filter(pl.col("users") >= min_users)
    pairs = pairs.join(items.select("item_id"), on="item_id", how="semi")
    user_ids = np.sort(pairs["uid"].unique().to_numpy()).astype(np.int64)
    item_ids = np.sort(pairs["item_id"].unique().to_numpy()).astype(np.int64)
    rows = np.searchsorted(user_ids, pairs["uid"].to_numpy())
    cols = np.searchsorted(item_ids, pairs["item_id"].to_numpy())
    matrix = csr_matrix((np.log1p(pairs["n"].to_numpy()).astype(np.float32), (rows, cols)), shape=(len(user_ids), len(item_ids)))
    u, s, vt = randomized_svd(matrix, n_components=dim, n_iter=5, random_state=config.SEED)
    print(f"item factors: {len(user_ids)} users, {len(item_ids)} items, nnz {matrix.nnz}, dim {dim}")
    return ItemFactors(item_ids, (vt.T * s).astype(np.float32), user_ids, u.astype(np.float32), cutoff_ts)


def cf_features(cf, uid, item, lp, objects, user_start):
    n_obj = len(objects)
    item_row = cf.item_rows(item)
    target = item_row[objects]
    user_row = cf.user_rows(uid[objects])
    known = target >= 0
    out = {
        "cf_known": known.astype(np.float32),
        "cf_user_score": np.full(n_obj, np.nan, dtype=np.float32),
        "cf_sim_prev": np.full(n_obj, np.nan, dtype=np.float32),
        "cf_sim_mean10": np.full(n_obj, np.nan, dtype=np.float32),
        "cf_sim_lp10": np.full(n_obj, np.nan, dtype=np.float32),
        "cf_sim_skip10": np.full(n_obj, np.nan, dtype=np.float32),
    }
    has_user = known & (user_row >= 0)
    out["cf_user_score"][has_user] = np.einsum("ij,ij->i", cf.user_vecs[user_row[has_user]], cf.item_vecs[target[has_user]])

    offsets = np.arange(1, RECENT + 1)
    for b in range(0, n_obj, BATCH):
        sl = slice(b, b + BATCH)
        o, t = objects[sl], target[sl]
        hist = o[:, None] - offsets[None, :]
        in_user = hist >= user_start[o - 1][:, None]
        hist = np.where(in_user, hist, 0)
        h_row = np.where(in_user, item_row[hist], -1)
        valid = (h_row >= 0) & (t >= 0)[:, None]
        sims = np.einsum("bkd,bd->bk", cf.item_unit[np.maximum(h_row, 0)], cf.item_unit[np.maximum(t, 0)])
        sims = np.where(valid, sims, 0.0)
        w_lp = np.where(valid, lp[hist], 0.0)
        w_skip = np.where(valid, 1.0 - lp[hist], 0.0)
        out["cf_sim_prev"][sl] = np.where(valid[:, 0], sims[:, 0], np.nan)
        out["cf_sim_mean10"][sl] = safe_ratio(sims.sum(1), valid.sum(1))
        out["cf_sim_lp10"][sl] = safe_ratio((sims * w_lp).sum(1), w_lp.sum(1))
        out["cf_sim_skip10"][sl] = safe_ratio((sims * w_skip).sum(1), w_skip.sum(1))
    return out


def safe_ratio(num, den):
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
