import gc

import numpy as np
import polars as pl

from . import config
from .windows import GroupIndex, WindowQuery
from .user_features import WINDOWS

D = config.DAY


def smooth(num, den, prior):
    return ((num + config.SMOOTH_M * prior) / (den + config.SMOOTH_M)).astype(np.float32)


def entity_window_sums(group, ts, values, q_group, day_start, windows, valid=None):
    idx = GroupIndex(group, ts)
    wq = WindowQuery(idx, q_group, day_start, end_ts=day_start, valid=valid)
    out = {}
    for name, v in values.items():
        csum = idx.cumsum(v)
        for w in windows[name]:
            out[(name, w)] = wq.sum(csum, WINDOWS[w])
        del csum
    first = wq.first_ts()
    del idx, wq
    gc.collect()
    return out, first


def entity_features(listens, explicit, objects, prior):
    day_start = (objects["ref_ts"].to_numpy().astype(np.int64) // D) * D
    data = listens.select("item_id", "artist_id", "album_id", "timestamp", "is_organic", "listen_plus").collect()
    ts = data["timestamp"].to_numpy()
    rec = (data["is_organic"].to_numpy() == 0).astype(np.uint8)
    lp_rec = (data["listen_plus"].to_numpy() & (rec == 1)).astype(np.uint8)
    one = np.ones(len(ts), dtype=np.uint8)
    f = {}

    sums, first = entity_window_sums(
        data["item_id"].to_numpy(), ts, {"one": one, "rec": rec, "lp": lp_rec},
        objects["item_id"].to_numpy(), day_start,
        {"one": ["1d", "7d", "30d", "all"], "rec": ["7d", "30d", "all"], "lp": ["7d", "30d", "all"]},
    )
    for w in ["1d", "7d", "30d", "all"]:
        f[f"i_cnt_{w}"] = sums[("one", w)]
    for w in ["7d", "30d", "all"]:
        f[f"i_rec_cnt_{w}"] = sums[("rec", w)]
        f[f"i_rec_lp_smooth_{w}"] = smooth(sums[("lp", w)], sums[("rec", w)], prior)
    f["i_trend_7d_30d"] = (f["i_cnt_7d"] / (f["i_cnt_30d"] + 1)).astype(np.float32)
    f["i_age_days"] = np.nan_to_num((day_start - first) / D, nan=0.0).astype(np.float32)

    sums, _ = entity_window_sums(
        explicit["item_id"].to_numpy(), explicit["timestamp"].to_numpy(),
        {"like": explicit["is_like"].to_numpy(), "dislike": explicit["is_dislike"].to_numpy()},
        objects["item_id"].to_numpy(), day_start, {"like": ["all"], "dislike": ["all"]},
    )
    f["i_like_rate_all"] = (sums[("like", "all")] / (f["i_cnt_all"] + 10)).astype(np.float32)
    f["i_dislike_rate_all"] = (sums[("dislike", "all")] / (f["i_cnt_all"] + 10)).astype(np.float32)

    for prefix, col, cnt_windows, lp_windows in [
        ("a", "artist_id", ["7d", "30d"], ["30d", "all"]),
        ("al", "album_id", ["30d"], ["all"]),
    ]:
        keep = data[col].is_not_null().to_numpy()
        valid = objects[col].is_not_null().to_numpy()
        sums, _ = entity_window_sums(
            data[col].to_numpy()[keep], ts[keep], {"one": one[keep], "rec": rec[keep], "lp": lp_rec[keep]},
            objects[col].fill_null(0).to_numpy(), day_start,
            {"one": cnt_windows, "rec": lp_windows, "lp": lp_windows}, valid=valid,
        )
        for w in cnt_windows:
            f[f"{prefix}_cnt_{w}"] = np.where(valid, sums[("one", w)], np.nan).astype(np.float32)
        for w in lp_windows:
            f[f"{prefix}_rec_lp_smooth_{w}"] = np.where(valid, smooth(sums[("lp", w)], sums[("rec", w)], prior), np.nan).astype(np.float32)
        del keep, sums
        gc.collect()
    del data, ts, rec, lp_rec, one
    gc.collect()
    return pl.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
