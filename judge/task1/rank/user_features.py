import numpy as np
import polars as pl

from . import config
from .cf import cf_features
from .windows import ALL, GroupIndex, WindowQuery

H = config.HOUR
D = config.DAY
WINDOWS = {"1h": H, "1d": D, "7d": 7 * D, "30d": 30 * D, "all": ALL}
TAUS = {"1h": H, "6h": 6 * H, "1d": D, "3d": 3 * D, "7d": 7 * D, "30d": 30 * D, "90d": 90 * D}
LISTEN_COLS = [
    "uid", "timestamp", "item_id", "is_organic", "played_ratio_pct", "track_length_seconds",
    "artist_id", "album_id", "listen_plus", "track_seen_n", "artist_seen_n",
]
DECAY_SPECS = {
    "u": {"sums": {"cnt": "one", "rec_cnt": "rec", "rec_lp": "lp_rec"}, "rates": {"rec_lp_rate": ("rec_lp", "rec_cnt")}, "hidden": ["rec_lp"]},
    "ui": {"sums": {"cnt": "one", "org_cnt": "org", "lp": "lp"}, "rates": {"lp_rate": ("lp", "cnt")}, "hidden": ["lp"]},
    "ua": {"sums": {"cnt": "one", "org_cnt": "org", "lp": "lp"}, "rates": {"lp_rate": ("lp", "cnt")}, "hidden": ["lp"]},
    "ual": {"sums": {"cnt": "one", "lp": "lp"}, "rates": {"lp_rate": ("lp", "cnt")}, "hidden": ["lp"]},
}
DECAY_MIN_MASS = 0.05


def rate(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan).astype(np.float32)


def dense_pairs(*code_arrays):
    codes = np.concatenate(code_arrays)
    _, inverse = np.unique(codes, return_inverse=True)
    out, pos = [], 0
    for arr in code_arrays:
        out.append(inverse[pos:pos + len(arr)])
        pos += len(arr)
    return out


def pair_code(uid, other):
    return (uid.astype(np.int64) << 32) | other.astype(np.int64)


class History:
    def __init__(self, chunk, t_end):
        chunk = chunk.select(LISTEN_COLS).sort("uid", "timestamp", maintain_order=True)
        self.n = chunk.height
        self.row = np.arange(self.n, dtype=np.int64)
        self.uid = chunk["uid"].to_numpy().astype(np.int64)
        self.ts = chunk["timestamp"].to_numpy().astype(np.int64)
        self.item = chunk["item_id"].to_numpy().astype(np.int64)
        self.has_artist = chunk["artist_id"].is_not_null().to_numpy()
        self.has_album = chunk["album_id"].is_not_null().to_numpy()
        self.artist = chunk["artist_id"].fill_null(0).to_numpy().astype(np.int64)
        self.album = chunk["album_id"].fill_null(0).to_numpy().astype(np.int64)
        self.org = chunk["is_organic"].to_numpy().astype(np.float64)
        self.rec = 1.0 - self.org
        self.ratio = chunk["played_ratio_pct"].to_numpy().astype(np.float64)
        self.length = chunk["track_length_seconds"].fill_null(0).to_numpy().astype(np.float64)
        self.lp = chunk["listen_plus"].to_numpy().astype(np.float64)
        self.new_track = (chunk["track_seen_n"].to_numpy() == 0).astype(np.float64)
        self.new_artist = (chunk["artist_seen_n"].fill_null(1).to_numpy() == 0).astype(np.float64)

        self.has_prev = np.r_[False, self.uid[1:] == self.uid[:-1]]
        self.prev_ts = np.r_[0, self.ts[:-1]]
        self.user_start = np.maximum.accumulate(np.where(~self.has_prev, self.row, 0))
        self.days_before_end = (t_end - self.ts) // D
        is_obj = (self.org == 0) & (self.days_before_end < config.OBJECT_DAYS) & self.has_prev
        self.objects = np.flatnonzero(is_obj)
        self.ref = self.prev_ts[self.objects]

    def values(self):
        return {
            "one": np.ones(self.n), "rec": self.rec, "org": self.org, "lp": self.lp,
            "lp_rec": self.lp * self.rec, "lp_org": self.lp * self.org,
            "new_track": self.new_track, "new_artist": self.new_artist,
            "lp_new_artist": self.lp * self.new_artist, "len_lp": self.length * self.lp,
        }


class Relation:
    def __init__(self, history, rows, group_rows, group_objects, valid=None):
        h = history
        self.rows = rows
        self.index = GroupIndex(group_rows, h.ts[rows], row=h.row[rows])
        self.query = WindowQuery(self.index, group_objects, h.ref, end_row=h.objects, valid=valid)

    def cumsum(self, values):
        return self.index.cumsum(values[self.rows])

    def window_sum(self, values, window):
        return self.query.masked(self.query.sum(self.cumsum(values), WINDOWS[window])).astype(np.float32)

    def decayed_sums(self, values, tau):
        prefixes = self.index.decay_prefixes({k: v[self.rows] for k, v in values.items()}, tau)
        return {k: self.query.decayed(p) for k, p in prefixes.items()}

    def recency_hours(self):
        return self.query.masked((self.query.ref_ts - self.query.last_ts()) / H).astype(np.float32)


def build_relations(h):
    o = h.objects
    everything = h.row
    relations = {"u": Relation(h, everything, h.uid, h.uid[o])}
    track_rows, track_obj = dense_pairs(pair_code(h.uid, h.item), pair_code(h.uid[o], h.item[o]))
    relations["ui"] = Relation(h, everything, track_rows, track_obj)
    for name, ent, has in [("ua", h.artist, h.has_artist), ("ual", h.album, h.has_album)]:
        rows = np.flatnonzero(has)
        g_rows, g_obj = dense_pairs(pair_code(h.uid[rows], ent[rows]), pair_code(h.uid[o], ent[o]))
        relations[name] = Relation(h, rows, g_rows, g_obj, valid=has[o])
    return relations


def user_window_features(h, rel, v):
    u = rel["u"]
    f = {}
    for w in ["1d", "7d", "30d"]:
        f[f"u_cnt_{w}"] = u.window_sum(v["one"], w)
    for w in ["1h", "1d", "7d", "30d", "all"]:
        f[f"u_rec_cnt_{w}"] = u.window_sum(v["rec"], w)
        f[f"u_rec_lp_rate_{w}"] = rate(u.window_sum(v["lp_rec"], w), f[f"u_rec_cnt_{w}"])
    n30 = f["u_cnt_30d"]
    org30 = u.window_sum(v["org"], "30d")
    new_artist30 = u.window_sum(v["new_artist"], "30d")
    f["u_org_share_30d"] = rate(org30, n30)
    f["u_org_lp_rate_30d"] = rate(u.window_sum(v["lp_org"], "30d"), org30)
    f["u_new_track_share_30d"] = rate(u.window_sum(v["new_track"], "30d"), n30)
    f["u_new_artist_share_30d"] = rate(new_artist30, n30)
    f["u_new_artist_lp_rate_30d"] = rate(u.window_sum(v["lp_new_artist"], "30d"), new_artist30)
    f["u_mean_len_lp_30d"] = rate(u.window_sum(v["len_lp"], "30d"), u.window_sum(v["lp"], "30d"))
    f["u_tenure_days"] = ((h.ref - u.query.first_ts()) / D).astype(np.float32)
    return f


def pair_window_features(h, rel, v):
    f = {}
    ui = rel["ui"]
    for w in ["1h", "1d", "7d", "30d", "all"]:
        f[f"ui_cnt_{w}"] = ui.window_sum(v["one"], w)
    for w in ["30d", "all"]:
        f[f"ui_org_cnt_{w}"] = ui.window_sum(v["org"], w)
        f[f"ui_lp_rate_{w}"] = rate(ui.window_sum(v["lp"], w), f[f"ui_cnt_{w}"])
    f["ui_last_ratio"] = ui.query.last(ui.index.sorted_values(h.ratio)).astype(np.float32)
    f["ui_recency_h"] = ui.recency_hours()

    ua = rel["ua"]
    for w in ["1d", "7d", "30d", "all"]:
        f[f"ua_cnt_{w}"] = ua.window_sum(v["one"], w)
    for w in ["30d", "all"]:
        f[f"ua_lp_rate_{w}"] = rate(ua.window_sum(v["lp"], w), f[f"ua_cnt_{w}"])
        f[f"ua_org_cnt_{w}"] = ua.window_sum(v["org"], w)
    f["ua_share_30d"] = rate(f["ua_cnt_30d"], rel["u"].window_sum(v["one"], "30d"))
    f["ua_recency_h"] = ua.recency_hours()

    ual = rel["ual"]
    for w in ["30d", "all"]:
        f[f"ual_cnt_{w}"] = ual.window_sum(v["one"], w)
    f["ual_lp_rate_all"] = rate(ual.window_sum(v["lp"], "all"), f["ual_cnt_all"])
    f["ual_recency_h"] = ual.recency_hours()
    return f


def decay_features(rel, v, taus):
    f = {}
    for name, spec in DECAY_SPECS.items():
        for tau_name in taus:
            sums = rel[name].decayed_sums({key: v[source] for key, source in spec["sums"].items()}, TAUS[tau_name])
            for key, value in sums.items():
                if key not in spec["hidden"]:
                    f[f"d_{name}_{key}_{tau_name}"] = value
            for key, (num, den) in spec["rates"].items():
                f[f"d_{name}_{key}_{tau_name}"] = np.where(sums[den] >= DECAY_MIN_MASS, sums[num] / np.maximum(sums[den], 1e-12), np.nan).astype(np.float32)
    return f


def explicit_features(h, explicit):
    o, q = h.objects, h.ref
    names = ["u_like_cnt_30d", "u_dislike_cnt_30d", "ui_like_state", "ui_dislike_state", "ua_liked_tracks", "ua_disliked_tracks"]
    if explicit.height == 0:
        return {c: np.zeros(len(o), dtype=np.float32) for c in names}
    e_uid = explicit["uid"].to_numpy().astype(np.int64)
    e_ts = explicit["timestamp"].to_numpy().astype(np.int64)
    e_item = explicit["item_id"].to_numpy().astype(np.int64)
    e_has_artist = explicit["artist_id"].is_not_null().to_numpy()
    e_artist = explicit["artist_id"].fill_null(0).to_numpy().astype(np.int64)
    e = {c: explicit[c].to_numpy().astype(np.float64) for c in ["is_like", "is_dislike", "like_delta", "dislike_delta"]}
    f = {}

    idx = GroupIndex(e_uid, e_ts)
    wq = WindowQuery(idx, h.uid[o], q, end_ts=q)
    f["u_like_cnt_30d"] = wq.sum(idx.cumsum(e["is_like"]), WINDOWS["30d"])
    f["u_dislike_cnt_30d"] = wq.sum(idx.cumsum(e["is_dislike"]), WINDOWS["30d"])

    g_e, g_o = dense_pairs(pair_code(e_uid, e_item), pair_code(h.uid[o], h.item[o]))
    idx = GroupIndex(g_e, e_ts)
    wq = WindowQuery(idx, g_o, q, end_ts=q)
    f["ui_like_state"] = np.clip(wq.sum(idx.cumsum(e["like_delta"]), ALL), 0, 1)
    f["ui_dislike_state"] = np.clip(wq.sum(idx.cumsum(e["dislike_delta"]), ALL), 0, 1)

    er = np.flatnonzero(e_has_artist)
    g_e, g_o = dense_pairs(pair_code(e_uid[er], e_artist[er]), pair_code(h.uid[o], h.artist[o]))
    idx = GroupIndex(g_e, e_ts[er])
    wq = WindowQuery(idx, g_o, q, end_ts=q, valid=h.has_artist[o])
    f["ua_liked_tracks"] = wq.masked(np.maximum(wq.sum(idx.cumsum(e["like_delta"][er]), ALL), 0)).astype(np.float32)
    f["ua_disliked_tracks"] = wq.masked(np.maximum(wq.sum(idx.cumsum(e["dislike_delta"][er]), ALL), 0)).astype(np.float32)
    return f


def context_features(h):
    o, q = h.objects, h.ref
    p = o - 1
    played_s = h.length * np.minimum(h.ratio, 100) / 100
    idle = h.ts - h.prev_ts - played_s
    new_session = ~h.has_prev | (idle > config.SESSION_IDLE)
    session_start = np.maximum.accumulate(np.where(new_session, h.row, 0))
    lp_cs = np.r_[0.0, np.cumsum(h.lp)]
    pos_prev = p - session_start[p]
    f = {
        "c_pos_in_session": (pos_prev + 1).astype(np.float32),
        "c_session_age_min": ((q - h.ts[session_start[p]]) / 60).astype(np.float32),
        "c_session_lp_rate": ((lp_cs[o] - lp_cs[session_start[p]]) / (pos_prev + 1)).astype(np.float32),
    }
    for k in [1, 3, 10]:
        lo = np.maximum(o - k, h.user_start[p])
        f[f"c_prev{k}_lp_rate"] = ((lp_cs[o] - lp_cs[lo]) / (o - lo)).astype(np.float32)
    f["c_prev1_ratio"] = h.ratio[p].astype(np.float32)
    f["c_prev1_organic"] = h.org[p].astype(np.float32)
    f["c_prev_same_artist"] = ((h.artist[p] == h.artist[o]) & h.has_artist[o]).astype(np.float32)
    f["c_hour"] = ((q % D) // H).astype(np.float32)
    f["c_weekday"] = ((q // D) % 7).astype(np.float32)
    f["i_len"] = h.length[o].astype(np.float32)
    return f


def canary_features(h):
    o = h.objects
    next_ratio = np.r_[h.ratio[1:], np.nan]
    same_next = np.r_[h.uid[1:] == h.uid[:-1], False]
    return {
        "canary_next_ratio": np.where(same_next, next_ratio, np.nan)[o].astype(np.float32),
        "canary_gap_to_prev": (h.ts[o] - h.ref).astype(np.float32),
    }


def object_ids(h):
    o = h.objects
    ids = pl.DataFrame({
        "uid": h.uid[o].astype(np.uint32),
        "timestamp": h.ts[o].astype(np.uint32),
        "ref_ts": h.ref.astype(np.uint32),
        "item_id": h.item[o].astype(np.uint32),
        "artist_id": pl.Series(h.artist[o].astype(np.uint32)).set(pl.Series(~h.has_artist[o]), None),
        "album_id": pl.Series(h.album[o].astype(np.uint32)).set(pl.Series(~h.has_album[o]), None),
        "days_before_end": h.days_before_end[o].astype(np.int16),
        "label": h.lp[o].astype(np.int8),
    })
    split = (
        pl.when(pl.col("days_before_end") < config.TEST_DAYS).then(pl.lit("test"))
        .when(pl.col("days_before_end") < config.TEST_DAYS + config.VAL_DAYS).then(pl.lit("val"))
        .otherwise(pl.lit("train"))
    )
    return ids.with_columns(split.alias("split"))


def user_part_features(chunk, explicit, t_end, cf=None, taus=tuple(TAUS), canary=False):
    h = History(chunk, t_end)
    v = h.values()
    rel = build_relations(h)
    f = {}
    f.update(user_window_features(h, rel, v))
    f.update(pair_window_features(h, rel, v))
    f.update(explicit_features(h, explicit))
    f.update(context_features(h))
    if taus:
        f.update(decay_features(rel, v, taus))
    if cf is not None:
        f.update(cf_features(cf, h.uid, h.item, h.lp, h.objects, h.user_start))
    if canary:
        f.update(canary_features(h))
    return object_ids(h).hstack(pl.DataFrame({k: np.asarray(x, dtype=np.float32) for k, x in f.items()}))
