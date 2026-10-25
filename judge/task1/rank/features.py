import gc
import json
import shutil
import time
from pathlib import Path

import polars as pl

from . import config
from .cf import fit_item_factors
from .entity_features import entity_features
from .user_features import TAUS, WINDOWS, user_part_features

D = config.DAY
ID_COLS = ["obj_id", "uid", "timestamp", "ref_ts", "item_id", "artist_id", "album_id", "days_before_end", "split", "label"]
GROUPS = {
    "user": ("u_",),
    "item": ("i_", "a_", "al_"),
    "pair_track": ("ui_",),
    "pair_artist_album": ("ua_", "ual_"),
    "context": ("c_",),
    "cross": ("x_",),
    "decay": ("d_",),
    "cf": ("cf_",),
}
WINDOW_BASES = [
    "u_cnt", "u_rec_cnt", "u_rec_lp_rate", "ui_cnt", "ui_org_cnt", "ui_lp_rate",
    "ua_cnt", "ua_org_cnt", "ua_lp_rate", "ual_cnt", "ual_lp_rate",
]


def dataset_end(events_path):
    return int(pl.scan_parquet(events_path).select(pl.col("timestamp").max()).collect().item())


def load_explicit(events_path, artist_of, max_ts=None):
    lf = pl.scan_parquet(events_path).filter(pl.col("event_type") != "listen")
    if max_ts is not None:
        lf = lf.filter(pl.col("timestamp") < max_ts)
    event = pl.col("event_type").cast(pl.Utf8)
    return (
        lf.select("uid", "item_id", "timestamp", "event_type")
        .with_columns(
            (event == "like").cast(pl.Int8).alias("is_like"),
            (event == "dislike").cast(pl.Int8).alias("is_dislike"),
            pl.when(event == "like").then(1).when(event == "unlike").then(-1).otherwise(0).cast(pl.Int8).alias("like_delta"),
            pl.when(event == "dislike").then(1).when(event == "undislike").then(-1).otherwise(0).cast(pl.Int8).alias("dislike_delta"),
        )
        .drop("event_type")
        .collect()
        .join(artist_of, on="item_id", how="left")
        .sort("uid", "timestamp", maintain_order=True)
    )


def add_cross(df):
    return df.with_columns(
        pl.when(pl.col("ui_cnt_all") > 0).then(0).when(pl.col("ua_cnt_all") > 0).then(1).otherwise(2).cast(pl.Float32).alias("x_novelty"),
        (pl.col("u_rec_lp_rate_30d") - pl.col("i_rec_lp_smooth_30d")).alias("x_user_minus_item"),
        (pl.col("ui_lp_rate_all") - pl.col("u_rec_lp_rate_all")).alias("x_pair_vs_user"),
        (pl.col("ua_lp_rate_all") - pl.col("u_rec_lp_rate_all")).alias("x_artist_vs_user"),
        (pl.col("i_len") / pl.col("u_mean_len_lp_30d")).alias("x_len_ratio"),
    )


def feature_columns(df):
    return [c for c in df.columns if c not in ID_COLS and not c.startswith("canary_")]


def window_features(cols):
    names = {f"{base}_{w}" for base in WINDOW_BASES for w in WINDOWS}
    return [c for c in cols if c in names]


def decay_features_for(cols, taus):
    suffixes = tuple(f"_{t}" for t in taus)
    return [c for c in cols if c.startswith("d_") and c.endswith(suffixes)]


def history_prior(listens, t_end):
    return float(
        listens.filter(pl.col("is_organic") == 0, pl.col("timestamp") <= t_end - config.OBJECT_DAYS * D)
        .select(pl.col("listen_plus").mean())
        .collect()
        .item()
    )


def cf_cutoff(t_end):
    return t_end - config.OBJECT_DAYS * D


def scan_listens_dir(listens_dir, max_ts=None):
    lf = pl.scan_parquet(sorted(Path(listens_dir).glob("part_*.parquet")))
    return lf.filter(pl.col("timestamp") < max_ts) if max_ts is not None else lf


def build_features(listens_dir, events_path, mappings_dir, out_dir, t_end=None, max_ts=None, max_parts=None,
                   prior=None, cf="fit", taus=tuple(TAUS), canary=False):
    from .prepare import load_mappings

    started = time.time()
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "parts").mkdir(parents=True)
    parts = sorted(Path(listens_dir).glob("part_*.parquet"))[:max_parts]

    listens = scan_listens_dir(listens_dir, max_ts)
    t_end = t_end if t_end is not None else dataset_end(events_path)
    prior = prior if prior is not None else history_prior(listens, t_end)
    if cf == "fit":
        cf = fit_item_factors(listens, cf_cutoff(t_end))
    artist_of, _ = load_mappings(events_path, mappings_dir)
    explicit = load_explicit(events_path, artist_of, max_ts=max_ts)
    print(f"t_end {t_end}, prior {prior:.4f}, explicit events {explicit.height}, parts {len(parts)}, taus {list(taus)}")

    offset = 0
    for path in parts:
        lf = pl.scan_parquet(path)
        chunk = (lf.filter(pl.col("timestamp") < max_ts) if max_ts is not None else lf).collect()
        if chunk.height == 0:
            continue
        part_explicit = explicit.filter(pl.col("uid").is_between(chunk["uid"].min(), chunk["uid"].max()))
        feats = user_part_features(chunk, part_explicit, t_end, cf=cf, taus=taus, canary=canary)
        feats = feats.with_columns((pl.int_range(feats.height, dtype=pl.UInt32) + offset).alias("obj_id"))
        offset += feats.height
        feats.write_parquet(out_dir / "parts" / path.name)
        print(f"{path.name}: rows {chunk.height}, objects {feats.height}, {time.time() - started:.0f}s")
        del chunk, feats, part_explicit
        gc.collect()

    user_feats = pl.scan_parquet(out_dir / "parts" / "*.parquet")
    objects = user_feats.select("obj_id", "item_id", "artist_id", "album_id", "ref_ts").collect().sort("obj_id")
    ent = entity_features(listens, explicit, objects, prior).with_columns(objects["obj_id"])
    print(f"entity features: {ent.width - 1} columns, {time.time() - started:.0f}s")
    del objects, explicit
    gc.collect()

    for split in config.SPLITS:
        df = add_cross(user_feats.filter(pl.col("split") == split).collect().join(ent, on="obj_id", how="left"))
        df.write_parquet(out_dir / f"{split}.parquet")
        rate = df["label"].mean() if df.height else float("nan")
        print(f"{split}: {df.height} objects, positive rate {rate:.4f}")
        cols = feature_columns(df)
        del df
        gc.collect()

    meta = {
        "t_end": t_end,
        "prior": prior,
        "max_ts": max_ts,
        "cf_cutoff_ts": cf.cutoff_ts if cf is not None else None,
        "taus": list(taus),
        "features": cols,
        "groups": {g: [c for c in cols if c.startswith(prefixes)] for g, prefixes in GROUPS.items()},
        "window_features": window_features(cols),
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    shutil.rmtree(out_dir / "parts")
    print(f"done: {len(cols)} features -> {out_dir}, {meta['seconds']}s")
    return meta
