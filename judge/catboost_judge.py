import os

os.environ.setdefault("_RJEM_MALLOC_CONF", "background_thread:true,dirty_decay_ms:500,muzzy_decay_ms:500")
os.environ.setdefault("POLARS_MAX_THREADS", "4")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
TASK1 = ROOT / "judge" / "task1"
sys.path.insert(0, str(TASK1))
sys.path.insert(0, str(ROOT))

from rank.cf import cf_features, fit_item_factors
from rank.entity_features import entity_features
from rank.features import add_cross, cf_cutoff, history_prior, load_explicit, scan_listens_dir
from rank.prepare import load_mappings, prepare_listens
from rank.user_features import (LISTEN_COLS, TAUS, History, build_relations, context_features, decay_features,
                                explicit_features, pair_window_features, user_window_features)

from rl.data import Data
from rl.metrics import bootstrap



def last_row_features(chunk, explicit, t_end, cf, real_uid):
    h = History(chunk, t_end)
    h.objects = np.flatnonzero(np.r_[h.uid[1:] != h.uid[:-1], True])
    h.ref = h.prev_ts[h.objects]
    v = h.values()
    rel = build_relations(h)
    f = {**user_window_features(h, rel, v), **pair_window_features(h, rel, v), **explicit_features(h, explicit),
         **context_features(h), **decay_features(rel, v, TAUS),
         **cf_features(cf, real_uid[h.uid], h.item, h.lp, h.objects, h.user_start)}
    o = h.objects
    ids = pl.DataFrame({
        "syn": h.uid[o].astype(np.int64),
        "item_id": h.item[o].astype(np.uint32),
        "artist_id": pl.Series(h.artist[o].astype(np.uint32)).set(pl.Series(~h.has_artist[o]), None),
        "album_id": pl.Series(h.album[o].astype(np.uint32)).set(pl.Series(~h.has_album[o]), None),
        "ref_ts": h.ref.astype(np.uint32),
    })
    return ids.hstack(pl.DataFrame({k: np.asarray(x, dtype=np.float32) for k, x in f.items()}))


def synthetic_chunk(history, queries, candidates, items):
    hist = (
        history.join(queries.select("q", "uid", "k"), on="uid")
        .filter(pl.col("row") < pl.col("k"))
        .join(candidates.select("q", "syn"), on="q")
        .with_columns(pl.col("syn").alias("uid"))
        .select(LISTEN_COLS)
    )
    rows = (
        candidates.join(queries.select("q", "ts"), on="q")
        .join(items, on="item_id", how="left")
        .select(
            pl.col("syn").alias("uid"), pl.col("ts").alias("timestamp"), "item_id",
            pl.lit(0, pl.UInt8).alias("is_organic"), pl.lit(0, pl.UInt16).alias("played_ratio_pct"),
            pl.col("length").alias("track_length_seconds"), "artist_id", "album_id",
            pl.lit(False).alias("listen_plus"), pl.lit(0, pl.UInt32).alias("track_seen_n"),
            pl.lit(0, pl.UInt32).alias("artist_seen_n"),
        )
    )
    return pl.concat([hist, rows.cast(hist.schema)])


def score(queries, candidates, items, explicit, t_end, cf, prior, model, cols, listens_dir, batch=20):
    history = (
        pl.scan_parquet(sorted(listens_dir.glob("part_*.parquet")))
        .filter(pl.col("uid").is_in(queries["uid"].unique().to_list()))
        .select(LISTEN_COLS)
        .collect()
        .with_columns(pl.int_range(pl.len()).over("uid").alias("row"))
    )
    real_uid = np.zeros(candidates["syn"].max() + 1, dtype=np.int64)
    owners = candidates.join(queries.select("q", "uid"), on="q")
    real_uid[owners["syn"].to_numpy()] = owners["uid"].to_numpy()
    feats = []
    started = time.time()
    for lo in range(0, queries.height, batch):
        q = queries.slice(lo, batch)
        c = candidates.join(q.select("q"), on="q")
        chunk = synthetic_chunk(history, q, c, items)
        syn_explicit = explicit.join(c.join(q.select("q", "uid"), on="q").select("uid", "syn"), on="uid")
        syn_explicit = syn_explicit.with_columns(pl.col("syn").alias("uid")).drop("syn").sort("uid", "timestamp")
        feats.append(last_row_features(chunk, syn_explicit, t_end, cf, real_uid))
        print(f"queries {lo + q.height}/{queries.height}: {chunk.height} rows, {time.time() - started:.0f}s", flush=True)
    feats = pl.concat(feats)
    listens = scan_listens_dir(listens_dir)
    ent = entity_features(listens, explicit, feats.select("item_id", "artist_id", "album_id", "ref_ts"), prior)
    df = add_cross(feats.hstack(ent))
    proba = model.predict_proba(df.select(cols).to_numpy().astype(np.float32))[:, 1]
    return df.select("syn").with_columns(pl.Series("p", proba))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--topk", help="npz from rl.evaluate --export-topk; without it only logged targets are scored")
    p.add_argument("--split", default="test")
    p.add_argument("--max-queries", type=int, default=500)
    p.add_argument("--model", default="L0_plus_L1")
    p.add_argument("--yambda", default=str(ROOT / "data" / "yambda"))
    p.add_argument("--catboost-dir", default=str(ROOT / "outputs" / "catboost"))
    p.add_argument("--out", required=True)
    a = p.parse_args()

    data = Data(a.data)
    if a.topk:
        with np.load(a.topk) as z:
            methods = {k: z[k] for k in z.files if k != "positions"}
            positions = z["positions"]
    else:
        methods, positions = {}, np.sort(np.random.default_rng(0).permutation(data.queries[a.split]))
    keep = np.sort(np.random.default_rng(1).permutation(len(positions))[:a.max_queries])
    positions = positions[keep]
    methods = {k: v[keep] for k, v in methods.items()}
    methods["logged"] = data.ev["item"][positions][:, None]

    user_ids = np.load(Path(a.data) / "user_ids.npy")
    users = data.user_of(positions)
    queries = pl.DataFrame({
        "q": np.arange(len(positions)), "uid": user_ids[users].astype(np.uint32),
        "k": (positions - data.offsets[users]).astype(np.int64), "ts": data.ev["ts"][positions].astype(np.uint32),
    })
    item_of = data.catalog["item_id"]
    pairs = pl.concat([
        pl.DataFrame({"q": np.repeat(np.arange(len(positions)), v.shape[1]), "item_id": item_of[v.ravel()]})
        for v in methods.values()
    ]).unique().sort("q", "item_id")
    candidates = pairs.with_columns(pl.int_range(pl.len()).cast(pl.Int64).alias("syn") + 1)
    print(f"queries {len(positions)}, methods {list(methods)}, candidates {candidates.height}")

    yambda, catboost_dir = Path(a.yambda), Path(a.catboost_dir)
    events = yambda / "flat" / "50m" / "multi_event.parquet"
    listens_dir = yambda / "derived" / "50m" / "listens"
    if not listens_dir.exists():
        prepare_listens(events, yambda, listens_dir)
    t_end = data.stats["t_end"]
    listens = scan_listens_dir(listens_dir)
    artist_of, album_of = load_mappings(events, yambda)
    lengths = pl.DataFrame({"item_id": item_of[2:], "length": data.catalog["dur"][2:].astype(np.uint32)})
    items = candidates.select("item_id").unique().join(artist_of, on="item_id", how="left").join(
        album_of, on="item_id", how="left").join(lengths, on="item_id", how="left")
    explicit = load_explicit(events, artist_of)
    prior = history_prior(listens, t_end)
    cf = fit_item_factors(listens, cf_cutoff(t_end))
    model = CatBoostClassifier()
    model.load_model(str(catboost_dir / f"{a.model}.cbm"))
    cols = json.loads((TASK1 / "features_meta.json").read_text())["features"]
    if a.model != "L0_plus_L1":
        raise ValueError("feature order is stored only for the full L0+L1 set")

    scored = candidates.join(score(queries, candidates, items, explicit, t_end, cf, prior, model, cols, listens_dir),
                               on="syn")
    lookup = {(q, i): p for q, i, p in scored.select("q", "item_id", "p").iter_rows()}
    per_query = {name: np.array([np.mean([lookup[(q, i)] for i in item_of[row]]) for q, row in enumerate(v)])
                 for name, v in methods.items()}
    result = {name: bootstrap(v, users) for name, v in per_query.items()}
    if "sft" in per_query:
        for name, v in per_query.items():
            result[name]["delta_vs_sft"] = bootstrap(v - per_query["sft"], users)
    if a.split == "test":
        logged = queries.with_columns(pl.Series("p", per_query["logged"]), pl.Series("item_id", item_of[methods["logged"][:, 0]]))
        reference = pl.read_parquet(catboost_dir / "test_predictions.parquet").select(
            "uid", "timestamp", "item_id", pl.col(f"pred_{a.model}").alias("p_ref"))
        check = logged.join(reference, left_on=["uid", "ts", "item_id"], right_on=["uid", "timestamp", "item_id"])
        diff = (check["p"] - check["p_ref"]).abs()
        result["logged_check"] = {"matched": check.height, "exact_share": float((diff < 1e-4).mean()),
                                  "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}
        print(check.with_columns(diff.alias("diff")).filter(pl.col("diff") >= 1e-4))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
