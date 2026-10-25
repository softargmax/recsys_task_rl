import os

os.environ.setdefault("_RJEM_MALLOC_CONF", "background_thread:true,dirty_decay_ms:500,muzzy_decay_ms:500")
os.environ.setdefault("POLARS_MAX_THREADS", "4")

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import polars as pl

from . import config as C


def scan_listens(events_path):
    return (
        pl.scan_parquet(events_path)
        .filter(pl.col("event_type") == "listen")
        .select("uid", "timestamp", "item_id", "is_organic", "played_ratio_pct", "track_length_seconds")
    )


def days_before_end(t_end):
    return (pl.lit(t_end, pl.Int64) - pl.col("timestamp").cast(pl.Int64)) // C.DAY


def split_of(t_end):
    days = days_before_end(t_end)
    return pl.when(days < C.TEST_DAYS).then(2).when(days < C.TEST_DAYS + C.VAL_DAYS).then(1).otherwise(0)


def bucketize(values, edges):
    return (1 + np.searchsorted(np.asarray(edges), values, side="right")).astype(np.uint8)


def build_item_table(listens, artist_path, t_end, catalog_size, artist_vocab):
    is_train = days_before_end(t_end) >= C.TEST_DAYS + C.VAL_DAYS
    items = (
        listens.group_by("item_id")
        .agg(
            pl.col("track_length_seconds").median().fill_null(0).cast(pl.Int64).alias("dur"),
            is_train.sum().alias("train_count"),
        )
        .collect(engine="streaming")
    )
    artist_of = (
        pl.scan_parquet(artist_path)
        .join(items.lazy(), on="item_id", how="semi")
        .group_by("item_id")
        .agg(pl.col("artist_id").min())
        .collect()
    )
    items = items.join(artist_of, on="item_id", how="left")

    catalog = (
        items.filter(pl.col("train_count") > 0)
        .sort(["train_count", "item_id"], descending=[True, False])
        .head(catalog_size)
        .with_row_index("item", offset=C.N_SPECIAL)
        .select("item_id", "item")
    )
    artists = (
        items.filter(pl.col("artist_id").is_not_null())
        .group_by("artist_id")
        .agg(pl.col("train_count").sum())
        .filter(pl.col("train_count") > 0)
        .sort(["train_count", "artist_id"], descending=[True, False])
        .head(artist_vocab)
        .with_row_index("artist", offset=C.N_SPECIAL)
        .select("artist_id", "artist")
    )
    return (
        items.join(catalog, on="item_id", how="left")
        .join(artists, on="artist_id", how="left")
        .with_columns(
            pl.col("item").fill_null(C.UNK).cast(pl.Int32),
            pl.col("artist").fill_null(C.UNK).cast(pl.Int32),
        )
    )


def encode_chunk(chunk, items, t_end):
    df = (
        chunk.join(items.select("item_id", "item", "artist"), on="item_id", how="left")
        .sort("uid", "timestamp", maintain_order=True)
        .with_columns(
            pl.col("played_ratio_pct").clip(0, 100).cast(pl.Int64).alias("ratio"),
            pl.col("track_length_seconds").fill_null(0).cast(pl.Int64).alias("dur"),
            pl.col("timestamp").cast(pl.Int64).alias("ts"),
        )
        .with_columns(
            (pl.col("ts") - pl.col("ratio") * pl.col("dur") // 100).alias("start"),
            split_of(t_end).alias("split"),
        )
        .with_columns(
            (pl.col("start") - pl.col("ts").shift(1).over("uid")).alias("gap"),
            (pl.col("start").shift(-1).over("uid") - pl.col("ts")).alias("next_gap"),
            (pl.col("split").shift(-1).over("uid") != pl.col("split")).alias("next_crosses_split"),
        )
    )
    gap_b = bucketize(df["gap"].fill_null(0).clip(lower_bound=0).to_numpy(), C.GAP_EDGES)
    gap_b[df["gap"].is_null().to_numpy()] = 0
    dur = df["dur"].to_numpy()
    dur_b = bucketize(dur, C.DUR_EDGES)
    dur_b[dur <= 0] = 0
    ratio = df["ratio"].to_numpy()
    has_next = df["next_gap"].is_not_null().to_numpy()
    ts = df["ts"].to_numpy()
    cont = (has_next & (df["next_gap"].fill_null(0).to_numpy() < C.SESSION_IDLE)).astype(np.uint8)
    cont[~has_next & (ts > t_end - C.SESSION_IDLE)] = C.CONT_UNKNOWN
    cont[df["next_crosses_split"].fill_null(False).to_numpy()] = C.CONT_UNKNOWN
    return {
        "uid": df["uid"].to_numpy().astype(np.uint32),
        "item": df["item"].to_numpy().astype(np.int32),
        "artist": df["artist"].to_numpy().astype(np.int32),
        "ts": ts.astype(np.uint32),
        "ratio": ratio.astype(np.uint8),
        "organic": df["is_organic"].to_numpy().astype(np.uint8),
        "dur_b": dur_b,
        "gap_b": gap_b,
        "ratio_b": bucketize(ratio, C.RATIO_EDGES),
        "cont": cont,
    }


def sample_per_user(positions, user_of_pos, k, rng):
    order = rng.permutation(len(positions))
    order = order[np.argsort(user_of_pos[order], kind="stable")]
    users = user_of_pos[order]
    group_start = np.r_[0, np.flatnonzero(np.diff(users)) + 1]
    rank = np.arange(len(order)) - np.repeat(group_start, np.diff(np.r_[group_start, len(order)]))
    return np.sort(positions[order[rank < k]])


def build_queries(ev, offsets, t_end, rng):
    n = len(ev["ts"])
    user_of = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    days = (t_end - ev["ts"].astype(np.int64)) // C.DAY
    has_history = np.ones(n, dtype=bool)
    has_history[offsets[:-1]] = False
    recommended = (ev["organic"] == 0) & has_history
    in_catalog = ev["item"] >= C.N_SPECIAL
    windows = {
        "test": days < C.TEST_DAYS,
        "val": (days >= C.TEST_DAYS) & (days < C.TEST_DAYS + C.VAL_DAYS),
        "rl": (days >= C.TEST_DAYS + C.VAL_DAYS) & (days < C.TEST_DAYS + C.VAL_DAYS + C.RL_DAYS),
    }
    queries = {}
    for name in ("val", "test"):
        queries[f"rm_{name}"] = np.flatnonzero(recommended & windows[name] & (user_of % 5 == 0)).astype(np.int64)
    for name, window in windows.items():
        eligible = recommended & window
        if name != "rl":
            eligible &= in_catalog
        pos = np.flatnonzero(eligible)
        queries[name] = sample_per_user(pos, user_of[pos], C.QUERIES_PER_USER, rng).astype(np.int64)
    return queries


def prepare(events_path, artist_path, out_dir, catalog_size=C.CATALOG_SIZE, artist_vocab=C.ARTIST_VOCAB,
            max_users=None, users_per_chunk=1000, seed=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    listens = scan_listens(events_path)
    uids = listens.select("uid").unique().sort("uid").collect(engine="streaming")["uid"].to_numpy()
    if max_users is not None:
        uids = uids[:max_users]
        listens = listens.filter(pl.col("uid").is_in(uids.tolist()))
    t_end = int(listens.select(pl.col("timestamp").max()).collect().item())
    items = build_item_table(listens, artist_path, t_end, catalog_size, artist_vocab)
    print(f"users {len(uids)}, items {items.height}, t_end {t_end}")

    parts = {f: [] for f in ("uid",) + C.EVENT_FIELDS}
    for i in range(0, len(uids), users_per_chunk):
        lo, hi = int(uids[i]), int(uids[min(i + users_per_chunk, len(uids)) - 1])
        chunk = listens.filter(pl.col("uid").is_between(lo, hi)).collect()
        for f, v in encode_chunk(chunk, items, t_end).items():
            parts[f].append(v)
        print(f"users {lo}-{hi}: {chunk.height} events")
        del chunk
        gc.collect()

    ev = {f: np.concatenate(v) for f, v in parts.items()}
    del parts
    user_ids, counts = np.unique(ev.pop("uid"), return_counts=True)
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    for f, v in ev.items():
        np.save(out_dir / f"{f}.npy", v)
    np.save(out_dir / "offsets.npy", offsets)
    np.save(out_dir / "user_ids.npy", user_ids)

    catalog = items.filter(pl.col("item") >= C.N_SPECIAL).sort("item")
    n_items = C.N_SPECIAL + catalog.height
    item_artist = np.full(n_items, C.UNK, dtype=np.int32)
    item_dur = np.zeros(n_items, dtype=np.int32)
    item_count = np.zeros(n_items, dtype=np.int64)
    idx = catalog["item"].to_numpy()
    item_artist[idx] = catalog["artist"].to_numpy()
    item_dur[idx] = catalog["dur"].to_numpy()
    item_count[idx] = catalog["train_count"].to_numpy()
    item_dur_b = bucketize(item_dur, C.DUR_EDGES)
    item_dur_b[item_dur <= 0] = 0
    np.savez(out_dir / "catalog.npz", item_id=np.r_[0, 0, catalog["item_id"].to_numpy()].astype(np.uint32),
             artist=item_artist, dur=item_dur, dur_b=item_dur_b, train_count=item_count)

    queries = build_queries(ev, offsets, t_end, np.random.default_rng(seed))
    np.savez(out_dir / "queries.npz", **queries)

    in_catalog = ev["item"] >= C.N_SPECIAL
    stats = {
        "t_end": t_end,
        "users": int(len(user_ids)),
        "events": int(len(ev["ts"])),
        "catalog_size": int(catalog.height),
        "n_artists": int(items["artist"].max()) + 1,
        "catalog_event_share": float(in_catalog.mean()),
        "queries": {k: int(len(v)) for k, v in queries.items()},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True)
    p.add_argument("--artists", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--catalog-size", type=int, default=C.CATALOG_SIZE)
    p.add_argument("--max-users", type=int)
    a = p.parse_args()
    prepare(a.events, a.artists, a.out, catalog_size=a.catalog_size, max_users=a.max_users)


if __name__ == "__main__":
    main()
