import gc
import shutil
from pathlib import Path

import polars as pl

from . import config


def first_mapping(path, key, items):
    return (
        pl.scan_parquet(path)
        .join(items.lazy(), on="item_id", how="semi")
        .group_by("item_id")
        .agg(pl.col(key).min())
        .collect()
    )


def enrich(chunk, artist_of, album_of):
    return (
        chunk.join(artist_of, on="item_id", how="left")
        .join(album_of, on="item_id", how="left")
        .sort("uid", "timestamp", maintain_order=True)
        .with_columns(
            (pl.col("played_ratio_pct") >= config.LISTEN_PLUS).alias("listen_plus"),
            pl.int_range(pl.len(), dtype=pl.UInt32).over("uid", "item_id").alias("track_seen_n"),
            pl.when(pl.col("artist_id").is_not_null())
            .then(pl.int_range(pl.len(), dtype=pl.UInt32).over("uid", "artist_id"))
            .alias("artist_seen_n"),
            pl.col("timestamp").cast(pl.Int64).diff().over("uid", "item_id").cast(pl.UInt32).alias("dt_track"),
            pl.when(pl.col("artist_id").is_not_null())
            .then(pl.col("timestamp").cast(pl.Int64).diff().over("uid", "artist_id"))
            .cast(pl.UInt32)
            .alias("dt_artist"),
            (pl.col("timestamp").cast(pl.Int64).diff().over("uid").fill_null(10**9) > config.SESSION_IDLE)
            .cast(pl.UInt32)
            .cum_sum()
            .over("uid")
            .alias("session_idx"),
        )
        .with_columns(
            pl.int_range(pl.len(), dtype=pl.UInt32).over("uid", "session_idx").alias("pos_in_session"),
            pl.col("listen_plus").shift(1).over("uid", "session_idx").alias("prev_listen_plus"),
        )
    )


def scan_listens(events_path, max_ts=None):
    lf = pl.scan_parquet(events_path).filter(pl.col("event_type") == "listen")
    if max_ts is not None:
        lf = lf.filter(pl.col("timestamp") < max_ts)
    return lf.select("uid", "timestamp", "item_id", "is_organic", "played_ratio_pct", "track_length_seconds")


def load_mappings(events_path, mappings_dir):
    items = scan_listens(events_path).select("item_id").unique().collect()
    mappings_dir = Path(mappings_dir)
    artist_of = first_mapping(mappings_dir / "artist_item_mapping.parquet", "artist_id", items)
    album_of = first_mapping(mappings_dir / "album_item_mapping.parquet", "album_id", items)
    return artist_of, album_of


def prepare_listens(events_path, mappings_dir, out_dir, users_per_chunk=1000):
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    listens = scan_listens(events_path)
    artist_of, album_of = load_mappings(events_path, mappings_dir)
    uids = listens.select("uid").unique().sort("uid").collect()["uid"].to_list()
    print(f"users: {len(uids)}, artist map: {artist_of.height}, album map: {album_of.height}")

    for i in range(0, len(uids), users_per_chunk):
        lo, hi = uids[i], uids[min(i + users_per_chunk, len(uids)) - 1]
        chunk = listens.filter(pl.col("uid").is_between(lo, hi)).collect()
        part = enrich(chunk, artist_of, album_of)
        path = out_dir / f"part_{i // users_per_chunk:03d}.parquet"
        part.write_parquet(path, compression="zstd")
        print(f"{path.name}: users {lo}-{hi}, rows {part.height}")
        del chunk, part
        gc.collect()

    print(f"done -> {out_dir}")
