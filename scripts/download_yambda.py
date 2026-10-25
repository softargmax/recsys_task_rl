import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "yandex/yambda"
FILES = ["flat/50m/multi_event.parquet", "artist_item_mapping.parquet", "album_item_mapping.parquet"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "yambda"))
    args = parser.parse_args()
    for name in FILES:
        path = hf_hub_download(REPO, name, repo_type="dataset", local_dir=args.out)
        print(f"{name} -> {path} ({Path(path).stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
