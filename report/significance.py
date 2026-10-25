import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl import config as C
from rl.data import Data
from rl.evaluate import relevance
from rl.metrics import ranking

RES = ROOT / "outputs" / "results"
SOURCES = {
    "full_logs": ("full_logs/{split}_top10.npz", ["lp_future_ndcg@10", "lp_next_ndcg@10", "future_ndcg@10", "next_ndcg@10"]),
    "night": ("night_v1/night_{split}_top10.npz", ["lp_future_ndcg@10", "future_ndcg@10", "next_ndcg@10", "organic_lp"]),
    "control": ("control_v1/control_{split}_top10.npz", ["lp_future_ndcg@10", "future_ndcg@10", "next_ndcg@10", "organic_lp"]),
}
BASELINES = ("positions", "sft", "most_pop", "recent")
HIT_RUNS = ("hit_b0.03__step_00600", "hit_b0.03_s1__step_00600", "hit_b0.01__step_00600")


def paired(diff, users, n_boot=2000):
    keep = np.isfinite(diff)
    diff, users = diff[keep], users[keep]
    uniq, inv = np.unique(users, return_inverse=True)
    s, c = np.bincount(inv, weights=diff), np.bincount(inv)
    idx = np.random.default_rng(0).integers(0, len(uniq), size=(n_boot, len(uniq)))
    m = s[idx].sum(1) / c[idx].sum(1)
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return diff.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975), max(p, 1 / n_boot)


def holm(ps):
    ps = np.asarray(ps)
    out, run = np.empty_like(ps), 0.0
    for rank, i in enumerate(np.argsort(ps)):
        run = max(run, min(1.0, ps[i] * (len(ps) - rank)))
        out[i] = run
    return out


def metrics_for(data, pos):
    items, ratios = data.future(pos)
    idx = np.minimum(pos[:, None] + np.arange(items.shape[1]), len(data.ev["organic"]) - 1)
    organic = np.where((data.ev["organic"][idx] == 1) & (ratios >= C.LISTEN_PLUS), items, C.PAD)

    def compute(topk):
        r = relevance(topk, data, pos)
        r["organic_lp"] = ranking(topk, organic, ks=(10,))["ndcg@10"]
        return r
    return compute


def short(name):
    return name.replace("__step_00600", "").replace("__policy", "")


def main():
    data = Data(ROOT / "data" / "full")
    result = {}
    for source, (pattern, metrics) in SOURCES.items():
        for split in ("test", "val"):
            z = np.load(RES / pattern.format(split=split))
            pos = z["positions"]
            users = data.user_of(pos)
            compute = metrics_for(data, pos)
            base = compute(z["sft"])
            rows = []
            for name in (k for k in z.files if k not in BASELINES):
                r = compute(z[name])
                for k in metrics:
                    mean, lo, hi, p = paired(r[k] - base[k], users)
                    b0 = np.nanmean(base[k])
                    rows.append({"model": short(name), "metric": k, "delta_pct": 100 * mean / b0,
                                 "ci_pct": [100 * lo / b0, 100 * hi / b0], "p": p})
            for row, h in zip(rows, holm([r["p"] for r in rows])):
                row["p_holm"] = float(h)
            result[f"{source}/{split}"] = rows
            print(f"{source} {split}: {len(pos)} queries, {len(rows)} comparisons")
            for r in rows:
                mark = "*" if r["p_holm"] < 0.05 else " "
                print(f"  {r['model']:20s} {r['metric']:18s} {r['delta_pct']:+6.1f}% "
                      f"[{r['ci_pct'][0]:+.1f}; {r['ci_pct'][1]:+.1f}] holm={r['p_holm']:.3f} {mark}")
            if source == "control":
                ref = compute(z["sft_cont_e2__policy"])
                rows = []
                for name in HIT_RUNS:
                    r = compute(z[name])
                    for k in ("lp_future_ndcg@10", "future_ndcg@10"):
                        mean, lo, hi, p = paired(r[k] - ref[k], users)
                        b0 = np.nanmean(base[k])
                        rows.append({"model": short(name), "metric": k, "delta_pct": 100 * mean / b0,
                                     "ci_pct": [100 * lo / b0, 100 * hi / b0], "p": p})
                result[f"hit_vs_sft_cont_e2/{split}"] = rows
                print(f"hit vs sft_cont_e2 {split}:")
                for r in rows:
                    print(f"  {r['model']:20s} {r['metric']:18s} {r['delta_pct']:+6.1f}% "
                          f"[{r['ci_pct'][0]:+.1f}; {r['ci_pct'][1]:+.1f}] p={r['p']:.4f}")
    (RES / "significance.json").write_text(json.dumps(result, indent=1))
    print("written", RES / "significance.json")


if __name__ == "__main__":
    main()
