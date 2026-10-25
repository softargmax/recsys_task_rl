import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .data import Data, to_device
from .metrics import bootstrap, bootstrap_weighted, concentration, gauc, group_aucs, ranking
from .model import load
from .optim import autocast, device_of, seed_everything
from .reward import Rewards, discounted, judge_reward, per_item_reward, slate_reward
from .rollout import sample_slates, slate_logprobs


def batches(data, positions, ctx, device, batch_size):
    for i in range(0, len(positions), batch_size):
        pos = positions[i:i + batch_size]
        yield pos, to_device(data.window(pos, ctx), device)


@torch.no_grad()
def policy_outputs(policy, data, positions, device, ref=None, k=20, batch_size=512):
    out = {"topk": [], "logp_target": [], "entropy": [], "kl": []}
    for pos, batch in batches(data, positions, policy.cfg.ctx, device, batch_size):
        with autocast(device):
            logp = torch.log_softmax(policy.next_logits(batch).float(), -1)
            target = torch.from_numpy(data.ev["item"][pos].astype(np.int64)).to(device)
            out["topk"].append(logp.topk(k, -1).indices.cpu())
            out["logp_target"].append(logp.gather(1, target[:, None]).squeeze(1).cpu())
            out["entropy"].append(-(logp.exp() * logp).sum(-1).cpu())
            if ref is not None:
                ref_logp = torch.log_softmax(ref.next_logits(batch).float(), -1)
                out["kl"].append((logp.exp() * (logp - ref_logp)).sum(-1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items() if v}


def popular_topk(data, n, k=20):
    order = np.argsort(-data.catalog["train_count"][C.N_SPECIAL:], kind="stable")[:k] + C.N_SPECIAL
    return np.tile(order, (n, 1))


def recent_topk(data, positions, ctx, k=20):
    popular = popular_topk(data, 1, 4 * k)[0]
    out = np.empty((len(positions), k), dtype=np.int64)
    for row, pos in enumerate(positions):
        start = max(data.offsets[data.user_of(pos)], pos - ctx)
        recent = data.ev["item"][start:pos][::-1]
        recent = recent[recent >= C.N_SPECIAL]
        _, first = np.unique(recent, return_index=True)
        items = list(recent[np.sort(first)][:k])
        items += [i for i in popular if i not in set(items)][: k - len(items)]
        out[row] = items
    return out


def relevance(topk, data, positions):
    items, ratios = data.future(positions)
    listened = np.where(ratios >= C.LISTEN_PLUS, items, C.PAD)
    skipped = np.where(ratios < C.LISTEN_PLUS, items, C.PAD)
    target = items[:, :1]
    out = {}
    for name, targets in (("next", target), ("future", items), ("lp_next", listened[:, :1]), ("lp_future", listened)):
        out.update({f"{name}_{m}": v for m, v in ranking(topk, targets).items()})
    top10 = topk[:, :10, None]
    only_skipped = (top10 == skipped[:, None, :]).any(-1) & ~(top10 == listened[:, None, :]).any(-1)
    out["skipped_share@10"] = only_skipped.mean(1)
    return out


def history_kinds(data, positions, ctx):
    target = data.ev["item"][positions]
    kinds = np.zeros(len(positions), dtype=np.int8)
    for row, pos in enumerate(positions):
        start = data.offsets[data.user_of(pos)]
        if target[row] in data.ev["item"][max(start, pos - ctx):pos]:
            kinds[row] = 2
        elif target[row] in data.ev["item"][start:pos]:
            kinds[row] = 1
    return kinds


def next_by_history(top10, data, positions, ctx):
    ndcg = ranking(top10, data.ev["item"][positions][:, None], ks=(10,))["ndcg@10"]
    kinds = history_kinds(data, positions, ctx)
    return {f"next_ndcg@10_{name}": np.where(kinds == k, ndcg, np.nan)
            for k, name in enumerate(("new", "old_repeat", "ctx_repeat"))}


def behavior(topk, data, positions, ctx):
    in_history = np.zeros(topk.shape, dtype=bool)
    new_artist = np.zeros(topk.shape, dtype=bool)
    artists = data.catalog["artist"][topk]
    for row, pos in enumerate(positions):
        start = max(data.offsets[data.user_of(pos)], pos - ctx)
        in_history[row] = np.isin(topk[row], data.ev["item"][start:pos])
        new_artist[row] = ~np.isin(artists[row], data.ev["artist"][start:pos])
    return {
        "repeat_share": in_history.mean(1),
        "new_artist_share": new_artist.mean(1),
        "duration_s": data.catalog["dur"][topk].mean(1),
        "log_popularity": np.log1p(data.catalog["train_count"][topk]).mean(1),
    }


def skip_auc(logp_target, data, positions):
    y = (data.ev["ratio"][positions] >= C.LISTEN_PLUS).astype(int)
    users = data.user_of(positions)
    dur_b = data.ev["dur_b"][positions]
    within = [gauc(y[dur_b == b], logp_target[dur_b == b], users[dur_b == b]) for b in np.unique(dur_b)]
    return group_aucs(y, logp_target, users), float(np.nanmean(within))


@torch.no_grad()
def topk_rewards(rewards, data, positions, topk, ctx, device, batch_size=256):
    out = {"proxy": [], "judge": []}
    for i, (pos, batch) in enumerate(batches(data, positions, ctx, device, batch_size)):
        items = torch.from_numpy(topk[i * batch_size:i * batch_size + len(pos)]).to(device)[:, None, :]
        with autocast(device):
            scores = rewards.scores(batch, items, far=rewards.far(pos, device))
        out["proxy"].append(per_item_reward("ratio", scores).mean(-1).squeeze(1).cpu())
        out["judge"].append(judge_reward(scores).mean(-1).squeeze(1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


@torch.no_grad()
def slate_eval(policy, ref, rewards, data, positions, device, kind="ratio", n_best=1, slate=5, sequences=512,
               seed=0):
    torch.manual_seed(seed)
    batch_size = max(1, sequences // n_best)
    out = {"reward": [], "proxy": [], "judge": [], "kl": [], "unique": [], "adjacent_repeat": []}
    for pos, batch in batches(data, positions, policy.cfg.ctx, device, batch_size):
        with autocast(device):
            slates = sample_slates(policy, batch, data, n_best, slate)
            scores = rewards.scores(batch, slates, far=rewards.far(pos, device))
            r = slate_reward(kind, scores)
            best = r.argmax(1)
            rows = torch.arange(len(pos), device=device)
            chosen = slates[rows, best][:, None]
            if ref is not policy:
                logp = slate_logprobs(policy, batch, chosen, data)
                ref_logp = slate_logprobs(ref, batch, chosen, data)
                out["kl"].append((logp.exp() * (logp - ref_logp)).sum(-1).sum(-1).cpu())
        out["reward"].append(r[rows, best].cpu())
        out["proxy"].append(slate_reward("ratio", scores)[rows, best].cpu())
        out["judge"].append(discounted(judge_reward(scores))[rows, best].cpu())
        s = chosen[:, 0]
        out["unique"].append(torch.tensor([len(set(x.tolist())) for x in s.cpu()], dtype=torch.float))
        out["adjacent_repeat"].append((s[:, 1:] == s[:, :-1]).float().mean(1).cpu())
    result = {k: torch.cat(v).numpy() for k, v in out.items() if v}
    if n_best > 1:
        result["kl_bound"] = np.full(len(positions), np.log(n_best) - (n_best - 1) / n_best)
    return result


def summarize(per_query, users, scalars=None):
    out = {k: bootstrap(v.astype(float), users) for k, v in per_query.items()}
    out.update(scalars or {})
    return out


def evaluate_policy(policy, ref, rewards, data, positions, device, slate_positions, ctx, batch_size):
    outs = policy_outputs(policy, data, positions, device, ref=ref, batch_size=batch_size)
    top10 = outs["topk"][:, :10]
    per_query = {**relevance(outs["topk"], data, positions), **next_by_history(top10, data, positions, ctx),
                 **behavior(top10, data, positions, ctx), "entropy": outs["entropy"]}
    if "kl" in outs:
        per_query["kl_first_step"] = outs["kl"]
    if rewards is not None:
        per_query.update({f"top10_{k}": v for k, v in
                          topk_rewards(rewards, data, positions, top10, ctx, device, batch_size).items()})
    skip, skip_within = skip_auc(outs["logp_target"], data, positions)
    run = {"per_query": per_query, "skip": skip, "top10": top10, "slates": None,
           "scalars": {**concentration(top10, data.n_items - C.N_SPECIAL, np.random.default_rng(0)),
                       "skip_gauc": bootstrap_weighted(*skip), "skip_gauc_within_duration": skip_within}}
    if rewards is not None:
        run["slates"] = slate_eval(policy, ref, rewards, data, slate_positions, device, sequences=2 * batch_size)
    return run


def evaluate_baseline(topk, data, positions, ctx):
    per_query = {**relevance(topk, data, positions), **next_by_history(topk[:, :10], data, positions, ctx),
                 **behavior(topk[:, :10], data, positions, ctx)}
    return {"per_query": per_query, "skip": None, "top10": topk[:, :10], "slates": None,
            "scalars": concentration(topk[:, :10], data.n_items - C.N_SPECIAL, np.random.default_rng(0))}


def report(run, base, users, slate_users):
    out = {"ranking": summarize(run["per_query"], users, run["scalars"])}
    if run["slates"] is not None:
        out["slate"] = summarize(run["slates"], slate_users)
    if base is None or run is base:
        return out
    delta = {k: bootstrap(v - base["per_query"][k], users) for k, v in run["per_query"].items() if k in base["per_query"]}
    if run["skip"] is not None:
        delta["skip_gauc"] = bootstrap_weighted(run["skip"][0] - base["skip"][0], run["skip"][1])
    if run["slates"] is not None and base["slates"] is not None:
        delta.update({f"slate_{k}": bootstrap(v - base["slates"][k], slate_users)
                      for k, v in run["slates"].items() if k in base["slates"]})
    out["delta_vs_sft"] = delta
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--sft", required=True)
    p.add_argument("--rm-dir")
    p.add_argument("--policies", nargs="*", default=[])
    p.add_argument("--best-of", type=int, nargs="*", default=[])
    p.add_argument("--split", default="val")
    p.add_argument("--max-queries", type=int, default=20000)
    p.add_argument("--slate-queries", type=int, default=3000)
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", required=True)
    p.add_argument("--export-topk", help="npz with top-10 of every method, input for the CatBoost judge")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    seed_everything(0)
    device = device_of(a.device)
    data = Data(a.data)
    rng = np.random.default_rng(0)
    positions = np.sort(rng.permutation(data.queries[a.split])[:a.max_queries])
    slate_positions = positions[:a.slate_queries]
    ref = load(a.sft, device).eval()
    ctx = ref.cfg.ctx
    rewards = Rewards.load(a.rm_dir, data, device) if a.rm_dir else None

    users, slate_users = data.user_of(positions), data.user_of(slate_positions)
    sft = evaluate_policy(ref, ref, rewards, data, positions, device, slate_positions, ctx, a.batch_size)
    results = {"sft": report(sft, sft, users, slate_users)}
    top10 = {"sft": sft["top10"]}
    if a.baselines:
        for name, topk in (("most_pop", popular_topk(data, len(positions))), ("recent", recent_topk(data, positions, ctx))):
            run = evaluate_baseline(topk, data, positions, ctx)
            results[name] = report(run, sft, users, slate_users)
            top10[name] = run["top10"]
    for n in a.best_of:
        slates = slate_eval(ref, ref, rewards, data, slate_positions, device, n_best=n, sequences=2 * a.batch_size)
        results[f"best_of_{n}"] = {"slate": summarize(slates, slate_users)}
    for path in a.policies:
        name = Path(path).parent.name + "/" + Path(path).stem
        run = evaluate_policy(load(path, device).eval(), ref, rewards, data, positions, device, slate_positions, ctx,
                              a.batch_size)
        results[name] = report(run, sft, users, slate_users)
        top10[name] = run["top10"]
        print(f"evaluated {path}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.export_topk:
        Path(a.export_topk).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.export_topk, positions=positions, **{k.replace("/", "__"): v for k, v in top10.items()})
    Path(a.out).write_text(json.dumps(results, indent=2))
    for name, r in results.items():
        rank = r.get("ranking", {})
        line = {m: round(rank[m]["mean"], 4) for m in ("next_ndcg@10", "lp_future_ndcg@10", "top10_proxy", "duration_s",
                                                         "skip_gauc") if m in rank}
        if "slate" in r:
            line.update({f"slate_{m}": round(r["slate"][m]["mean"], 4) for m in ("proxy", "judge", "kl", "kl_bound")
                         if m in r["slate"]})
        print(name, line)


if __name__ == "__main__":
    main()
