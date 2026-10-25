import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .data import Data, append_items, to_device
from .metrics import auc, ranking
from .model import load
from .optim import autocast, device_of, seed_everything
from .reward import LOGGED_FIELDS, Rewards
from .rollout import sample_slates, slate_logprobs

STEPS = 5
PROTOCOLS = ("real", "listened", "predicted")


def chains(data, split, rng, n, steps=STEPS):
    pos = np.sort(data.queries[f"rm_{split}"])
    idx = np.minimum(pos[:, None] + np.arange(steps), len(data.session) - 1)
    ok = np.isin(idx, pos).all(1) & (data.session[idx] == data.session[pos][:, None]).all(1)
    return np.sort(rng.permutation(pos[ok])[:n])


def with_feedback(batch, ratio):
    k = ratio.shape[1]
    b = {**batch, "ratio": batch["ratio"].clone(), "ratio_b": batch["ratio_b"].clone()}
    b["ratio"][:, -k:] = ratio
    b["ratio_b"][:, -k:] = 1 + torch.bucketize(ratio, torch.tensor(C.RATIO_EDGES, device=ratio.device), right=True)
    return b


def batched(starts, batch_size):
    return [starts[i:i + batch_size] for i in range(0, len(starts), batch_size)]


@torch.no_grad()
def simulator_error(rm, data, starts, device, batch_size=256):
    pred = {m: [] for m in ("teacher", "listened", "predicted")}
    for s in batched(starts, batch_size):
        own = {m: torch.zeros(len(s), STEPS, device=device) for m in pred}
        for k in range(STEPS):
            pos = s + k
            batch = to_device(data.window(pos, rm.cfg.ctx), device)
            far = to_device(data.far(pos, rm.cfg.ctx, rm.far_len), device) if rm.far_len else None
            target = data.targets(pos)
            cand = {f: target[f][:, None].to(device) for f in LOGGED_FIELDS}
            for mode in pred:
                b = batch
                if k and mode == "listened":
                    b = with_feedback(batch, torch.full((len(s), k), 100, device=device, dtype=torch.long))
                if k and mode == "predicted":
                    b = with_feedback(batch, (own[mode][:, :k] * 100).round().long())
                with autocast(device):
                    own[mode][:, k] = torch.sigmoid(rm.score(b, cand, far)[:, 0, 0].float())
        for mode in pred:
            pred[mode].append(own[mode].cpu().numpy())
    y = data.ev["ratio"][starts[:, None] + np.arange(STEPS)] / 100
    out = {}
    for mode, chunks in pred.items():
        p = np.concatenate(chunks)
        out[mode] = {
            "mae": np.abs(p - y).mean(0).tolist(),
            "bias": (p - y).mean(0).tolist(),
            "auc": [auc(y[:, k] >= C.LISTEN_PLUS / 100, p[:, k]) for k in range(STEPS)],
        }
    out["actual_ratio"] = y.mean(0).tolist()
    return out


@torch.no_grad()
def policy_error(policy, data, starts, device, batch_size=256):
    ctx = policy.cfg.ctx
    hits = {m: [] for m in ("teacher", "hidden_feedback", "generated")}
    for s in batched(starts, batch_size):
        targets = data.ev["item"][s[:, None] + np.arange(STEPS)]
        gen = to_device(data.window(s, ctx), device)
        chosen = []
        step = {m: [] for m in hits}
        for k in range(STEPS):
            real = to_device(data.window(s + k, ctx), device)
            hidden = {**real, "ratio_b": real["ratio_b"].clone()}
            if k:
                hidden["ratio_b"][:, -k:] = 0
            with autocast(device):
                logits = {m: policy.next_logits(b).float() for m, b in
                          (("teacher", real), ("hidden_feedback", hidden), ("generated", gen))}
            if chosen:
                logits["generated"] = logits["generated"].scatter(1, torch.stack(chosen, 1), -1e4)
            for m, lg in logits.items():
                topk = lg.topk(10).indices.cpu().numpy()
                step[m].append(ranking(topk, targets[:, k:k + 1], ks=(10,))["ndcg@10"])
            item = logits["generated"].argmax(-1)
            chosen.append(item)
            gen = append_items(gen, item, data)
        for m in hits:
            hits[m].append(np.stack(step[m], 1))
    return {m: np.nanmean(np.concatenate(v), 0).tolist() for m, v in hits.items()}


@torch.no_grad()
def rollout_stats(policy, ref, rewards, data, positions, device, greedy, batch_size=128):
    ctx = policy.cfg.ctx
    keys = [f"proxy_{p}" for p in PROTOCOLS] + ["judge_real", "kl", "history_repeat", "duration", "log_popularity"]
    acc = {k: [] for k in keys + ["unique", "adjacent_repeat"]}
    cat = data.catalog
    for pos in batched(positions, batch_size):
        batch = to_device(data.window(pos, ctx), device)
        far = rewards.far(pos, device)
        with autocast(device):
            slates = sample_slates(policy, batch, data, 1, STEPS, greedy)
            for protocol in PROTOCOLS:
                scores = rewards.scores(batch, slates, protocol, far)
                acc[f"proxy_{protocol}"].append(scores["ratio"][0][:, 0].cpu().numpy())
                if protocol == "real":
                    acc["judge_real"].append(scores["ratio"][1:].mean(0)[:, 0].cpu().numpy())
            logp = slate_logprobs(policy, batch, slates, data)
            ref_logp = slate_logprobs(ref, batch, slates, data)
        acc["kl"].append((logp.exp() * (logp - ref_logp)).sum(-1).cpu().numpy())
        s = slates[:, 0].cpu().numpy()
        history = batch["item"].cpu().numpy()
        acc["history_repeat"].append((s[:, :, None] == history[:, None, :]).any(-1).astype(float))
        acc["duration"].append(cat["dur"][s].astype(float))
        acc["log_popularity"].append(np.log1p(cat["train_count"][s]))
        acc["unique"].append(np.array([len(set(row)) for row in s], dtype=float))
        acc["adjacent_repeat"].append((s[:, 1:] == s[:, :-1]).mean(1))
    out = {k: np.concatenate(acc[k]).mean(0).tolist() for k in keys}
    out.update({k: float(np.concatenate(acc[k]).mean()) for k in ("unique", "adjacent_repeat")})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--sft", required=True)
    p.add_argument("--rm-dir", required=True)
    p.add_argument("--policies", nargs="*", default=[])
    p.add_argument("--split", default="test")
    p.add_argument("--chains", type=int, default=5000)
    p.add_argument("--contexts", type=int, default=2000)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    seed_everything(0)
    device = device_of(a.device)
    data = Data(a.data)
    rng = np.random.default_rng(0)
    starts = chains(data, a.split, rng, a.chains)
    contexts = np.sort(rng.permutation(data.queries[a.split])[:a.contexts])
    ref = load(a.sft, device).eval()
    rewards = Rewards.load(a.rm_dir, data, device)
    result = {"n_chains": int(len(starts)), "simulator": simulator_error(rewards.members[0], data, starts, device),
              "policy": {}, "rollout": {}}
    print("simulator", json.dumps(result["simulator"]), flush=True)
    for path in [a.sft] + a.policies:
        name = "sft" if path == a.sft else Path(path).parent.name + "/" + Path(path).stem
        policy = ref if path == a.sft else load(path, device).eval()
        result["policy"][name] = policy_error(policy, data, starts, device)
        result["rollout"][name] = {mode: rollout_stats(policy, ref, rewards, data, contexts, device, mode == "greedy")
                                   for mode in ("greedy", "sample")}
        print(name, json.dumps(result["policy"][name]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
