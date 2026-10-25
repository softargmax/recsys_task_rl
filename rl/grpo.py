import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .data import Data, to_device
from .model import load, save
from .optim import Log, Optimizer, autocast, device_of, seed_everything
from .reward import REWARD_KINDS, Rewards, discounted, hit_reward, judge_reward, per_item_reward, slate_reward
from .rollout import sample_slates, slate_logprobs
from .sft import sft_loss


@dataclass
class GRPOConfig:
    reward: str = "ratio"
    beta: float = 0.1
    steps: int = 600
    batch_size: int = 64
    group: int = 8
    slate: int = 5
    lr: float = 2e-5
    k_pess: float = 1.0
    hit_weight: float = 0.0
    ce_weight: float = 0.0
    ce_batch: int = 32
    chunk: int = 8
    save_every: int = 100
    seed: int = 0


def chunk_loss(policy, ref, batch, slates, advantage, data, beta, scale):
    slate = slates.shape[2]
    logp = slate_logprobs(policy, batch, slates, data)
    with torch.no_grad():
        ref_logp = slate_logprobs(ref, batch, slates, data)
    taken = logp.gather(-1, slates.reshape(-1, slate, 1)).squeeze(-1).sum(-1)
    kl = (logp.exp() * (logp - ref_logp)).sum(-1).sum(-1)
    loss = (-(advantage.reshape(-1) * taken).sum() + beta * kl.sum()) * scale
    entropy = -(logp.exp() * logp).sum(-1).mean(-1)
    return loss, kl.detach(), entropy.detach()


def grpo_step(policy, ref, rewards, data, batch, future, opt, cfg, ce_batch=None, far=None):
    slates = sample_slates(policy, batch, data, cfg.group, cfg.slate)
    scores = rewards.scores(batch, slates, far=far)
    hits = hit_reward(slates, *future)
    reward = discounted(per_item_reward(cfg.reward, scores, cfg.k_pess) + cfg.hit_weight * hits)
    advantage = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-6)
    n = len(batch["item"])
    kls, entropies = [], []
    for i in range(0, n, cfg.chunk):
        part = {k: v[i:i + cfg.chunk] for k, v in batch.items()}
        loss, kl, entropy = chunk_loss(policy, ref, part, slates[i:i + cfg.chunk], advantage[i:i + cfg.chunk], data,
                                       cfg.beta, 1 / (n * cfg.group))
        opt.backward(loss)
        kls.append(kl)
        entropies.append(entropy)
    stats = {}
    if ce_batch is not None:
        ce = sft_loss(policy, ce_batch)
        opt.backward(cfg.ce_weight * ce)
        stats["ce"] = ce.detach()
    opt.step()
    return {
        "reward": reward.mean(),
        "proxy": slate_reward("ratio", scores).mean(),
        "judge": discounted(judge_reward(scores)).mean(),
        "hit_rate": (hits > 0).float().mean(),
        "kl": torch.cat(kls).mean(),
        "entropy": torch.cat(entropies).mean(),
        "flat_groups": (reward.std(1) < 1e-6).float().mean(),
        **stats,
    }


def train_grpo(data, sft_path, rm_dir, out_dir, cfg, device="auto"):
    device = device_of(device)
    seed_everything(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    policy = load(sft_path, device).eval()
    ref = load(sft_path, device).eval().requires_grad_(False)
    rewards = Rewards.load(rm_dir, data, device)
    opt = Optimizer(policy, cfg.lr, cfg.steps, warmup=20, weight_decay=0.0)
    contexts = data.queries["rl"]
    rng = np.random.default_rng(cfg.seed)
    windows = data.train_windows(policy.cfg.ctx + 1, rng) if cfg.ce_weight > 0 else None
    log = Log(every=20)
    for step in range(cfg.steps):
        pos = np.sort(rng.choice(contexts, cfg.batch_size, replace=False))
        batch = to_device(data.window(pos, policy.cfg.ctx), device)
        future = [torch.from_numpy(a.astype(np.int64)).to(device) for a in data.future(pos, within_train=True)]
        ce_batch = None
        if windows is not None:
            ce_batch = to_device(data.window(rng.choice(windows, cfg.ce_batch, replace=False), policy.cfg.ctx + 1), device)
        with autocast(device):
            stats = grpo_step(policy, ref, rewards, data, batch, future, opt, cfg, ce_batch, rewards.far(pos, device))
        log.add(step, **stats)
        if (step + 1) % cfg.save_every == 0 or step + 1 == cfg.steps:
            save(policy, out_dir / f"step_{step + 1:05d}.pt", step=step + 1, **asdict(cfg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--sft", required=True)
    p.add_argument("--rm-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="auto")
    for f in fields(GRPOConfig):
        kwargs = {"choices": REWARD_KINDS} if f.name == "reward" else {}
        p.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default, **kwargs)
    a = vars(p.parse_args())
    cfg = GRPOConfig(**{f.name: a[f.name] for f in fields(GRPOConfig)})
    train_grpo(Data(a["data"]), a["sft"], a["rm_dir"], a["out_dir"], cfg, a["device"])


if __name__ == "__main__":
    main()
