import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .data import Data, drop_feedback, shift, to_device
from .evaluate import policy_outputs, relevance
from .model import Policy, load, save
from .optim import Log, Optimizer, autocast, device_of, seed_everything


def sft_loss(policy, batch, feedback_dropout=0.15, rw_lambda=None):
    inputs, nxt = shift(batch)
    hidden = policy(drop_feedback(inputs, feedback_dropout) if policy.training else inputs)
    valid = nxt["mask"] & (nxt["item"] >= C.N_SPECIAL)
    loss = F.cross_entropy(policy.logits(hidden[valid]).float(), nxt["item"][valid], reduction="none")
    if rw_lambda is None:
        return loss.mean()
    weights = torch.exp(nxt["ratio"][valid].float() / 100 / rw_lambda)
    return (loss * weights / weights.mean()).mean()


def validate(policy, data, positions, device):
    policy.eval()
    topk = policy_outputs(policy, data, positions, device)["topk"]
    policy.train()
    return {k: float(np.nanmean(v)) for k, v in relevance(topk, data, positions).items()}


def train_sft(data, cfg, out_dir, epochs=3, batch_size=64, lr=1e-3, device="auto", init=None, rw_lambda=None,
              feedback_dropout=0.15, val_queries=10000, seed=0):
    device = device_of(device)
    seed_everything(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = (load(init) if init else Policy(data.n_items, data.n_artists, cfg)).to(device)
    cfg = policy.cfg
    rng = np.random.default_rng(seed)
    val_pos = np.sort(rng.permutation(data.queries["val"])[:val_queries])
    epochs_ends = [data.train_windows(cfg.ctx + 1, rng) for _ in range(epochs)]
    steps = sum(len(e) // batch_size for e in epochs_ends)
    opt = Optimizer(policy, lr, steps)
    log = Log(every=200)
    history, best, step = [], -1.0, 0
    policy.train()
    for epoch, ends in enumerate(epochs_ends):
        for i in range(0, len(ends) - batch_size + 1, batch_size):
            batch = to_device(data.window(ends[i:i + batch_size], cfg.ctx + 1), device)
            with autocast(device):
                loss = sft_loss(policy, batch, feedback_dropout, rw_lambda)
            log.add(step, loss=loss, grad_norm=opt.step(loss))
            step += 1
        metrics = {"epoch": epoch + 1, "steps": step, **validate(policy, data, val_pos, device)}
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if metrics["next_ndcg@10"] > best:
            best = metrics["next_ndcg@10"]
            policy.eval()
            save(policy, out_dir / "policy.pt", epoch=epoch + 1, rw_lambda=rw_lambda)
            policy.train()
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--size", default="base", choices=sorted(C.MODEL_SIZES))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--init", help="continue from a policy checkpoint, used for reward-weighted SFT")
    p.add_argument("--rw-lambda", type=float)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    train_sft(Data(a.data), C.MODEL_SIZES[a.size], a.out_dir, a.epochs, a.batch_size, a.lr, a.device, a.init,
              a.rw_lambda)


if __name__ == "__main__":
    main()
