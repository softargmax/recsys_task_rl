import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .data import Data, repeat, shift, to_device
from .metrics import auc, gauc
from .model import RewardModel, load, save
from .optim import Log, Optimizer, autocast, device_of, seed_everything
from .rollout import history_for_reward

REWARD_KINDS = ("ratio", "seconds", "norm", "pess", "continue", "hit")
LOGGED_FIELDS = ("item", "artist", "dur_b", "organic")


def user_halves(data, seed=0):
    perm = np.random.default_rng(seed).permutation(data.n_users)
    return perm[: data.n_users // 2], perm[data.n_users // 2:]


def member_users(data, member):
    users_a, _ = user_halves(data)
    if member == 0:
        return users_a
    return np.unique(np.random.default_rng(member).choice(users_a, size=len(users_a)))


def rm_loss(rm, batch, far=None):
    inputs, nxt = shift(batch)
    logits = rm.forward_train(inputs, nxt, far).float()
    valid = nxt["mask"]
    ratio_loss = F.binary_cross_entropy_with_logits(logits[..., 0][valid], nxt["ratio"][valid].float() / 100)
    cont = valid & (nxt["cont"] != C.CONT_UNKNOWN)
    cont_loss = F.binary_cross_entropy_with_logits(logits[..., 1][cont], nxt["cont"][cont].float())
    return ratio_loss, cont_loss


def train_reward_model(data, sft_path, out, member=0, duration_only=False, epochs=1, batch_size=64, lr=5e-4,
                       device="auto", match_window=None):
    device = device_of(device)
    seed_everything(member)
    sft = load(sft_path)
    rm = RewardModel(data.n_items, data.n_artists, sft.cfg, duration_only, match_window).to(device)
    rm.encoder.load_state_dict(sft.encoder.state_dict())
    rng = np.random.default_rng(member)
    users = member_users(data, member)
    epochs_ends = [data.train_windows(sft.cfg.ctx + 1, rng, users) for _ in range(epochs)]
    steps = sum(len(e) // batch_size for e in epochs_ends)
    opt = Optimizer(rm, lr, steps)
    log = Log(every=200)
    rm.train()
    step = 0
    for ends in epochs_ends:
        for i in range(0, len(ends) - batch_size + 1, batch_size):
            part = ends[i:i + batch_size]
            batch = to_device(data.window(part, sft.cfg.ctx + 1), device)
            far = to_device(data.far(part, sft.cfg.ctx + 1, rm.match_window), device) if rm.far_len else None
            with autocast(device):
                ratio_loss, cont_loss = rm_loss(rm, batch, far)
            opt.step(ratio_loss + cont_loss)
            log.add(step, ratio_bce=ratio_loss, cont_bce=cont_loss)
            step += 1
    rm.eval()
    save(rm, out, member=member, users=len(users))
    return rm


@torch.no_grad()
def predict_logged(rm, data, positions, device, batch_size=512):
    out = []
    for i in range(0, len(positions), batch_size):
        pos = positions[i:i + batch_size]
        batch = to_device(data.window(pos, rm.cfg.ctx), device)
        far = to_device(data.far(pos, rm.cfg.ctx, rm.far_len), device) if rm.far_len else None
        target = data.targets(pos)
        cand = {k: target[k][:, None].to(device) for k in LOGGED_FIELDS}
        with autocast(device):
            out.append(torch.sigmoid(rm.score(batch, cand, far)[:, 0].float()).cpu())
    return torch.cat(out).numpy()


def rm_report(rm, data, split, device, base=None):
    pos = data.queries[f"rm_{split}"]
    pred = predict_logged(rm, data, pos, device)
    ratio = data.ev["ratio"][pos] / 100
    y = (ratio >= C.LISTEN_PLUS / 100).astype(int)
    users, sessions = data.user_of(pos), data.session[pos]
    cont = data.ev["cont"][pos]
    known = cont != C.CONT_UNKNOWN
    dur_b = data.ev["dur_b"][pos]
    report = {
        "n": int(len(pos)),
        "auc": auc(y, pred[:, 0]),
        "gauc_user": gauc(y, pred[:, 0], users),
        "gauc_session": gauc(y, pred[:, 0], sessions),
        "mae_ratio": float(np.abs(pred[:, 0] - ratio).mean()),
        "auc_continue": auc(cont[known], pred[known, 1]),
        "auc_by_duration_bucket": {int(b): auc(y[dur_b == b], pred[dur_b == b, 0]) for b in np.unique(dur_b)},
    }
    if base is not None:
        residual = pred[:, 0] - predict_logged(base, data, pos, device)[:, 0]
        report["auc_residual"] = auc(y, residual)
        report["gauc_session_residual"] = gauc(y, residual, sessions)
    return report


class Rewards:
    def __init__(self, members, base, data):
        self.members = members
        self.base = base
        self.data = data
        self.far_len = max(m.far_len for m in members)

    def far(self, pos, device):
        if not self.far_len:
            return None
        return to_device(self.data.far(pos, self.members[0].cfg.ctx, self.far_len), device)

    @classmethod
    def load(cls, rm_dir, data, device):
        rm_dir = Path(rm_dir)
        members = [load(p, device).eval() for p in sorted(rm_dir.glob("member_*.pt"))]
        base_path = rm_dir / "base.pt"
        base = load(base_path, device).eval() if base_path.exists() else None
        return cls(members, base, data)

    def _score(self, model, batch, slates, protocol, predicted=None, far=None):
        b, n, k = slates.shape
        far = {key: v[:, -model.far_len:] for key, v in far.items()} if far is not None and model.far_len else None
        if protocol == "real":
            cand = self.data.item_fields(slates.reshape(b, n * k))
            return torch.sigmoid(model.score(batch, cand, far).float()).view(b, n, k, 2)
        far = repeat(far, n) if far is not None else None
        steps = []
        for j in range(k):
            hist = history_for_reward(batch, slates, j, protocol, self.data, predicted)
            cand = self.data.item_fields(slates[:, :, j].reshape(-1, 1))
            steps.append(torch.sigmoid(model.score(hist, cand, far).float()).view(b, n, 2))
        return torch.stack(steps, 2)

    @torch.no_grad()
    def scores(self, batch, slates, protocol="real", far=None):
        if protocol == "predicted":
            proxy = self._predicted_rollout(batch, slates, far)
        else:
            proxy = self._score(self.members[0], batch, slates, protocol, far=far)
        others = [self._score(m, batch, slates, protocol, proxy[..., 0], far) for m in self.members[1:]]
        ratio = torch.stack([proxy[..., 0]] + [o[..., 0] for o in others])
        out = {"ratio": ratio, "continue": proxy[..., 1],
               "dur": self.data.catalog_on(slates.device)["dur"][slates].float()}
        if self.base is not None:
            out["base"] = self._score(self.base, batch, slates, protocol, proxy[..., 0], far)[..., 0]
        return out

    def _predicted_rollout(self, batch, slates, far=None):
        b, n, k = slates.shape
        far = repeat(far, n) if far is not None else None
        predicted = torch.zeros(b, n, k, device=slates.device)
        steps = []
        for j in range(k):
            hist = history_for_reward(batch, slates, j, "predicted", self.data, predicted)
            cand = self.data.item_fields(slates[:, :, j].reshape(-1, 1))
            step = torch.sigmoid(self.members[0].score(hist, cand, far).float()).view(b, n, 2)
            predicted[:, :, j] = step[..., 0]
            steps.append(step)
        return torch.stack(steps, 2)


def per_item_reward(kind, scores, k_pess=1.0):
    ratio = scores["ratio"]
    if kind == "ratio":
        return ratio[0]
    if kind == "seconds":
        return ratio[0] * scores["dur"] / 240
    if kind == "norm":
        return ratio[0] - scores["base"]
    if kind == "pess":
        return ratio.mean(0) - k_pess * ratio.std(0)
    if kind == "continue":
        return scores["continue"]
    if kind == "hit":
        return torch.zeros_like(ratio[0])
    raise ValueError(kind)


def discounted(r, gamma=0.9):
    return (r * gamma ** torch.arange(r.shape[-1], device=r.device, dtype=r.dtype)).sum(-1)


def slate_reward(kind, scores, gamma=0.9, k_pess=1.0):
    return discounted(per_item_reward(kind, scores, k_pess), gamma)


def hit_reward(slates, future_items, future_ratio):
    match = slates[..., None] == future_items[:, None, None, :]
    return (match * future_ratio[:, None, None, :]).amax(-1).float() / 100


def judge_reward(scores):
    return scores["ratio"][1:].mean(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--sft", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--member", type=int, default=0)
    p.add_argument("--base", action="store_true")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--match-window", type=int, help="history length for track and artist match features")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    data = Data(a.data)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "base.pt" if a.base else f"member_{a.member}.pt"
    rm = train_reward_model(data, a.sft, out_dir / name, a.member, a.base, a.epochs, a.batch_size, a.lr, a.device,
                            a.match_window)
    base_path = out_dir / "base.pt"
    base = load(base_path, device_of(a.device)).eval() if base_path.exists() and not a.base else None
    report = rm_report(rm, data, "val", device_of(a.device), base)
    (out_dir / name.replace(".pt", "_val.json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
