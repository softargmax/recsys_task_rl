import math
import os
import random
import time

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_of(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def autocast(device):
    enabled = device.type == "cuda" and os.environ.get("RL_FP16", "1") == "1"
    return torch.autocast(device.type, dtype=torch.float16, enabled=enabled)


class Optimizer:
    def __init__(self, model, lr, steps, warmup=200, weight_decay=0.01, clip=1.0):
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=weight_decay)
        warmup = min(warmup, max(1, steps // 10))
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
        )
        self.scaler = torch.amp.GradScaler(enabled=next(model.parameters()).is_cuda)
        self.clip = clip

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def step(self, loss=None):
        if loss is not None:
            self.backward(loss)
        self.scaler.unscale_(self.opt)
        norm = torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        self.sched.step()
        return float(norm)


class Log:
    def __init__(self, every):
        self.every = every
        self.sums = {}
        self.n = 0
        self.started = time.time()

    def add(self, step, **values):
        for k, v in values.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
        self.n += 1
        if (step + 1) % self.every == 0:
            text = ", ".join(f"{k} {v / self.n:.4f}" for k, v in self.sums.items())
            print(f"step {step + 1}: {text}, {time.time() - self.started:.0f}s", flush=True)
            self.sums, self.n = {}, 0
