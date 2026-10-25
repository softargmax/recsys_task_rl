import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl.data import Data
from rl.evaluate import relevance

RES = ROOT / "outputs" / "results"
OUT = ROOT / "report"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "legend.frameon": False, "legend.labelcolor": INK,
})


def load(*paths):
    out = {}
    for p in paths:
        out.update(json.loads((RES / p).read_text()))
    return out


def stat(run, key):
    v = run["ranking"][key]
    return v["mean"] if isinstance(v, dict) else v


def overoptimization():
    v1 = load("final_v1/runs/eval/final_0.json", "final_v1/runs/eval/final_1.json")
    v2 = load("rm2eval_v1/dl/runs/eval/rm2_0.json", "rm2eval_v1/dl/runs/eval/rm2_1.json")
    j1 = load("final_v1/runs/eval/judge_val.json")
    j2 = load("rm2eval_v1/judge_val.json")
    curves = [
        ("RM v1 (256 событий)", BLUE, "o", v1, j1, "", [1.0, 0.3, 0.1, 0.03, 0.01, 0.0]),
        ("RM v2 (4096 событий)", ORANGE, "s", v2, j2, "rm2_", [0.3, 0.1, 0.03]),
    ]
    sft_ndcg = stat(v1["sft"], "lp_future_ndcg@10")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for label, color, marker, ev, judge, prefix, betas in curves:
        names = [f"ratio_b{b}/step_00600" for b in betas]
        kl = np.array([max(stat(ev[n], "kl_first_step"), 1e-3) for n in names])
        jd = np.array([judge[prefix + n.replace("/", "__")]["delta_vs_sft"]["mean"] for n in names])
        nd = np.array([100 * (stat(ev[n], "lp_future_ndcg@10") / sft_ndcg - 1) for n in names])
        px = np.array([stat(ev[n], "top10_proxy") - stat(ev["sft"], "top10_proxy") for n in names])
        for ax, y in zip(axes, (px, jd, nd)):
            ax.plot(kl, y, color=color, marker=marker, lw=2, ms=6, label=label)
        for x, y, b in zip(kl, jd, betas):
            if b == 0.01:
                continue
            axes[1].annotate(f"β={b:g}", (x, y), textcoords="offset points", xytext=(4, 6), fontsize=8, color=MUTED)
    titles = ["Proxy (оптимизируемая награда), Δ к SFT", "CatBoost-судья, Δ P(Listen+) к SFT",
              "NDCG@10 по дослушанным, Δ % к SFT"]
    for ax, t in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_xlabel("KL к SFT (лог. шкала)")
        ax.set_title(t, loc="left")
        ax.axhline(0, color=MUTED, lw=0.8)
    axes[0].legend(loc="upper left")
    fig.suptitle("Переоптимизация: proxy растёт монотонно, независимый судья - до KL ≈ 1, точность по логам падает",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_overoptimization.png", dpi=150)


def behavior():
    v1 = load("final_v1/runs/eval/final_0.json", "final_v1/runs/eval/final_1.json")
    betas = [1.0, 0.3, 0.1, 0.03, 0.01, 0.0]
    names = ["sft"] + [f"ratio_b{b}/step_00600" for b in betas]
    labels = ["SFT"] + [f"{b:g}" for b in betas]
    x = np.arange(len(names))
    panels = [("duration_s", "Средняя длительность трека в top-10, с", 1),
              ("top100_share", "Доля top-10 из 100 самых частых треков, %", 100),
              ("repeat_share", "Доля повторов из истории, %", 100)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (key, title, scale) in zip(axes, panels):
        y = np.array([stat(v1[n], key) * scale for n in names])
        ax.bar(x, y, color=[MUTED] + [BLUE] * len(betas), width=0.6)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8,
                        color=INK)
        ax.set_xticks(x, labels)
        ax.set_xlabel("β (сила KL-штрафа)")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Что делает RL с наградой ratio при ослаблении KL (val): короткие, популярные, одинаковые для всех треки",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_behavior.png", dpi=150)


def five_steps():
    r = json.loads((RES / "test_v1/eval/multistep_rm.json").read_text())
    steps = np.arange(1, 6)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    sim = r["simulator"]
    for key, label, color, marker in (("teacher", "реальные отклики", BLUE, "o"),
                                      ("listened", "«всё дослушано» (P1)", ORANGE, "s"),
                                      ("predicted", "свои предсказания RM (P2)", AQUA, "^")):
        axes[0].plot(steps, sim[key]["bias"], color=color, marker=marker, lw=2, ms=6, label=label)
    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].set_title("Смещение RM на k-м реальном треке цепочки", loc="left")
    axes[0].set_ylabel("предсказание − реальность")
    pol = r["policy"]["sft"]
    for key, label, color, marker in (("teacher", "реальные треки с откликом", BLUE, "o"),
                                      ("hidden_feedback", "реальные треки без отклика", ORANGE, "s"),
                                      ("generated", "собственные выдачи политики", AQUA, "^")):
        axes[1].plot(steps, pol[key], color=color, marker=marker, lw=2, ms=6, label=label)
    axes[1].set_title("NDCG@10 SFT для k-го реального трека", loc="left")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel("шаг k")
        ax.set_xticks(steps)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("Пять шагов вперёд (test, 5000 реальных цепочек)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_five_steps.png", dpi=150)


def paired(diff, users, n_boot=1000):
    keep = np.isfinite(diff)
    diff, users = diff[keep], users[keep]
    uniq, inv = np.unique(users, return_inverse=True)
    s, c = np.bincount(inv, weights=diff), np.bincount(inv)
    idx = np.random.default_rng(0).integers(0, len(uniq), size=(n_boot, len(uniq)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return diff.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


def log_gains():
    data = Data(ROOT / "data" / "full")
    models = [("control", "hit_b0.01__step_00600", "GRPO, hit-награда, β=0.01"),
              ("night", "hit_b0.03__step_00600", "GRPO, hit-награда, β=0.03"),
              ("control", "hit_b0.03_s1__step_00600", "GRPO, hit-награда, β=0.03, seed 1"),
              ("night", "hit_b0.1__step_00600", "GRPO, hit-награда, β=0.1"),
              ("night", "rwsft_l0.5_e2__policy", "RW-SFT, λ=0.5, 2 эпохи"),
              ("night", "rwsft_l0.25__policy", "RW-SFT, λ=0.25"),
              ("control", "sft_cont_e2__policy", "SFT ещё 2 эпохи (контроль)"),
              ("full", "ratio_b1.0__step_00600", "GRPO, RM, β=1"),
              ("full", "rm2_ratio_b0.3__step_00600", "GRPO, RM v2, β=0.3"),
              ("full", "ratio_b0.3__step_00600", "GRPO, RM, β=0.3")]
    metrics = [("lp_future_ndcg@10", "по дослушанным трекам сессии"), ("future_ndcg@10", "по всем трекам сессии")]
    res = {}
    for split in ("test", "val"):
        z = {"full": np.load(RES / f"full_logs/{split}_top10.npz"),
             "night": np.load(RES / f"night_v1/night_{split}_top10.npz"),
             "control": np.load(RES / f"control_v1/control_{split}_top10.npz")}
        pos = z["night"]["positions"]
        assert all((pos == v["positions"]).all() for v in z.values())
        users = data.user_of(pos)
        base = {k: relevance(z[k]["sft"], data, pos) for k in z}
        for src, key, _ in models:
            r = relevance(z[src][key], data, pos)
            for metric, _ in metrics:
                mean, lo, hi = paired(r[metric] - base[src][metric], users)
                b0 = np.nanmean(base[src][metric])
                res.setdefault((split, metric), []).append((100 * mean / b0, 100 * lo / b0, 100 * hi / b0))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, (metric, title) in zip(axes, metrics):
        for off, split, color, marker in ((0.17, "test", BLUE, "o"), (-0.17, "val", ORANGE, "s")):
            m = np.array(res[(split, metric)])
            ax.errorbar(m[:, 0], y + off, xerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt=marker, color=color,
                        ecolor=color, elinewidth=1.5, capsize=3, ms=6, label=split)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(f"Δ NDCG@10 {title}, %", loc="left")
        ax.set_xlabel("разница с SFT, 95% интервал (бутстрэп по пользователям)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [m[2] for m in models])
    axes[0].legend(loc="lower right")
    fig.suptitle("Прирост по логам на всех ~54k запросах test и val: выигрывает только награда из логов",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_log_gains.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    overoptimization()
    behavior()
    five_steps()
    log_gains()
    print("figures written to", OUT)
