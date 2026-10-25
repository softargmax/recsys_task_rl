import argparse
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
MODULES = sorted(p.name for p in (ROOT / "rl").glob("*.py"))
USER = "rabochi1024"
DATASET = f"{USER}/yambda-50m-flat"

SETUP = """
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

WORK = Path("/kaggle/working")
sys.path.insert(0, str(WORK / "code"))
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)


def run_queue(commands, gpu, log_name):
    with open(WORK / "logs" / log_name, "w") as log:
        for cmd in commands:
            log.write(f"$ {cmd}\\n")
            log.flush()
            code = subprocess.run(f"CUDA_VISIBLE_DEVICES={gpu} {cmd}", shell=True, cwd=WORK / "code",
                                  stdout=log, stderr=subprocess.STDOUT).returncode
            log.write(f"exit {code}\\n")


def run_parallel(queues):
    (WORK / "logs").mkdir(exist_ok=True)
    threads = [threading.Thread(target=run_queue, args=(cmds, gpu, f"gpu{gpu}_{i}.log"))
               for i, (gpu, cmds) in enumerate(queues)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i, (gpu, _) in enumerate(queues):
        text = (WORK / "logs" / f"gpu{gpu}_{i}.log").read_text()
        print(f"gpu {gpu}:")
        print("\\n".join(line for line in text.splitlines() if not line.startswith("step ")))
"""

COPY_PREVIOUS = """
prev = next(Path("/kaggle/input").rglob("runs")).parent
!cp -r {prev}/data {prev}/runs /kaggle/working/
SFT = "../runs/sft_base/policy.pt"
"""

RM2_EVAL = """
policies = [str(p) for p in sorted((WORK / "runs" / "grpo_rm2").glob("*/step_*.pt"))]
evaluate = f"python -m rl.evaluate --data ../data --sft {SFT} --rm-dir ../runs/rm2 --max-queries 10000 --slate-queries 2000"
run_parallel([
    (0, [f"{evaluate} --best-of 4 16 --policies {' '.join(policies[0::2])} "
         "--out ../runs/eval/rm2_0.json --export-topk ../runs/eval/rm2_top10_0.npz"]),
    (1, [f"{evaluate} --policies {' '.join(policies[1::2])} --out ../runs/eval/rm2_1.json "
         "--export-topk ../runs/eval/rm2_top10_1.npz"]),
])
"""

STAGES = {
    "sft": {
        "title": "recsys-rl-sft",
        "sources": [],
        "intro": """
# RL для музыкальных рекомендаций: данные, SFT и reward model

Сгенерировано `kaggle/build_notebook.py` из пакета `rl/`.

1. `rl.prepare`: события Yambda-50M -> массивы `npy`, каталог из 50k треков, запросы для оценки.
2. SFT GPT-2 двух размеров параллельно на двух T4, выбор по NDCG@10 на val.
3. Reward model: baseline по длительности и ансамбль из 5 членов.
4. Оценка SFT, бейзлайнов и best-of-N на val.
""",
        "cells": [
            """
events = next(Path("/kaggle/input").rglob("multi_event.parquet"))
artists = next(Path("/kaggle/input").rglob("artist_item_mapping.parquet"))
!cd /kaggle/working/code && python -m rl.prepare --events {events} --artists {artists} --out /kaggle/working/data
""",
            """
run_parallel([
    (0, ["python -m rl.sft --data ../data --out-dir ../runs/sft_base --size base --epochs 3"]),
    (1, ["python -m rl.sft --data ../data --out-dir ../runs/sft_large --size large --epochs 3 --batch-size 32"]),
])
scores = {}
for name in ("sft_base", "sft_large"):
    history = json.loads((WORK / "runs" / name / "history.json").read_text())
    scores[name] = max(h["next_ndcg@10"] for h in history)
    print(name, json.dumps(history, indent=1))
best = "sft_large" if scores["sft_large"] > 1.02 * scores["sft_base"] else "sft_base"
shutil.copytree(WORK / "runs" / best, WORK / "runs" / "sft")
print("selected", best, scores)
""",
            """
rm = "python -m rl.reward --data ../data --sft ../runs/sft/policy.pt --out-dir ../runs/rm"
run_parallel([
    (0, [f"{rm} --base", f"{rm} --member 0", f"{rm} --member 2"]),
    (1, [f"{rm} --member 1", f"{rm} --member 3", f"{rm} --member 4"]),
])
for path in sorted((WORK / "runs" / "rm").glob("*_val.json")):
    print(path.name, path.read_text())
""",
            """
!cd /kaggle/working/code && python -m rl.evaluate --data ../data --sft ../runs/sft/policy.pt --rm-dir ../runs/rm --baselines --best-of 1 4 16 64 --out ../runs/eval/sft_val.json
""",
        ],
    },
    "grpo": {
        "title": "recsys-rl-grpo",
        "sources": [f"{USER}/recsys-rl-sft"],
        "intro": """
# RL для музыкальных рекомендаций: GRPO

Использует данные, SFT и reward model из ноутбука `recsys-rl-sft`. Сетка по beta и видам reward идёт двумя очередями, по одной на каждую T4.
""",
        "cells": [
            COPY_PREVIOUS,
            """
grpo = f"python -m rl.grpo --data ../data --sft {SFT} --rm-dir ../runs/rm --save-every 200"
grid = [("ratio", b) for b in (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)] + [(k, 0.1) for k in ("seconds", "norm", "pess", "continue")]
commands = [f"{grpo} --reward {k} --beta {b} --out-dir ../runs/grpo/{k}_b{b}" for k, b in grid]
run_parallel([(0, commands[0::2]), (1, commands[1::2])])
""",
            """
policies = [str(p) for p in sorted((WORK / "runs" / "grpo").glob("*/step_*.pt"))]
evaluate = f"python -m rl.evaluate --data ../data --sft {SFT} --rm-dir ../runs/rm --max-queries 10000 --slate-queries 2000"
run_parallel([
    (0, [f"{evaluate} --baselines --best-of 1 4 16 64 --policies {' '.join(policies[0::2])} --out ../runs/eval/val_0.json"]),
    (1, [f"{evaluate} --policies {' '.join(policies[1::2])} --out ../runs/eval/val_1.json"]),
])
""",
        ],
    },
    "fix": {
        "title": "recsys-rl-fix",
        "sources": [f"{USER}/recsys-rl-sft"],
        "intro": """
# RL для музыкальных рекомендаций: защита от взлома награды

Два способа привязать RL к реальным логам при beta = 0.1: CE на реальных следующих треках и награда за попадание
в реальное будущее сессии. Плюс reward-weighted SFT без reward model как контроль.
""",
        "cells": [
            COPY_PREVIOUS,
            """
grpo = f"python -m rl.grpo --data ../data --sft {SFT} --rm-dir ../runs/rm --save-every 200 --beta 0.1"
runs = {"ce0.3": "--ce-weight 0.3", "ce1": "--ce-weight 1", "hit1": "--hit-weight 1",
        "ce1_hit1": "--ce-weight 1 --hit-weight 1"}
commands = [f"{grpo} {args} --out-dir ../runs/fix/{name}" for name, args in runs.items()]
rwsft = f"python -m rl.sft --data ../data --init {SFT} --rw-lambda 0.5 --epochs 1 --lr 1e-4 --out-dir ../runs/fix/rwsft_l0.5"
run_parallel([(0, [commands[0], commands[2], rwsft]), (1, [commands[1], commands[3]])])
""",
            """
policies = [str(p) for p in sorted((WORK / "runs" / "fix").glob("*/*.pt"))]
evaluate = f"python -m rl.evaluate --data ../data --sft {SFT} --rm-dir ../runs/rm --max-queries 10000 --slate-queries 2000"
run_parallel([
    (0, [f"{evaluate} --policies {' '.join(policies[0::2])} --out ../runs/eval/fix_0.json"]),
    (1, [f"{evaluate} --policies {' '.join(policies[1::2])} --out ../runs/eval/fix_1.json"]),
])
""",
        ],
    },
    "rm2": {
        "title": "recsys-rl-rm2",
        "sources": [f"{USER}/recsys-rl-sft"],
        "intro": """
# RL для музыкальных рекомендаций: reward model с длинной историей

Признаки совпадения кандидата с историей считаются по последним 4096 событиям вместо 256: так reward model видит
94% повторов вместо 20%. Ансамбль из 5 членов обучается заново, затем GRPO с новой наградой на тех же beta.
""",
        "cells": [
            COPY_PREVIOUS,
            """
(WORK / "runs" / "rm2").mkdir(parents=True, exist_ok=True)
shutil.copy(WORK / "runs" / "rm" / "base.pt", WORK / "runs" / "rm2" / "base.pt")
rm = f"python -m rl.reward --data ../data --sft {SFT} --out-dir ../runs/rm2 --match-window 4096"
run_parallel([
    (0, [f"{rm} --member 0", f"{rm} --member 2", f"{rm} --member 4"]),
    (1, [f"{rm} --member 1", f"{rm} --member 3"]),
])
for name in ("rm", "rm2"):
    print(name, (WORK / "runs" / name / "member_0_val.json").read_text())
""",
            """
grpo = f"python -m rl.grpo --data ../data --sft {SFT} --rm-dir ../runs/rm2 --save-every 200"
grid = [("ratio", 0.3), ("ratio", 0.1), ("ratio", 0.03), ("norm", 0.1)]
commands = [f"{grpo} --reward {k} --beta {b} --out-dir ../runs/grpo_rm2/{k}_b{b}" for k, b in grid]
run_parallel([(0, commands[0::2]), (1, commands[1::2])])
""",
            RM2_EVAL,
        ],
    },
    "rm2eval": {
        "title": "recsys-rl-rm2-eval",
        "sources": [f"{USER}/recsys-rl-rm2"],
        "intro": """
# RL для музыкальных рекомендаций: оценка GRPO с reward model на длинной истории

Повтор последнего шага `recsys-rl-rm2`: там оценка упала на сохранении top-10 в несуществующую папку.
""",
        "cells": [COPY_PREVIOUS, RM2_EVAL],
    },
    "test": {
        "title": "recsys-rl-test",
        "sources": [f"{USER}/recsys-rl-final", f"{USER}/recsys-rl-rm2"],
        "intro": """
# RL для музыкальных рекомендаций: финальная оценка на test и пять шагов

1. SFT, бейзлайны и ключевые чекпоинты GRPO на test по тому же протоколу, что и на val, с выгрузкой top-10 для CatBoost-судьи.
2. `rl.multistep`: ошибка reward model и политики по шагам на реальных цепочках из 5 рекомендаций и поведение SFT и RL
   при генерации пяти треков подряд, для старой reward model и для reward model с длинной историей.
""",
        "cells": [
            """
def source_with(name):
    return next(p for p in Path("/kaggle/input").rglob("runs") if (p / name).is_dir()).parent

final_root, rm2_root = source_with("grpo"), source_with("rm2")
!cp -r {final_root}/data {final_root}/runs /kaggle/working/
!cp -r {rm2_root}/runs/rm2 {rm2_root}/runs/grpo_rm2 /kaggle/working/runs/
SFT = "../runs/sft_base/policy.pt"
""",
            """
runs = WORK / "runs"
old = [runs / "grpo" / f"{r}/step_00600.pt" for r in
       ("ratio_b1.0", "ratio_b0.3", "ratio_b0.1", "ratio_b0.0", "norm_b0.1", "seconds_b0.1")]
old += [runs / "fix" / "ce1_hit1" / "step_00600.pt", runs / "fix" / "rwsft_l0.5" / "policy.pt"]
new = [runs / "grpo_rm2" / f"{r}/step_00600.pt" for r in ("ratio_b0.3", "ratio_b0.1", "ratio_b0.03", "norm_b0.1")]
evaluate = f"python -m rl.evaluate --data ../data --sft {SFT} --split test --max-queries 10000 --slate-queries 2000"
multistep = f"python -m rl.multistep --data ../data --sft {SFT} --split test"
paths = lambda ps: " ".join(str(p) for p in ps)
run_parallel([
    (0, [f"{evaluate} --rm-dir ../runs/rm --baselines --best-of 4 16 --policies {paths(old)} "
         "--out ../runs/eval/test_rm.json --export-topk ../runs/eval/test_top10_rm.npz",
         f"{multistep} --rm-dir ../runs/rm --policies {paths(old[1:4])} --out ../runs/eval/multistep_rm.json"]),
    (1, [f"{evaluate} --rm-dir ../runs/rm2 --policies {paths(new)} "
         "--out ../runs/eval/test_rm2.json --export-topk ../runs/eval/test_top10_rm2.npz",
         f"{multistep} --rm-dir ../runs/rm2 --policies {paths(new[:3])} --out ../runs/eval/multistep_rm2.json"]),
])
""",
        ],
    },
    "night": {
        "title": "recsys-rl-night",
        "sources": [f"{USER}/recsys-rl-final"],
        "intro": """
# RL для музыкальных рекомендаций: награда из логов против выученной награды

1. Reward-weighted SFT с разными lambda: единственный прирост, подтверждённый логами.
2. GRPO только на hit-награде (попал ли трек в реальное будущее сессии), без reward model.
3. Повтор ratio beta 0.3 с другим seed.
4. Оценка на всех запросах test и val, только по логам.
""",
        "cells": [
            COPY_PREVIOUS,
            """
rwsft = f"python -m rl.sft --data ../data --init {SFT} --epochs 1 --lr 1e-4"
grpo = f"python -m rl.grpo --data ../data --sft {SFT} --rm-dir ../runs/rm --save-every 200"
run_parallel([
    (0, [f"{rwsft} --rw-lambda 0.25 --out-dir ../runs/night/rwsft_l0.25",
         f"{rwsft} --rw-lambda 1.0 --out-dir ../runs/night/rwsft_l1.0",
         f"{rwsft} --rw-lambda 0.5 --epochs 2 --out-dir ../runs/night/rwsft_l0.5_e2"]),
    (1, [f"{grpo} --reward hit --hit-weight 1 --group 16 --beta 0.1 --out-dir ../runs/night/hit_b0.1",
         f"{grpo} --reward hit --hit-weight 1 --group 16 --beta 0.03 --out-dir ../runs/night/hit_b0.03",
         f"{grpo} --reward ratio --beta 0.3 --seed 1 --out-dir ../runs/night/ratio_b0.3_s1"]),
])
""",
            """
runs = WORK / "runs"
policies = sorted(runs.glob("night/rwsft*/policy.pt")) + sorted(runs.glob("night/*/step_00600.pt"))
policies += [runs / "fix" / "rwsft_l0.5" / "policy.pt", runs / "grpo" / "ratio_b0.3" / "step_00600.pt"]
evaluate = f"python -m rl.evaluate --data ../data --sft {SFT} --max-queries 60000 --policies {' '.join(map(str, policies))}"
run_parallel([
    (0, [f"{evaluate} --split test --out ../runs/eval/night_test.json --export-topk ../runs/eval/night_test_top10.npz"]),
    (1, [f"{evaluate} --split val --out ../runs/eval/night_val.json --export-topk ../runs/eval/night_val_top10.npz"]),
])
""",
        ],
    },
    "control": {
        "title": "recsys-rl-control",
        "sources": [f"{USER}/recsys-rl-final", f"{USER}/recsys-rl-night"],
        "intro": """
# RL для музыкальных рекомендаций: контроли для GRPO на hit-награде

1. Обычный SFT, дообученный ещё на 1 и 2 эпохи: даёт ли тот же прирост просто дополнительное обучение.
2. GRPO на hit-награде с другим seed и с меньшим beta.
3. Оценка на всех запросах test и val вместе с моделями ночного запуска.
""",
        "cells": [
            """
def source_with(name):
    return next(p for p in Path("/kaggle/input").rglob("runs") if (p / name).is_dir()).parent

final_root, night_root = source_with("grpo"), source_with("night")
!cp -r {final_root}/data {final_root}/runs /kaggle/working/
!cp -r {night_root}/runs/night /kaggle/working/runs/
SFT = "../runs/sft_base/policy.pt"
""",
            """
sft = f"python -m rl.sft --data ../data --init {SFT} --lr 1e-4"
grpo = f"python -m rl.grpo --data ../data --sft {SFT} --rm-dir ../runs/rm --save-every 200 --reward hit --hit-weight 1 --group 16"
run_parallel([
    (0, [f"{sft} --epochs 1 --out-dir ../runs/control/sft_cont_e1",
         f"{sft} --epochs 2 --out-dir ../runs/control/sft_cont_e2",
         f"{grpo} --beta 0.01 --out-dir ../runs/control/hit_b0.01"]),
    (1, [f"{grpo} --beta 0.03 --seed 1 --out-dir ../runs/control/hit_b0.03_s1"]),
])
""",
            """
runs = WORK / "runs"
policies = sorted(runs.glob("control/sft_cont*/policy.pt")) + sorted(runs.glob("control/hit*/step_00600.pt"))
policies += [runs / "night" / "hit_b0.03" / "step_00600.pt", runs / "night" / "rwsft_l0.5_e2" / "policy.pt"]
evaluate = f"python -m rl.evaluate --data ../data --sft {SFT} --max-queries 60000 --policies {' '.join(map(str, policies))}"
run_parallel([
    (0, [f"{evaluate} --split test --out ../runs/eval/control_test.json --export-topk ../runs/eval/control_test_top10.npz"]),
    (1, [f"{evaluate} --split val --out ../runs/eval/control_val.json --export-topk ../runs/eval/control_val_top10.npz"]),
])
""",
        ],
    },
    "final": {
        "title": "recsys-rl-final",
        "sources": [f"{USER}/recsys-rl-grpo", f"{USER}/recsys-rl-fix"],
        "intro": """
# RL для музыкальных рекомендаций: норм-награда с якорями и переоценка всех чекпоинтов

1. Норм-награда (дослушивание относительно ожидаемого для такой длины трека) при beta 0.3 и вместе с CE-якорем при beta 0.1.
2. Все чекпоинты GRPO, CE-якоря, награды за попадание, reward-weighted SFT и новые запуски оцениваются заново с метриками
   по дослушанным целям, парными интервалами относительно SFT и выгрузкой top-10 для CatBoost-судьи.
""",
        "cells": [
            """
def source_with(name):
    return next(p for p in Path("/kaggle/input").rglob("runs") if (p / name).is_dir()).parent

grpo_root, fix_root = source_with("grpo"), source_with("fix")
!cp -r {grpo_root}/data {grpo_root}/runs /kaggle/working/
!cp -r {fix_root}/runs/fix /kaggle/working/runs/
SFT = "../runs/sft_base/policy.pt"
""",
            """
grpo = f"python -m rl.grpo --data ../data --sft {SFT} --rm-dir ../runs/rm --save-every 200 --reward norm"
run_parallel([
    (0, [f"{grpo} --beta 0.3 --out-dir ../runs/fix2/norm_b0.3"]),
    (1, [f"{grpo} --beta 0.1 --ce-weight 1 --out-dir ../runs/fix2/norm_ce1_b0.1"]),
])
""",
            """
runs = WORK / "runs"
policies = [str(p) for p in sorted(runs.glob("grpo/*/step_*.pt")) + sorted(runs.glob("fix/*/step_*.pt"))
            + sorted(runs.glob("fix/rwsft*/policy.pt")) + sorted(runs.glob("fix2/*/step_*.pt"))]
evaluate = f"python -m rl.evaluate --data ../data --sft {SFT} --rm-dir ../runs/rm --max-queries 10000 --slate-queries 2000"
run_parallel([
    (0, [f"{evaluate} --baselines --best-of 1 4 16 64 --policies {' '.join(policies[0::2])} "
         "--out ../runs/eval/final_0.json --export-topk ../runs/eval/top10_0.npz"]),
    (1, [f"{evaluate} --policies {' '.join(policies[1::2])} --out ../runs/eval/final_1.json "
         "--export-topk ../runs/eval/top10_1.npz"]),
])
""",
        ],
    },
}


def build(stage):
    spec = STAGES[stage]
    kernel_dir = ROOT / "kaggle" / stage
    kernel_dir.mkdir(parents=True, exist_ok=True)
    cells = [nbf.v4.new_markdown_cell(spec["intro"].strip()),
             nbf.v4.new_code_cell("!pip install -q polars==1.38.1"),
             nbf.v4.new_code_cell(SETUP.strip())]
    cells.append(nbf.v4.new_code_cell("!mkdir -p /kaggle/working/code/rl && touch /kaggle/working/code/rl/__init__.py"))
    for name in MODULES:
        source = (ROOT / "rl" / name).read_text()
        if source:
            cells.append(nbf.v4.new_code_cell(f"%%writefile /kaggle/working/code/rl/{name}\n{source}"))
    cells += [nbf.v4.new_code_cell(c.strip()) for c in spec["cells"]]
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    notebook = f"{spec['title']}.ipynb"
    nbf.write(nb, kernel_dir / notebook)
    metadata = {
        "id": f"{USER}/{spec['title']}", "title": spec["title"], "code_file": notebook, "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET], "competition_sources": [],
        "kernel_sources": spec["sources"],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"written {len(cells)} cells -> {kernel_dir / notebook}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=sorted(STAGES))
    build(p.parse_args().stage)


if __name__ == "__main__":
    main()
