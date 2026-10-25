# Воспроизведение

Три уровня проверки: пересчитать цифры отчёта по сохранённым выдачам top-10, оценить скачанные чекпоинты, обучить всё с нуля.

## 1. Окружение и данные

Python 3.12, CUDA-GPU для оценки и обучения.

```bash
pip install -r requirements.txt
python scripts/download_yambda.py
python -m rl.prepare --events data/yambda/flat/50m/multi_event.parquet \
    --artists data/yambda/artist_item_mapping.parquet --out data/full
```

`rl.prepare` строит массивы событий, каталог из 50 000 треков и запросы val/test с фиксированным seed, поэтому позиции запросов совпадают с сохранёнными выгрузками.

## 2. Артефакты из Releases

Архивы лежат в Releases репозитория. Все распаковываются в `outputs/`:

```bash
sha256sum -c SHA256SUMS.txt
for z in policies reward_models_v1 reward_models_v2 catboost_judge top10_exports; do unzip -q $z.zip -d outputs; done
```

| Архив | Размер | Куда | Что внутри |
|---|---|---|---|
| `policies.zip` | 1.4 ГБ | `outputs/runs/` | 17 политик: SFT; GRPO с RM: ratio β = 1 / 0.3 / 0.1 / 0, norm β=0.1; GRPO с RM v2: ratio β = 0.3 / 0.1 / 0.03; GRPO с hit-наградой: β = 0.1 / 0.03 / 0.01 и β=0.03 seed 1; RW-SFT λ = 0.5 / 0.25 / 0.5×2 эпохи; SFT ещё 2 эпохи |
| `reward_models_v1.zip` | 1.5 ГБ | `outputs/runs/rm/` | RM v1: ансамбль из 5 моделей и базовая модель без трека |
| `reward_models_v2.zip` | 0.7 ГБ | `outputs/runs/rm2/` | RM v2 (история 4096 событий) |
| `catboost_judge.zip` | 25 МБ | `outputs/catboost/` | CatBoost из первого задания и его предсказания на test для сверки |
| `top10_exports.zip` | 80 МБ | `outputs/results/` | top-10 каждой модели на всех запросах test и val |

Каждый `.pt` хранит архитектуру вместе с весами и загружается `rl.model.load(path)`.

Метрики всех запусков (JSON) и логи Kaggle лежат в репозитории, в `outputs/results/`. Откуда какая таблица [отчёта](REPORT.md):

| Таблица отчёта | Файлы в `outputs/results/` |
|---|---|
| качество RM v1 / v2 (раздел 4) | `final_v1/runs/rm/*_val.json`, `rm2_v1/runs/rm2/*_val.json` |
| 8.1-8.3, val, графики 1-2 | `final_v1/runs/eval/final_{0,1}.json`, CatBoost - `final_v1/runs/eval/judge_val.json` |
| 8.4, RM v2 | `rm2eval_v1/dl/runs/eval/rm2_{0,1}.json`, `rm2eval_v1/judge_val.json` |
| 8.5, test | `test_v1/eval/test_rm.json`, `test_v1/eval/test_rm2.json`, `test_v1/judge_test.json` |
| 8.6, пять шагов, график 3 | `test_v1/eval/multistep_rm.json`, `test_v1/eval/multistep_rm2.json` |
| 8.7, все запросы, GRPO с RM | `full_logs/{test,val}.json` |
| 8.8, hit-награда и RW-SFT | `night_v1/night_{test,val}.json`, `night_v1/judge_test.json` |
| 8.9, контроли | `control_v1/control_{test,val}.json` |
| значимость с поправкой Холма, график 4 | `significance.json` (строится `report/significance.py`) |

## 3. Пересчитать цифры отчёта

Нужны данные из шага 1 и `top10_exports.zip`.

```bash
python report/significance.py
python report/make_figures.py
```

- `significance.py` - таблица 4.3: Δ к SFT на всех ~54k запросах test и val, парный бутстрэп по пользователям (2000 повторов), поправка Холма внутри каждой выгрузки, сравнение GRPO с hit-наградой против «SFT ещё 2 эпохи». Результат в `outputs/results/significance.json`.
- `make_figures.py` - четыре графика в `report/`.

## 4. Оценить чекпоинты

Лучшая модель (GRPO с hit-наградой, β=0.03) против SFT на всех запросах test:

```bash
python -m rl.evaluate --data data/full --sft outputs/runs/sft_base/policy.pt --split test --max-queries 60000 \
    --policies outputs/runs/night/hit_b0.03/step_00600.pt --out eval_hit.json --export-topk top10_hit.npz
```

С `--rm-dir outputs/runs/rm` добавляются метрики reward model (proxy, KL, Best-of-N через `--best-of 4 16`). Пять шагов вперёд:

```bash
python -m rl.multistep --data data/full --sft outputs/runs/sft_base/policy.pt --rm-dir outputs/runs/rm --split test \
    --policies outputs/runs/grpo/ratio_b0.3/step_00600.pt outputs/runs/grpo/ratio_b0.1/step_00600.pt --out multistep.json
```

CatBoost-судья по выгрузке top-10 (код признаков первого задания лежит в `judge/task1/` в том виде, в каком обучалась модель):

```bash
python judge/catboost_judge.py --data data/full --topk top10_hit.npz --split test --max-queries 2000 --out judge.json
```

При первом запуске судья строит таблицу прослушиваний в `data/yambda/derived/` кодом первого задания. На test он сверяет свои предсказания для реально показанных треков с сохранёнными предсказаниями первого задания (`logged_check` в выводе).

На GPU без быстрого fp16 (GTX 10xx/16xx) стоит выставить `RL_FP16=0`.

## 5. Обучить с нуля

Порядок и параметры совпадают с запусками в отчёте.

```bash
D="--data data/full"; SFT=runs/sft_base/policy.pt

python -m rl.sft $D --out-dir runs/sft_base --size base --epochs 3

python -m rl.reward $D --sft $SFT --out-dir runs/rm --base
for m in 0 1 2 3 4; do python -m rl.reward $D --sft $SFT --out-dir runs/rm --member $m; done

mkdir -p runs/rm2 && cp runs/rm/base.pt runs/rm2/
for m in 0 1 2 3 4; do python -m rl.reward $D --sft $SFT --out-dir runs/rm2 --member $m --match-window 4096; done

G="python -m rl.grpo $D --sft $SFT --save-every 200"
for b in 0.0 0.01 0.03 0.1 0.3 1.0; do $G --rm-dir runs/rm --reward ratio --beta $b --out-dir runs/grpo/ratio_b$b; done
for k in seconds norm pess continue; do $G --rm-dir runs/rm --reward $k --beta 0.1 --out-dir runs/grpo/${k}_b0.1; done
for b in 0.3 0.1 0.03; do $G --rm-dir runs/rm2 --reward ratio --beta $b --out-dir runs/grpo_rm2/ratio_b$b; done
$G --rm-dir runs/rm --beta 0.1 --ce-weight 1 --hit-weight 1 --out-dir runs/fix/ce1_hit1

H="$G --rm-dir runs/rm --reward hit --hit-weight 1 --group 16"
$H --beta 0.1 --out-dir runs/night/hit_b0.1
$H --beta 0.03 --out-dir runs/night/hit_b0.03
$H --beta 0.03 --seed 1 --out-dir runs/control/hit_b0.03_s1
$H --beta 0.01 --out-dir runs/control/hit_b0.01

R="python -m rl.sft $D --init $SFT --epochs 1 --lr 1e-4"
$R --rw-lambda 0.5 --out-dir runs/fix/rwsft_l0.5
$R --rw-lambda 0.25 --out-dir runs/night/rwsft_l0.25
$R --rw-lambda 0.5 --epochs 2 --out-dir runs/night/rwsft_l0.5_e2
$R --epochs 2 --out-dir runs/control/sft_cont_e2
```

`--hit-weight` при `--reward hit` - вес награды за попадание в реальное будущее сессии; reward model в обучении не участвует, по `--rm-dir` только логируется proxy.

## 6. Kaggle

Все запуски в отчёте шли на Kaggle (2×T4, две очереди параллельно). В каждом ноутбуке видны код, логи обучения и оценки, а в выводе лежат чекпоинты и JSON с метриками. Ноутбуки собираются из пакета `rl/`: модули вставляются ячейками `%%writefile`, так что код в ноутбуке совпадает с кодом в репозитории.

```bash
python kaggle/build_notebook.py <этап>
kaggle kernels push -p kaggle/<этап>
```

Перед сборкой в `kaggle/build_notebook.py` нужно указать свой `USER` и датасет с файлами Yambda (`multi_event.parquet`, `artist_item_mapping.parquet`). Каждый этап берёт данные и чекпоинты из вывода предыдущих.

| Этап | Ноутбук | Что делает | Результаты в `outputs/results/` |
|---|---|---|---|
| `sft` | [recsys-rl-sft](https://www.kaggle.com/code/rabochi1024/recsys-rl-sft) | данные, SFT двух размеров, ансамбль RM v1, оценка SFT и Best-of-N | `sft_v2` |
| `grpo` | [recsys-rl-grpo](https://www.kaggle.com/code/rabochi1024/recsys-rl-grpo) | сетка наград и β | `grpo_v2` |
| `fix` | [recsys-rl-fix](https://www.kaggle.com/code/rabochi1024/recsys-rl-fix) | CE-якорь, hit-награда вместе с RM, RW-SFT | `fix_v1` |
| `final` | [recsys-rl-final](https://www.kaggle.com/code/rabochi1024/recsys-rl-final) | norm-варианты и переоценка всех чекпоинтов на val | `final_v1` |
| `rm2` | [recsys-rl-rm2](https://www.kaggle.com/code/rabochi1024/recsys-rl-rm2) | RM v2 с историей 4096 событий и GRPO с ней | `rm2_v1` |
| `rm2eval` | [recsys-rl-rm2-eval](https://www.kaggle.com/code/rabochi1024/recsys-rl-rm2-eval) | оценка GRPO с RM v2 | `rm2eval_v1` |
| `test` | [recsys-rl-test](https://www.kaggle.com/code/rabochi1024/recsys-rl-test) | оценка на test и пять шагов вперёд | `test_v1` |
| `night` | [recsys-rl-night](https://www.kaggle.com/code/rabochi1024/recsys-rl-night) | GRPO на hit-награде, подбор RW-SFT, оценка на всех запросах | `night_v1` |
| `control` | [recsys-rl-control](https://www.kaggle.com/code/rabochi1024/recsys-rl-control) | «SFT ещё 1-2 эпохи», второй seed и β=0.01 для hit-награды | `control_v1` |

`full_logs` - оценка политик с RM на всех запросах test и val, сделанная локально (`--max-queries 60000`, `RL_FP16=0`).
