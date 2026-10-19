# Configuration Reference (XGBoost)

Where every training setting lives and how to change it. Read this only when you want to change a setting. The commands to run are in [`PIPELINE_FLOW.md`](PIPELINE_FLOW.md).

There are two layers:

```text
Layer 1  DEFAULT   params.yaml -> train:        edit the file by hand. Tracked by DVC.
            |  overridden by
Layer 2  ONE-OFF   --set key=value              typed on the command line. Tracked by MLflow only.
            |
            v
      final config -> train.py / set_champion.py -> saved in the MLflow run as config.yaml
```

Precedence: `--set` beats `params.yaml`. There are no profile files; to try other parameters you edit `params.yaml` or use `--set`.

---

## 1. `params.yaml -> train`

```yaml
train:
  data:
    path: data/processed/dataset.csv   # DVC-tracked via dataset.csv.dvc
    target: Label
    drop_columns: [Stime, Ltime]       # timestamps are not features
    float_columns: [dur, sbytes, dbytes, Sload, Dload, Spkts, Dpkts, smeansz, dmeansz]
    test_size: 0.2
    seed: 42
  xgb:                                 # passed straight to xgboost.XGBClassifier(**xgb)
    tree_method: hist
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.3
    random_state: 42
  mlflow:
    tracking_uri: sqlite:///mlflow.db  # relative paths are resolved from the project root
    experiment: ids-unsw-nb15
    model_name: ids-xgboost
    register: true                     # false = log the run but do not create a registry version
    run_name: null                     # null -> "xgb-<data version tag>"
  promote:                             # gate used by set_champion.py --eval
    metric: recall                     # primary: attack detection
    min_gain: 0.002                    # challenger must beat the champion by at least this much
    guard_metric: f1
    max_guard_drop: 0.01               # ...without losing more than this on the guard metric
```

| Section | Used by | What it controls | Retriggers `dvc repro train`? |
|---|---|---|---|
| `data` | `train.py`, `set_champion.py` | which CSV, which columns, the split | yes |
| `xgb` | `train.py` | XGBoost hyperparameters. **Add any `XGBClassifier` argument here** | yes |
| `mlflow` | `train.py`, `set_champion.py` | where to log, the registry name, whether to register | no |
| `promote` | `set_champion.py --eval` | the promotion rule | no |

`split:` (at the top of `params.yaml`) is unrelated to training. It only controls how the dataset is cut into the base set and the stream batches.

### Adding or changing XGBoost parameters

Add the key under `xgb:`. Any argument of `xgboost.XGBClassifier` works, for example:

```yaml
  xgb:
    tree_method: hist
    n_estimators: 300
    max_depth: 8
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
    scale_pos_weight: 12      # roughly negatives / positives; helps recall on the attack class
    random_state: 42
```

### Why `float_columns` exists

The MLflow model signature is strict: it rejects `int64` for a `double` column and `float64` for a `long` column. The same column can be an integer in one data version and a float in another (live Zeek rows are floats, UNSW rows are integers). `train.py` and `set_champion.py` therefore cast these nine columns to `float64` before they use the data, so the signature always says `double` for them. **Anything that calls the model (your own prediction code, a serving client) must cast the same columns to `float64` first.**

---

## 2. One-off overrides: `--set`

```powershell
python src/production/train.py --set xgb.max_depth=10 --set xgb.n_estimators=200 --set mlflow.run_name=xgb-deep
python src/production/train.py --set mlflow.register=false          # smoke test: logged, but no registry version
python src/production/set_champion.py --eval <csv> --set promote.min_gain=0
```

- Keys are dotted paths relative to `train`.
- Values are parsed as YAML: `10` is an int, `0.05` a float, `false` a bool, `[a,b]` a list.
- `--set` is not seen by DVC. With `dvc exp run`, use the full path instead: `-S train.xgb.max_depth=10`.
- Every override is recorded in the run (tag `set_overrides`, and the full result in the `config.yaml` artifact).

### Which way should I change a setting?

| I want to... | Do this |
|---|---|
| Change what the pipeline trains by default | Edit `params.yaml`, then `dvc repro train` |
| Try a quick variation and keep the result in MLflow only | `train.py --set ...` |
| Try variations and let DVC remember each one | `dvc exp run -S train.xgb.max_depth=10 -n xgb-deep` |
| Run a throw-away test without polluting the registry | `train.py --set mlflow.register=false` |

---

## 3. The files behind this

| File | Role |
|---|---|
| `params.yaml` | the defaults (above) |
| `src/config.py` | `load_config()`: reads `params.yaml -> train`, applies `--set` overrides. Also `flatten()` and `resolve_tracking_uri()` |
| `src/production/train.py` | reads the config, trains, logs to MLflow, writes `metrics/train_metrics.json` |
| `src/production/set_champion.py` | reads the same config for the model name, tracking URI, float columns and the `promote` gate |
| `dvc.yaml` (stage `train`) | tracks `train.data` and `train.xgb` as the stage's parameters |
