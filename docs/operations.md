# Operations Guide

Run commands from the repository root with a Python 3.11 environment active.
This guide consolidates the former source and production README instructions.

## Environment and storage setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-automation.txt
```

Use `requirements-test.txt` for unit tests alone, or `requirements.txt` for the
full exploration environment. These profiles serve different purposes; the test
and automation profiles share pinned versions through the full requirements file.
For uv environments without pip, use
`uv pip install --python .venv/bin/python -r requirements-automation.txt`.
The [dev container](../.devcontainer/devcontainer.json) installs the full environment.

The DVC remote is `s3://dvc-store`, served locally by SeaweedFS:

1. Copy `.env.example` to `.env` and choose the S3 access and secret keys.
2. Start storage with `docker compose up -d`.
3. Configure matching credentials locally:

   ```bash
   dvc remote modify --local s3store access_key_id '<access-key>'
   dvc remote modify --local s3store secret_access_key '<secret-key>'
   ```

Keep `.env` and `.dvc/config.local` out of Git. The endpoint defaults to
`http://127.0.0.1:8333`. Restore the active training dataset with
`dvc pull data/processed/dataset.csv.dvc`, or run `dvc pull` to restore all
versioned outputs. The remote must already contain the requested objects.

## Offline data and streaming simulation

Raw UNSW-NB15 partitions and their feature definition belong in `data/raw/`.
Existing `.dvc` pointers identify the required files.

```bash
dvc repro preprocess split
python src/simulation/simulate_stream.py --mode list-stream
```

The `preprocess` stage cleans the four CSV partitions and writes
`data/processed/preprocessed_data.csv`. The `split` stage sorts by `Stime`,
uses `split.train_ratio` for `base_train.csv`, and divides the remaining records
into `split.num_batches` stream batches. Defaults are 50% base training and
seven batches, configured in [params.yaml](../params.yaml).

To merge a selected batch into the active dataset:

```bash
python src/simulation/simulate_stream.py --mode ingest-batch --batch-id 2 --merge-latest
dvc add data/processed/dataset.csv
```

`--merge-latest` appends to `dataset.csv`, falling back to `base_train.csv` if
needed. Omitting it replaces the active dataset with the selected batch. Do not
ingest batch 7 into training; it is reserved for final testing. Commit the `.dvc`
pointer, dataset metrics, and relevant DVC ignore-file changes, use the next
unused dataset tag (`vX.Y.Z`), then run `dvc push`.

## Live traffic capture and preprocessing

Install Zeek separately and check `zeek --version`. Live capture needs packet
capture privileges; use a suitable network interface for your machine.

```bash
python src/production/capture.py --list-interfaces
sudo python3 src/production/capture.py --interface wlo1 --duration 60
python src/production/ingest_data.py --zeek-log zeek_logs/run_1/conn.log --no-preprocess
python src/production/preprocess_data.py --raw-file data/raw/zeek/raw_zeek_run_1_conn.csv
```

Capture creates numbered folders under `zeek_logs/`. Ingestion writes raw CSVs
to `data/raw/zeek/`; without `--no-preprocess`, it also preprocesses automatically.
Preprocessing writes `data/processed/dataset.csv` and `dataset_metrics.json`.
Use `--merge-latest` to accumulate records instead of replacing the dataset.

`--default-label` defaults to `0` (Normal). This is a placeholder label and must
be verified before using live traffic as labeled training or evaluation data.
Track raw captures converted to CSV and processed datasets with DVC, then commit
their pointers and metrics. For an ingestion example without live capture, use
`zeek_logs/example/conn.log`; pass a separate `--processed-dir` when experimenting
to avoid replacing the active training dataset.

## Configuration and local training

[params.yaml](../params.yaml) holds the shared settings:

| Section | Controls |
|---|---|
| `split` | Base training fraction and number of stream batches. |
| `train.data` | Dataset path, label, excluded timestamps, float types, validation fraction, and seed. |
| `train.xgb` | Parameters passed to `XGBClassifier`. |
| `train.mlflow` | Tracking URI, experiment, model names, and registration behavior. |
| `train.validation` | Validation source, PR-AUC definition, and absolute Staging thresholds. |
| `train.promote` | Relative champion promotion gate used by `set_champion.py --eval`. |

Training CLIs accept repeated `--set key=value` options relative to `train`:

```bash
python src/production/train.py --set xgb.max_depth=10 --set mlflow.register=false
```

Values use YAML types (`false`, `null`, numbers, and lists). Overrides apply only
to that invocation. `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` override
file defaults; explicit `--set` options take precedence. Relative SQLite paths
resolve from the project root.

The complete local candidate pipeline is:

```bash
python src/production/check_training_data.py --output reports/data_validation.json
python src/production/train.py --set mlflow.register=false --data-report reports/data_validation.json
python src/production/evaluate_model.py
python src/production/register_validated_model.py
python src/production/summarize_pipeline.py --reports reports
```

Run each command only after the previous command succeeds. Training writes a
manifest identifying the candidate and data snapshot. Evaluation verifies that
identity and requires Recall, F1, and PR-AUC (Average Precision) each to reach
0.95. Registration verifies the validation receipt and puts the candidate into
the serving registry's Staging stage with alias `challenger`.

The deterministic `GroupShuffleSplit` reserves 20% of flow groups, with seed 42;
the exact row fraction can differ. Flow identity includes timestamps and excludes
labels, keeping duplicate flows on the same side. Batch 7 is excluded from this
gate. Local reports live in `reports/`, model intermediates in `artifacts/`,
and training metrics in `metrics/train_metrics.json`.

`dvc repro train` runs training with the defaults, including registration into
`ids-xgboost`. It does not run the separate evaluation or Staging registration
steps. Set `mlflow.register=false` when following the candidate workflow above.

## GitHub Actions setup

The [automation workflow](../.github/workflows/mlops-automation.yaml) has two jobs:

| Job | Trigger and runner |
|---|---|
| `automated-testing` | Push/PR targeting `main`, or manual run; GitHub-hosted Python 3.11. |
| `train-validate-register` | After tests pass on a push or manual run on `main`; self-hosted runner. |

Before running the training job:

1. Start SeaweedFS and publish the pinned source snapshot with
   `dvc push data/processed/dataset.csv.dvc`.
2. Register a Linux x64 self-hosted runner with the custom label `adaptive-nids`.
   Install the runner outside this checkout and keep it online.
3. Add these repository Actions settings:

   | Type | Name | Value |
   |---|---|---|
   | Variable | `MLFLOW_TRACKING_URI` | Persistent absolute SQLite URI, e.g. `sqlite:////srv/nids/mlflow.db`, or server URI. |
   | Variable | `MLFLOW_ARTIFACT_ROOT` | Absolute persistent directory outside the Actions checkout, e.g. `/srv/nids/artifacts`. |
   | Optional variable | `DVC_S3_ENDPOINT` | Defaults to `http://127.0.0.1:8333`. |
   | Secret | `S3_ACCESS_KEY` | The SeaweedFS access key from local `.env`. |
   | Secret | `S3_SECRET_KEY` | The corresponding SeaweedFS secret key. |

Reuse the existing persistent MLflow database to retain registry history. Both
SQLite storage and artifact storage must be outside the Actions checkout. The
workflow creates an isolated Python environment, pulls the pinned dataset, checks
data, trains, evaluates, and registers passing candidates. It leaves Production
promotion manual. A failed gate stops registration without retrying or tuning.

JUnit output is uploaded after tests; JSON pipeline evidence and the job summary
are collected even when a later step fails. An offline runner leaves training
queued. Save the URL of the actual Actions run for evidence; local tests do not
prove a successful remote workflow run. Coursework instructions and evidence
status are in [Code as a Trigger](code-as-trigger/README.md).

## Model registry and inference

The registry names have distinct roles:

- `ids-xgboost`: experimental versions registered by direct/default training.
- `adaptive-nids-xgb`: serving candidates and Production versions, configured by
  `train.mlflow.serving_model_name`.

The exported metadata records serving v1 as Production with alias `champion`,
trained on dataset `v1.1.0` (MD5 `21be6cc162573182b8ae89df7f65278b`). Its XGBoost
parameters are `n_estimators=100`, `max_depth=6`, and `learning_rate=0.3`.
See [production_model.yaml](../models/production_model.yaml) for identity and lineage,
and [production.dvc](../models/production.dvc) for the exported model pointer.

Historical final-test results on batch 7:

| Serving version | Recall | F1 | Precision | Recorded decision |
|---|---:|---:|---:|---|
| v1 | 0.9736 | 0.9775 | 0.9815 | Production / `champion`. |
| v2 | 0.9733 | 0.9777 | 0.9822 | Archived; recall failed the relative promotion gate. |

v2 used the same data with `n_estimators=200`, `max_depth=8`, and
`learning_rate=0.1`. Experiments retrained on datasets `v1.2.2` through `v1.6.0`
also failed to improve recall by the required 0.002 on the next unseen batch;
their recall was 0.13–0.24 percentage points lower than the champion.
These are recorded historical results, separate from the automatic Staging gate.

To load the registered Production model:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.pyfunc.load_model("models:/adaptive-nids-xgb/Production")
predictions = model.predict(df_raw)
```

Inputs must match the raw UNSW-mapped feature schema. Cast columns listed in
`train.data.float_columns` to `float64` to match the logged MLflow signature.

Useful registry commands:

```bash
python src/production/transition_stage.py --list
python src/production/verify_inference.py
```

Inference verification uses batch 7 for final testing, measures latency and
throughput, and compares predictions from the stage, alias, and available DVC
copy. Restore the exported model with `dvc pull models/production.dvc` if needed.

After reviewing a serving version, promote or roll back explicitly (replace `1`
with the selected version), then refresh and version the export:

```bash
python src/production/transition_stage.py 1 Production --archive-existing
python src/production/export_model_metadata.py --reason "Reviewed Production selection"
dvc add models/production
```

Commit `models/production.dvc`, `models/production_model.yaml`, and any DVC ignore
changes together, then run `dvc push`.

`set_champion.py` serves a separate experiment workflow: it moves the
`ids-xgboost` alias, optionally comparing models on a batch neither trained on.
For example:

```bash
python src/production/set_champion.py --eval data/processed/simulation_stream/batch_4.csv --dry-run
```

Its relative gate requires recall gain of at least 0.002 with F1 loss at most
0.01. Without `--eval`, it forces the alias move. Use `transition_stage.py` for
serving stage changes and synchronized aliases.

## CLI and shared-module reference

Each CLI supports `--help` for its complete options.

| Script under `src/` | Purpose and key options |
|---|---|
| `simulation/preprocess.py` | Clean UNSW-NB15; `--raw-dir`, `--output-file`. |
| `simulation/simulate_stream.py` | `--mode prepare-split`, `list-stream`, or `ingest-batch`; `--train-ratio`, `--num-batches`, `--batch-id`, `--merge-latest`. |
| `production/capture.py` | Capture Zeek logs; `--interface`, `--duration`, `--output-dir`, `--prefix`, `--filter`, `--ipv4-only`. |
| `production/ingest_data.py` | Parse `conn.log`; `--zeek-log`, `--raw-dir`, `--no-preprocess`, `--processed-dir`. |
| `production/preprocess_data.py` | Map Zeek features; `--raw-file`, `--raw-dir`, `--processed-dir`, `--default-label`, `--merge-latest`. |
| `production/check_training_data.py` | Validate snapshot and split; `--output`, `--set`. |
| `production/train.py` | Train and log candidate; `--manifest`, `--data-report`, `--artifacts-dir`, `--metrics-output`, `--set`. |
| `production/evaluate_model.py` | Enforce Staging thresholds; `--manifest`, `--output`, `--set`. |
| `production/register_validated_model.py` | Verify receipt and register; `--manifest`, `--validation`, `--output`, `--set`. |
| `production/summarize_pipeline.py` | Render report evidence as Markdown; `--reports`. |
| `production/set_champion.py` | Move experimental alias; optional version, `--eval`, `--dry-run`, `--set`. |
| `production/transition_stage.py` | Move serving stage; version and stage, `--model`, `--archive-existing`, `--list`. |
| `production/export_model_metadata.py` | Export model and lineage; `--model`, `--stage`, `--out`, `--meta`, `--reason`. |
| `production/verify_inference.py` | Verify serving predictions; `--model`, `--stage`, `--eval`, `--rows`, `--local`. |

Shared modules: `config.py` loads settings and overrides; `dataset_io.py` saves
datasets and metrics; `production/pipeline_utils.py` handles snapshots, splits,
and validation; `production/ids_model.py` packages encoding and inference into an
MLflow pyfunc. Keep those helpers shared rather than duplicating their behavior
in individual CLIs or notebooks.
