# Source Scripts (`src/`) CLI Documentation

The `src/` directory contains the core Python scripts responsible for managing the data preprocessing pipeline, streaming data simulation, live network traffic capture (via Zeek), and dataset I/O for the **MLOps-AdaptiveNIDS** project.

---

## Script Overview

1. [`simulation/preprocess.py`](#1-preprocesspy) - UNSW-NB15 Raw Dataset Cleaning & Standardization
2. [`simulation/simulate_stream.py`](#2-simulate_streampy) - Historical Data Streaming Simulation & Partitioning
3. [`production/ingest_data.py`](#3-ingest_datapy) - Live Zeek Connection Log Raw Ingestion
4. [`production/preprocess_data.py`](#4-preprocess_datapy) - Live Zeek Raw Dataset Preprocessing
5. [`production/capture.py`](#5-capturepy) - Live Network Traffic Packet Capture
6. [`dataset_io.py`](#6-dataset_iopy) - Dataset Output Helper (Core Module)
7. [`production/train.py`](#7-productiontrainpy) - XGBoost training, logged and registered in MLflow
8. [`production/set_champion.py`](#8-productionset_championpy) - Move the `champion` model alias, optionally gated by an evaluation
9. [`config.py`](#9-configpy) - Loads `params.yaml -> train` plus `--set` overrides (Core Module)

> Training and promotion are explained step by step in [`../PIPELINE_FLOW.md`](../PIPELINE_FLOW.md); the settings in [`../CONFIG_GUIDE.md`](../CONFIG_GUIDE.md).

> Dataset versioning is handled by **DVC + Git tags**
---

### 1. `simulation/preprocess.py`
**Function**: Loads the 4 raw UNSW-NB15 CSV splits, performs stateless offline cleaning (drops IP addresses and unmappable TCP metrics, parses hex ports `0x...` to integers), adds domain port features, and exports the standardized dataset to CSV.

#### CLI Arguments (`argparse`):
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--raw-dir` | `str` | `data/raw` | Path to the directory containing raw UNSW-NB15 CSV files (`UNSW-NB15_1.csv` through `4.csv`). |
| `--output-file` | `str` | `data/processed/preprocessed_data.csv` | Output file path for saving the preprocessed dataset. |

#### Usage Examples:
```bash
# Run preprocessing with default configuration
python3 src/simulation/preprocess.py

# Specify custom raw data directory and output path
python3 src/simulation/preprocess.py --raw-dir dataset/raw --output-file data/processed/custom_unsw.csv
```

---

### 2. `simulation/simulate_stream.py`
**Function**: Manages simulated moving data streams from historical datasets. Splits the preprocessed dataset chronologically into a **Base Training Dataset** (`data/processed/base_train.csv`, version `v1.0.0` = Git tag) and a series of virtual simulation batches in `data/processed/simulation_stream/`. `prepare-split` is also the `split` stage in `dvc.yaml`.

#### CLI Arguments (`argparse`):
| Argument | Type | Default | Modes | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--mode` | `str` | *(Required)* | `All` | Select primary action: `prepare-split`, `ingest-batch`, or `list-stream`. |
| `--input-file` | `str` | `data/processed/preprocessed_data.csv` | `prepare-split` | Path to the preprocessed dataset CSV file. |
| `--train-ratio` | `float` | `0.5` (50%) | `prepare-split` | Fraction of initial data allocated to Base Training (`dvc repro` passes the value from `params.yaml`). |
| `--num-batches` | `int` | `7` | `prepare-split` | Number of simulation batch splits for the remaining stream pool. |
| `--batch-id` | `int` | `1` | `ingest-batch` | ID number of the simulation batch to ingest (1, 2, ...). |
| `--merge-latest` | `flag` | `False` | `ingest-batch` | Accumulate/merge the new batch with the current `data/processed/dataset.csv` (falls back to `base_train.csv`). |

#### Usage Examples:
```bash
# 1. Split preprocessed dataset into Base Train (50%) and 7 streaming batches
#    (normally done through `dvc repro`, which reads train_ratio / num_batches from params.yaml)
python3 src/simulation/simulate_stream.py --mode prepare-split --train-ratio 0.5 --num-batches 7

# 2. List available simulation stream batches and attack ratios
python3 src/simulation/simulate_stream.py --mode list-stream

# 3. Ingest Batch 1 as the standalone active dataset (data/processed/dataset.csv)
python3 src/simulation/simulate_stream.py --mode ingest-batch --batch-id 1

# 4. Ingest Batch 2 accumulatively (merge with the current dataset.csv)
python3 src/simulation/simulate_stream.py --mode ingest-batch --batch-id 2 --merge-latest

# 5. Record it as a new dataset version
dvc add data/processed/dataset.csv
git add data/processed/dataset.csv.dvc data/processed/dataset_metrics.json
git commit -m "dataset: ingest batch 2" && git tag v1.2.0 && dvc push
```

---

### 3. `production/ingest_data.py`
**Function**: Reads live network inference connection logs (`conn.log`) produced by Zeek, parses tab-separated headers, and persists the raw un-processed CSV dataset directly into `data/raw/zeek/` (track it with `dvc add`, e.g. `data/raw/zeek/raw_zeek_example_conn.csv`). Automatically triggers preprocessing via `preprocess_data.py` unless `--no-preprocess` is passed.

#### CLI Arguments (`argparse`):
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--zeek-log` | `str` | `zeek_logs/example/conn.log` | Path to the target Zeek connection log file (`conn.log`). |
| `--raw-dir` | `str` | `data/raw/zeek` | Destination directory for storing raw ingested Zeek CSV files. |
| `--no-preprocess` | `flag` | `False` | Ingest raw log to `data/raw/zeek` only without running preprocessing. |
| `--processed-dir` | `str` | `data/processed` | Destination directory for the preprocessed `dataset.csv` (tracked by DVC). |
| `--default-label` | `int` | `0` | Default ground truth label before SOC verification (`0` = Normal, `1` = Attack). |
| `--merge-latest` | `flag` | `False` | Merge this inference log batch with the current `dataset.csv`. |

#### Usage Examples:
```bash
# 1. Full pipeline: Ingest raw log to data/raw/zeek and preprocess to data/processed/dataset.csv
python3 src/production/ingest_data.py --zeek-log zeek_logs/example/conn.log

# 2. Raw ingestion only (persists raw CSV in data/raw/zeek without preprocessing)
python3 src/production/ingest_data.py --zeek-log zeek_logs/run_1/conn.log --no-preprocess
```

---

### 4. `production/preprocess_data.py`
**Function**: Loads raw Zeek CSV datasets from `data/raw/zeek/`, maps features to match the UNSW-NB15 schema, calculates derived traffic metrics (Sload, Dload, smeansz, dmeansz, Ltime, is_sm_ips_ports), computes rolling connection window aggregations, and writes the output to `data/processed/dataset.csv` (plus `dataset_metrics.json`).

#### CLI Arguments (`argparse`):
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--raw-file` | `str` | *(Latest in raw-dir)* | Path to the raw Zeek CSV file in `data/raw/zeek` to preprocess. |
| `--raw-dir` | `str` | `data/raw/zeek` | Directory containing raw ingested CSV files. |
| `--processed-dir` | `str` | `data/processed` | Destination directory for the preprocessed `dataset.csv` (tracked by DVC). |
| `--default-label` | `int` | `0` | Default ground truth label (`0` = Normal, `1` = Attack). |
| `--merge-latest` | `flag` | `False` | Merge preprocessed dataset with the current `dataset.csv`. |

#### Usage Examples:
```bash
# 1. Preprocess latest raw file in data/raw/zeek/
python3 src/production/preprocess_data.py

# 2. Preprocess specific raw file and merge with the current dataset.csv
python3 src/production/preprocess_data.py --raw-file data/raw/zeek/raw_zeek_run_1_conn.csv --merge-latest

# 3. List version history (versions are Git tags)
git tag -l "v*"
```

---

### 4. `production/capture.py`
**Function**: Performs live packet capture on a specified network interface using the Zeek engine and writes structured logs into numbered run folders under `zeek_logs/`.

#### CLI Arguments (`argparse`):
| Argument | Short Option | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--interface` | `-i` | `str` | *(Required unless -l)* | Target network interface name (e.g., `eth0`, `wlo1`, `lo`). |
| `--list-interfaces` | `-l` | `flag` | `False` | List active system network interfaces and exit. |
| `--output-dir` | `-o` | `str` | `zeek_logs` | Parent directory for storing output Zeek log runs. |
| `--prefix` | `-p` | `str` | `run_` | Folder prefix for numbered capture directories (e.g., `run_1`). |
| `--duration` | `-d` | `int` | `None` (Continuous) | Capture duration limit in seconds. Continuous until manually stopped (Ctrl+C). |
| `--filter` | `-f` | `str` | `None` | BPF (Berkeley Packet Filter) filter string passed to Zeek (e.g., `'ip'` or `'not ip6'`). |
| `--ipv4-only` / `--no-ipv6` | - | `flag` | `False` | Restrict packet capture to IPv4 packets only (equivalent to `-f 'ip'`). |

#### Usage Examples:
```bash
# 1. List available network interfaces
python3 src/production/capture.py -l

# 2. Capture traffic on interface wlo1 for 60 seconds (requires root/sudo)
sudo python3 src/production/capture.py -i wlo1 -d 60

# 3. Capture IPv4 packets only on eth0 continuously
sudo python3 src/production/capture.py -i eth0 --ipv4-only
```

---

### 6. `dataset_io.py`
**Function**: Small helper module (`save_dataset`, `write_metrics`). Contains no CLI parser; imported by `simulation/simulate_stream.py` and `production/preprocess_data.py`. It writes `data/processed/dataset.csv` (optionally merged with the current one) and `dataset_metrics.json` (record count, feature count, label distribution). It does **not** version anything: versions are DVC-tracked files plus Git tags (`v1.0.0`, `v1.1.0`, ...), see [`DVC_MIGRATION.md`](../DVC_MIGRATION.md).


---

### 7. `production/train.py`
**Function**: Trains an XGBoost classifier on `data/processed/dataset.csv` (path and all settings come from `params.yaml -> train`), frequency-encodes `proto`/`state`/`service`, logs params, metrics, the final config and version tags (`data_md5`, `data_git_tag`, `git_commit`) to MLflow, and logs the model as an `IDSModel` pyfunc (model + encoding maps in one). It registers a new version of `ids-xgboost` unless `mlflow.register` is `false`, and writes `metrics/train_metrics.json` for DVC. It is also the `train` stage in `dvc.yaml`.

#### CLI Arguments (`argparse`):
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--set` | `str` (repeatable) | none | One-off override of `params.yaml -> train`, as `key=value`, e.g. `xgb.max_depth=10`, `mlflow.register=false`. |

#### Usage Examples:
```bash
# Train with the defaults from params.yaml (through DVC, recorded in dvc.lock)
dvc repro train

# Train directly, with a one-off change
python src/production/train.py --set xgb.max_depth=10 --set mlflow.run_name=xgb-deep

# Smoke test: logged to MLflow, but no registry version
python src/production/train.py --set mlflow.register=false
```

---

### 8. `production/set_champion.py`
**Function**: Moves the `champion` alias of the registered model `ids-xgboost`, which is what `models:/ids-xgboost@champion` loads. Without `--eval` it is a forced move (first champion, rollback). With `--eval` it first compares the champion and the challenger on a CSV that neither model trained on, and moves the alias only if the gate in `params.yaml -> train.promote` passes.

#### CLI Arguments (`argparse`):
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `version` | `str` (positional) | newest version | Model version to promote (the challenger in `--eval` mode). |
| `--eval` | `str` | none | CSV (relative to the project root) to compare champion and challenger on, e.g. `data/processed/simulation_stream/batch_4.csv`. |
| `--dry-run` | `flag` | `False` | With `--eval`: print the comparison and verdict, change nothing. |
| `--set` | `str` (repeatable) | none | Override `params.yaml -> train`, e.g. `promote.min_gain=0`. |

Exit code with `--eval`: `0` = gate passed, `1` = failed.

#### Usage Examples:
```bash
# Force the newest version to be the champion (first champion)
python src/production/set_champion.py

# Roll back to version 1
python src/production/set_champion.py 1

# Compare on batch 4 without changing anything, then for real
python src/production/set_champion.py --eval data/processed/simulation_stream/batch_4.csv --dry-run
python src/production/set_champion.py --eval data/processed/simulation_stream/batch_4.csv
```

---

### 9. `config.py`
**Function**: Small helper module (`load_config`, `flatten`, `resolve_tracking_uri`). Contains no CLI parser; imported by `production/train.py` and `production/set_champion.py`. `load_config()` reads `params.yaml -> train` and applies `--set key=value` overrides.
