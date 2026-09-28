# Live Zeek Data Collection & Processing Pipeline (`src/production/`)

This directory contains the production scripts for capturing live network traffic using **Zeek**, ingesting raw connection logs, preprocessing features to match the **UNSW-NB15** schema, and versioning the dataset into the catalog.

---

## Pipeline Overview

```
 ┌────────────────────────┐
 │ 1. capture.py          │  --> Captures live packets via Zeek & outputs conn.log into zeek_logs/run_N/
 └───────────┬────────────┘
             │
             ▼
 ┌────────────────────────┐
 │ 2. ingest_data.py      │  --> Parses conn.log & persists raw dataset into data/raw/zeek/
 └───────────┬────────────┘
             │
             ▼
 ┌────────────────────────┐
 │ 3. preprocess_data.py  │  --> Engineers UNSW-NB15 features & registers dataset version in data/processed/
 └────────────────────────┘
```

---

## Step-by-Step Guide: Collecting & Processing Live Traffic

### Step 1: Discover Active Network Interfaces
Before capturing traffic, list the active network interfaces available on your machine:

```bash
python3 src/production/capture.py -l
```

**Example Output:**
```text
🌐 Available Network Interfaces:
===================================
  [1] eth0
  [2] wlo1
  [3] lo
===================================
```

---

### Step 2: Capture Live Traffic with Zeek

Capture live traffic from your target network interface (e.g. `wlo1` or `eth0`). Packet capture requires root / `sudo` privileges.

#### Option A: Timed Capture Session (e.g., 60 seconds)
```bash
sudo python3 src/production/capture.py -i wlo1 -d 60
```

#### Option B: Continuous Traffic Capture
```bash
sudo python3 src/production/capture.py -i wlo1
```
*(Press `Ctrl+C` to stop capturing manually).*

#### Option C: Capture IPv4 Packets Only with Custom Prefix
```bash
sudo python3 src/production/capture.py -i eth0 -d 120 --ipv4-only --prefix production_run_
```

**Output**: Zeek will automatically generate structured log files (including `conn.log`, `dns.log`, `http.log`, `ssl.log`) in a new numbered folder under `zeek_logs/` (e.g., `zeek_logs/run_1/conn.log`).

---

### Step 3: Ingest Raw Zeek Connection Logs (`data/raw/zeek/`)

Parse the captured `conn.log` header structure and save the un-processed raw tabular dataset into `data/raw/zeek/`.

#### Full Automatic Pipeline (Ingest + Preprocess + Version):
```bash
python3 src/production/ingest_data.py --zeek-log zeek_logs/run_1/conn.log --desc "Live capture run 1"
```

#### Raw Ingestion Only (Save to `data/raw/zeek/` without preprocessing):
```bash
python3 src/production/ingest_data.py --zeek-log zeek_logs/run_1/conn.log --no-preprocess
```

**Resulting File**: `data/raw/zeek/raw_zeek_run_1_conn.csv`

---

### Step 4: Preprocess Raw Dataset (`data/processed/`)

Map raw Zeek features to the UNSW-NB15 schema, compute load metrics (`Sload`, `Dload`), derive packet sizes (`smeansz`, `dmeansz`), calculate 100-connection rolling window aggregations (`ct_srv_src`, `ct_dst_ltm`, etc.), and register a new dataset version:

#### Preprocess the Latest Raw Dataset:
```bash
python3 src/production/preprocess_data.py
```

#### Preprocess Specific Raw File & Merge with Active Version:
```bash
python3 src/production/preprocess_data.py \
  --raw-file data/raw/zeek/raw_zeek_run_1_conn.csv \
  --merge-latest \
  --bump-type minor \
  --desc "Accumulated live traffic run 1"
```

**CLI Options for Preprocessing**:
- `--default-label`: Set ground truth label before SOC triage (`0` = Normal, `1` = Attack).
- `--bump-type`: Semantic version bump (`major`, `minor`, `patch`).
- `--merge-latest`: Accumulate new records into the previous active dataset version.

---

### Step 5: Verify Dataset Versions

Inspect the dataset catalog history to verify newly registered versions:

```bash
python3 src/production/preprocess_data.py --list-versions
```

**Example Output:**
```text
================================================================================
DATASET VERSIONS CATALOG (Current Active: v1.1.0)
================================================================================
- v1.0.0   | Created: 2026-09-28T10:24:15 | Rows: 35        | Source: zeek_live
- v1.1.0   | Created: 2026-09-28T10:24:30 | Rows: 494       | Source: zeek_live
================================================================================
```

Registered datasets are stored in `data/processed/versions/vX.X.X/dataset.csv` with complete `metadata.json` manifests.

---

## Quick Reference Summary

| Task | Script Command | Output Location |
| :--- | :--- | :--- |
| **1. List Interfaces** | `python3 src/production/capture.py -l` | Stdout |
| **2. Capture Packets** | `sudo python3 src/production/capture.py -i <iface> -d <sec>` | `zeek_logs/run_N/conn.log` |
| **3. Ingest Raw Logs** | `python3 src/production/ingest_data.py --zeek-log zeek_logs/run_N/conn.log` | `data/raw/zeek/raw_zeek_run_N_conn.csv` |
| **4. Preprocess Data** | `python3 src/production/preprocess_data.py` | `data/processed/versions/vX.X.X/` |
| **5. View Catalog** | `python3 src/production/preprocess_data.py --list-versions` | Catalog Summary |
