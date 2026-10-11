import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# Ensure src root is in sys.path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dataset_io import save_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("PreprocessZeek")


def map_zeek_to_unsw_schema(df_zeek: pd.DataFrame, default_label: int = 0) -> pd.DataFrame:
    df = pd.DataFrame()

    duration = df_zeek["duration"].fillna(0.0).astype(float).values
    orig_bytes = df_zeek["orig_bytes"].fillna(0).astype(float).values
    resp_bytes = df_zeek["resp_bytes"].fillna(0).astype(float).values
    orig_pkts = df_zeek["orig_pkts"].fillna(0).astype(float).values
    resp_pkts = df_zeek["resp_pkts"].fillna(0).astype(float).values
    ts = df_zeek["ts"].fillna(0.0).astype(float).values

    # Direct mappings
    df["sport"] = df_zeek["id.orig_p"]
    df["dsport"] = df_zeek["id.resp_p"]
    df["proto"] = df_zeek["proto"].astype(str)
    df["state"] = df_zeek["conn_state"].astype(str)
    df["dur"] = duration
    df["sbytes"] = orig_bytes
    df["dbytes"] = resp_bytes
    df["Spkts"] = orig_pkts
    df["Dpkts"] = resp_pkts
    df["service"] = df_zeek["service"].fillna("-").astype(str)
    df["Stime"] = ts

    # Derivable features
    df["Sload"] = np.divide(orig_bytes * 8, duration, out=np.zeros_like(orig_bytes, dtype=float), where=duration > 0)
    df["Dload"] = np.divide(resp_bytes * 8, duration, out=np.zeros_like(resp_bytes, dtype=float), where=duration > 0)
    df["smeansz"] = np.divide(orig_bytes, orig_pkts, out=np.zeros_like(orig_bytes, dtype=float), where=orig_pkts > 0)
    df["dmeansz"] = np.divide(resp_bytes, resp_pkts, out=np.zeros_like(resp_bytes, dtype=float), where=resp_pkts > 0)
    df["Ltime"] = ts + duration
    df["is_sm_ips_ports"] = (
        (df_zeek["id.orig_h"] == df_zeek["id.resp_h"]) &
        (df_zeek["id.orig_p"] == df_zeek["id.resp_p"])
    ).astype(int)

    # Windowed count aggregations (100-connection rolling window)
    n = len(df_zeek)
    ct_srv_src = np.zeros(n, dtype=int)
    ct_srv_dst = np.zeros(n, dtype=int)
    ct_dst_ltm = np.zeros(n, dtype=int)
    ct_src_ltm = np.zeros(n, dtype=int)
    ct_src_dport_ltm = np.zeros(n, dtype=int)
    ct_dst_sport_ltm = np.zeros(n, dtype=int)
    ct_dst_src_ltm = np.zeros(n, dtype=int)
    ct_state_ttl = np.zeros(n, dtype=int)

    orig_h = df_zeek["id.orig_h"].values
    resp_h = df_zeek["id.resp_h"].values
    orig_p = df_zeek["id.orig_p"].values
    resp_p = df_zeek["id.resp_p"].values
    services = df["service"].values
    states = df["state"].values

    for i in range(n):
        start_idx = max(0, i - 100)
        w_orig_h = orig_h[start_idx : i + 1]
        w_resp_h = resp_h[start_idx : i + 1]
        w_orig_p = orig_p[start_idx : i + 1]
        w_resp_p = resp_p[start_idx : i + 1]
        w_serv = services[start_idx : i + 1]
        w_state = states[start_idx : i + 1]

        ct_srv_src[i] = np.sum((w_serv == services[i]) & (w_orig_h == orig_h[i]))
        ct_srv_dst[i] = np.sum((w_serv == services[i]) & (w_resp_h == resp_h[i]))
        ct_dst_ltm[i] = np.sum(w_resp_h == resp_h[i])
        ct_src_ltm[i] = np.sum(w_orig_h == orig_h[i])
        ct_src_dport_ltm[i] = np.sum((w_orig_h == orig_h[i]) & (w_resp_p == resp_p[i]))
        ct_dst_sport_ltm[i] = np.sum((w_resp_h == resp_h[i]) & (w_orig_p == orig_p[i]))
        ct_dst_src_ltm[i] = np.sum((w_orig_h == orig_h[i]) & (w_resp_h == resp_h[i]))
        ct_state_ttl[i] = np.sum(w_state == states[i])

    df["ct_srv_src"] = ct_srv_src
    df["ct_srv_dst"] = ct_srv_dst
    df["ct_dst_ltm"] = ct_dst_ltm
    df["ct_src_ltm"] = ct_src_ltm
    df["ct_src_dport_ltm"] = ct_src_dport_ltm
    df["ct_dst_sport_ltm"] = ct_dst_sport_ltm
    df["ct_dst_src_ltm"] = ct_dst_src_ltm
    df["ct_state_ttl"] = ct_state_ttl

    df["sport_is_well_known"] = (df["sport"] < 1024).astype(int)
    df["dsport_is_well_known"] = (df["dsport"] < 1024).astype(int)
    df["sport_is_ephemeral"] = (df["sport"] >= 49152).astype(int)
    df["dsport_is_ephemeral"] = (df["dsport"] >= 49152).astype(int)

    critical_ports = {80, 443, 22, 23, 53, 445, 3389}
    df["dsport_is_critical"] = df["dsport"].isin(critical_ports).astype(int)


    df["Label"] = default_label

    return df


def preprocess_zeek_dataset(
    raw_input: Union[str, Path, pd.DataFrame],
    default_label: int = 0,
    processed_dir: Union[str, Path] = "data/processed",
    merge_latest: bool = False
) -> Path:
    if isinstance(raw_input, (str, Path)):
        raw_path = Path(raw_input)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")
        logger.info(f"Loading raw dataset from {raw_path}...")
        df_raw = pd.read_csv(raw_path, low_memory=False)
    else:
        df_raw = raw_input

    logger.info(f"Preprocessing raw Zeek dataset ({len(df_raw)} records)...")
    df_processed = map_zeek_to_unsw_schema(df_raw, default_label=default_label)
    logger.info(f"Preprocessing complete. Dimensions: {df_processed.shape[0]} rows x {df_processed.shape[1]} columns.")

    return save_dataset(df_processed, processed_dir=processed_dir, merge_with_latest=merge_latest)


def main():
    parser = argparse.ArgumentParser(
        description="MLOps-AdaptiveNIDS Production Preprocessing Pipeline (data/raw/zeek -> data/processed)"
    )
    parser.add_argument(
        "--raw-file",
        type=str,
        default=None,
        help="Path to raw Zeek CSV file in data/raw/zeek (default: latest raw_zeek_*.csv in data/raw/zeek)"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw/zeek",
        help="Directory containing raw ingested CSV files (default: data/raw/zeek)"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory to store the preprocessed dataset, tracked by DVC (default: data/processed)"
    )
    parser.add_argument(
        "--default-label",
        type=int,
        default=0,
        help="Default ground truth label (0=Normal, 1=Attack) (default: 0)"
    )
    parser.add_argument(
        "--merge-latest",
        action="store_true",
        help="Merge with the current dataset.csv in --processed-dir"
    )

    args = parser.parse_args()

    raw_file = args.raw_file
    if not raw_file:
        raw_path = Path(args.raw_dir)
        raw_files = sorted(raw_path.glob("raw_zeek_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not raw_files and raw_path != Path("data/raw"):
            # Fallback check in parent data/raw if no files in data/raw/zeek
            parent_raw = Path("data/raw")
            raw_files = sorted(parent_raw.glob("raw_zeek_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not raw_files:
            raise FileNotFoundError(
                f"No raw Zeek files (raw_zeek_*.csv) found in '{raw_path}'.\n"
                f"Run 'python3 src/production/ingest_data.py' first to ingest raw data into '{raw_path}'."
            )
        raw_file = raw_files[0]
        logger.info(f"No --raw-file specified. Selected latest raw dataset: {raw_file}")

    out_path = preprocess_zeek_dataset(
        raw_input=raw_file,
        default_label=args.default_label,
        processed_dir=args.processed_dir,
        merge_latest=args.merge_latest
    )
    print(f"\n[SUCCESS] Raw Zeek dataset successfully preprocessed & saved to: {out_path}")
    print(f"Next: dvc add {out_path} && git add {out_path}.dvc && git commit && git tag vX.Y.Z\n")


if __name__ == "__main__":
    main()
