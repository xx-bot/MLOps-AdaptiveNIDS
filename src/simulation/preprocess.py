import argparse
import logging
import sys
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("PreprocessUNSW")


def parse_port(val: Union[str, int, float]) -> int:
    """Konversi nilai port (hexadecimal string '0x20205321', float, string, atau '-' ) menjadi integer (0-65535)."""
    if pd.isna(val) or str(val).strip() in ('-', '', 'NaN', 'nan'):
        return 0
    val_str = str(val).strip()
    try:
        if val_str.startswith(('0x', '0X')):
            return int(val_str, 16)
        return int(float(val_str))
    except ValueError:
        return 0


def preprocess_unsw_dataset(
    raw_dir: Union[str, Path] = "data/raw",
    output_file: Union[str, Path] = "data/processed/preprocessed_data.csv"
) -> Path:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data folder not found in: {raw_path}")

    features_file = raw_path / "NUSW-NB15_features.csv"
    if not features_file.exists():
        raise FileNotFoundError(f"Feature definition file not found: {features_file}")

    logger.info(f"Reading feature names list from {features_file.name}...")
    features_df = pd.read_csv(features_file, encoding="latin1")
    features = [str(name).strip() for name in features_df["Name"]]

    # specify data type to prevent Dtype warning
    dtype_spec = {
        "sport": str,
        "dsport": str,
        "ct_ftp_cmd": str,
        "Label": np.int64
    }

    logger.info(f"Reading CSV partitions from {raw_path}...")
    dfs = []
    for i in range(1, 5):
        split_file = raw_path / f"UNSW-NB15_{i}.csv"
        if not split_file.exists():
            raise FileNotFoundError(f"Dataset partition not found: {split_file}")
        logger.info(f"  - Loading {split_file.name}...")
        df_part = pd.read_csv(split_file, names=features, dtype=dtype_spec, low_memory=False)
        dfs.append(df_part)

    logger.info("Combining all dataset partitions...")
    df_combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total raw rows: {len(df_combined):,} rows | {len(df_combined.columns)} columns.")

    # fix column name typo in UNSW NB15 dataset
    df_combined = df_combined.rename(columns={"ct_src_ ltm": "ct_src_ltm"})

    # remove non-existent columns in zeek logs and columns that can potentially cause data leak
    excluded_col = [
        "sttl", "dttl", "sloss", "dloss", "swin", "dwin", "stcpb", "dtcpb",
        "trans_depth", "res_bdy_len", "Sjit", "Djit", "Sintpkt", "Dintpkt",
        "tcprtt", "synack", "ackdat", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"
    ]
    logger.info("Removing irrelevant / data leak columns (IP, internal TCP metrics)...")
    df_cleaned = df_combined.drop(columns=excluded_col, errors="ignore")
    df_cleaned = df_cleaned.drop(columns=["srcip", "dstip"], errors="ignore")

    # convert hexadecimal port number into decimal
    logger.info("Parsing port number values (string/hex -> int)...")
    df_cleaned["sport"] = df_cleaned["sport"].apply(parse_port)
    df_cleaned["dsport"] = df_cleaned["dsport"].apply(parse_port)

    # feature engineering port numbers
    logger.info("Engineering network domain features (well-known, ephemeral, & critical ports)...")
    df_cleaned["sport_is_well_known"] = (df_cleaned["sport"] < 1024).astype(int)
    df_cleaned["dsport_is_well_known"] = (df_cleaned["dsport"] < 1024).astype(int)
    df_cleaned["sport_is_ephemeral"] = (df_cleaned["sport"] >= 49152).astype(int)
    df_cleaned["dsport_is_ephemeral"] = (df_cleaned["dsport"] >= 49152).astype(int)

    critical_ports = {80, 443, 22, 23, 53, 445, 3389}
    df_cleaned["dsport_is_critical"] = df_cleaned["dsport"].isin(critical_ports).astype(int)

    # Save Output
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving preprocessed dataset to {out_path}...")
    df_cleaned.to_csv(out_path, index=False)

    logger.info(
        f"[SUCCESS] Dataset successfully preprocessed & saved to {out_path}\n"
        f"        Dimensions: {df_cleaned.shape[0]:,} rows x {df_cleaned.shape[1]} columns"
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="MLOps-AdaptiveNIDS UNSW-NB15 Data Preprocessing Pipeline"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Raw dataset directory path (default: data/raw)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/processed/preprocessed_data.csv",
        help="Destination path for preprocessed output file (default: data/processed/preprocessed_data.csv)"
    )

    args = parser.parse_args()
    preprocess_unsw_dataset(raw_dir=args.raw_dir, output_file=args.output_file)


if __name__ == "__main__":
    main()
