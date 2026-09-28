import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Union

import pandas as pd

# Ensure src root is in sys.path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from production.preprocess import preprocess_zeek_dataset
except ImportError:
    from production.preprocess import preprocess_zeek_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("IngestZeek")


def parse_zeek_conn_log(filepath: Union[str, Path]) -> pd.DataFrame:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Zeek Logs Not Found: {filepath}")

    field_names = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#fields"):
                field_names = line.strip().split("\t")[1:]
                break

    if not field_names:
        raise ValueError(f"#fields Header Not Found in Zeek Logs: {filepath}")

    df = pd.read_csv(
        filepath,
        sep="\t",
        comment="#",
        names=field_names,
        na_values="-",
        low_memory=False
    )
    logger.info(f"Loaded {len(df)} records from Zeek logs: {filepath.name}")
    return df


def ingest_raw_zeek_log(
    zeek_log_path: Union[str, Path],
    raw_dir: Union[str, Path] = "data/raw/zeek"
) -> Tuple[pd.DataFrame, Path]:
    log_path = Path(zeek_log_path)
    df_raw = parse_zeek_conn_log(log_path)

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    if log_path.parent.name and log_path.parent.name not in (".", "zeek_logs"):
        raw_filename = f"raw_zeek_{log_path.parent.name}_{log_path.stem}.csv"
    else:
        raw_filename = f"raw_zeek_{log_path.stem}.csv"

    out_raw_path = raw_path / raw_filename
    df_raw.to_csv(out_raw_path, index=False)
    logger.info(f"Saved raw Zeek dataset ({len(df_raw)} records) to: {out_raw_path}")

    return df_raw, out_raw_path


def ingest_zeek_logs(
    zeek_log_path: Union[str, Path] = "zeek_logs/example/conn.log",
    raw_dir: Union[str, Path] = "data/raw/zeek",
    run_preprocess: bool = True,
    default_label: int = 0,
    processed_dir: Union[str, Path] = "data/processed",
    merge_latest: bool = False,
    bump_type: str = "minor",
    desc: str = ""
) -> Tuple[Path, str]:
    log_path = Path(zeek_log_path)

    # Ingest data from captured zeek logs
    logger.info("=== Phase 1: Raw Data Ingestion ===")
    df_raw, raw_file_path = ingest_raw_zeek_log(log_path, raw_dir=raw_dir)

    new_version = ""
    if run_preprocess:
        logger.info("=== Phase 2: Preprocessing & Dataset Versioning ===")
        new_version = preprocess_zeek_dataset(
            raw_input=raw_file_path,
            default_label=default_label,
            processed_dir=processed_dir,
            merge_latest=merge_latest,
            bump_type=bump_type,
            desc=desc or f"Ingested & preprocessed from {log_path.name}"
        )

    return raw_file_path, new_version


def main():
    parser = argparse.ArgumentParser(
        description="MLOps-AdaptiveNIDS Live Zeek Inference Log Ingestion Pipeline (Zeek conn.log -> data/raw/zeek)"
    )
    parser.add_argument(
        "--zeek-log",
        type=str,
        default="zeek_logs/example/conn.log",
        help="Path to Zeek conn.log file (default: zeek_logs/example/conn.log)"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw/zeek",
        help="Directory to store raw ingested files (default: data/raw/zeek)"
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Only ingest raw files to data/raw without running preprocessing"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory to store versioned preprocessed datasets (default: data/processed)"
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
        help="Merge into previous active dataset version"
    )
    parser.add_argument(
        "--bump-type",
        choices=["major", "minor", "patch"],
        default="minor",
        help="Semantic version bump type (default: minor)"
    )
    parser.add_argument(
        "--desc",
        type=str,
        default="",
        help="Description note in dataset version metadata"
    )

    args = parser.parse_args()

    raw_file, new_ver = ingest_zeek_logs(
        zeek_log_path=args.zeek_log,
        raw_dir=args.raw_dir,
        run_preprocess=not args.no_preprocess,
        default_label=args.default_label,
        processed_dir=args.processed_dir,
        merge_latest=args.merge_latest,
        bump_type=args.bump_type,
        desc=args.desc
    )

    print(f"\n[SUCCESS] Raw Zeek log ingested to: {raw_file}")
    if new_ver:
        print(f"[SUCCESS] Dataset version created: {new_ver}\n")


if __name__ == "__main__":
    main()
