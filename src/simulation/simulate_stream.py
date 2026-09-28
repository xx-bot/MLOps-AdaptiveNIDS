import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from version_manager import DatasetVersionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SimulateStream")


def prepare_and_split(
    input_file: Union[str, Path] = "data/processed/preprocessed_data.csv",
    train_ratio: float = 0.5,
    num_stream_batches: int = 7,
    processed_dir: Union[str, Path] = "data/processed",
    sort_by_time: bool = True
) -> Dict[str, Union[str, int, List[str]]]:
    
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Preprocessed file not found: {input_path}\n"
            f"Make sure you have saved the dataframe result from notebook/preprocessing.ipynb "
            f"to that path first."
        )

    logger.info(f"Reading preprocessed dataset from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    total_rows = len(df)
    logger.info(f"Total preprocessed data rows: {total_rows}")

    # Urutkan berdasarkan waktu jika kolom Stime tersedia
    if sort_by_time and "Stime" in df.columns:
        logger.info("Sorting dataset chronologically based on 'Stime' column...")
        df = df.sort_values(by="Stime").reset_index(drop=True)

    # 1. Pisahkan Data Base Training vs Data Bergerak
    split_idx = int(total_rows * train_ratio)
    df_base_train = df.iloc[:split_idx].copy()
    df_stream_pool = df.iloc[split_idx:].copy().reset_index(drop=True)

    logger.info(
        f"Data split: Base Train = {len(df_base_train)} rows ({train_ratio*100:.1f}%), "
        f"Stream Pool = {len(df_stream_pool)} rows ({(1-train_ratio)*100:.1f}%)"
    )

    # 2. Daftarkan Base Train sebagai Dataset Versi v1.0.0
    version_mgr = DatasetVersionManager(base_processed_dir=processed_dir)
    base_version = version_mgr.save_new_version(
        df=df_base_train,
        source_type="base_training",
        description=f"Base training dataset ({train_ratio*100:.0f}% initial preprocessed data)",
        explicit_version="v1.0.0"
    )

    # 3. Pecah sisa data menjadi N batch simulasi data bergerak
    stream_dir = Path(processed_dir) / "simulation_stream"
    stream_dir.mkdir(parents=True, exist_ok=True)

    # Split DataFrame into N batch chunks using index splits to preserve DataFrame type
    batch_indices = np.array_split(df_stream_pool.index, num_stream_batches)
    stream_batches = [df_stream_pool.loc[idx] for idx in batch_indices]
    batch_files = []
    manifest_batches = []

    for i, batch_df in enumerate(stream_batches):
        batch_id = i + 1
        batch_filename = f"batch_{batch_id}.csv"
        batch_filepath = stream_dir / batch_filename
        batch_df.to_csv(batch_filepath, index=False)
        batch_files.append(str(batch_filepath))

        attack_count = int((batch_df["Label"] == 1).sum()) if "Label" in batch_df.columns else 0
        normal_count = int((batch_df["Label"] == 0).sum()) if "Label" in batch_df.columns else len(batch_df)

        manifest_batches.append({
            "batch_id": batch_id,
            "filename": batch_filename,
            "num_records": len(batch_df),
            "normal_count": normal_count,
            "attack_count": attack_count,
            "attack_ratio": round(attack_count / len(batch_df), 4) if len(batch_df) > 0 else 0
        })

    # Simpan manifest antrean batch
    manifest_file = stream_dir / "stream_manifest.json"
    manifest_data = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_stream_records": len(df_stream_pool),
        "num_batches": num_stream_batches,
        "base_version": base_version,
        "batches": manifest_batches
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(
        f"Successfully split streaming data into {num_stream_batches} batches in {stream_dir}/ "
        f"(manifest: {manifest_file.name})"
    )

    return {
        "base_version": base_version,
        "base_records": len(df_base_train),
        "stream_batches": batch_files
    }


def ingest_batch(
    batch_id: int,
    stream_dir: Union[str, Path] = "data/processed/simulation_stream",
    processed_dir: Union[str, Path] = "data/processed",
    merge_latest: bool = False,
    bump_type: str = "minor",
    desc: str = ""
) -> str:

    batch_file = Path(stream_dir) / f"batch_{batch_id}.csv"
    if not batch_file.exists():
        raise FileNotFoundError(
            f"Batch file not found: {batch_file}\n"
            f"Run 'prepare-split' mode first to create batch files."
        )

    df_batch = pd.read_csv(batch_file, low_memory=False)
    version_mgr = DatasetVersionManager(base_processed_dir=processed_dir)

    description = desc or f"Simulated moving data stream batch {batch_id}"
    new_version = version_mgr.save_new_version(
        df=df_batch,
        source_type="simulation_stream",
        description=description,
        bump_type=bump_type,
        merge_with_latest=merge_latest
    )
    return new_version


def list_simulation_stream(stream_dir: Union[str, Path] = "data/processed/simulation_stream") -> None:

    manifest_file = Path(stream_dir) / "stream_manifest.json"
    if not manifest_file.exists():
        print(f"\n[INFO] No simulated batch data available in {stream_dir} yet.")
        print("Please run 'prepare-split' mode first.\n")
        return

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("\n" + "=" * 80)
    print(f"SIMULATION STREAM BATCHES (Base Version: {manifest.get('base_version', 'N/A')})")
    print(f"Total Stream Records: {manifest.get('total_stream_records', 0)} | Total Batches: {manifest.get('num_batches', 0)}")
    print("=" * 80)
    for b in manifest.get("batches", []):
        print(
            f"Batch {b['batch_id']:2} | File: {b['filename']:14} | Records: {b['num_records']:<8} | "
            f"Normal: {b['normal_count']:<8} | Attack: {b['attack_count']:<6} (Ratio: {b['attack_ratio']*100:.2f}%)"
        )
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="MLOps-AdaptiveNIDS Simulation Data Streamer (Prepare Split & Ingest Batch)"
    )
    parser.add_argument(
        "--mode",
        choices=["prepare-split", "ingest-batch", "list-stream", "list-versions"],
        required=True,
        help="Select action: 'prepare-split', 'ingest-batch', 'list-stream', or 'list-versions'"
    )

    # Argumen mode prepare-split
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/processed/preprocessed_data.csv",
        help="Path to preprocessed dataset file (default: data/processed/preprocessed_data.csv)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.5,
        help="Data proportion for Base Training Model, e.g., 0.5 (50%) (default: 0.5)"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=7,
        help="Number of simulation batch partitions for remaining data (default: 7 virtual batches)"
    )

    # Argumen mode ingest-batch
    parser.add_argument(
        "--batch-id",
        type=int,
        default=1,
        help="Batch ID number to ingest into new version (1, 2, ...)"
    )
    parser.add_argument(
        "--merge-latest",
        action="store_true",
        help="Accumulate this batch with previous dataset version"
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
        help="Description note for dataset version"
    )

    args = parser.parse_args()

    if args.mode == "prepare-split":
        res = prepare_and_split(
            input_file=args.input_file,
            train_ratio=args.train_ratio,
            num_stream_batches=args.num_batches
        )
        print(f"\n[SUCCESS] Base training dataset registered as version: {res['base_version']}")
        print(f"Total base train data rows: {res['base_records']}")
        print(f"Total streaming simulation batches: {len(res['stream_batches'])} batches ready for use.\n")

    elif args.mode == "ingest-batch":
        new_ver = ingest_batch(
            batch_id=args.batch_id,
            merge_latest=args.merge_latest,
            bump_type=args.bump_type,
            desc=args.desc
        )
        print(f"\n[SUCCESS] Batch {args.batch_id} successfully ingested as dataset version: {new_ver}\n")

    elif args.mode == "list-stream":
        list_simulation_stream()

    elif args.mode == "list-versions":
        version_mgr = DatasetVersionManager()
        version_mgr.list_versions()


if __name__ == "__main__":
    main()
