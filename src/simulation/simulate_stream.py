import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd

# Ensure src root is in sys.path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dataset_io import BASE_TRAIN_NAME, DATASET_METRICS_NAME, save_dataset, write_metrics

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

    # 2. Simpan Base Train (dilacak DVC sebagai output stage `split`, versi v1.0.0 = git tag)
    base_train_path = Path(processed_dir) / BASE_TRAIN_NAME
    base_train_path.parent.mkdir(parents=True, exist_ok=True)
    df_base_train.to_csv(base_train_path, index=False)
    write_metrics(df_base_train, Path(processed_dir) / "base_train_metrics.json")

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
    # tanpa timestamp supaya output stage deterministik (hash DVC tidak berubah tiap run)
    manifest_data = {
        "total_stream_records": len(df_stream_pool),
        "num_batches": num_stream_batches,
        "batches": manifest_batches
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(
        f"Successfully split streaming data into {num_stream_batches} batches in {stream_dir}/ "
        f"(manifest: {manifest_file.name})"
    )

    return {
        "base_train_file": str(base_train_path),
        "base_records": len(df_base_train),
        "stream_batches": batch_files
    }


def ingest_batch(
    batch_id: int,
    stream_dir: Union[str, Path] = "data/processed/simulation_stream",
    processed_dir: Union[str, Path] = "data/processed",
    merge_latest: bool = False
) -> Path:

    batch_file = Path(stream_dir) / f"batch_{batch_id}.csv"
    if not batch_file.exists():
        raise FileNotFoundError(
            f"Batch file not found: {batch_file}\n"
            f"Run 'prepare-split' mode first to create batch files."
        )

    df_batch = pd.read_csv(batch_file, low_memory=False)
    return save_dataset(df_batch, processed_dir=processed_dir, merge_with_latest=merge_latest)


def list_simulation_stream(stream_dir: Union[str, Path] = "data/processed/simulation_stream") -> None:

    manifest_file = Path(stream_dir) / "stream_manifest.json"
    if not manifest_file.exists():
        print(f"\n[INFO] No simulated batch data available in {stream_dir} yet.")
        print("Please run 'prepare-split' mode first.\n")
        return

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("\n" + "=" * 80)
    print("SIMULATION STREAM BATCHES")
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
        choices=["prepare-split", "ingest-batch", "list-stream"],
        required=True,
        help="Select action: 'prepare-split', 'ingest-batch', or 'list-stream'"
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
        help="Batch ID number to ingest into data/processed/dataset.csv (1, 2, ...)"
    )
    parser.add_argument(
        "--merge-latest",
        action="store_true",
        help="Accumulate this batch with the current data/processed/dataset.csv"
    )

    args = parser.parse_args()

    if args.mode == "prepare-split":
        res = prepare_and_split(
            input_file=args.input_file,
            train_ratio=args.train_ratio,
            num_stream_batches=args.num_batches
        )
        print(f"\n[SUCCESS] Base training dataset saved to: {res['base_train_file']}")
        print(f"Total base train data rows: {res['base_records']}")
        print(f"Total streaming simulation batches: {len(res['stream_batches'])} batches ready for use.\n")

    elif args.mode == "ingest-batch":
        out_path = ingest_batch(
            batch_id=args.batch_id,
            merge_latest=args.merge_latest
        )
        print(f"\n[SUCCESS] Batch {args.batch_id} successfully ingested to: {out_path}")
        print(f"Next: dvc add {out_path} && git add {out_path}.dvc data/processed/{DATASET_METRICS_NAME} "
              f"&& git commit && git tag vX.Y.Z\n")

    elif args.mode == "list-stream":
        list_simulation_stream()


if __name__ == "__main__":
    main()
