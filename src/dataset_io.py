import json
import logging
from pathlib import Path
from typing import Dict, Union

import pandas as pd

logger = logging.getLogger("DatasetIO")

# Dataset versioning itself is handled by DVC + Git tags (see DVC_MIGRATION.md).
# This module only decides where files live and writes the small metrics file
# that `dvc metrics diff` compares between versions.
DATASET_NAME = "dataset.csv"
BASE_TRAIN_NAME = "base_train.csv"
DATASET_METRICS_NAME = "dataset_metrics.json"


def compute_metrics(df: pd.DataFrame) -> Dict[str, object]:
    label_dist = {}
    if "Label" in df.columns:
        label_dist = {str(k): int(v) for k, v in df["Label"].value_counts().sort_index().items()}

    return {
        "num_records": int(len(df)),
        "num_features": int(len(df.columns)),
        "label_distribution": label_dist,
    }


def write_metrics(df: pd.DataFrame, metrics_path: Union[str, Path]) -> Path:
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(compute_metrics(df), f, indent=2)
    return metrics_path


def save_dataset(
    df: pd.DataFrame,
    processed_dir: Union[str, Path] = "data/processed",
    merge_with_latest: bool = False
) -> Path:
    """Write the active dataset (data/processed/dataset.csv) and its metrics file.

    With merge_with_latest the new rows are appended to the current dataset.csv,
    falling back to base_train.csv when no dataset.csv exists yet.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = processed_dir / DATASET_NAME

    df_to_save = df
    if merge_with_latest:
        prev_path = dataset_path if dataset_path.exists() else processed_dir / BASE_TRAIN_NAME
        if prev_path.exists():
            logger.info(f"Merging new data with previous dataset ({prev_path.name})...")
            df_prev = pd.read_csv(prev_path, low_memory=False)
            df_to_save = pd.concat([df_prev, df], ignore_index=True)
        else:
            logger.warning("--merge-latest given but no previous dataset exists; saving new data only.")

    df_to_save.to_csv(dataset_path, index=False)
    write_metrics(df_to_save, processed_dir / DATASET_METRICS_NAME)

    logger.info(f"Dataset saved to {dataset_path} | Records: {len(df_to_save)}")
    return dataset_path
