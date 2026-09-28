import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger("VersionManager")


class DatasetVersionManager:
    def __init__(self, base_processed_dir: Union[str, Path] = "data/processed"):
        self.base_dir = Path(base_processed_dir)
        self.versions_dir = self.base_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.base_dir / "catalog.json"
        self._init_catalog()

    def _init_catalog(self):
        if not self.catalog_path.exists():
            initial_catalog = {
                "latest_version": None,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "versions": {}
            }
            with open(self.catalog_path, "w", encoding="utf-8") as f:
                json.dump(initial_catalog, f, indent=2)

    def get_catalog(self) -> dict:
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_catalog(self, catalog: dict):
        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)

    def compute_sha256(self, filepath: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def generate_next_version(self, bump_type: str = "minor") -> str:
        catalog = self.get_catalog()
        latest = catalog.get("latest_version")
        if not latest or bump_type == "init":
            return "v1.0.0"

        parts = latest.lstrip("v").split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        else:
            patch += 1

        return f"v{major}.{minor}.{patch}"

    def save_new_version(
        self,
        df: pd.DataFrame,
        source_type: str,
        description: str = "",
        bump_type: str = "minor",
        merge_with_latest: bool = False,
        explicit_version: Optional[str] = None
    ) -> str:
        catalog = self.get_catalog()
        latest_version = catalog.get("latest_version")

        df_to_save = df.copy()

        # merge with latest version
        if merge_with_latest and latest_version:
            prev_file = self.versions_dir / latest_version / "dataset.csv"
            if prev_file.exists():
                logger.info(f"Merging new data with previous version ({latest_version})...")
                df_prev = pd.read_csv(prev_file, low_memory=False)
                df_to_save = pd.concat([df_prev, df_to_save], ignore_index=True)
                source_type = f"accumulated_{source_type}"

        new_version = explicit_version or self.generate_next_version(bump_type=bump_type)
        version_dir = self.versions_dir / new_version
        version_dir.mkdir(parents=True, exist_ok=True)

        dataset_file = version_dir / "dataset.csv"
        metadata_file = version_dir / "metadata.json"

        # Save Output
        df_to_save.to_csv(dataset_file, index=False)
        sha256_hash = self.compute_sha256(dataset_file)

        # manifest metadata
        label_dist = {}
        if "Label" in df_to_save.columns:
            label_dist = {str(k): int(v) for k, v in df_to_save["Label"].value_counts().items()}

        metadata = {
            "version": new_version,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_type": source_type,
            "description": description,
            "parent_version": latest_version,
            "num_records": int(len(df_to_save)),
            "num_features": int(len(df_to_save.columns)),
            "columns": list(df_to_save.columns),
            "label_distribution": label_dist,
            "file_name": dataset_file.name,
            "file_size_bytes": dataset_file.stat().st_size,
            "sha256": sha256_hash
        }

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Update master catalog: update latest version and add newest version
        catalog["latest_version"] = new_version
        catalog["versions"][new_version] = {
            "created_at": metadata["created_at"],
            "source_type": source_type,
            "num_records": metadata["num_records"],
            "sha256": sha256_hash,
            "path": str(version_dir.relative_to(self.base_dir.parent))
        }
        self._save_catalog(catalog)

        logger.info(
            f"Dataset version {new_version} successfully registered | "
            f"Records: {len(df_to_save)} | Checksum: {sha256_hash[:8]}..."
        )
        return new_version

    def list_versions(self) -> None:
        catalog = self.get_catalog()
        print("\n" + "=" * 80)
        print(f"DATASET VERSIONS CATALOG (Current Active: {catalog.get('latest_version', 'None')})")
        print("=" * 80)
        if not catalog.get("versions"):
            print("No dataset versions registered yet.")
        for ver, info in catalog.get("versions", {}).items():
            print(f"- {ver:8} | Created: {info['created_at'][:19]} | Rows: {info['num_records']:<10} | Source: {info['source_type']}")
        print("=" * 80 + "\n")
