"""Shared data, lineage, and validation checks for the automated pipeline."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

from config import BASE_PATH, load_config


def load_pipeline_config(overrides=()):
    cfg = load_config(overrides)
    cfg["mlflow"]["register"] = False
    return cfg


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else BASE_PATH / path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_md5(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(path, expected_md5):
    if not expected_md5 or len(expected_md5) != 32:
        raise ValueError("A pinned file MD5 is required")
    actual = file_md5(project_path(path))
    if actual != expected_md5:
        raise ValueError(f"Data checksum mismatch for {path}: expected {expected_md5}, got {actual}")
    return {"path": str(path), "md5": actual}


def training_snapshot(cfg):
    path = cfg["data"]["path"]
    pointer = project_path(str(path) + ".dvc")
    outputs = yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"]
    if len(outputs) != 1 or Path(outputs[0]["path"]).name != Path(path).name:
        raise ValueError("Training DVC pointer must describe exactly this dataset")
    return snapshot(path, outputs[0]["md5"])


def prepare_frame(frame, data_cfg, require_both_classes=True):
    target = data_cfg["target"]
    if frame.empty or target not in frame:
        raise ValueError("A nonempty, labeled dataset is required")
    labels = frame[target]
    classes = set(labels.unique())
    if labels.isna().any() or not classes <= {0, 1} or (require_both_classes and classes != {0, 1}):
        raise ValueError("Validated binary labels with both classes 0 and 1 are required")
    features = frame.drop(columns=[target, *data_cfg["drop_columns"]])
    features[data_cfg["float_columns"]] = features[data_cfg["float_columns"]].astype("float64")
    numeric = features.select_dtypes(include="number")
    if features.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Features must not contain missing or non-finite values")
    return features, labels


def row_hashes(frame, target):
    # Include timestamps for flow identity, exclude the label. Normalize numeric
    # dtypes so inference of int vs float in different CSV chunks cannot hide overlap.
    frame = frame.drop(columns=[target]).sort_index(axis=1).copy()
    for column in frame.select_dtypes(include="number"):
        frame[column] = frame[column].astype("float64")
    return pd.util.hash_pandas_object(frame, index=False).to_numpy()


def training_split_indices(frame, data_cfg):
    """Keep every copy of the same flow on one side of the validation split."""
    groups = row_hashes(frame, data_cfg["target"])
    splitter = GroupShuffleSplit(n_splits=1, test_size=data_cfg["test_size"],
                                 random_state=data_cfg["seed"])
    training, validation = next(splitter.split(frame, groups=groups))
    if np.isin(groups[training], groups[validation]).any():
        raise ValueError("Training and validation contain overlapping rows")
    labels = frame[data_cfg["target"]]
    if any(set(labels.iloc[indices].unique()) != {0, 1} for indices in (training, validation)):
        raise ValueError("Both training and validation splits must contain both classes")
    return training, validation


def evaluation_snapshot(cfg, training=None):
    policy = cfg["validation"]
    source = policy.get("source", "external_file")
    if source == "training_split":
        training = training if training is not None else training_snapshot(cfg)
        return {**training, "split": {"method": "group_shuffle_by_flow", "subset": "validation",
                                      "test_size": cfg["data"]["test_size"],
                                      "seed": cfg["data"]["seed"]}}
    if source == "external_file":
        return snapshot(policy["data_path"], policy["data_md5"])
    raise ValueError(f"Unknown validation data source: {source}")


def load_evaluation_frame(cfg):
    evaluation = evaluation_snapshot(cfg)
    frame = pd.read_csv(project_path(evaluation["path"]))
    if "split" in evaluation:
        _, validation = training_split_indices(frame, cfg["data"])
        frame = frame.iloc[validation]
    return evaluation, frame


def check_datasets(cfg):
    training = training_snapshot(cfg)
    evaluation = evaluation_snapshot(cfg, training)
    if "split" in evaluation:
        frame = pd.read_csv(project_path(training["path"]))
        features, _ = prepare_frame(frame, cfg["data"])
        del features
        fit_indices, validation_indices = training_split_indices(frame, cfg["data"])
        return {"passed": True, "training_data": training, "evaluation_data": evaluation,
                "n_source_rows": len(frame), "n_training": len(fit_indices),
                "n_evaluation": len(validation_indices), "overlapping_rows": 0}
    eval_frame = pd.read_csv(project_path(evaluation["path"]))
    eval_features, _ = prepare_frame(eval_frame, cfg["data"])
    evaluation_hashes = row_hashes(eval_frame, cfg["data"]["target"])
    n_train = 0
    training_classes = set()
    for chunk in pd.read_csv(project_path(training["path"]), chunksize=200_000):
        features, labels = prepare_frame(chunk, cfg["data"], require_both_classes=False)
        training_classes.update(labels.unique())
        if list(features.columns) != list(eval_features.columns):
            raise ValueError("Training and evaluation feature schemas differ")
        if np.isin(row_hashes(chunk, cfg["data"]["target"]), evaluation_hashes).any():
            raise ValueError("Training and evaluation contain overlapping rows")
        n_train += len(chunk)
    if not n_train or training_classes != {0, 1}:
        raise ValueError("Training data must contain both classes")
    return {"passed": True, "training_data": training, "evaluation_data": evaluation,
            "n_training": n_train, "n_evaluation": len(eval_frame), "overlapping_rows": 0}


def metric_gate(metrics, minimums):
    if set(minimums) != {"recall", "f1", "pr_auc"}:
        raise ValueError("Thresholds for recall, f1, and pr_auc must all be configured")
    reasons = []
    for name, threshold in minimums.items():
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"Invalid threshold for {name}")
        value = metrics.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            reasons.append(f"{name}: missing or invalid metric")
        elif value < threshold:
            reasons.append(f"{name}={value:.6f} < {threshold:.6f}")
    return not reasons, reasons


def verify_manifest(manifest, cfg):
    if not manifest.get("run_id") or not manifest.get("model_uri") or not manifest.get("logged_model_id"):
        raise ValueError("Candidate run/model identity is missing")
    if manifest["model_uri"] != f"models:/{manifest['logged_model_id']}":
        raise ValueError("Candidate URI and logged model ID differ")
    if manifest.get("config_sha256") != fingerprint(manifest["config"]):
        raise ValueError("Manifest configuration fingerprint differs")
    if manifest["config_sha256"] != fingerprint(cfg):
        raise ValueError("Current configuration differs from the training configuration")


def verify_candidate(client, manifest):
    run = client.get_run(manifest["run_id"])
    if run.info.status != "FINISHED":
        raise ValueError("Candidate training run has not finished successfully")
    for key, value in {"git_commit": manifest["git_commit"],
                       "data_md5": manifest["training_data"]["md5"]}.items():
        if run.data.tags.get(key) != value:
            raise ValueError(f"Candidate run {key} differs from manifest")
    logged = client.get_logged_model(manifest["logged_model_id"])
    if logged.source_run_id != manifest["run_id"]:
        raise ValueError("Logged model belongs to a different training run")


def verify_validation(manifest, report, cfg):
    verify_manifest(manifest, cfg)
    if report.get("passed") is not True:
        raise ValueError("Candidate did not pass validation")
    if report.get("manifest_sha256") != fingerprint(manifest):
        raise ValueError("Validation belongs to a different candidate manifest")
    if report.get("run_id") != manifest["run_id"] or report.get("model_uri") != manifest["model_uri"]:
        raise ValueError("Validation run/model identity differs from manifest")
    if report.get("evaluation_data") != manifest.get("evaluation_data"):
        raise ValueError("Validation evaluation dataset differs from manifest")
    if report.get("thresholds") != cfg["validation"]["min_metrics"]:
        raise ValueError("Validation thresholds differ from current policy")
    passed, reasons = metric_gate(report.get("metrics", {}), report["thresholds"])
    if not passed:
        raise ValueError("Validation metrics failed: " + "; ".join(reasons))
