import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

# train.py lives in src/production/, so make src/ importable too (for config.py)
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import BASE_PATH, flatten, load_config, resolve_tracking_uri
from ids_model import IDSModel
from pipeline_utils import (fingerprint, prepare_frame, project_path, read_json,
                            training_snapshot, write_json)


class Frequency_encoder:
    def __init__(self):
        self.frequency_map = {}
        self.cat_cols = None

    def fit(self, X):
        self.cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        for col in self.cat_cols:
            self.frequency_map[f'frequency_{col}'] = X[col].value_counts(normalize=True).to_dict()

        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cat_cols:
            X[f'{col}_freq'] = X[col].map(self.frequency_map[f'frequency_{col}']).fillna(0)

        X = X.drop(columns=self.cat_cols)
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def git(*cmd):
    try:
        return subprocess.check_output(["git", *cmd], text=True, cwd=BASE_PATH,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def data_version(data_path):
    """(md5 from the .dvc file, newest git tag whose .dvc file has the same md5)."""
    dvc_file = BASE_PATH / (data_path + ".dvc")
    if not dvc_file.exists():
        return "untracked", "untracked"
    md5 = re.search(r"md5:\s*(\w+)", dvc_file.read_text()).group(1)
    for tag in git("tag", "--sort=-v:refname", "-l", "v*").splitlines():
        if md5 in git("show", f"{tag}:{data_path}.dvc"):
            return md5, tag
    return md5, "untagged"


def train_model(cfg, manifest_path, artifacts_dir, metrics_path, overrides=(), data_report=None):
    training_data = training_snapshot(cfg)
    evaluation_data = {"path": cfg["validation"]["data_path"], "md5": cfg["validation"]["data_md5"]}
    if data_report is not None:
        report = read_json(data_report)
        if report.get("passed") is not True or report.get("training_data") != training_data or report.get("evaluation_data") != evaluation_data:
            raise ValueError("Data validation report does not match this training/evaluation snapshot")
    df = pd.read_csv(project_path(cfg["data"]["path"]))
    X, y = prepare_frame(df, cfg["data"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=cfg["data"]["test_size"], random_state=cfg["data"]["seed"])

    fe = Frequency_encoder().fit(X_train)
    X_train_encoded = fe.transform(X_train)
    X_test_encoded = fe.transform(X_test)

    mlflow.set_tracking_uri(resolve_tracking_uri(cfg["mlflow"]["tracking_uri"]))
    artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT")
    if artifact_root and mlflow.get_experiment_by_name(cfg["mlflow"]["experiment"]) is None:
        # Keep artifacts outside the Actions checkout, which is cleaned each run.
        location = Path(artifact_root).expanduser().resolve().as_uri()
        mlflow.create_experiment(cfg["mlflow"]["experiment"], artifact_location=location)
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    params = cfg["xgb"]
    data_md5, data_tag = data_version(cfg["data"]["path"])
    run_name = cfg["mlflow"]["run_name"] or f"xgb-{data_tag}"
    commit = git("rev-parse", "HEAD")

    with mlflow.start_run(run_name=run_name) as run:
        # 1) Train
        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(X_train_encoded, y_train)

        # 2) Evaluate on the held-out test set
        y_pred = xgb_model.predict(X_test_encoded)
        y_proba = xgb_model.predict_proba(X_test_encoded)[:, 1]
        metrics = {
            "f1": f1_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),   # use probabilities, not hard labels
            "pr_auc": average_precision_score(y_test, y_proba),
        }

        # 3) Track params + metrics + the exact config + data/code version
        mlflow.log_params(flatten({"data": cfg["data"], "xgb": params}))
        mlflow.log_metrics(metrics)
        mlflow.log_params({"n_train": len(X_train), "n_test": len(X_test),
                           "n_features": X_train_encoded.shape[1]})
        mlflow.log_dict(cfg, "config.yaml")                   # the final config, verbatim
        mlflow.set_tags({"data_md5": data_md5, "data_git_tag": data_tag,
                         "git_commit": commit,
                         "github_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
                         "set_overrides": ",".join(overrides) or "none"})

        # 4) Save the pieces the wrapper needs
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        xgb_model.save_model(str(artifacts_dir / "xgb_model.json"))
        with open(artifacts_dir / "encoding_maps.json", "w") as f:
            json.dump({
                "freq_maps": {c: fe.frequency_map[f"frequency_{c}"] for c in fe.cat_cols},
                "feature_names": list(X_train_encoded.columns),
            }, f)

        # 5) Signature = input/output schema of the RAW input (before encoding)
        signature = infer_signature(X_test.head(100), y_pred[:100])

        # 6) Log the combined model; register it only if params.yaml / --set says so
        log_kwargs = {"registered_model_name": cfg["mlflow"]["model_name"]} if cfg["mlflow"]["register"] else {}
        model_info = mlflow.pyfunc.log_model(
            name="model",                                  # (older MLflow 2.x: artifact_path="model")
            python_model=IDSModel(),
            artifacts={
                "xgb_model": str(artifacts_dir / "xgb_model.json"),
                "encoding_maps": str(artifacts_dir / "encoding_maps.json"),
            },
            code_paths=[str(Path(__file__).parent / "ids_model.py")],            # ships the class with the model
            signature=signature,
            input_example=X_test.head(5),
            pip_requirements=[f"mlflow=={mlflow.__version__}", f"xgboost=={xgb.__version__}",
                              f"pandas=={pd.__version__}", f"numpy=={np.__version__}"],
            **log_kwargs,
        )

        # 7) Small file that DVC tracks as this stage's metrics (see dvc.yaml)
        metrics = {k: float(v) for k, v in metrics.items()}
        write_json(metrics_path, metrics)
        manifest = {
            "schema_version": 1,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "logged_model_id": model_info.model_id,
            "git_commit": commit,
            "working_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "training_data": training_data,
            "evaluation_data": evaluation_data,
            "config": cfg,
            "config_sha256": fingerprint(cfg),
            "metrics": metrics,
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        }

    # Publish only after the MLflow run has exited successfully.
    write_json(manifest_path, manifest)
    print("TRAIN SUCCESS | run_id:", manifest["run_id"], "| model_uri:", manifest["model_uri"])
    print("Internal test-split metrics:", json.dumps(metrics))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost, log it to MLflow and (optionally) register it.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="override params.yaml -> train, e.g. --set xgb.max_depth=10 (repeatable)")
    parser.add_argument("--manifest", default="reports/training_manifest.json")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--metrics-output", default="metrics/train_metrics.json")
    parser.add_argument("--data-report", help="report produced by check_training_data.py")
    args = parser.parse_args()
    Path(args.manifest).unlink(missing_ok=True)
    train_model(load_config(args.overrides), args.manifest, args.artifacts_dir,
                args.metrics_output, args.overrides, args.data_report)


if __name__ == "__main__":
    main()
