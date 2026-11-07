"""Evaluate the exact candidate on pinned hold-out data and enforce the Staging gate."""

import argparse
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import resolve_tracking_uri
from pipeline_utils import (fingerprint, load_pipeline_config, metric_gate, prepare_frame,
                            project_path, read_json, snapshot, verify_candidate,
                            verify_manifest, write_json)


def evaluate(manifest, cfg):
    verify_manifest(manifest, cfg)
    policy = cfg["validation"]
    if policy["pr_auc_definition"] != "average_precision":
        raise ValueError("Only average_precision PR-AUC is supported")
    evaluation_data = snapshot(policy["data_path"], policy["data_md5"])
    if evaluation_data != manifest["evaluation_data"]:
        raise ValueError("Evaluation snapshot differs from training manifest")
    features, labels = prepare_frame(pd.read_csv(project_path(evaluation_data["path"])), cfg["data"])
    mlflow.set_tracking_uri(resolve_tracking_uri(cfg["mlflow"]["tracking_uri"]))
    client = mlflow.MlflowClient()
    verify_candidate(client, manifest)
    model = mlflow.pyfunc.load_model(manifest["model_uri"])
    predicted = model.predict(features)
    probabilities = model.unwrap_python_model().predict_proba(features)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Model produced invalid probabilities")
    metrics = {
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
    }
    passed, reasons = metric_gate(metrics, policy["min_metrics"])
    for key, value in metrics.items():
        client.log_metric(manifest["run_id"], "validation_" + key, value)
    client.set_tag(manifest["run_id"], "validation_status", "passed" if passed else "failed")
    return {"schema_version": 1, "passed": passed, "reasons": reasons,
            "manifest_sha256": fingerprint(manifest), "run_id": manifest["run_id"],
            "model_uri": manifest["model_uri"], "evaluation_data": evaluation_data,
            "n_evaluation": len(labels), "pr_auc_definition": "average_precision",
            "thresholds": policy["min_metrics"], "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/training_manifest.json")
    parser.add_argument("--output", default="reports/validation.json")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    try:
        report = evaluate(read_json(args.manifest), load_pipeline_config(args.overrides))
    except Exception as error:
        write_json(args.output, {"passed": False, "reasons": [str(error)]})
        raise
    write_json(args.output, report)
    print("EVALUATION", "PASS" if report["passed"] else "FAIL")
    print(json.dumps({"metrics": report["metrics"], "thresholds": report["thresholds"],
                      "reasons": report["reasons"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
