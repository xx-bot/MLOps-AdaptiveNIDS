"""Register a validated candidate in Staging; retain the Production model."""

import argparse
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import resolve_tracking_uri
from pipeline_utils import (fingerprint, load_pipeline_config, read_json, verify_candidate,
                            verify_validation, write_json)


def register_validated_model(manifest, validation, cfg, client=None, register_model=None, load_model=None):
    # Check every condition before creating or moving a registry version.
    verify_validation(manifest, validation, cfg)
    client = client or mlflow.MlflowClient()
    verify_candidate(client, manifest)
    name = cfg["mlflow"]["serving_model_name"]
    if "'" in name:
        raise ValueError("Model name must not contain a quote")
    candidate_id = fingerprint(manifest)
    versions = client.search_model_versions(f"name='{name}'")
    existing = [version for version in versions
                if version.tags.get("pipeline_candidate_id") == candidate_id]
    if len(existing) > 1:
        raise ValueError("Duplicate versions for this candidate require inspection")
    if existing:
        version = existing[0]
        if version.run_id != manifest["run_id"]:
            raise ValueError("Existing candidate registry version has a different run")
        if version.current_stage in {"Production", "Archived"}:
            raise ValueError("Will not move an already promoted/archived version back to Staging")
    else:
        register_model = register_model or mlflow.register_model
        version = register_model(manifest["model_uri"], name, tags={
            "pipeline_candidate_id": candidate_id,
            "git_commit": manifest["git_commit"],
            "data_md5": manifest["training_data"]["md5"],
            "validation_status": "passed",
            "validation_sha256": fingerprint(validation),
        })
    client.transition_model_version_stage(name, version.version, "Staging", archive_existing_versions=False)
    client.set_registered_model_alias(name, "challenger", version.version)
    actual = client.get_model_version(name, version.version)
    alias = client.get_model_version_by_alias(name, "challenger")
    if actual.current_stage != "Staging" or str(alias.version) != str(version.version):
        raise RuntimeError("Registry verification failed: stage or alias differs")
    (load_model or mlflow.pyfunc.load_model)(f"models:/{name}/{version.version}")
    return {"passed": True, "model_name": name, "version": str(version.version),
            "stage": actual.current_stage, "alias": "challenger", "run_id": manifest["run_id"],
            "model_uri": f"models:/{name}/{version.version}", "candidate_id": candidate_id,
            "validation_sha256": fingerprint(validation)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/training_manifest.json")
    parser.add_argument("--validation", default="reports/validation.json")
    parser.add_argument("--output", default="reports/registration.json")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    try:
        cfg = load_pipeline_config(args.overrides)
        mlflow.set_tracking_uri(resolve_tracking_uri(cfg["mlflow"]["tracking_uri"]))
        receipt = register_validated_model(read_json(args.manifest), read_json(args.validation), cfg)
    except Exception as error:
        write_json(args.output, {"passed": False, "reasons": [str(error)]})
        raise
    write_json(args.output, receipt)
    print(f"REGISTRY PASS: {receipt['model_name']} v{receipt['version']} -> Staging (@challenger)")


if __name__ == "__main__":
    main()
