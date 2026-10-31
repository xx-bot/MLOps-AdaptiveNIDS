import argparse
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient

# export_model_metadata.py lives in src/production/, so make src/ importable too (for config.py)
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import BASE_PATH, load_config, resolve_tracking_uri


if __name__ == "__main__":
    # python src/production/export_model_metadata.py --reason "..."
    #   models/production_model.yaml   which model is in Production and what made it (Git)
    #   models/production/             the model files themselves (dvc add, not Git)
    parser = argparse.ArgumentParser(description="Export the reference to (and a copy of) the model in a stage.")
    parser.add_argument("--model", help="registered model name (default: params.yaml -> train.mlflow.serving_model_name)")
    parser.add_argument("--stage", default="Production")
    parser.add_argument("--out", default="models/production", help="model copy, tracked with dvc add")
    parser.add_argument("--meta", default="models/production_model.yaml", help="reference file, tracked with Git")
    parser.add_argument("--reason", default="", help="why this version is the active one")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="override params.yaml -> train (repeatable)")
    args = parser.parse_args()

    cfg = load_config(args.overrides)
    mlflow.set_tracking_uri(resolve_tracking_uri(cfg["mlflow"]["tracking_uri"]))
    client, name = MlflowClient(), args.model or cfg["mlflow"]["serving_model_name"]

    in_stage = [mv for mv in client.search_model_versions(f"name='{name}'") if mv.current_stage == args.stage]
    if not in_stage:
        sys.exit(f"No version of '{name}' is in {args.stage}. Run transition_stage.py first.")
    mv = client.get_model_version(name, max(in_stage, key=lambda v: int(v.version)).version)  # search omits aliases
    run = client.get_run(mv.run_id)

    # copy the model files (MLmodel, xgb_model.json, encoding_maps.json, code/) for DVC
    out_dir = BASE_PATH / args.out
    shutil.rmtree(out_dir, ignore_errors=True)
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{mv.version}", dst_path=tmp)
        shutil.copytree(local, out_dir)

    tags, params = run.data.tags, run.data.params
    meta = {
        "model": {
            "name": name,
            "version": int(mv.version),
            "stage": mv.current_stage,
            "aliases": list(mv.aliases),
            "uri": f"models:/{name}/{mv.version}",
            "stage_uri": f"models:/{name}/{mv.current_stage}",
            "source": mv.source,                     # the MLflow logged model, models:/m-...
            "run_id": mv.run_id,
            "run_name": run.info.run_name,
            "description": mv.description or "",
        },
        "reason": args.reason,
        "local_copy": {"path": args.out, "dvc_file": f"{args.out}.dvc"},
        "metrics": {k: round(v, 4) for k, v in sorted(run.data.metrics.items())},
        "params": {k: v for k, v in sorted(params.items()) if k.startswith("xgb.")},
        "lineage": {
            "data_path": params.get("data.path"),
            "data_dvc_file": f"{params.get('data.path')}.dvc",
            "data_md5": tags.get("data_md5"),
            "data_git_tag": tags.get("data_git_tag"),
            "git_commit": tags.get("git_commit"),
        },
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path = BASE_PATH / args.meta
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")

    print(f"{name} v{mv.version} ({args.stage}) -> {args.out}/ + {args.meta}")
    print(f"Next: dvc add {args.out} && git add {args.out}.dvc models/.gitignore {args.meta} "
          f"&& git commit && git tag model-v{mv.version}")
