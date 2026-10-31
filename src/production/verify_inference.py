import argparse
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# verify_inference.py lives in src/production/, so make src/ importable too (for config.py)
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import BASE_PATH, load_config, resolve_tracking_uri

STAGE_ALIAS = {"Staging": "challenger", "Production": "champion"}


if __name__ == "__main__":
    # python src/production/verify_inference.py                       Production, full batch 7
    # python src/production/verify_inference.py --stage Staging --rows 20000
    parser = argparse.ArgumentParser(description="Load the model by stage, the way an inference service would, and use it.")
    parser.add_argument("--model", help="registered model name (default: params.yaml -> train.mlflow.serving_model_name)")
    parser.add_argument("--stage", default="Production")
    parser.add_argument("--eval", default="data/processed/simulation_stream/batch_7.csv", metavar="CSV")
    parser.add_argument("--rows", type=int, default=0, help="only the first N rows (0 = all)")
    parser.add_argument("--local", default="models/production", help="DVC copy to compare with (Production only)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="override params.yaml -> train (repeatable)")
    args = parser.parse_args()

    cfg = load_config(args.overrides)
    mlflow.set_tracking_uri(resolve_tracking_uri(cfg["mlflow"]["tracking_uri"]))
    client, name = MlflowClient(), args.model or cfg["mlflow"]["serving_model_name"]

    in_stage = [mv for mv in client.search_model_versions(f"name='{name}'") if mv.current_stage == args.stage]
    if not in_stage:
        sys.exit(f"No version of '{name}' is in {args.stage}.")
    mv = max(in_stage, key=lambda v: int(v.version))

    # 1) load by stage, never by version number
    uri = f"models:/{name}/{args.stage}"
    start = time.perf_counter()
    model = mlflow.pyfunc.load_model(uri)
    load_s = time.perf_counter() - start
    print(f"loaded {uri} -> v{mv.version} (run {mv.run_id}, data {mv.tags.get('data_git_tag', '?')}) in {load_s:.2f}s")

    # 2) raw features in, labels out
    df = pd.read_csv(BASE_PATH / args.eval, nrows=args.rows or None)
    df = df.drop(columns=cfg["data"]["drop_columns"], errors="ignore")
    df[cfg["data"]["float_columns"]] = df[cfg["data"]["float_columns"]].astype("float64")  # the signature is strict
    target = cfg["data"]["target"]
    X, y = df.drop(columns=[target]), df[target]

    start = time.perf_counter()
    pred = model.predict(X)
    batch_s = time.perf_counter() - start
    start = time.perf_counter()
    one = model.predict(X.head(1))
    single_ms = (time.perf_counter() - start) * 1000

    print(f"eval set: {args.eval} ({len(X)} rows, {int(y.sum())} attacks)")
    print(f"  recall {recall_score(y, pred):.4f} | f1 {f1_score(y, pred):.4f} | "
          f"precision {precision_score(y, pred):.4f} | accuracy {accuracy_score(y, pred):.4f}")
    print(f"  single request: {single_ms:.1f} ms -> prediction {int(one[0])} (label {int(y.iloc[0])})")
    print(f"  batch: {batch_s:.2f}s, {len(X) / batch_s:,.0f} rows/s")

    # 3) the alias and the DVC copy must serve the very same model
    checks = {}
    if args.stage in STAGE_ALIAS:
        checks[f"alias @{STAGE_ALIAS[args.stage]}"] = f"models:/{name}@{STAGE_ALIAS[args.stage]}"
    if args.stage == "Production" and (BASE_PATH / args.local).exists():
        checks[f"DVC copy {args.local}/"] = str(BASE_PATH / args.local)

    sample = X.head(1000)
    ok = True
    for label, src in checks.items():
        same = np.array_equal(mlflow.pyfunc.load_model(src).predict(sample), pred[:len(sample)])
        ok &= same
        print(f"  {label:<28} same predictions: {same}")

    print("READY" if ok else "MISMATCH: the stage, alias and DVC copy do not point to the same model")
    sys.exit(0 if ok else 1)
