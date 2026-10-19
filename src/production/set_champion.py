import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# set_champion.py lives in src/production/, so make src/ importable too (for config.py)
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import BASE_PATH, load_config, resolve_tracking_uri

ALIAS = "champion"


def score(model, X, y):
    pred = model.predict(X)
    return {"f1": f1_score(y, pred), "recall": recall_score(y, pred),
            "precision": precision_score(y, pred), "accuracy": accuracy_score(y, pred)}


def verdict(res, gate):
    gain = res["challenger"][gate["metric"]] - res["champion"][gate["metric"]]
    guard = res["challenger"][gate["guard_metric"]] - res["champion"][gate["guard_metric"]]
    passed = gain >= gate["min_gain"] and guard >= -gate["max_guard_drop"]
    print(f"{gate['metric']} gain {gain:+.4f} (need >= {gate['min_gain']}), "
          f"{gate['guard_metric']} change {guard:+.4f} (need >= -{gate['max_guard_drop']}) -> "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    # python src/production/set_champion.py                 newest version -> champion (forced)
    # python src/production/set_champion.py 3              version 3 -> champion (forced, e.g. rollback)
    # python src/production/set_champion.py --eval data/processed/simulation_stream/batch_4.csv [--dry-run] [3]
    #                                                        compare on a batch neither model trained on; move
    #                                                        the alias only if the gate in params.yaml passes
    parser = argparse.ArgumentParser(description="Move the champion alias, optionally gated by an evaluation.")
    parser.add_argument("version", nargs="?", help="model version (default: newest)")
    parser.add_argument("--eval", metavar="CSV", help="a batch neither model has trained on; enables the gate")
    parser.add_argument("--dry-run", action="store_true", help="with --eval: compare only, never move the alias")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="override params.yaml -> train, e.g. --set promote.min_gain=0 (repeatable)")
    args = parser.parse_args()
    if args.dry_run and not args.eval:
        parser.error("--dry-run only makes sense together with --eval")

    cfg = load_config(args.overrides)
    mlflow.set_tracking_uri(resolve_tracking_uri(cfg["mlflow"]["tracking_uri"]))
    client, name = MlflowClient(), cfg["mlflow"]["model_name"]

    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        sys.exit(f"No registered versions of '{name}'. Run src/production/train.py first.")
    version = args.version or max(versions, key=lambda v: int(v.version)).version

    # mode 1: forced move (first champion, rollback, manual override)
    if not args.eval:
        client.set_registered_model_alias(name, ALIAS, version)
        print(f"{name}@{ALIAS} -> v{version}")
        sys.exit(0)

    # mode 2: gated move
    try:
        champion = client.get_model_version_by_alias(name, ALIAS).version
    except MlflowException:
        sys.exit(f"No '{ALIAS}' yet. Set the first one with: python src/production/set_champion.py <version>")
    if str(version) == str(champion):
        sys.exit(f"v{version} is already the champion; nothing to compare.")

    df = pd.read_csv(BASE_PATH / args.eval).drop(columns=cfg["data"]["drop_columns"], errors="ignore")
    df[cfg["data"]["float_columns"]] = df[cfg["data"]["float_columns"]].astype("float64")  # the signature is strict
    target = cfg["data"]["target"]
    X, y = df.drop(columns=[target]), df[target]

    res = {
        "champion": score(mlflow.pyfunc.load_model(f"models:/{name}/{champion}"), X, y),
        "challenger": score(mlflow.pyfunc.load_model(f"models:/{name}/{version}"), X, y),
    }
    print(f"eval set: {args.eval} ({len(df)} rows) | champion v{champion} vs challenger v{version}")
    print(pd.DataFrame(res).T.round(4))
    passed = verdict(res, cfg["promote"])

    promote = passed and not args.dry_run
    if not args.dry_run:                                      # a dry run leaves no trace on the run
        run_id = client.get_model_version(name, version).run_id
        for k, v in res["challenger"].items():
            client.log_metric(run_id, f"eval_{Path(args.eval).stem}_{k}", float(v))
        client.set_tag(run_id, "promoted", str(promote))
    if promote:
        client.set_registered_model_alias(name, ALIAS, version)
        print(f"{name}@{ALIAS} -> v{version}")
    sys.exit(0 if passed else 1)
