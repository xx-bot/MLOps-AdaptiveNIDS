import argparse
import sys
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow import MlflowClient

# transition_stage.py lives in src/production/, so make src/ importable too (for config.py)
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import load_config, resolve_tracking_uri

STAGES = ["None", "Staging", "Production", "Archived"]
# MLflow deprecated stages in favour of aliases, so every stage move is mirrored on an alias
STAGE_ALIAS = {"Staging": "challenger", "Production": "champion"}


def show(client, name):
    print(f"\n{name}")
    # search results leave out aliases, so fetch each version in full
    for found in sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version)):
        mv = client.get_model_version(name, found.version)
        print(f"  v{mv.version:<3} stage={mv.current_stage:<10} aliases={','.join(mv.aliases) or '-':<20} "
              f"data={mv.tags.get('data_git_tag', '-')}")


if __name__ == "__main__":
    # python src/production/transition_stage.py 1 Staging
    # python src/production/transition_stage.py 2 Production --archive-existing
    # python src/production/transition_stage.py --list [--model ids-xgboost]
    parser = argparse.ArgumentParser(description="Move a model version through None -> Staging -> Production (-> Archived).")
    parser.add_argument("version", nargs="?")
    parser.add_argument("stage", nargs="?", choices=STAGES)
    parser.add_argument("--model", help="registered model name (default: params.yaml -> train.mlflow.serving_model_name)")
    parser.add_argument("--archive-existing", action="store_true",
                        help="move the version currently in that stage to Archived (Staging/Production only)")
    parser.add_argument("--list", action="store_true", help="only print every version with its stage and aliases")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="override params.yaml -> train (repeatable)")
    args = parser.parse_args()
    if not args.list and not (args.version and args.stage):
        parser.error("give VERSION STAGE, or --list")
    if args.archive_existing and args.stage not in STAGE_ALIAS:
        parser.error("--archive-existing only works when moving to Staging or Production")

    cfg = load_config(args.overrides)
    mlflow.set_tracking_uri(resolve_tracking_uri(cfg["mlflow"]["tracking_uri"]))
    client, name = MlflowClient(), args.model or cfg["mlflow"]["serving_model_name"]

    if args.list:
        show(client, name)
        sys.exit(0)

    before = client.get_model_version(name, args.version)
    client.transition_model_version_stage(name, args.version, args.stage,
                                          archive_existing_versions=args.archive_existing)

    # keep the aliases in step with the stage: drop the one of the stage it left, set the one of the stage it entered
    for alias in before.aliases:
        if alias in STAGE_ALIAS.values() and alias != STAGE_ALIAS.get(args.stage):
            client.delete_registered_model_alias(name, alias)
    if args.stage in STAGE_ALIAS:
        client.set_registered_model_alias(name, STAGE_ALIAS[args.stage], args.version)

    client.set_model_version_tag(name, args.version, "last_transition",
                                 f"{before.current_stage} -> {args.stage} at {datetime.now().isoformat(timespec='seconds')}")
    print(f"{name} v{args.version}: {before.current_stage} -> {args.stage}")
    show(client, name)
