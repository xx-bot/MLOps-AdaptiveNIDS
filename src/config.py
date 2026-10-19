from pathlib import Path

import yaml

BASE_PATH = Path(__file__).resolve().parents[1]
PARAMS_FILE = BASE_PATH / "params.yaml"


def set_by_path(cfg: dict, dotted_key: str, value) -> None:
    *parents, leaf = dotted_key.split(".")
    node = cfg
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def load_config(overrides=()) -> dict:
    """Defaults from params.yaml (train section), then one-off `key=value` overrides.

    Keys are dotted paths relative to `train`, e.g. "xgb.max_depth=10".
    Values are parsed as YAML, so 10 is an int, 0.05 a float, false a bool.
    """
    with open(PARAMS_FILE, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["train"]

    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got: {item!r}")
        set_by_path(cfg, key.strip(), yaml.safe_load(raw))

    return cfg


def flatten(d: dict, prefix: str = "") -> dict:
    """{'xgb': {'max_depth': 10}} -> {'xgb.max_depth': 10}  (for mlflow.log_params)."""
    flat = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, name + "."))
        else:
            flat[name] = value
    return flat


def resolve_tracking_uri(uri: str) -> str:
    """sqlite:///mlflow.db -> absolute path under the project root, so the cwd never matters."""
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not Path(uri[len(prefix):]).is_absolute():
        return prefix + (BASE_PATH / uri[len(prefix):]).as_posix()
    return uri
