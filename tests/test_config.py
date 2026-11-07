import pytest
import yaml

import config


@pytest.fixture
def params_file(tmp_path, monkeypatch):
    path = tmp_path / "params.yaml"
    path.write_text(
        yaml.safe_dump({
            "train": {
                "data": {"drop_columns": ["Stime", "Ltime"]},
                "xgb": {"max_depth": 6, "learning_rate": 0.3},
                "mlflow": {"register": True, "run_name": "baseline"},
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PARAMS_FILE", path)
    return path


def test_overrides_keep_yaml_types_and_do_not_change_defaults(params_file):
    original = params_file.read_bytes()

    result = config.load_config([
        "xgb.max_depth=10",
        "xgb.learning_rate=0.05",
        "mlflow.register=false",
        "mlflow.run_name=null",
        "data.drop_columns=[Stime]",
    ])

    assert result["xgb"] == {"max_depth": 10, "learning_rate": 0.05}
    assert type(result["xgb"]["max_depth"]) is int
    assert result["mlflow"]["register"] is False
    assert result["mlflow"]["run_name"] is None
    assert result["data"]["drop_columns"] == ["Stime"]
    assert params_file.read_bytes() == original
    assert config.load_config()["mlflow"]["register"] is True


def test_override_without_equals_is_rejected(params_file):
    with pytest.raises(ValueError, match="expects key=value"):
        config.load_config(["xgb.max_depth"])


def test_sqlite_uri_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert config.resolve_tracking_uri("sqlite:///mlflow.db") == (
        "sqlite:///" + (config.BASE_PATH / "mlflow.db").as_posix()
    )


@pytest.mark.parametrize("uri", [
    "https://mlflow.example.test",
    "sqlite:////var/lib/adaptive-nids/mlflow.db",
])
def test_external_and_absolute_tracking_uris_are_preserved(uri):
    assert config.resolve_tracking_uri(uri) == uri


def test_flatten_preserves_non_dict_values():
    result = config.flatten({
        "data": {"drop_columns": ["Stime", "Ltime"]},
        "xgb": {"max_depth": 6},
        "register": False,
    })

    assert result == {
        "data.drop_columns": ["Stime", "Ltime"],
        "xgb.max_depth": 6,
        "register": False,
    }


def test_runner_environment_and_explicit_override_precedence(params_file, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.example.test")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "ids-automation")
    assert config.load_config()["mlflow"]["tracking_uri"] == "http://tracking.example.test"
    assert config.load_config()["mlflow"]["experiment"] == "ids-automation"
    assert config.load_config(["mlflow.tracking_uri=sqlite:///custom.db"])["mlflow"]["tracking_uri"] == "sqlite:///custom.db"
