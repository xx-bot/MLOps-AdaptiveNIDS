import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

from pipeline_utils import (check_datasets, file_md5, fingerprint, metric_gate,
                            snapshot, verify_manifest, verify_validation)
from register_validated_model import register_validated_model


@pytest.fixture
def candidate():
    cfg = {
        "mlflow": {"serving_model_name": "adaptive-nids-xgb", "register": False},
        "validation": {"min_metrics": {"recall": 0.95, "f1": 0.95, "pr_auc": 0.95}},
    }
    manifest = {
        "run_id": "run-a", "model_uri": "models:/m-a", "logged_model_id": "m-a",
        "git_commit": "commit-a", "training_data": {"md5": "training-md5"},
        "evaluation_data": {"path": "evaluation.csv", "md5": "evaluation-md5"},
        "config": copy.deepcopy(cfg), "config_sha256": fingerprint(cfg),
    }
    report = {
        "passed": True, "manifest_sha256": fingerprint(manifest),
        "run_id": manifest["run_id"], "model_uri": manifest["model_uri"],
        "evaluation_data": manifest["evaluation_data"].copy(),
        "thresholds": cfg["validation"]["min_metrics"].copy(),
        "metrics": {"recall": 0.97, "f1": 0.98, "pr_auc": 0.99},
    }
    return cfg, manifest, report


def test_metric_equal_to_threshold_passes():
    minimums = {"recall": 0.95, "f1": 0.95, "pr_auc": 0.95}
    assert metric_gate(minimums, minimums) == (True, [])


@pytest.mark.parametrize("value", [0.94, None, float("nan"), float("inf"), -0.1, 1.1, True])
def test_missing_nonfinite_or_below_threshold_metric_fails(value):
    metrics = {"recall": value, "f1": 0.98, "pr_auc": 0.99}
    passed, reasons = metric_gate(metrics, {"recall": 0.95, "f1": 0.95, "pr_auc": 0.95})
    assert passed is False
    assert reasons and "recall" in reasons[0]


def test_absent_metric_fails():
    assert metric_gate({"recall": 0.98, "f1": 0.98},
                       {"recall": 0.95, "f1": 0.95, "pr_auc": 0.95})[0] is False


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="Invalid threshold"):
        metric_gate({}, {"recall": threshold, "f1": 0.95, "pr_auc": 0.95})


def test_missing_threshold_is_rejected():
    with pytest.raises(ValueError, match="must all be configured"):
        metric_gate({}, {"recall": 0.95})


@pytest.mark.parametrize("change", [
    {"passed": False}, {"passed": "true"}, {"manifest_sha256": "old-candidate"},
    {"run_id": "another-run"}, {"model_uri": "models:/m-other"},
    {"evaluation_data": {"md5": "another-dataset"}},
    {"metrics": {"recall": 0.1, "f1": 0.99, "pr_auc": 0.99}},
    {"thresholds": {"recall": 0.1, "f1": 0.1, "pr_auc": 0.1}},
])
def test_invalid_validation_never_contacts_registry(candidate, change):
    cfg, manifest, report = candidate
    report.update(change)
    client, register = Mock(), Mock()

    with pytest.raises(ValueError):
        register_validated_model(manifest, report, cfg, client=client, register_model=register)

    assert client.mock_calls == []
    register.assert_not_called()


def test_changed_configuration_is_rejected(candidate):
    cfg, manifest, _ = candidate
    cfg["validation"]["min_metrics"]["recall"] = 0.96
    with pytest.raises(ValueError, match="configuration differs"):
        verify_manifest(manifest, cfg)


def test_tampered_manifest_configuration_is_rejected(candidate):
    cfg, manifest, _ = candidate
    manifest["config"]["mlflow"]["register"] = True
    with pytest.raises(ValueError, match="fingerprint differs"):
        verify_manifest(manifest, cfg)


def test_valid_report_is_bound_to_candidate(candidate):
    cfg, manifest, report = candidate
    verify_validation(manifest, report, cfg)


def registry_client(manifest):
    client = Mock()
    client.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(status="FINISHED"),
        data=SimpleNamespace(tags={"git_commit": manifest["git_commit"],
                                   "data_md5": manifest["training_data"]["md5"]}),
    )
    client.get_logged_model.return_value = SimpleNamespace(source_run_id=manifest["run_id"])
    client.search_model_versions.return_value = []
    client.get_model_version.return_value = SimpleNamespace(current_stage="Staging")
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="7")
    return client


def test_only_validated_candidate_is_registered_and_verified(candidate):
    cfg, manifest, report = candidate
    client = registry_client(manifest)
    register = Mock(return_value=SimpleNamespace(version="7"))
    load = Mock()

    receipt = register_validated_model(manifest, report, cfg, client, register, load)

    assert receipt["stage"] == "Staging" and receipt["version"] == "7"
    assert register.call_args.args == ("models:/m-a", "adaptive-nids-xgb")
    client.transition_model_version_stage.assert_called_once_with(
        "adaptive-nids-xgb", "7", "Staging", archive_existing_versions=False,
    )
    client.set_registered_model_alias.assert_called_once_with("adaptive-nids-xgb", "challenger", "7")
    load.assert_called_once_with("models:/adaptive-nids-xgb/7")


def test_retry_after_stage_failure_reuses_version(candidate):
    cfg, manifest, report = candidate
    client = registry_client(manifest)
    version = SimpleNamespace(version="7", run_id="run-a", current_stage="None",
                              tags={"pipeline_candidate_id": fingerprint(manifest)})
    client.search_model_versions.side_effect = [[], [version]]
    client.transition_model_version_stage.side_effect = [RuntimeError("connection lost"), None]
    register = Mock(return_value=version)
    with pytest.raises(RuntimeError, match="connection lost"):
        register_validated_model(manifest, report, cfg, client, register, Mock())

    receipt = register_validated_model(manifest, report, cfg, client, register, Mock())

    register.assert_called_once()
    assert receipt["version"] == "7"


def test_model_from_another_run_is_rejected(candidate):
    cfg, manifest, report = candidate
    client = registry_client(manifest)
    client.get_logged_model.return_value.source_run_id = "another-run"
    register = Mock()
    with pytest.raises(ValueError, match="different training run"):
        register_validated_model(manifest, report, cfg, client, register)
    register.assert_not_called()
    client.transition_model_version_stage.assert_not_called()


def test_checksum_mismatch_is_rejected(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("changed data", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        snapshot(path, "0" * 32)


@pytest.mark.parametrize("overlap", [False, True])
def test_data_readiness_enforces_separation(tmp_path, overlap):
    training = tmp_path / "training.csv"
    evaluation = tmp_path / "evaluation.csv"
    pd.DataFrame({"dur": [1.0, 2.0], "Label": [0, 1]}).to_csv(training, index=False)
    pd.DataFrame({"dur": [1.0 if overlap else 3.0, 4.0], "Label": [1, 0]}).to_csv(evaluation, index=False)
    Path(str(training) + ".dvc").write_text(yaml.safe_dump({
        "outs": [{"path": training.name, "md5": file_md5(training)}],
    }))
    cfg = {"data": {"path": str(training), "target": "Label", "drop_columns": [], "float_columns": ["dur"]},
           "validation": {"data_path": str(evaluation), "data_md5": file_md5(evaluation)}}
    if overlap:
        with pytest.raises(ValueError, match="overlapping rows"):
            check_datasets(cfg)
    else:
        assert check_datasets(cfg)["passed"] is True
