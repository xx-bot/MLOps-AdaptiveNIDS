from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pipeline_utils import (check_datasets, evaluation_snapshot, file_md5,
                            load_evaluation_frame, row_hashes, training_split_indices)


@pytest.fixture
def split_dataset(tmp_path):
    # Each flow occurs three times: copies must stay together despite different row IDs.
    frame = pd.DataFrame({"Stime": np.repeat(np.arange(50), 3),
                          "dur": np.repeat(np.arange(50) / 10, 3),
                          "proto": "tcp", "Label": np.repeat(np.arange(50) % 2, 3)})
    path = tmp_path / "dataset.csv"
    frame.to_csv(path, index=False)
    Path(str(path) + ".dvc").write_text(yaml.safe_dump({
        "outs": [{"path": path.name, "md5": file_md5(path)}],
    }))
    cfg = {"data": {"path": str(path), "target": "Label", "drop_columns": ["Stime"],
                    "float_columns": ["dur"], "test_size": 0.2, "seed": 42},
           "validation": {"source": "training_split"}}
    return cfg, pd.read_csv(path)


def test_split_is_reproducible_and_keeps_duplicate_flows_together(split_dataset):
    cfg, frame = split_dataset
    training, validation = training_split_indices(frame, cfg["data"])
    repeated_training, repeated_validation = training_split_indices(frame.copy(), cfg["data"])
    hashes = row_hashes(frame, cfg["data"]["target"])

    np.testing.assert_array_equal(training, repeated_training)
    np.testing.assert_array_equal(validation, repeated_validation)
    assert set(training).isdisjoint(validation)
    assert set(training) | set(validation) == set(range(len(frame)))
    assert not np.isin(hashes[training], hashes[validation]).any()
    assert set(frame.iloc[training].Label) == set(frame.iloc[validation].Label) == {0, 1}


def test_data_check_succeeds_without_hidden_test_file(split_dataset):
    cfg, frame = split_dataset
    report = check_datasets(cfg)

    assert report["passed"] is True
    assert report["n_source_rows"] == len(frame)
    assert report["n_training"] + report["n_evaluation"] == len(frame)
    assert report["n_evaluation"] == 30
    assert report["overlapping_rows"] == 0
    assert report["evaluation_data"]["split"]["method"] == "group_shuffle_by_flow"
    assert not (Path(cfg["data"]["path"]).parent / "simulation_stream").exists()


def test_evaluation_loads_only_validation_partition(split_dataset):
    cfg, frame = split_dataset
    training, validation = training_split_indices(frame, cfg["data"])
    identity, evaluation = load_evaluation_frame(cfg)

    pd.testing.assert_frame_equal(evaluation, frame.iloc[validation])
    assert not set(evaluation.index).intersection(training)
    assert identity == check_datasets(cfg)["evaluation_data"]


@pytest.mark.parametrize("key,value", [("seed", 123), ("test_size", 0.4)])
def test_evaluation_identity_binds_split_policy(split_dataset, key, value):
    cfg, _ = split_dataset
    original = evaluation_snapshot(cfg)
    cfg["data"][key] = value
    changed = evaluation_snapshot(cfg)

    assert changed["md5"] == original["md5"]
    assert changed["split"] != original["split"]


def test_unknown_evaluation_source_is_rejected(split_dataset):
    cfg, _ = split_dataset
    cfg["validation"]["source"] = "typo"
    with pytest.raises(ValueError, match="Unknown validation data source"):
        check_datasets(cfg)


def test_split_rejects_partition_without_both_classes(split_dataset):
    cfg, frame = split_dataset
    frame["Label"] = 0
    with pytest.raises(ValueError, match="both classes"):
        training_split_indices(frame, cfg["data"])
