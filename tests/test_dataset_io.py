import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from dataset_io import save_dataset


@pytest.mark.parametrize("previous_file", ["dataset.csv", "base_train.csv", None])
def test_merge_uses_latest_dataset_then_base_training_data(tmp_path, previous_file):
    previous = pd.DataFrame({"dur": [1.0, 2.0], "Label": [0, 1]})
    incoming = pd.DataFrame({"dur": [3.0], "Label": [1]})
    before = incoming.copy(deep=True)
    if previous_file:
        previous.to_csv(tmp_path / previous_file, index=False)
    if previous_file == "dataset.csv":
        # The base must not be used when a newer cumulative dataset exists.
        pd.DataFrame({"dur": [99.0], "Label": [0]}).to_csv(
            tmp_path / "base_train.csv", index=False,
        )

    result_path = save_dataset(incoming, processed_dir=tmp_path, merge_with_latest=True)

    expected = pd.concat([previous, incoming], ignore_index=True) if previous_file else incoming
    assert_frame_equal(pd.read_csv(result_path), expected.reset_index(drop=True))
    assert_frame_equal(incoming, before)
    metrics = json.loads((tmp_path / "dataset_metrics.json").read_text())
    assert metrics == {
        "num_records": 3 if previous_file else 1,
        "num_features": 2,
        "label_distribution": {"0": 1, "1": 2} if previous_file else {"1": 1},
    }


def test_save_without_merge_replaces_previous_dataset(tmp_path):
    pd.DataFrame({"dur": [1.0], "Label": [0]}).to_csv(
        tmp_path / "dataset.csv", index=False,
    )
    incoming = pd.DataFrame({"dur": [2.0], "Label": [1]})

    path = save_dataset(incoming, processed_dir=tmp_path)

    assert_frame_equal(pd.read_csv(path), incoming)
