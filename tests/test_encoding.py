import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ids_model import IDSModel
from train import Frequency_encoder


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
def test_frequency_encoding_uses_training_distribution_for_unseen_data(dtype):
    training = pd.DataFrame({
        "proto": pd.Series(["tcp", "tcp", "udp"], dtype=dtype),
        "dur": [1.0, 2.0, 3.0],
    })
    evaluation = pd.DataFrame({
        "proto": pd.Series(["tcp", "icmp", "tcp", "tcp"], dtype=dtype),
        "dur": [4.0, 5.0, 6.0, 7.0],
    })
    training_before = training.copy(deep=True)
    evaluation_before = evaluation.copy(deep=True)

    encoder = Frequency_encoder().fit(training)
    encoded = encoder.transform(evaluation)

    assert list(encoded.columns) == ["dur", "proto_freq"]
    assert encoded["proto_freq"].tolist() == pytest.approx([2 / 3, 0, 2 / 3, 2 / 3])
    assert encoded["dur"].tolist() == evaluation["dur"].tolist()
    assert_frame_equal(training, training_before)
    assert_frame_equal(evaluation, evaluation_before)
    assert encoder.transform(training)["proto_freq"].tolist() == pytest.approx([2 / 3, 2 / 3, 1 / 3])


def test_numeric_only_input_keeps_values_and_columns():
    data = pd.DataFrame({"dur": [0.0, 1.5], "sbytes": [0, 80]})

    assert_frame_equal(Frequency_encoder().fit_transform(data), data)


def test_serving_encoding_matches_training_and_preserves_feature_order():
    training = pd.DataFrame({
        "proto": ["tcp", "tcp", "udp"],
        "service": ["http", "http", "dns"],
        "dur": [1.0, 2.0, 3.0],
    })
    evaluation = pd.DataFrame({
        "service": ["http", "unknown"],
        "dur": [4.0, 5.0],
        "proto": ["tcp", "icmp"],
    })
    before = evaluation.copy(deep=True)
    encoder = Frequency_encoder().fit(training)
    expected = encoder.transform(evaluation)[["dur", "proto_freq", "service_freq"]]

    wrapper = IDSModel()
    wrapper.freq_maps = {
        column: encoder.frequency_map[f"frequency_{column}"]
        for column in encoder.cat_cols
    }
    wrapper.feature_names = ["dur", "proto_freq", "service_freq"]

    class RecordingClassifier:
        def predict(self, features):
            assert_frame_equal(features, expected)
            return np.array([1, 0])

    wrapper.model = RecordingClassifier()

    np.testing.assert_array_equal(wrapper.predict(None, evaluation), [1, 0])
    assert_frame_equal(evaluation, before)


def test_serving_rejects_missing_required_numeric_feature():
    wrapper = IDSModel()
    wrapper.freq_maps = {"proto": {"tcp": 1.0}}
    wrapper.feature_names = ["dur", "proto_freq"]

    with pytest.raises(KeyError, match="dur"):
        wrapper._encode(pd.DataFrame({"proto": ["tcp"]}))
