import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from preprocess_data import map_zeek_to_unsw_schema


@pytest.fixture
def zeek_connections():
    return pd.DataFrame({
        "duration": [0.0, np.nan, 2.0],
        "orig_bytes": [0.0, np.nan, 100.0],
        "resp_bytes": [0.0, np.nan, 50.0],
        "orig_pkts": [0.0, np.nan, 4.0],
        "resp_pkts": [0.0, np.nan, 2.0],
        "ts": [1000.0, 1001.0, 1002.0],
        "id.orig_p": [50000, 50000, 50000],
        "id.resp_p": [443, 443, 443],
        "id.orig_h": ["192.0.2.1"] * 3,
        "id.resp_h": ["198.51.100.1"] * 3,
        "proto": ["tcp"] * 3,
        "conn_state": ["SF"] * 3,
        "service": ["ssl"] * 3,
    })


def test_zero_and_missing_counters_produce_finite_features(zeek_connections):
    before = zeek_connections.copy(deep=True)

    result = map_zeek_to_unsw_schema(zeek_connections)
    features = result[["Sload", "Dload", "smeansz", "dmeansz"]]

    assert np.isfinite(features.to_numpy()).all()
    np.testing.assert_array_equal(features.iloc[:2].to_numpy(), np.zeros((2, 4)))
    assert features.iloc[2].tolist() == pytest.approx([400.0, 200.0, 25.0, 25.0])
    assert_frame_equal(zeek_connections, before)


def test_mapping_preserves_traffic_semantics(zeek_connections):
    result = map_zeek_to_unsw_schema(zeek_connections, default_label=1)

    assert result["Ltime"].tolist() == [1000.0, 1001.0, 1004.0]
    assert result["sport_is_ephemeral"].tolist() == [1, 1, 1]
    assert result["dsport_is_well_known"].tolist() == [1, 1, 1]
    assert result["dsport_is_critical"].tolist() == [1, 1, 1]
    assert result["ct_dst_src_ltm"].tolist() == [1, 2, 3]
    assert result["Label"].tolist() == [1, 1, 1]
