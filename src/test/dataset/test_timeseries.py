import pytest
import os
import shutil
import numpy as np

from sdg_core_lib.dataset.datasets import TimeSeries
from sdg_core_lib.preprocess.strategies.vae_strategy import (
    TabularVAEPreprocessingStrategy,
)
from sdg_core_lib.preprocess.table_processor import TableProcessor

current_folder = os.path.dirname(os.path.abspath(__file__))

# Sample time series data
time_series_data = [
    {
        "column_name": "experiment_id",
        "column_type": "group_index",
        "column_data": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "column_datatype": "int32",
    },
    {
        "column_name": "time",
        "column_type": "primary_key",
        "column_data": [0, 1, 2, 0, 1, 2, 0, 1, 2],
        "column_datatype": "int32",
    },
    {
        "column_name": "value1",
        "column_type": "continuous",
        "column_data": [1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3],
        "column_datatype": "float32",
    },
    {
        "column_name": "category",
        "column_type": "categorical",
        "column_data": ["a", "b", "c", "a", "b", "c", "a", "b", "c"],
        "column_datatype": "str",
    },
]


@pytest.fixture()
def temp_folder():
    folder = os.path.join(current_folder, "temp_timeseries")
    os.makedirs(folder, exist_ok=True)
    yield folder
    shutil.rmtree(folder)


@pytest.fixture()
def sample_timeseries(temp_folder):
    return TimeSeries.from_json(time_series_data)


def test_timeseries_creation(temp_folder):
    ts = TimeSeries.from_json(time_series_data)

    # Verify basic properties
    assert len(ts.columns) == 4
    assert ts.group_index is not None
    assert ts._get_experiment_length() == 3  # 3 time steps per experiment


def test_timeseries_invalid_creation(temp_folder):
    # Missing group_index
    invalid_data = [d for d in time_series_data if d["column_name"] != "experiment_id"]

    with pytest.raises(ValueError, match="Time series must have a group index"):
        TimeSeries.from_json(invalid_data)


def test_timeseries_clone(sample_timeseries):
    # Create a 3D array: (batch=3, features=2, time_steps=3)
    new_data = np.random.rand(3, 2, 3)  # 3 experiments, 2 features, 3 time steps
    cloned_ts = sample_timeseries.clone(new_data)

    # Verify the shape and structure
    assert cloned_ts.get_computing_data().shape == (
        3,
        2,
        3,
    )  # (batch, features, time_steps)
    assert len(cloned_ts.columns) == 4  # Original columns should be preserved
    assert cloned_ts._get_experiment_length() == 3  # Time steps should be preserved


def test_timeseries_computing_data(sample_timeseries):
    # Test the shape transformation of computing data
    computing_data = sample_timeseries.get_computing_data()
    assert computing_data.shape == (3, 2, 3)  # (batch, features, time_steps)
    print(computing_data.dtype)

    # Verify data is grouped by experiment
    test1 = [1.1, 1.2, 1.3]
    test2 = [2.1, 2.2, 2.3]
    assert np.array_equal(
        computing_data[0, 0, :].astype("float"), test1
    )  # First experiment, first feature
    assert np.array_equal(
        computing_data[1, 0, :].astype("float"), test2
    )  # Second experiment, first feature


def test_timeseries_invalid_computing_data(sample_timeseries):
    # Test with invalid data shape
    invalid_data = np.random.rand(3, 2)  # 2D instead of 3D
    with pytest.raises(ValueError, match="Data must be a 3D array"):
        sample_timeseries.clone(invalid_data)


def test_timeseries_preprocess(sample_timeseries, temp_folder):
    # Test preprocessing (should normalize continuous columns)
    processor = TableProcessor(temp_folder).set_strategy(
        TabularVAEPreprocessingStrategy()
    )
    preprocessed = sample_timeseries.preprocess(processor)
    computing_data = preprocessed.get_computing_data()

    # Check if one-hot encoding is applied to categorical columns and changed shape
    assert computing_data.shape == (3, 4, 3)

    # Verify continuous data is normalized (mean ~0, std ~1 per feature)
    for numeric in preprocessed.get_numeric_columns():
        feature_data = numeric.get_data().flatten()
        assert np.isclose(np.mean(feature_data), 0, atol=1e-7)
        assert np.isclose(np.std(feature_data), 1, atol=1e-7)

    for categoric in preprocessed.get_categorical_columns():
        feature_data = categoric.get_data().flatten()
        assert np.all(
            np.isin(feature_data, [0, 1])
        )  # Categorical data should be one-hot encoded


def test_timeseries_postprocess(sample_timeseries, temp_folder):
    # Test round-trip preprocessing and postprocessing
    processor = TableProcessor(temp_folder).set_strategy(
        TabularVAEPreprocessingStrategy()
    )
    preprocessed = sample_timeseries.preprocess(processor)
    postprocessed = preprocessed.postprocess(processor)

    # Verify the data is restored (approximately due to floating point)
    original_data = sample_timeseries.get_computing_data()
    restored_data = postprocessed.get_computing_data()
    assert np.allclose(
        original_data[:, 0, :].astype("float"),
        restored_data[:, 0, :].astype("float"),
        atol=1e-6,
    )
    assert np.all(original_data[:, 1, :] == restored_data[:, 1, :])


def test_timeseries_invalid_experiment_lengths(temp_folder):
    # Create data with inconsistent experiment lengths
    invalid_data = [
        {
            "column_name": "experiment_id",
            "column_type": "group_index",
            "column_data": [1, 1, 2, 2, 2],  # Different lengths
            "column_datatype": "int32",
        },
        {
            "column_name": "time",
            "column_type": "primary_key",
            "column_data": [0, 1, 0, 1, 2],  # Different lengths
            "column_datatype": "int32",
        },
        {
            "column_name": "value",
            "column_type": "continuous",
            "column_data": [1.1, 1.2, 2.1, 2.2, 2.3],
            "column_datatype": "float32",
        },
    ]

    ts = TimeSeries.from_json(invalid_data)

    # Should raise when trying to get computing data due to inconsistent lengths
    with pytest.raises(ValueError, match="Experiments have different lengths"):
        ts.get_computing_data()
