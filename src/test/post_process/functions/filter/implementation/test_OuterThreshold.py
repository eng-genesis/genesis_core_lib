import pytest
import numpy as np

from sdg_core_lib.post_process.functions.filter.implementation.OuterThreshold import (
    OuterThreshold,
)


@pytest.fixture
def instance_strict():
    params = [
        {"name": "upper_bound", "value": "10.0", "parameter_type": "float"},
        {"name": "lower_bound", "value": "2.0", "parameter_type": "float"},
        {"name": "upper_strict", "value": "True", "parameter_type": "bool"},
        {"name": "lower_strict", "value": "True", "parameter_type": "bool"},
        {"name": "extra_param", "value": "ignored", "parameter_type": "str"},
    ]
    return OuterThreshold.from_json(json_params=params)


@pytest.fixture
def instance_non_strict():
    params = [
        {"name": "upper_bound", "value": "10.0", "parameter_type": "float"},
        {"name": "lower_bound", "value": "2.0", "parameter_type": "float"},
        {"name": "upper_strict", "value": "False", "parameter_type": "bool"},
        {"name": "lower_strict", "value": "False", "parameter_type": "bool"},
    ]
    return OuterThreshold.from_json(json_params=params)


def test_check_parameters(instance_strict):
    assert instance_strict.upper_bound == 10.0
    assert isinstance(instance_strict.upper_bound, float)
    assert instance_strict.lower_bound == 2.0
    assert isinstance(instance_strict.lower_bound, float)
    assert instance_strict.upper_strict is True
    assert isinstance(instance_strict.upper_strict, bool)
    assert instance_strict.lower_strict is True
    assert isinstance(instance_strict.lower_strict, bool)
    assert "extra_param" not in instance_strict.__dict__


def test_apply_strict(instance_strict):
    # Test with strict=True (<= lower_bound or >= upper_bound)
    data = np.array([1.0, 2.0, 5.0, 10.0, 11.0])
    result_data, indexes, success = instance_strict.apply(n_rows=len(data), data=data)

    # OuterThreshold removes values inside interval (2.0 < x < 10.0)
    expected_data = np.array([1.0, 2.0, np.nan, 10.0, 11.0])
    expected_indexes = np.array([False, False, True, False, False])

    np.testing.assert_array_equal(result_data, expected_data)
    np.testing.assert_array_equal(indexes, expected_indexes)
    assert success is True


def test_apply_non_strict(instance_non_strict):
    # Test with strict=False (< lower_bound or > upper_bound)
    data = np.array([1.0, 2.0, 5.0, 10.0, 11.0])
    result_data, indexes, success = instance_non_strict.apply(
        n_rows=len(data), data=data
    )

    # OuterThreshold removes values inside interval (2.0 <= x <= 10.0)
    expected_data = np.array([1.0, np.nan, np.nan, np.nan, 11.0])
    expected_indexes = np.array([False, True, True, True, False])

    np.testing.assert_array_equal(result_data, expected_data)
    np.testing.assert_array_equal(indexes, expected_indexes)
    assert success is True


def test_apply_edge_cases(instance_strict):
    # Test with empty array
    empty_data = np.array([])
    result_data, indexes, success = instance_strict.apply(n_rows=0, data=empty_data)
    assert result_data.shape == (0,)
    assert indexes.shape == (0,)
    assert success is True

    # Test with all values inside interval
    inside_data = np.array([3.0, 4.0, 5.0])
    result_data, indexes, success = instance_strict.apply(
        n_rows=len(inside_data), data=inside_data
    )
    # OuterThreshold removes all values inside interval
    expected_data = np.array([np.nan, np.nan, np.nan])
    expected_indexes = np.array([True, True, True])
    np.testing.assert_array_equal(result_data, expected_data)
    np.testing.assert_array_equal(indexes, expected_indexes)
    assert success is True

    # Test with all values outside interval
    outside_data = np.array([1.0, 11.0])
    result_data, indexes, success = instance_strict.apply(
        n_rows=len(outside_data), data=outside_data
    )
    # OuterThreshold keeps all values (none to remove)
    expected_data = np.array([1.0, 11.0])
    expected_indexes = np.array([False, False])
    np.testing.assert_array_equal(result_data, expected_data)
    np.testing.assert_array_equal(indexes, expected_indexes)
    assert success is True


def test_check_parameters_invalid_boundary():
    # Test with lower_bound > upper_bound
    params = [
        {"name": "upper_bound", "value": "2.0", "parameter_type": "float"},
        {"name": "lower_bound", "value": "10.0", "parameter_type": "float"},
        {"name": "upper_strict", "value": "True", "parameter_type": "bool"},
        {"name": "lower_strict", "value": "True", "parameter_type": "bool"},
    ]
    with pytest.raises(ValueError):
        OuterThreshold.from_json(json_params=params)
