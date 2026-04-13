import shutil
import pytest
from unittest.mock import Mock, patch
from sdg_core_lib.job import Job
from sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE import (
    TabularVAE,
)
from sdg_core_lib.data_generator.models.VAEs.implementation.TimeSeriesVAE import (
    TimeSeriesVAE,
)
from sdg_core_lib.preprocess.table_processor import TableProcessor
from sdg_core_lib.preprocess.strategies.vae_strategy import (
    TabularVAEPreprocessingStrategy,
    TimeSeriesVAEPreprocessingStrategy,
)
from sdg_core_lib.preprocess.strategies.base_strategy import BasePreprocessingStrategy
import json
import os
from loguru import logger

current_folder = os.path.dirname(os.path.abspath(__file__))
current_folder = os.path.join(current_folder, "test_files")
train_request = json.load(open(os.path.join(current_folder, "train_test.json")))
train_request_2 = json.load(open(os.path.join(current_folder, "train_test_2.json")))
train_request_3 = json.load(open(os.path.join(current_folder, "train_test_3.json")))

infer_request = json.load(open(os.path.join(current_folder, "infer_test.json")))
infer_nodata_request = json.load(
    open(os.path.join(current_folder, "infer_test_nodata.json"))
)
infer_nodata_request_wrong = json.load(
    open(os.path.join(current_folder, "infer_test_nodata_wrong.json"))
)
output_folder = os.path.join(current_folder, "outputs")


@pytest.fixture()
def setup():
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)


@pytest.fixture()
def teardown():
    yield
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)


def test_train_timeseries(setup):
    model_info = train_request_2["model"]
    dataset = train_request_2["dataset"]
    n_rows = train_request_2["n_rows"]
    save_filepath = output_folder

    results, metrics, model, data = Job(
        model_info=model_info,
        dataset=dataset,
        n_rows=n_rows,
        save_filepath=save_filepath,
    ).train()
    assert isinstance(results, list)
    assert results is not None
    assert model is not None
    assert data is not None


def test_train(setup):
    model_info = train_request["model"]
    dataset = train_request["dataset"]
    n_rows = train_request["n_rows"]
    save_filepath = output_folder

    results, metrics, model, data = Job(
        model_info=model_info,
        dataset=dataset,
        n_rows=n_rows,
        save_filepath=save_filepath,
    ).train()

    logger.add(
        os.path.join(current_folder, "out.log"),
    )
    assert isinstance(results, list)
    assert results is not None
    logger.info(results)
    assert metrics is not None
    logger.info(metrics)
    assert model is not None
    logger.info(model.training_info.to_json())
    assert data is not None
    logger.info(data)


def test_infer(setup):
    model_info = infer_request["model"]
    model_info["image"] = output_folder
    dataset = infer_request["dataset"]
    n_rows = infer_request["n_rows"]
    save_filepath = output_folder

    (
        results,
        metrics,
    ) = Job(
        model_info=model_info,
        dataset=dataset,
        n_rows=n_rows,
        save_filepath=save_filepath,
    ).infer()
    assert isinstance(results, list)
    assert results is not None
    assert metrics is not None


def test_infer_nodata_wrong(setup):
    model_info = infer_nodata_request_wrong["model"]
    model_info["image"] = output_folder
    n_rows = infer_nodata_request_wrong["n_rows"]
    save_filepath = output_folder

    with pytest.raises(ValueError) as exception_info:
        (
            _,
            _,
        ) = Job(
            model_info=model_info,
            dataset={"dataset_type": "table", "data": []},
            n_rows=n_rows,
            save_filepath=save_filepath,
        ).infer()
    assert exception_info.type is ValueError


def test_infer_nodata(setup, teardown):
    model_info = infer_nodata_request["model"]
    model_info["image"] = output_folder
    n_rows = infer_nodata_request["n_rows"]
    save_filepath = output_folder

    results, metrics = Job(
        model_info=model_info,
        dataset={"dataset_type": "table", "data": []},
        n_rows=n_rows,
        save_filepath=save_filepath,
    ).infer()
    assert isinstance(results, list)
    assert results is not None
    print(results)
    assert metrics is not None
    print(metrics)


def test_generate_from_functions():
    functions = [
        {
            "feature": "test_feature",
            "function_reference": "sdg_core_lib.post_process.functions.generation.implementation.NormalDistributionSample.NormalDistributionSample",
            "parameters": [
                {"name": "mean", "value": "0.0", "parameter_type": "float"},
                {
                    "name": "standard_deviation",
                    "value": "1.0",
                    "parameter_type": "float",
                },
            ],
        }
    ]
    n_rows = 100
    dataset = Job(
        n_rows, functions=functions, dataset={"dataset_type": "", "data": []}
    ).generate_from_functions()
    assert len(dataset) == 1
    dataset_data = dataset[0]
    assert len(dataset_data.get("column_data")) == n_rows
    assert dataset_data.get("column_name") == "test_feature"


def test_model_factory_with_tabular_vae():
    """Test model factory creates TabularVAE with correct parameters"""
    job = Job(
        n_rows=100,
        model_info={
            "algorithm_name": "sdg_core_lib.data_generator.models.keras.implementation.TabularVAE.TabularVAE",
            "model_name": "test_tabular_vae",
            "input_shape": "(10,)",
        },
        dataset={"dataset_type": "table", "data": []},
        save_filepath="/tmp/test",
    )

    mock_preprocess_data = Mock()
    mock_preprocess_data.get_shape_for_model.return_value = (10,)
    mock_preprocess_data.to_skeleton.return_value = {"metadata": "test"}

    with patch.object(job, "_get_model_class", return_value=TabularVAE):
        model = job._model_factory(mock_preprocess_data)

        assert isinstance(model, TabularVAE)
        assert model.model_name == "test_tabular_vae"
        assert model.input_shape == (10,)
        assert model._metadata == {"metadata": "test"}

    # Verify that the correct mapping is used
    from sdg_core_lib.mappings import TableMapping

    assert job._Job__dataset_mapping == TableMapping


def test_model_factory_with_timeseries_vae():
    """Test model factory creates TimeSeriesVAE with correct parameters"""
    job = Job(
        n_rows=100,
        model_info={
            "algorithm_name": "sdg_core_lib.data_generator.models.keras.implementation.TimeSeriesVAE.TimeSeriesVAE",
            "model_name": "test_timeseries_vae",
        },
        dataset={"dataset_type": "time_series", "data": []},
        save_filepath="/tmp/test",
    )

    mock_preprocess_data = Mock()
    mock_preprocess_data.get_shape_for_model.return_value = "(20, 30)"
    mock_preprocess_data.to_skeleton.return_value = {"timeseries": "metadata"}

    with patch.object(job, "_get_model_class", return_value=TimeSeriesVAE):
        model = job._model_factory(mock_preprocess_data)

        assert isinstance(model, TimeSeriesVAE)
        assert model.model_name == "test_timeseries_vae"
        assert model.input_shape == (20, 30)
        assert model._metadata == {"timeseries": "metadata"}

    # Verify that the correct mapping is used
    from sdg_core_lib.mappings import TimeSeriesMapping

    assert job._Job__dataset_mapping == TimeSeriesMapping


def test_model_factory_without_preprocess_data():
    """Test model factory when no preprocess data is provided"""
    job = Job(
        n_rows=100,
        model_info={
            "algorithm_name": "sdg_core_lib.data_generator.models.keras.implementation.TabularVAE.TabularVAE",
            "model_name": "test_model",
            "input_shape": "(5,)",
        },
        dataset={"dataset_type": "", "data": []},
        save_filepath="/tmp/test",
    )

    # Create a mock preprocess_data that returns None for to_skeleton
    mock_preprocess_data = Mock()
    mock_preprocess_data.to_skeleton.return_value = None

    with patch.object(job, "_get_model_class", return_value=TabularVAE):
        model = job._model_factory(mock_preprocess_data)

        assert isinstance(model, TabularVAE)
        assert model.model_name == "test_model"
        assert model.input_shape == (5,)
        assert model._metadata is None

    # Verify that the default mapping is used when dataset_type is empty
    from sdg_core_lib.mappings import DatasetMapping

    assert job._Job__dataset_mapping == DatasetMapping


def test_model_factory_input_shape_from_data():
    """Test model factory uses input_shape from preprocess data when not provided"""
    job = Job(
        n_rows=100,
        model_info={
            "algorithm_name": "sdg_core_lib.data_generator.models.keras.implementation.TabularVAE.TabularVAE",
            "model_name": "test_model",
            "input_shape": None,  # Not provided
        },
        dataset={"dataset_type": "table", "data": []},
        save_filepath="/tmp/test",
    )

    mock_preprocess_data = Mock()
    mock_preprocess_data.get_shape_for_model.return_value = "(15,)"
    mock_preprocess_data.to_skeleton.return_value = {"shape": "from_data"}

    with patch.object(job, "_get_model_class", return_value=TabularVAE):
        model = job._model_factory(mock_preprocess_data)

        assert isinstance(model, TabularVAE)
        assert model.input_shape == (15,)  # Should come from preprocess data
        assert model._metadata == {"shape": "from_data"}

    # Verify that the table mapping is used
    from sdg_core_lib.mappings import TableMapping

    assert job._Job__dataset_mapping == TableMapping


def test_get_model_class_dynamic_import():
    """Test that _get_model_class dynamically imports the correct class"""
    job = Job(
        n_rows=100,
        model_info={
            "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE.TabularVAE"
        },
        dataset={"dataset_type": "", "data": []},
    )

    model_class = job._get_model_class()
    assert model_class == TabularVAE

    # Verify that default mapping is used when no dataset_type provided
    from sdg_core_lib.mappings import DatasetMapping

    assert job._Job__dataset_mapping == DatasetMapping


def test_get_model_class_invalid_module():
    """Test _get_model_class with invalid module name"""
    job = Job(
        n_rows=100,
        model_info={"algorithm_name": "invalid.module.InvalidClass"},
        dataset={"dataset_type": "", "data": []},
    )

    with pytest.raises(ModuleNotFoundError):
        job._get_model_class()


def test_dataset_mapping_selection():
    """Test that correct dataset mapping is selected based on dataset_type"""
    from sdg_core_lib.mappings import TableMapping, TimeSeriesMapping, DatasetMapping

    # Test table mapping
    job_table = Job(n_rows=100, dataset={"dataset_type": "table", "data": []})
    assert job_table._Job__dataset_mapping == TableMapping

    # Test time_series mapping
    job_timeseries = Job(
        n_rows=100, dataset={"dataset_type": "time_series", "data": []}
    )
    assert job_timeseries._Job__dataset_mapping == TimeSeriesMapping

    # Test default mapping for unknown dataset_type
    job_unknown = Job(n_rows=100, dataset={"dataset_type": "unknown", "data": []})
    assert job_unknown._Job__dataset_mapping == DatasetMapping

    # Test default mapping for empty dataset_type
    job_empty = Job(n_rows=100, dataset={"dataset_type": "", "data": []})
    assert job_empty._Job__dataset_mapping == DatasetMapping


def test_get_processor_tabular_vae():
    """Test processor creation for TabularVAE with correct strategy"""
    job = Job(
        n_rows=100,
        save_filepath="/tmp/test",
        dataset={"dataset_type": "table", "data": []},
    )

    processor = job._get_processor(TabularVAE)

    assert isinstance(processor, TableProcessor)
    assert hasattr(processor, "strategy")
    assert isinstance(processor.strategy, TabularVAEPreprocessingStrategy)


def test_get_processor_timeseries_vae():
    """Test processor creation for TimeSeriesVAE with correct strategy"""
    job = Job(
        n_rows=100,
        save_filepath="/tmp/test",
        dataset={"dataset_type": "time_series", "data": []},
    )

    processor = job._get_processor(TimeSeriesVAE)

    assert isinstance(processor, TableProcessor)
    assert hasattr(processor, "strategy")
    assert isinstance(processor.strategy, TimeSeriesVAEPreprocessingStrategy)


def test_get_processor_unknown_model():
    """Test processor creation with unknown model class"""
    job = Job(
        n_rows=100,
        save_filepath="/tmp/test",
        dataset={"dataset_type": "table", "data": []},
    )

    class UnknownModel:
        pass

    processor = job._get_processor(UnknownModel)

    assert isinstance(processor, TableProcessor)
    assert hasattr(processor, "strategy")
    # Should fall back to BasePreprocessingStrategy
    assert isinstance(processor.strategy, BasePreprocessingStrategy)


def test_get_processor_invalid_dataset_type():
    """Test processor creation with invalid dataset type"""
    job = Job(
        n_rows=100,
        save_filepath="/tmp/test",
        dataset={"dataset_type": "table", "data": []},
    )

    # This should work fine since _get_processor only takes model_class now
    processor = job._get_processor(TabularVAE)
    assert isinstance(processor, TableProcessor)


def test_processor_strategy_mapping_completeness():
    """Test that all model classes have corresponding strategies"""
    from sdg_core_lib.mappings import ModelStrategyMapping

    # Check that TabularVAE has a strategy
    assert TabularVAE in ModelStrategyMapping.mapping
    assert ModelStrategyMapping.mapping[TabularVAE] == TabularVAEPreprocessingStrategy

    # Check that TimeSeriesVAE has a strategy
    assert TimeSeriesVAE in ModelStrategyMapping.mapping
    assert (
        ModelStrategyMapping.mapping[TimeSeriesVAE]
        == TimeSeriesVAEPreprocessingStrategy
    )


def test_processor_filepath_setting():
    """Test that processor receives correct filepath"""
    test_filepath = "/tmp/custom_test_path"
    job = Job(
        n_rows=100,
        save_filepath=test_filepath,
        dataset={"dataset_type": "table", "data": []},
    )

    processor = job._get_processor(TabularVAE)

    assert processor.dir_path == test_filepath


@patch("sdg_core_lib.preprocess.table_processor.TableProcessor")
def test_processor_initialization_called_correctly(mock_table_processor):
    """Test that TableProcessor is initialized with correct parameters"""
    mock_processor_instance = Mock()
    mock_table_processor.return_value = mock_processor_instance

    job = Job(
        n_rows=100,
        save_filepath="/tmp/test",
        dataset={"dataset_type": "table", "data": []},
    )

    processor = job._get_processor(TabularVAE)
    assert isinstance(processor, TableProcessor)
    assert isinstance(processor.strategy, TabularVAEPreprocessingStrategy)


def test_full_model_and_processor_creation_flow():
    """Test the complete flow of creating both model and processor"""
    job = Job(
        n_rows=100,
        model_info={
            "algorithm_name": "sdg_core_lib.data_generator.models.keras.implementation.TabularVAE.TabularVAE",
            "model_name": "integration_test_model",
        },
        dataset={"dataset_type": "table", "data": []},
        save_filepath="/tmp/integration_test",
    )

    # Test processor creation
    processor = job._get_processor(TabularVAE)
    assert isinstance(processor, TableProcessor)
    assert isinstance(processor.strategy, TabularVAEPreprocessingStrategy)

    # Test model creation
    mock_preprocess_data = Mock()
    mock_preprocess_data.get_shape_for_model.return_value = "(10,)"
    mock_preprocess_data.to_skeleton.return_value = {"test": "metadata"}

    with patch.object(job, "_get_model_class", return_value=TabularVAE):
        model = job._model_factory(mock_preprocess_data)
        assert isinstance(model, TabularVAE)
        assert model.model_name == "integration_test_model"


def test_different_dataset_types_same_model():
    """Test that the same model works with different dataset types"""
    job_table = Job(
        n_rows=100,
        save_filepath="/tmp/test",
        dataset={"dataset_type": "table", "data": []},
    )
    job_ts = Job(
        n_rows=100,
        save_filepath="/tmp/test",
        dataset={"dataset_type": "time_series", "data": []},
    )

    # TabularVAE should work with both table and time_series processors
    table_processor = job_table._get_processor(TabularVAE)
    ts_processor = job_ts._get_processor(TabularVAE)

    assert isinstance(table_processor, TableProcessor)
    assert isinstance(ts_processor, TableProcessor)
    assert isinstance(table_processor.strategy, TabularVAEPreprocessingStrategy)
    # TimeSeriesVAE strategy should only be used with TimeSeriesVAE model
    assert isinstance(ts_processor.strategy, TabularVAEPreprocessingStrategy)


def test_tabular_function_applier_with_inner_threshold():
    """Test TabularFunctionApplier with InnerThreshold function payload"""
    from sdg_core_lib.post_process.TabularFunctionApplier import TabularFunctionApplier
    from sdg_core_lib.dataset.datasets import Table
    import numpy as np

    # Create test payload
    functions_payload = [
        {
            "feature": "fixed_acidity",
            "function_reference": "sdg_core_lib.post_process.functions.filter.implementation.InnerThreshold.InnerThreshold",
            "parameters": [
                {"name": "lower_bound", "value": "6.0", "parameter_type": "float"},
                {"name": "upper_bound", "value": "9.0", "parameter_type": "float"},
                {"name": "lower_strict", "value": "True", "parameter_type": "bool"},
                {"name": "upper_strict", "value": "True", "parameter_type": "bool"},
            ],
        }
    ]

    # Create test dataset with fixed_acidity feature
    test_data = [
        {
            "column_name": "fixed_acidity",
            "column_data": [
                5.0,
                6.5,
                7.0,
                8.5,
                10.0,
            ],  # Values outside and inside range
            "column_datatype": "float32",
            "column_type": "continuous",
        },
        {
            "column_name": "volatile_acidity",
            "column_data": [0.5, 0.6, 0.7, 0.8, 0.9],
            "column_datatype": "float32",
            "column_type": "continuous",
        },
    ]

    # Create TabularFunctionApplier instance
    applier = TabularFunctionApplier(
        function_feature_dict=functions_payload, n_rows=5, from_scratch=False
    )

    # Create test dataset
    dataset = Table.from_json(test_data)

    # Apply functions
    result_dataset = applier.apply_all(dataset)

    # Verify results
    result_json = result_dataset.to_json()

    # Check that fixed_acidity was filtered (values 6.5, 7.0, 8.5 should remain after removing rows with NaN)
    fixed_acidity_data = result_json[0]["column_data"]
    logger.info(fixed_acidity_data)
    assert fixed_acidity_data == [6.5, 7.0, 8.5]  # Only values within range 6-9 remain

    # Check that volatile_acidity was also filtered to match the remaining rows
    volatile_acidity_data = result_json[1]["column_data"]
    assert np.allclose(
        volatile_acidity_data, [0.6, 0.7, 0.8]
    )  # Corresponding rows to kept fixed_acidity values
