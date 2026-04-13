import pytest
import numpy as np
from unittest.mock import patch

from sdg_core_lib.post_process.TabularFunctionApplier import TabularFunctionApplier
from sdg_core_lib.dataset.datasets import Table
from sdg_core_lib.post_process.functions.generation.implementation.NormalDistributionSample import (
    NormalDistributionSample,
)
from sdg_core_lib.post_process.functions.modification.implementation.WhiteNoiseAdder import (
    WhiteNoiseAdder,
)


@pytest.fixture
def generative_function_dict():
    return {
        "feature": "test_feature",
        "function_reference": "sdg_core_lib.post_process.functions.generation.implementation.NormalDistributionSample.NormalDistributionSample",
        "parameters": [
            {"name": "mean", "value": "0.0", "parameter_type": "float"},
            {"name": "standard_deviation", "value": "1.0", "parameter_type": "float"},
        ],
    }


@pytest.fixture
def modification_function_dict():
    return {
        "feature": "test_feature",
        "function_reference": "sdg_core_lib.post_process.functions.modification.implementation.WhiteNoiseAdder.WhiteNoiseAdder",
        "parameters": [
            {"name": "mean", "value": "0.0", "parameter_type": "float"},
            {"name": "standard_deviation", "value": "0.1", "parameter_type": "float"},
        ],
    }


@pytest.fixture
def sample_dataset():
    """Create a sample Table dataset for testing."""
    json_data = [
        {
            "column_name": "feature1",
            "column_data": [1.0, 2.0, 3.0, 4.0, 5.0],
            "column_datatype": "float32",
            "column_type": "continuous",
        },
        {
            "column_name": "feature2",
            "column_data": [10.0, 20.0, 30.0, 40.0, 50.0],
            "column_datatype": "float32",
            "column_type": "continuous",
        },
    ]
    return Table.from_json(json_data)


class TestFunctionApplier:
    """Test cases for FunctionApplier class."""

    def test_init_from_scratch(self, generative_function_dict):
        """Test initialization for from_scratch mode."""
        applier = TabularFunctionApplier(
            function_feature_dict=[generative_function_dict],
            n_rows=100,
            from_scratch=True,
        )

        assert applier.n_rows == 100
        assert applier.from_scratch is True
        assert "test_feature" in applier.function_feature_mapping
        assert len(applier.function_feature_mapping["test_feature"]) == 1

    def test_init_modification_mode(self, modification_function_dict):
        """Test initialization for modification mode."""
        applier = TabularFunctionApplier(
            function_feature_dict=[modification_function_dict],
            n_rows=100,
            from_scratch=False,
        )

        assert applier.n_rows == 100
        assert applier.from_scratch is False
        assert "test_feature" in applier.function_feature_mapping

    def test_init_multiple_functions_same_feature(
        self, generative_function_dict, modification_function_dict
    ):
        """Test initialization with multiple functions for the same feature."""
        # Modify the function dicts to use the same feature name
        gen_dict = generative_function_dict.copy()
        gen_dict["feature"] = "shared_feature"
        mod_dict = modification_function_dict.copy()
        mod_dict["feature"] = "shared_feature"

        applier = TabularFunctionApplier(
            function_feature_dict=[gen_dict, mod_dict],
            n_rows=100,
            from_scratch=True,
        )

        functions = applier.function_feature_mapping["shared_feature"]
        assert len(functions) == 2
        # Functions should be sorted by priority (higher first)
        assert functions[0].priority.value >= functions[1].priority.value

    def test_apply_all_from_scratch_success(self, generative_function_dict):
        """Test successful generation from scratch."""
        applier = TabularFunctionApplier(
            function_feature_dict=[generative_function_dict],
            n_rows=50,
            from_scratch=True,
        )

        result = applier.apply_all()

        assert isinstance(result, Table)
        assert len(result.columns) == 1
        assert result.columns[0].name == "test_feature"
        assert result.columns[0].values.shape == (50, 1)

    def test_apply_all_modification_success(
        self, modification_function_dict, sample_dataset
    ):
        """Test successful modification of existing dataset."""
        applier = TabularFunctionApplier(
            function_feature_dict=[modification_function_dict],
            n_rows=5,
            from_scratch=False,
        )

        # Update function dict to target feature1 from sample dataset
        mod_dict = modification_function_dict.copy()
        mod_dict["feature"] = "feature1"
        applier.function_feature_dict = [mod_dict]
        applier._initialize()

        result = applier.apply_all(dataset=sample_dataset)

        assert isinstance(result, Table)
        assert len(result.columns) == 2  # Both features should be preserved
        assert result.columns[0].name == "feature1"
        assert result.columns[1].name == "feature2"
        # feature1 should be modified, feature2 should be unchanged
        assert not np.array_equal(
            result.columns[0].values, sample_dataset.columns[0].values
        )
        assert np.array_equal(
            result.columns[1].values, sample_dataset.columns[1].values
        )

    def test_apply_all_missing_dataset_error(self, modification_function_dict):
        """Test error when dataset is missing in modification mode."""
        applier = TabularFunctionApplier(
            function_feature_dict=[modification_function_dict],
            n_rows=100,
            from_scratch=False,
        )

        with pytest.raises(
            ValueError, match="Dataset is required if from_scratch is False"
        ):
            applier.apply_all()

    def test_generate_from_scratch_invalid_function_sequence(
        self, modification_function_dict
    ):
        """Test error when first function is not generative in from_scratch mode."""
        # Use modification function (non-generative) as first function
        applier = TabularFunctionApplier(
            function_feature_dict=[modification_function_dict],
            n_rows=100,
            from_scratch=True,
        )

        with pytest.raises(ValueError, match="First function must be generative"):
            applier._generate_from_scratch()

    def test_generate_from_scratch_multiple_generative_functions(
        self, generative_function_dict
    ):
        """Test error when multiple generative functions are provided."""
        gen_dict2 = generative_function_dict.copy()

        applier = TabularFunctionApplier(
            function_feature_dict=[generative_function_dict, gen_dict2],
            n_rows=100,
            from_scratch=True,
        )

        with pytest.raises(
            ValueError, match="Only the first function can be generative"
        ):
            applier._generate_from_scratch()

    def test_modify_existing_dataset_unmapped_features(self, sample_dataset):
        """Test that unmapped features are preserved."""
        # Create function that only targets feature1
        function_dict = {
            "feature": "feature1",
            "function_reference": "sdg_core_lib.post_process.functions.modification.implementation.WhiteNoiseAdder.WhiteNoiseAdder",
            "parameters": [
                {"name": "mean", "value": "0.0", "parameter_type": "float"},
                {
                    "name": "standard_deviation",
                    "value": "0.0",
                    "parameter_type": "float",
                },  # No change
            ],
        }

        applier = TabularFunctionApplier(
            function_feature_dict=[function_dict],
            n_rows=5,
            from_scratch=False,
        )

        with patch(
            "sdg_core_lib.post_process.TabularFunctionApplier.logger"
        ) as mock_logger:
            _ = applier.apply_all(dataset=sample_dataset)

            # Check that unmapped feature preservation was logged
            mock_logger.info.assert_any_call(
                "Preserving 1 unmapped features: ['feature2']"
            )

    def test_safe_concatenate_success(self):
        """Test successful array concatenation."""
        arrays = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5], [6]]),
        ]

        result = TabularFunctionApplier._safe_concatenate(arrays)

        expected = np.array([[1, 2, 5], [3, 4, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_safe_concatenate_mismatched_rows(self):
        """Test error when arrays have different row counts."""
        arrays = [
            np.array([[1, 2], [3, 4]]),  # 2 rows
            np.array([[5], [6], [7]]),  # 3 rows
        ]

        with pytest.raises(ValueError, match="Array 1 has 3 rows, expected 2"):
            TabularFunctionApplier._safe_concatenate(arrays)

    def test_safe_concatenate_empty_list(self):
        """Test error when array list is empty."""
        with pytest.raises(ValueError, match="Cannot concatenate empty array list"):
            TabularFunctionApplier._safe_concatenate([])

    def test_safe_concatenate_1d_arrays(self):
        """Test concatenation of 1D arrays."""
        arrays = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
        ]

        result = TabularFunctionApplier._safe_concatenate(arrays)

        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_validate_function_sequence_from_scratch(
        self, generative_function_dict, modification_function_dict
    ):
        """Test function sequence validation for from_scratch mode."""
        # Valid sequence: generative first, then modification
        gen_func = NormalDistributionSample.from_json(
            generative_function_dict["parameters"]
        )
        mod_func = WhiteNoiseAdder.from_json(modification_function_dict["parameters"])

        # Should not raise
        TabularFunctionApplier._validate_function_sequence(
            [gen_func, mod_func], from_scratch=True
        )

    def test_validate_function_sequence_from_scratch_invalid_first(
        self, modification_function_dict
    ):
        """Test validation error when first function is not generative."""
        mod_func = WhiteNoiseAdder.from_json(modification_function_dict["parameters"])

        with pytest.raises(ValueError, match="First function must be generative"):
            TabularFunctionApplier._validate_function_sequence(
                [mod_func], from_scratch=True
            )

    def test_validate_function_sequence_from_scratch_multiple_generative(
        self, generative_function_dict
    ):
        """Test validation error when multiple generative functions."""
        gen_func = NormalDistributionSample.from_json(
            generative_function_dict["parameters"]
        )

        with pytest.raises(
            ValueError, match="Only the first function can be generative"
        ):
            TabularFunctionApplier._validate_function_sequence(
                [gen_func, gen_func], from_scratch=True
            )

    def test_validate_function_sequence_empty_list(self):
        """Test validation error for empty function list."""
        with pytest.raises(ValueError, match="Function list cannot be empty"):
            TabularFunctionApplier._validate_function_sequence([], from_scratch=True)

    def test_infer_datatype(self):
        """Test datatype inference."""
        result = TabularFunctionApplier._infer_datatype(None)
        assert result == "float32"

    def test_infer_column_type(self):
        """Test column type inference."""
        result = TabularFunctionApplier._infer_column_type([])
        assert result == "continuous"

    def test_modify_existing_dataset_wrong_type(self):
        """Test error when dataset is not a Table."""
        from sdg_core_lib.dataset.datasets import Dataset

        class MockDataset(Dataset):
            def from_json(cls, json_data, save_path):
                pass

            def from_skeleton(cls, skeleton, save_path):
                pass

            def clone(self, new_data):
                pass

            def get_computing_data(self):
                pass

            def get_computing_shape(self):
                pass

            def to_json(self):
                pass

            def to_skeleton(self):
                pass

            def preprocess(self):
                pass

            def postprocess(self):
                pass

        mock_dataset = MockDataset()
        applier = TabularFunctionApplier([], 100, False)

        with pytest.raises(
            TypeError, match="Only Table datasets are currently supported"
        ):
            applier._modify_existing_dataset(mock_dataset)

    @patch("sdg_core_lib.post_process.TabularFunctionApplier.logger")
    def test_function_application_failure_logging(
        self, mock_logger, generative_function_dict
    ):
        """Test logging when function application fails."""
        applier = TabularFunctionApplier(
            function_feature_dict=[generative_function_dict],
            n_rows=10,
            from_scratch=True,
        )

        # Mock the function to raise an exception
        with patch.object(
            applier.function_feature_mapping["test_feature"][0],
            "apply",
            side_effect=Exception("Test error"),
        ):
            with pytest.raises(Exception, match="Test error"):
                applier._generate_from_scratch()

            # Check that error was logged
            mock_logger.error.assert_called()

    def test_apply_all_with_generative_functions_in_modification_mode(
        self, modification_function_dict, sample_dataset
    ):
        """Test that generative functions are properly skipped in modification mode."""
        # Create a generative function
        gen_dict = {
            "feature": "feature1",
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

        applier = TabularFunctionApplier(
            function_feature_dict=[gen_dict],
            n_rows=5,
            from_scratch=False,
        )

        with patch(
            "sdg_core_lib.post_process.TabularFunctionApplier.logger"
        ) as mock_logger:
            _ = applier.apply_all(dataset=sample_dataset)

            # Check that generative function skip was logged
            mock_logger.info.assert_any_call(
                "Skipping generative function NormalDistributionSample on existing dataset"
            )

    def test_remove_nan_rows_no_nan(self):
        """Test _remove_nan_rows with no NaN values."""
        arrays = [
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            np.array([[7.0], [8.0], [9.0]]),
        ]

        cleaned_arrays, removed_rows = TabularFunctionApplier._remove_nan_rows(arrays)

        assert removed_rows == 0
        assert len(cleaned_arrays) == 2
        np.testing.assert_array_equal(cleaned_arrays[0], arrays[0])
        np.testing.assert_array_equal(cleaned_arrays[1], arrays[1])

    def test_remove_nan_rows_with_nan(self):
        """Test _remove_nan_rows with NaN values in different columns."""
        arrays = [
            np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]]),
            np.array([[7.0], [8.0], [np.nan]]),
        ]

        cleaned_arrays, removed_rows = TabularFunctionApplier._remove_nan_rows(arrays)

        assert (
            removed_rows == 2
        )  # Row 1 (nan in first array) and row 2 (nan in second array)
        assert len(cleaned_arrays) == 2

        # Only first row should remain (no NaN in any column)
        expected_first = np.array([[1.0, 2.0]])
        expected_second = np.array([[7.0]])

        np.testing.assert_array_equal(cleaned_arrays[0], expected_first)
        np.testing.assert_array_equal(cleaned_arrays[1], expected_second)

    def test_remove_nan_rows_all_nan(self):
        """Test _remove_nan_rows when all rows contain NaN."""
        arrays = [
            np.array([[np.nan, 2.0], [np.nan, 4.0]]),
            np.array([[7.0], [8.0]]),
        ]

        cleaned_arrays, removed_rows = TabularFunctionApplier._remove_nan_rows(arrays)

        assert removed_rows == 2
        assert len(cleaned_arrays) == 2
        assert cleaned_arrays[0].shape[0] == 0
        assert cleaned_arrays[1].shape[0] == 0

    def test_remove_nan_rows_1d_arrays(self):
        """Test _remove_nan_rows with 1D arrays."""
        arrays = [
            np.array([1.0, np.nan, 3.0, 4.0]),
            np.array([5.0, 6.0, np.nan, 8.0]),
        ]

        cleaned_arrays, removed_rows = TabularFunctionApplier._remove_nan_rows(arrays)

        assert (
            removed_rows == 2
        )  # Row 1 (nan in first array) and row 2 (nan in second array)
        assert len(cleaned_arrays) == 2

        # Only rows 0 and 3 should remain
        expected_first = np.array([[1.0], [4.0]])
        expected_second = np.array([[5.0], [8.0]])

        np.testing.assert_array_equal(cleaned_arrays[0], expected_first)
        np.testing.assert_array_equal(cleaned_arrays[1], expected_second)

    def test_remove_nan_rows_empty_list(self):
        """Test _remove_nan_rows with empty list."""
        cleaned_arrays, removed_rows = TabularFunctionApplier._remove_nan_rows([])

        assert cleaned_arrays == []
        assert removed_rows == 0

    def test_remove_nan_rows_mixed_shapes(self):
        """Test _remove_nan_rows with mixed 1D and 2D arrays."""
        arrays = [
            np.array([1.0, np.nan, 3.0]),  # 1D
            np.array([[4.0], [5.0], [6.0]]),  # 2D
        ]

        cleaned_arrays, removed_rows = TabularFunctionApplier._remove_nan_rows(arrays)

        assert removed_rows == 1
        assert len(cleaned_arrays) == 2
        assert cleaned_arrays[0].shape == (2, 1)  # Converted to 2D
        assert cleaned_arrays[1].shape == (2, 1)

    @patch("sdg_core_lib.post_process.TabularFunctionApplier.logger")
    def test_generate_from_scratch_with_nan_removal(
        self, mock_logger, generative_function_dict
    ):
        """Test that NaN rows are removed during generation from scratch."""
        # Mock the function to return data with NaN values
        mock_data_with_nan = np.array([[1.0], [np.nan], [3.0], [4.0], [5.0]])

        applier = TabularFunctionApplier(
            function_feature_dict=[generative_function_dict],
            n_rows=5,
            from_scratch=True,
        )

        with patch.object(
            applier.function_feature_mapping["test_feature"][0],
            "apply",
            return_value=(mock_data_with_nan, None, True),
        ):
            result = applier.apply_all()

            # Should have 4 rows after removing the NaN row
            assert result.columns[0].values.shape == (4, 1)

            # Check that warning was logged
            mock_logger.warning.assert_any_call(
                "Removed 1 rows containing NaN values during generation"
            )

    def test_modify_existing_dataset_with_nan_removal(self, modification_function_dict):
        """Test that NaN rows are removed during dataset modification."""
        # Create a dataset with NaN values
        json_data_with_nan = [
            {
                "column_name": "feature1",
                "column_data": [1.0, np.nan, 3.0, 4.0, 5.0],
                "column_datatype": "float32",
                "column_type": "continuous",
            },
            {
                "column_name": "feature2",
                "column_data": [10.0, 20.0, 30.0, np.nan, 50.0],
                "column_datatype": "float32",
                "column_type": "continuous",
            },
        ]
        dataset_with_nan = Table.from_json(json_data_with_nan)

        # Use modification function that doesn't change the data
        mod_dict = modification_function_dict.copy()
        mod_dict["feature"] = "feature1"
        mod_dict["parameters"][1]["value"] = "0.0"  # No actual modification

        applier = TabularFunctionApplier(
            function_feature_dict=[mod_dict],
            n_rows=5,
            from_scratch=False,
        )

        result = applier.apply_all(dataset=dataset_with_nan)

        print(result.to_json())

        # Should have 3 rows after removing rows with NaN (row 1 and row 3)
        assert result.columns[0].values.shape == (3, 1)
        assert result.columns[1].values.shape == (3, 1)
