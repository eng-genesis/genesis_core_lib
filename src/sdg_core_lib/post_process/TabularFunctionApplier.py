from sdg_core_lib.dataset.datasets import Dataset, Table
from sdg_core_lib.post_process.function_factory import function_factory
import numpy as np
from loguru import logger
from typing import Optional


class TabularFunctionApplier:
    """
    A class to apply functions to datasets.
    WARNING: this class is under construction. New dataset types will be supported in the future.
    """

    def __init__(
        self, function_feature_dict: list[dict], n_rows: int, from_scratch: bool = False
    ):
        self.function_feature_dict = function_feature_dict
        self.n_rows = n_rows
        self.from_scratch = from_scratch
        self._initialize()

    def _initialize(self):
        feature_function_mapping = {}
        for item in self.function_feature_dict:
            feature = item["feature"]
            if feature not in feature_function_mapping.keys():
                feature_function_mapping[feature] = []

            feature_function_mapping[feature].append(function_factory(item))

        for feature, functions in feature_function_mapping.items():
            functions.sort(key=lambda x: x.priority.value, reverse=True)

        self.function_feature_mapping = feature_function_mapping

    def apply_all(self, dataset: Optional[Dataset] = None) -> Dataset:
        """
        Apply all functions to either generate new data or modify existing dataset.
        Args:
            dataset: Existing dataset to modify (required if from_scratch=False)

        Returns:
            Dataset with functions applied
        Raises:
            ValueError: If input validation fails
        """
        if dataset is None and not self.from_scratch:
            raise ValueError("Dataset is required if from_scratch is False")

        if dataset is None and self.from_scratch:
            return self._generate_from_scratch()

        return self._modify_existing_dataset(dataset)

    def _generate_from_scratch(self) -> Dataset:
        """
        Generate a new Table from scratch using functions.
        # TODO: support other dataset types

        Returns:
            New Table dataset with generated data
        Raises:
            ValueError: If function configuration is invalid
        """
        new_data_json = []

        for feature, functions in self.function_feature_mapping.items():
            self._validate_function_sequence(functions, from_scratch=True)

            # Initialize with appropriate dtype based on first function
            data = np.empty((self.n_rows, 1), dtype=self._infer_datatype(functions[0]))

            for function in functions:
                try:
                    data, indexes, success = function.apply(
                        n_rows=self.n_rows, data=data
                    )
                    if not success:
                        logger.warning(
                            f"Function {function.__class__.__name__} failed to apply successfully"
                        )
                except Exception as e:
                    logger.error(
                        f"Error applying function {function.__class__.__name__}: {e}"
                    )
                    raise

            new_data_json.append(
                {
                    "column_name": feature,
                    "column_data": data.tolist(),
                    "column_datatype": str(data.dtype),
                    "column_type": self._infer_column_type(functions),
                }
            )

        try:
            # Remove rows with NaN values before creating dataset
            data_arrays = [
                np.array(item["column_data"]).reshape(-1, 1) for item in new_data_json
            ]
            cleaned_arrays, removed_rows = self._remove_nan_rows(data_arrays)

            if removed_rows > 0:
                logger.warning(
                    f"Removed {removed_rows} rows containing NaN values during generation"
                )
                # Update the JSON with cleaned data
                for i, item in enumerate(new_data_json):
                    item["column_data"] = cleaned_arrays[i].flatten().tolist()

            new_dataset = Table.from_json(new_data_json)
            logger.info(
                f"Successfully generated dataset with {len(new_data_json)} features"
            )
            return new_dataset
        except Exception as e:
            logger.error(f"Failed to create dataset from generated data: {e}")
            raise

    def _modify_existing_dataset(self, dataset: Dataset) -> Dataset:
        """
        Modify an existing dataset by applying functions to mapped features.

        Args:
            dataset: Existing dataset to modify

        Returns:
            Modified dataset

        Raises:
            ValueError: If data compatibility issues arise
        """
        if not type(dataset) is Table:
            # TODO: support other dataset types
            logger.error("Only Table datasets are currently supported")
            raise TypeError("Only Table datasets are currently supported")

        json_structure = dataset.to_json()
        modified_features = set()
        data_array = []
        unmapped_features = []

        for feature in json_structure:
            feature_name = feature["column_name"]

            if feature_name in self.function_feature_mapping:
                functions = self.function_feature_mapping[feature_name]
                self._validate_function_sequence(functions, from_scratch=False)

                feature_data = np.array(feature["column_data"])
                original_shape = feature_data.shape

                for function in functions:
                    if function.is_generative:
                        logger.info(
                            f"Skipping generative function {function.__class__.__name__} on existing dataset"
                        )
                        continue

                    try:
                        feature_data, indexes, success = function.apply(
                            n_rows=self.n_rows, data=feature_data
                        )
                        if not success:
                            logger.warning(
                                f"Function {function.__class__.__name__} failed to apply successfully"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error applying function {function.__class__.__name__}: {e}"
                        )
                        raise

                # Validate shape compatibility
                if feature_data.shape != original_shape:
                    logger.warning(
                        f"Feature {feature_name} shape changed from {original_shape} to {feature_data.shape}"
                    )

                data_array.append(feature_data)
                modified_features.add(feature_name)
            else:
                # Preserve unmapped features
                data_array.append(np.array(feature["column_data"]))
                unmapped_features.append(feature_name)

        if unmapped_features:
            logger.info(
                f"Preserving {len(unmapped_features)} unmapped features: {unmapped_features}"
            )

        try:
            # Remove rows with NaN values before concatenation
            cleaned_data_array, removed_rows = self._remove_nan_rows(data_array)

            if removed_rows > 0:
                logger.warning(f"Removed {removed_rows} rows containing NaN values")

            # Safe concatenation with shape validation
            final_array = self._safe_concatenate(cleaned_data_array)
            new_dataset = dataset.clone(final_array)

            logger.info(
                f"Successfully modified dataset. Updated {len(modified_features)} features, preserved {len(unmapped_features)} features"
            )
            return new_dataset
        except Exception as e:
            logger.error(f"Failed to concatenate modified data: {e}")
            raise

    @staticmethod
    def _validate_function_sequence(functions: list, from_scratch: bool) -> None:
        """
        Validate the sequence of functions for correctness.

        Args:
            functions: list of functions to validate
            from_scratch: Whether this is for generation from scratch

        Raises:
            ValueError: If function sequence is invalid
        """
        if not functions:
            raise ValueError("Function list cannot be empty")

        if from_scratch:
            if not functions[0].is_generative:
                raise ValueError(
                    "First function must be generative when generating from scratch"
                )

            for i, function in enumerate(functions[1:], 1):
                if function.is_generative:
                    raise ValueError(
                        f"Only the first function can be generative (found generative function at position {i})"
                    )
        else:
            generative_count = sum(1 for f in functions if f.is_generative)
            if generative_count > 0:
                logger.warning(
                    f"Found {generative_count} generative functions in modification mode - these will be skipped"
                )

    @staticmethod
    def _infer_datatype(function) -> str:
        """
        Infer appropriate datatype for a function's output.

        Args:
            function: The function to analyze

        Returns:
            String representation of numpy dtype
        """
        # Default to float32 for most cases, but this could be extended
        # to inspect function parameters or metadata
        return "float32"

    @staticmethod
    def _infer_column_type(functions: list) -> str:
        """
        Infer column type based on function sequence.

        Args:
            functions: list of functions applied to the column

        Returns:
            Column type string
        """
        # Default to continuous, but could be made smarter by analyzing
        # function types or parameters
        return "continuous"

    @staticmethod
    def _safe_concatenate(data_array: list[np.ndarray]) -> np.ndarray:
        """
        Safely concatenate arrays with proper validation.

        Args:
            data_array: list of arrays to concatenate

        Returns:
            Concatenated array

        Raises:
            ValueError: If arrays cannot be safely concatenated
        """
        if not data_array:
            raise ValueError("Cannot concatenate empty array list")

        # Validate all arrays have the same number of rows
        expected_rows = data_array[0].shape[0]
        for i, arr in enumerate(data_array):
            if arr.shape[0] != expected_rows:
                raise ValueError(
                    f"Array {i} has {arr.shape[0]} rows, expected {expected_rows}. "
                    f"All arrays must have the same number of rows."
                )

        # Ensure all arrays are 2D
        for i, arr in enumerate(data_array):
            if len(arr.shape) == 1:
                data_array[i] = arr.reshape(-1, 1)
            elif len(arr.shape) > 2:
                raise ValueError(
                    f"Array {i} has {len(arr.shape)} dimensions, expected 1 or 2"
                )

        try:
            return np.hstack(data_array)
        except Exception as e:
            raise ValueError(f"Failed to concatenate arrays: {e}")

    @staticmethod
    def _remove_nan_rows(data_array: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
        """
        Remove rows containing NaN values from all arrays in the list.

        Args:
            data_array: list of arrays to clean

        Returns:
            Tuple of (cleaned_data_array, number_of_removed_rows)

        Raises:
            ValueError: If arrays cannot be safely processed
        """
        if not data_array:
            return [], 0

        # Ensure all arrays are 2D for consistent processing
        processed_arrays = []
        for i, arr in enumerate(data_array):
            if len(arr.shape) == 1:
                processed_arrays.append(arr.reshape(-1, 1))
            else:
                processed_arrays.append(arr)

        # Find rows with NaN values across all columns
        nan_mask = np.zeros(processed_arrays[0].shape[0], dtype=bool)

        for arr in processed_arrays:
            nan_mask |= np.isnan(arr).any(axis=1)

        # Remove rows with NaN values
        rows_to_keep = ~nan_mask
        removed_rows = np.sum(nan_mask)

        if removed_rows == 0:
            return data_array, 0

        cleaned_arrays = []
        for arr in processed_arrays:
            cleaned_arrays.append(arr[rows_to_keep])

        return cleaned_arrays, removed_rows
