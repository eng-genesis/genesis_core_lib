from typing import Optional, Type
from loguru import logger

from sdg_core_lib.data_generator.models.UnspecializedModel import UnspecializedModel
from sdg_core_lib.dataset.datasets import Dataset
from sdg_core_lib.post_process.TabularFunctionApplier import TabularFunctionApplier
from sdg_core_lib.preprocess.base_processor import Processor
from sdg_core_lib.mappings import (
    DatasetMapping,
    ModelStrategyMapping,
    TableMapping,
    TimeSeriesMapping,
)
import importlib
import os


def get_hyperparameters() -> dict:
    hyperparams_name = ["EPOCHS", "LEARNING_RATE", "BATCH_SIZE"]
    return {
        hp.lower(): os.environ.get(hp)
        for hp in hyperparams_name
        if os.environ.get(hp) is not None
    }


class Job:
    def __init__(
        self,
        n_rows: int,
        model_info: Optional[dict] = None,
        dataset: Optional[dict] = None,
        save_filepath: Optional[str] = None,
        functions: Optional[list[dict]] = None,
    ):
        self.__model_info = model_info
        self.__dataset = dataset if dataset is not None else {}
        self.__n_rows = n_rows
        self.__save_filepath = save_filepath
        self.__functions = functions if functions is not None else []
        dataset_type = self.__dataset.get("dataset_type", "")
        self.__dataset_mapping = self._get_dataset_mapping(dataset_type)
        self.__dataset_class = self.__dataset_mapping.get_dataset_class()
        self.__evaluator_class = self.__dataset_mapping.get_evaluator_class()
        self.__processor_class = self.__dataset_mapping.get_processor_class()

    @staticmethod
    def _get_dataset_mapping(dataset_type: str) -> Type[DatasetMapping]:
        if dataset_type == "table":
            return TableMapping
        if dataset_type == "time_series":
            return TimeSeriesMapping
        return DatasetMapping

    def _get_model_class(self) -> type:
        """
        Dynamically imports a class given its name.

        :return: the class itself
        """
        model_type = self.__model_info.get("algorithm_name")
        module_name, class_name = model_type.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _model_factory(
        self, preprocess_data: Dataset | None = None, is_new_model=True
    ) -> UnspecializedModel:
        model_file = self.__save_filepath if not is_new_model else None
        model_name = self.__model_info.get("model_name")
        input_shape = self.__model_info.get("input_shape", None)
        metadata = None

        if self.__dataset is not None:
            if input_shape is None:
                input_shape = preprocess_data.get_shape_for_model()
            metadata = preprocess_data.to_skeleton()

        model_class = self._get_model_class()
        model = model_class(
            metadata=metadata,
            model_name=model_name,
            input_shape=input_shape,
            load_path=model_file,
        )
        return model

    def _get_processor(self, model_class: type) -> Processor:
        processor = self.__processor_class(self.__save_filepath)

        strategy = ModelStrategyMapping.get_strategy(model_class)()
        processor.set_strategy(strategy)

        return processor

    def _infer_and_evaluate(
        self,
        data: Dataset,
        preprocessed_data: Dataset,
        processor: Processor,
        model: UnspecializedModel,
    ) -> tuple[dict, list[dict]]:

        predicted_data = model.infer(self.__n_rows)
        synthetic_data = preprocessed_data.clone(predicted_data)
        synthetic_data = synthetic_data.postprocess(processor)
        function_generator = TabularFunctionApplier(
            self.__functions, self.__n_rows, from_scratch=False
        )
        try:
            filtered_synthetic_data = function_generator.apply_all(synthetic_data)
        except (TypeError, ValueError) as e:
            logger.error(f"Unable to apply functions to data: {e}")
            filtered_synthetic_data = synthetic_data

        report = {"available": "false"}
        if data is not None:
            evaluator = self.__evaluator_class(
                real_data=data,
                synthetic_data=synthetic_data,
            )
            report = evaluator.compute()

        results = filtered_synthetic_data.to_json()
        return report, results

    def train(self) -> tuple[list[dict], dict, UnspecializedModel, list[dict]]:
        """
        Runs a pre-defined training job.

        This function will run the Synthetic Data Generation job. It will create an instance of the specified model or
        load the specified dataset, pre-process the data, train the model (if specified to do so), generate synthetic
        data, evaluate the generated data and save the results to the specified location.


        :return: a tuple containing a list of metrics, a dictionary with the model's info, the trained model, and the generated dataset
        """

        data_payload = self.__dataset.get("data", [])
        model_class = self._get_model_class()

        data = self.__dataset_class.from_json(data_payload)
        processor = self._get_processor(model_class)
        preprocessed_data = data.preprocess(processor)
        preprocess_schema = preprocessed_data.to_skeleton()

        model = self._model_factory(preprocessed_data)
        model.set_hyperparameters(**get_hyperparameters())
        model.train(data=preprocessed_data.get_computing_data())
        model.save(self.__save_filepath)

        report, results = self._infer_and_evaluate(
            data, preprocessed_data, processor, model
        )

        return results, report, model, preprocess_schema

    def infer(self) -> tuple[list[dict], dict]:
        data_payload = self.__dataset.get("data", [])
        model_class = self._get_model_class()
        processor = self._get_processor(model_class)
        data = None
        if len(data_payload) == 0:
            data_skeleton = self.__model_info.get("training_data_info")
            preprocessed_data = self.__dataset_class.from_skeleton(data_skeleton)
        else:
            data = self.__dataset_class.from_json(data_payload)
            preprocessed_data = data.preprocess(processor)

        model = self._model_factory(preprocessed_data, is_new_model=False)
        report, results = self._infer_and_evaluate(
            data, preprocessed_data, processor, model
        )

        return results, report

    def generate_from_functions(self, dataset: Optional[Dataset] = None):
        """
        Generate a dataset from a list of functions.
        :param dataset: a Dataset object
        :return: a dataset in JSON format
        """
        from_scratch = False
        if dataset is None:
            from_scratch = True
        function_generator = TabularFunctionApplier(
            self.__functions, self.__n_rows, from_scratch=from_scratch
        )
        dataset = function_generator.apply_all(dataset)
        return dataset.to_json()
