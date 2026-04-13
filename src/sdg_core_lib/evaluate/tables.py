import numpy as np
import pandas as pd
import scipy.stats as ss

from sdg_core_lib.dataset.datasets import Table
from sdg_core_lib.evaluate.base_evaluator import BaseEvaluator
from sdg_core_lib.evaluate.metrics import (
    StatisticalMetric,
    AdherenceMetric,
    NoveltyMetric,
)


class TabularComparisonEvaluator(BaseEvaluator):
    """
    Evaluates the quality of a synthetic dataset with respect to a real one.

    The evaluation is based on the following metrics:
    - Statistical properties: wasserstein distance and Cramer's V
    - Adherence: evaluates how well the synthetic data adheres to the real data distribution
    - Novelty: evaluates how many new values are generated in the synthetic dataset

    The evaluation is performed on a per-column basis, and the results are aggregated.
    """

    def __init__(
        self,
        real_data: Table,
        synthetic_data: Table,
    ):
        if not isinstance(real_data, Table):
            raise TypeError("real_data must be a Table")
        if not isinstance(synthetic_data, Table):
            raise TypeError("synthetic_data must be a Table")
        super().__init__(real_data, synthetic_data)

    def compute(self) -> dict:
        numerical_columns = self._real_data.get_numeric_columns()
        categorical_columns = self._real_data.get_categorical_columns()
        if len(numerical_columns) < 1 and len(categorical_columns) < 1:
            return {"available": "false"}

        self._evaluate_wasserstein_distance()
        self._evaluate_cramer_v_distance()
        self._evaluate_categorical_frequency_absolute_difference()
        self._evaluate_adherence()
        self._evaluate_novelty()
        return self.report.to_json()

    def _evaluate_cramer_v_distance(self):
        """
        Evaluates Cramer's v with Bias Correction https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V on categorical data,
        evaluating pairwise columns. Each pair of columns is evaluated on both datasets, appending scores in a list
        and returning the aggregate.

        :return: A score ranging from 0 to 1. A score of 0 is the worst possible score, while 1 is the best possible score,
        meaning that category pairs are perfectly balanced
        """
        total_real_categorical = (
            self._real_data.all_to_categorical().get_categorical_columns()
        )
        total_synth_categorical = (
            self._synth_data.all_to_categorical().get_categorical_columns()
        )

        result_dict = {}
        if len(total_real_categorical) >= 2:
            contingency_scores_distances = []
            for idx, (col, synth_col) in enumerate(
                zip(total_real_categorical[:-1], total_synth_categorical[:-1])
            ):
                result_dict[col.name] = {}
                for col_2, synth_col_2 in zip(
                    total_real_categorical[idx + 1 :],
                    total_synth_categorical[idx + 1 :],
                ):
                    try:
                        v_real = compute_cramer_v(col.get_data(), col_2.get_data())
                        v_synth = compute_cramer_v(
                            synth_col.get_data(), synth_col_2.get_data()
                        )
                        contingency_scores_distances.append(np.abs(v_real - v_synth))
                        result = np.round((np.abs(v_real - v_synth)) * 100, 2).item()
                        if np.isnan(result):
                            result = "NaN"
                        result_dict[col.name][col_2.name] = result
                    except ValueError:
                        result_dict[col.name][col_2.name] = "NaN"

        if not len(total_real_categorical) == 0:
            self.report.add_metric(
                StatisticalMetric(
                    title="Association Distance Index (Cramer's V, Real vs Synthetic)",
                    unit_measure="% - Range: from 0 (Association is Perfectly Kept in Synthetic Data) to 100 (Association is not Kept in Synthetic Data)",
                    value=result_dict,
                )
            )

    def _evaluate_wasserstein_distance(self):
        """
        Computing the Wasserstein distance for each numerical column. The score is computed using a different approach,
        trying to clip the values between 0 and 1. With 1 it means that the distribution of data is aligned, while with
        0 means that the distribution of data are largely unaligned.
        In particular, the Wasserstein distance score will be clipped between 0 and |max - min|, where max and min
        are related to the real dataset distribution. In the end, the score is scaled between 0 and 1
        :return: A single score, computed as 1 - mean(scores)
        """
        numerical_columns = self._real_data.get_numeric_columns()
        synth_numerical_columns = self._synth_data.get_numeric_columns()
        result_dict = {}
        if len(numerical_columns) > 0:
            for col, synt_col in zip(numerical_columns, synth_numerical_columns):
                real_data = col.get_data().reshape(
                    -1,
                )
                synth_data = synt_col.get_data().reshape(
                    -1,
                )
                distance = (
                    np.abs(np.max(real_data) - np.min(real_data)) or np.finfo(float).eps
                )
                wass_dist = ss.wasserstein_distance(real_data, synth_data)
                wass_dist = np.clip(wass_dist, 0, distance) / distance
                result_dict[col.name] = np.round(wass_dist * 100, 2).item()

        if not len(numerical_columns) == 0:
            self.report.add_metric(
                StatisticalMetric(
                    title="Continuous Features Statical Distance (Wasserstein Distance)",
                    unit_measure="% - Range: from 0 (Synthetic features are distributed like Real Features) to 100 (Synthetic features are distributed differently from Real Features)",
                    value=result_dict,
                )
            )

    def _evaluate_novelty(self):
        """
        This function evaluates in two steps the following metrics
        1) The number of unique samples generated in the synthetic dataset with respect to the real data
        2) The number of duplicated samples in the synthetic dataset
        """
        synthetic_data = self._synth_data.get_computing_data()
        synth_len = synthetic_data.shape[0]
        synth_unique = give_stringed_unique_rows(synthetic_data)
        synth_unique_len = synth_unique.shape[0]

        real_data = self._real_data.get_computing_data()
        real_unique = give_stringed_unique_rows(real_data)

        new_synt_data = len([data for data in synth_unique if data not in real_unique])

        self.report.add_metric(
            NoveltyMetric(
                title="Synthetic Data Uniqueness Score (Unique Synthetic Rows / Total Synthetic Rows)",
                unit_measure="% - Range: from 0 (worst) to 100 (best)",
                value=np.round((synth_unique_len / synth_len) * 100, 2).item(),
            )
        )

        self.report.add_metric(
            NoveltyMetric(
                title="Synthetic Data Novelty Score (Synthetic Rows not in Original Data / Total Synthetic Rows)",
                unit_measure="% - Range: from 0 (worst) to 100 (best)",
                value=np.round((new_synt_data / synth_len) * 100, 2).item(),
            )
        )

    def _evaluate_adherence(self):
        """
        Computes adherence metrics such as:
        - Synthetic Categories Adherence to Real Categories
        - Numerical min-max boundaries adherence

        :return: A tuple containing:
            - category_adherence_score: dict mapping column name to adherence percentage.
            - boundary_adherence_score: dict mapping column name to adherence percentage.
        """
        total_records = self._synth_data.columns[0].get_data().shape[0]

        # --- Categorical Adherence ---
        # For each categorical column, compute the percentage of synthetic entries
        # that have values found in the real data.
        category_adherence_score: dict[str, float] = {}
        numerical_columns = self._real_data.get_numeric_columns()
        synth_numerical_columns = self._synth_data.get_numeric_columns()
        categorical_columns = self._real_data.get_categorical_columns()
        synth_categorical_columns = self._synth_data.get_categorical_columns()

        for real_cat, synth_cat in zip(categorical_columns, synth_categorical_columns):
            real_data = real_cat.get_data()
            synth_data = synth_cat.get_data()
            extra_values = np.array(
                set(np.unique(synth_data)) - set(np.unique(real_data))
            )
            # Count how many synthetic records use these extra values.
            extra_count = np.sum(np.isin(synth_data, extra_values))
            # Define adherence as the percentage of records that do NOT have extra values.
            adherence_percentage = np.round((1 - extra_count / total_records) * 100, 2)
            category_adherence_score[real_cat.name] = float(adherence_percentage)

        # --- Numerical Boundary Adherence ---
        # For each numerical column, compute the percentage of synthetic entries
        # that lie within the min-max boundaries of the real data.
        boundary_adherence_score: dict[str, float] = {}

        for real_num, synth_num in zip(numerical_columns, synth_numerical_columns):
            # Obtain min and max boundaries from the real data.
            min_boundary = real_num.get_data().min()
            max_boundary = real_num.get_data().max()
            # Filter synthetic records that fall within these boundaries.
            synth_data = synth_num.get_data()
            in_boundary = synth_data[
                (synth_data >= min_boundary) & (synth_data <= max_boundary)
            ]
            in_boundary_count = in_boundary.shape[0]
            adherence_percentage = np.round(in_boundary_count / total_records * 100, 2)
            boundary_adherence_score[real_num.name] = float(adherence_percentage)

        if not len(categorical_columns) == 0:
            self.report.add_metric(
                AdherenceMetric(
                    title="Synthetic Categories Adherence to Real Categories",
                    unit_measure="% - Range: from 0 (worst) to 100 (best)",
                    value=category_adherence_score,
                )
            )

        if not len(numerical_columns) == 0:
            self.report.add_metric(
                AdherenceMetric(
                    title="Synthetic Numerical Min-Max Boundaries Adherence",
                    unit_measure="% - Range: from 0 (worst) to 100 (best)",
                    value=boundary_adherence_score,
                )
            )

    def _evaluate_categorical_frequency_absolute_difference(self):
        categorical_columns = self._real_data.get_categorical_columns()
        synth_categorical_columns = self._synth_data.get_categorical_columns()

        result_dictionary = {}
        for real_cat, synth_cat in zip(categorical_columns, synth_categorical_columns):
            feature_name = real_cat.name
            result_dictionary[feature_name] = {}
            real_data = real_cat.get_data()
            synth_data = synth_cat.get_data()
            real_samples = real_data.shape[0]
            synthetic_samples = synth_data.shape[0]
            real_categories, real_counts = np.unique(real_data, return_counts=True)
            synthetic_categories, synthetic_counts = np.unique(
                synth_data, return_counts=True
            )
            real_frequencies = real_counts / real_samples
            real_frequencies = {k: v for k, v in zip(real_categories, real_frequencies)}
            synthetic_frequencies = synthetic_counts / synthetic_samples
            synthetic_frequencies = {
                k: v for k, v in zip(synthetic_categories, synthetic_frequencies)
            }
            for cat in real_categories:
                result_dictionary[feature_name][str(cat)] = {}
                if cat in synthetic_categories:
                    result_dictionary[feature_name][str(cat)]["difference"] = np.round(
                        (real_frequencies[cat] - synthetic_frequencies[cat]) * 100, 2
                    ).item()
                    result_dictionary[feature_name][str(cat)]["real_frequency"] = (
                        np.round(real_frequencies[cat] * 100, 2).item()
                    )
                    result_dictionary[feature_name][str(cat)]["synthetic_frequency"] = (
                        np.round(synthetic_frequencies[cat] * 100, 2).item()
                    )
                else:
                    result_dictionary[feature_name][str(cat)]["difference"] = (
                        "Not Caught"
                    )
                    result_dictionary[feature_name][str(cat)]["real_frequency"] = (
                        np.round(real_frequencies[cat] * 100, 2).item()
                    )
                    result_dictionary[feature_name][str(cat)]["synthetic_frequency"] = (
                        0.00
                    )
            for cat in synthetic_categories:
                if cat not in real_categories:
                    result_dictionary[feature_name][str(cat)] = {}
                    result_dictionary[feature_name][str(cat)]["difference"] = (
                        "Does Exists in real data"
                    )
                    result_dictionary[feature_name][str(cat)]["real_frequency"] = 0.00
                    result_dictionary[feature_name][str(cat)]["synthetic_frequency"] = (
                        np.round(synthetic_frequencies[cat] * 100, 2).item()
                    )

        if len(categorical_columns) > 0:
            self.report.add_metric(
                StatisticalMetric(
                    title="Categorical Frequency Difference",
                    unit_measure="% - Range: from -100 to 100. Negative values imply overrepresentation in synthetic data, positive values imply underrepresentation. 0 Is optimal",
                    value=result_dictionary,
                )
            )


def compute_cramer_v(data1: np.ndarray, data2: np.ndarray):
    """
    Computes Cramer's V on a pair of categorical columns
    :param data1: first column
    :param data2: second column
    :return: Cramer's V
    """
    confusion_matrix = pd.crosstab(
        data1.reshape(
            -1,
        ),
        data2.reshape(
            -1,
        ),
    )
    v = ss.contingency.association(confusion_matrix)
    return v


def give_stringed_unique_rows(dirty_data: np.ndarray):
    """
    Works on 0-axis of a numpy array to search unique values
    :param dirty_data:
    :return:
    """
    data = dirty_data.tolist()
    seen = {}
    unique = []
    for o in data:
        if str(o) not in seen:
            seen[str(o)] = True
            unique.append(str(o))

    return np.array(unique)
