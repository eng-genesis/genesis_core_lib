from tslearn.metrics import dtw
import numpy as np

from sdg_core_lib.evaluate.tables import TabularComparisonEvaluator
from sdg_core_lib.evaluate.metrics import TimeSeriesSpecificMetric
from sdg_core_lib.dataset.datasets import TimeSeries


class TimeSeriesComparisonEvaluator(TabularComparisonEvaluator):
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
        real_data: TimeSeries,
        synthetic_data: TimeSeries,
    ):
        if not isinstance(real_data, TimeSeries):
            raise ValueError("real_data must be a TimeSeries")
        if not isinstance(synthetic_data, TimeSeries):
            raise ValueError("synthetic_data must be a TimeSeries")
        super().__init__(real_data, synthetic_data)

    def compute(self) -> dict:
        numerical_columns = self._real_data.get_numeric_columns()
        categorical_columns = self._real_data.get_categorical_columns()
        if len(numerical_columns) < 1 and len(categorical_columns) < 1:
            return {"available": "false"}

        super().compute()
        self._compute_multivariate_dependent_dtw()

        return self.report.to_json()

    def _compute_multivariate_dependent_dtw(self):
        numerical_real_data = self._real_data.all_to_numeric()
        if len(numerical_real_data.get_numeric_columns()) < 1:
            return
        numerical_synth_data = self._synth_data.all_to_numeric()

        real_data = numerical_real_data.get_computing_data()
        synthetic_data = numerical_synth_data.get_computing_data()

        # Avoid small datasets
        min_size = 30
        if real_data.shape[0] < min_size or synthetic_data.shape[0] < min_size:
            return

        random_sample = np.random.choice(
            real_data.shape[0], size=min_size, replace=False
        )
        real_data = real_data[random_sample]
        random_sample = np.random.choice(
            synthetic_data.shape[0], size=min_size, replace=False
        )
        synthetic_data = synthetic_data[random_sample]
        generated_samples = min_size
        real_samples = min_size
        total_score = 0
        for real_sample in real_data:
            partial_score = 0
            for synth_sample in synthetic_data:
                partial_score += float(dtw(real_sample, synth_sample))
            total_score += partial_score / generated_samples
        result = total_score / real_samples

        self.report.add_metric(
            TimeSeriesSpecificMetric(
                title="Sample Mean Time Series Evolution Similarity (Multivariate Dependent Dynamic Time Warping",
                unit_measure="Pure Number - Range: from 0 (Synthetic features are IDENTICAL to Real Features), no upper bound. Lower is better",
                value=result,
            )
        )
