import numpy as np

from sdg_core_lib.post_process.functions.Parameter import Parameter
from sdg_core_lib.post_process.functions.filter.IntervalThreshold import (
    IntervalThreshold,
)


class InnerThreshold(IntervalThreshold):
    description = "Keeps data in a given interval"

    def __init__(self, parameters: list[Parameter]):
        super().__init__(parameters)

    def apply(
        self, n_rows: int, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        if self.lower_strict:
            upper_indexes = np.greater_equal(data, self.lower_bound)
        else:
            upper_indexes = np.greater(data, self.lower_bound)

        if self.upper_strict:
            lower_indexes = np.less_equal(data, self.upper_bound)
        else:
            lower_indexes = np.less(data, self.upper_bound)

        final_indexes = lower_indexes & upper_indexes
        data[final_indexes] = np.nan
        return data, final_indexes, True
