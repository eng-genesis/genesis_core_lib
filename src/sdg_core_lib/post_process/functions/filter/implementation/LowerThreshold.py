import numpy as np

from sdg_core_lib.post_process.functions.Parameter import Parameter
from sdg_core_lib.post_process.functions.filter.MonoThreshold import MonoThreshold


class LowerThreshold(MonoThreshold):
    description = "Keeps only values greater than a defined threshold"

    def __init__(self, parameters: list[Parameter]):
        super().__init__(parameters)

    def apply(
        self, n_rows: int, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        if self.strict:
            indexes = np.greater_equal(data, self.value)
        else:
            indexes = np.greater(data, self.value)

        removed_indexes = indexes
        data[removed_indexes] = np.nan
        return data, removed_indexes, True
