import numpy as np

from sdg_core_lib.post_process.functions.Parameter import Parameter
from sdg_core_lib.post_process.functions.filter.MonoThreshold import MonoThreshold


class UpperThreshold(MonoThreshold):
    description = "Keeps only values lower than a defined threshold"

    def __init__(self, parameters: list[Parameter]):
        super().__init__(parameters)

    def apply(
        self, n_rows: int, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        if self.strict:
            indexes = np.less_equal(data, self.value)
        else:
            indexes = np.less(data, self.value)

        removed_indexes = indexes
        data[removed_indexes] = np.nan
        return data, removed_indexes, True
