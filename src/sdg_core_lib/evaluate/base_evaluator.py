from sdg_core_lib.dataset.datasets import Dataset
from sdg_core_lib.evaluate.metrics import MetricReport
from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    def __init__(self, real_data: Dataset, synthetic_data: Dataset):
        self._real_data = real_data
        self._synth_data = synthetic_data
        self.report = MetricReport()

    @abstractmethod
    def compute(self) -> dict:
        raise NotImplementedError
