# Custom Evaluators

This guide explains how to create custom evaluators for the Genesis Core Library. Custom evaluators allow you to implement specialized evaluation metrics for specific dataset types beyond the built-in table and time series evaluators.

## Understanding the Base Evaluator and Metrics

### BaseEvaluator Class

The `BaseEvaluator` is the abstract base class that all evaluators must inherit from. It provides a common interface for comparing real and synthetic datasets.

```python
from abc import ABC, abstractmethod
from sdg_core_lib.dataset.datasets import Dataset
from sdg_core_lib.evaluate.metrics import MetricReport

class BaseEvaluator(ABC):
    def __init__(self, real_data: Dataset, synthetic_data: Dataset):
        self._real_data = real_data
        self._synth_data = synthetic_data
        self.report = MetricReport()

    @abstractmethod
    def compute(self) -> dict:
        raise NotImplementedError
```

**Key Components:**
- **`real_data`**: The original/reference dataset
- **`synthetic_data`**: The generated/synthetic dataset to evaluate
- **`report`**: A `MetricReport` instance that collects all evaluation metrics
- **`compute()`**: Abstract method that must implement the evaluation logic

### Metrics System

The metrics system provides different types of evaluation metrics organized by category:

#### Base Metric Class

```python
class Metric:
    def __init__(self, title: str, unit_measure: str, value: float | int | dict):
        self.title = title
        self.unit_measure = unit_measure
        self.value = value
        self.type = None

    def to_json(self):
        return {
            "title": self.title,
            "unit_measure": self.unit_measure,
            "value": self.value,
        }
```

#### MetricReport Class

The `MetricReport` automatically organizes metrics by type:

```python
class MetricReport:
    def __init__(self):
        self.report = {}

    def add_metric(self, metric: Metric):
        if metric.type not in self.report:
            self.report[metric.type] = [metric.to_json()]
        else:
            self.report[metric.type].append(metric.to_json())

    def to_json(self):
        return self.report
```

## Creating Custom Evaluators and Matrics

### Custom Metric Types

You can create custom metric types for domain-specific evaluations:

```python
class TextSpecificMetric(Metric):
    def __init__(self, title: str, unit_measure: str, value: float | int | dict):
        super().__init__(title, unit_measure, value)
        self.type = "text_metrics"
```


### Design Principles

Custom evaluators should follow these principles:

1. **Dataset Type Specific**: Each evaluator is designed for a specific dataset type
2. **Metric Categorization**: Use appropriate metric types for different evaluation aspects
3. **Data Validation**: Validate input datasets match the expected type
4. **Comprehensive Coverage**: Evaluate multiple aspects of data quality
5. **Standardized Output**: Return consistent JSON format through `MetricReport`

### Implementation Pattern

Most custom evaluators follow this pattern:

```python
class CustomEvaluator(BaseEvaluator):
    def __init__(self, real_data: SpecificDataset, synthetic_data: SpecificDataset):
        # Validate dataset types
        if not isinstance(real_data, SpecificDataset):
            raise TypeError("real_data must be a SpecificDataset")
        if not isinstance(synthetic_data, SpecificDataset):
            raise TypeError("synthetic_data must be a SpecificDataset")
        
        super().__init__(real_data, synthetic_data)

    def compute(self) -> dict:
        # Check if data is available for evaluation
        if not self._has_evaluatable_data():
            return {"available": "false"}
        
        # Compute various metrics
        self._compute_statistical_metrics()
        self._compute_adherence_metrics()
        self._compute_novelty_metrics()
        self._compute_domain_specific_metrics()
        
        return self.report.to_json()
```

## Example: TextEvaluator

Let's create a comprehensive `TextEvaluator` for evaluating text generation models. This evaluator will work with text datasets and compute metrics specific to text quality.

### Step 1: Define the Text Dataset

First, we need a text dataset that properly inherits from the Dataset abstract base class (refer to [Custom Datasets](custom-datasets.md) for complete implementation details):

```python
import numpy as np
from typing import List, Dict
from sdg_core_lib.dataset.datasets import Dataset
from sdg_core_lib.dataset.columns import Column, Numeric, Categorical
from sdg_core_lib.preprocess.base_processor import Processor
from sdg_core_lib.dataset.validation_schema import FeatureData, DataSkeleton, SkeletonOut

class TextDataset(Dataset):
    def __init__(self, columns: List[Column]):
        super().__init__()
        self.columns = columns
        self._processed_data = None

    # ---- So much code ---- (See Dataset Docs)
    
    def get_column_texts(self, column_name: str = None) -> List[str]:
        """Get text strings from a specific column or all text columns"""
        if column_name:
            for column in self.columns:
                if column.name == column_name and column.value_type == "string":
                    return column.get_data().tolist()
            return []
        else:
            # Return all text columns concatenated
            all_texts = []
            for column in self.columns:
                if column.value_type == "string":
                    all_texts.extend(column.get_data().tolist())
            return all_texts
```

### Step 2: Create the TextEvaluator

```python
import numpy as np
from typing import List, Dict
from collections import Counter
import math

from sdg_core_lib.evaluate.base_evaluator import BaseEvaluator
from sdg_core_lib.evaluate.metrics import (
    StatisticalMetric,
    AdherenceMetric, 
    NoveltyMetric,
    MetricReport
)
from sdg_core_lib.dataset.datasets import Dataset


class TextEvaluator(BaseEvaluator):
    """
    Evaluator for text generation quality.
    
    Evaluates synthetic text against real text using multiple metrics:
    - Statistical: Vocabulary overlap, length distribution
    - Adherence: N-gram consistency, topic similarity
    - Novelty: Unique sequences, diversity metrics
    """
    
    def __init__(self, real_data: Dataset, synthetic_data: Dataset):
        # Validate that datasets are TextDataset instances
        if not hasattr(real_data, 'columns') or not isinstance(real_data, TextDataset):
            raise TypeError("real_data must be a TextDataset with columns")
        if not hasattr(synthetic_data, 'columns') or not isinstance(synthetic_data, TextDataset):
            raise TypeError("synthetic_data must be a TextDataset with columns")
        
        super().__init__(real_data, synthetic_data)
        
        # Extract text data from columns
        self._real_texts = real_data.get_column_texts()
        self._synth_texts = synthetic_data.get_column_texts()
    
    def compute(self) -> dict:
        """Compute all text evaluation metrics"""
        if len(self._real_texts) == 0 or len(self._synth_texts) == 0:
            return {"available": "false"}
        
        # Statistical metrics
        self._compute_vocabulary_overlap()
        self._compute_length_distribution_similarity()
        self._compute_token_frequency_similarity()
        
        # Novelty metrics
        self._compute_sequence_uniqueness()
        self._compute_lexical_diversity()
        
        return self.report.to_json()
    
    def _compute_vocabulary_overlap(self):
        """Compute vocabulary overlap between real and synthetic texts"""
        real_vocab = set()
        synth_vocab = set()
        
        for text in self._real_texts:
            real_vocab.update(text.lower().split())
        
        for text in self._synth_texts:
            synth_vocab.update(text.lower().split())
        
        # Calculate overlap metrics
        intersection = real_vocab.intersection(synth_vocab)
        union = real_vocab.union(synth_vocab)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0
        vocab_coverage = len(intersection) / len(real_vocab) if real_vocab else 0
        
        self.report.add_metric(StatisticalMetric(
            title="Vocabulary Jaccard Similarity",
            unit_measure="ratio",
            value=jaccard_similarity
        ))
        
        self.report.add_metric(StatisticalMetric(
            title="Vocabulary Coverage",
            unit_measure="ratio", 
            value=vocab_coverage
        ))
    
    def _compute_length_distribution_similarity(self):
        """Compare text length distributions"""
        real_lengths = [len(text.split()) for text in self._real_texts]
        synth_lengths = [len(text.split()) for text in self._synth_texts]
        
        # Calculate statistical measures
        real_mean = np.mean(real_lengths)
        synth_mean = np.mean(synth_lengths)
        real_std = np.std(real_lengths)
        synth_std = np.std(synth_lengths)
        
        # Mean absolute difference
        mean_diff = abs(real_mean - synth_mean)
        std_diff = abs(real_std - synth_std)
        
        self.report.add_metric(StatisticalMetric(
            title="Length Mean Difference",
            unit_measure="tokens",
            value=mean_diff
        ))
        
        self.report.add_metric(StatisticalMetric(
            title="Length Std Difference", 
            unit_measure="tokens",
            value=std_diff
        ))
    
    def _compute_token_frequency_similarity(self):
        """Compare token frequency distributions"""
        real_tokens = []
        synth_tokens = []
        
        for text in self._real_texts:
            real_tokens.extend(text.lower().split())
        
        for text in self._synth_texts:
            synth_tokens.extend(text.lower().split())
        
        real_freq = Counter(real_tokens)
        synth_freq = Counter(synth_tokens)
        
        # Calculate frequency correlation
        all_tokens = set(real_freq.keys()).union(set(synth_freq.keys()))
        real_counts = [real_freq.get(token, 0) for token in all_tokens]
        synth_counts = [synth_freq.get(token, 0) for token in all_tokens]
        
        if len(real_counts) > 1 and len(synth_counts) > 1:
            correlation = np.corrcoef(real_counts, synth_counts)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        self.report.add_metric(StatisticalMetric(
            title="Token Frequency Correlation",
            unit_measure="pearson_r",
            value=correlation
        ))
    
    def _compute_sequence_uniqueness(self):
        """Compute how many unique sequences are generated"""
        unique_sequences = len(set(self._synth_texts))
        total_sequences = len(self._synth_texts)
        uniqueness_ratio = unique_sequences / total_sequences if total_sequences > 0 else 0
        
        self.report.add_metric(NoveltyMetric(
            title="Sequence Uniqueness",
            unit_measure="ratio",
            value=uniqueness_ratio
        ))
    
    def _compute_lexical_diversity(self):
        """Compute lexical diversity metrics"""
        def type_token_ratio(texts):
            all_tokens = []
            for text in texts:
                all_tokens.extend(text.lower().split())
            
            unique_types = len(set(all_tokens))
            total_tokens = len(all_tokens)
            
            return unique_types / total_tokens if total_tokens > 0 else 0
        
        real_diversity = type_token_ratio(self._real_texts)
        synth_diversity = type_token_ratio(self._synth_texts)
        
        diversity_ratio = synth_diversity / real_diversity if real_diversity > 0 else 0
        
        self.report.add_metric(NoveltyMetric(
            title="Lexical Diversity Ratio",
            unit_measure="ratio",
            value=diversity_ratio
        ))
```

### Step 3: Using the TextEvaluator

```python
from sdg_core_lib.dataset.datasets import TextDataset
from sdg_core_lib.evaluate.text import TextEvaluator


# Create datasets (imagine they contain data)
real_dataset = TextDataset()
synthetic_dataset = TextDataset()

# Create evaluator
evaluator = TextEvaluator(real_dataset, synthetic_dataset)

# Compute evaluation metrics
results = evaluator.compute()

```

## Integration with the GENESIS System

### Dataset Type Validation

Custom evaluators should validate that they receive appropriate dataset types:

```python
def __init__(self, real_data: SpecificDataset, synthetic_data: SpecificDataset):
    if not isinstance(real_data, SpecificDataset):
        raise TypeError(f"real_data must be a SpecificDataset, got {type(real_data)}")
    if not isinstance(synthetic_data, SpecificDataset):
        raise TypeError(f"synthetic_data must be a SpecificDataset, got {type(synthetic_data)}")
    
    super().__init__(real_data, synthetic_data)
```

### Metric Organization

Properly categorize metrics to ensure consistent reporting:

```python
# Statistical metrics - data properties and distributions
self.report.add_metric(StatisticalMetric(...))

# Adherence metrics - how well synthetic matches real
self.report.add_metric(AdherenceMetric(...))

# Novelty metrics - diversity and uniqueness
self.report.add_metric(NoveltyMetric(...))

# Domain-specific metrics for specialized data types
self.report.add_metric(DomainSpecificMetric(...))
```

### Error Handling

Handle edge cases gracefully:

```python
def compute(self) -> dict:
    # Check data availability
    if not self._has_evaluatable_data():
        return {"available": "false"}
    
    try:
        # Compute metrics
        self._compute_metrics()
        return self.report.to_json()
    except Exception as e:
        return {"error": str(e), "available": "false"}
```

## Best Practices

### 1. Domain-Specific Metrics

Choose metrics relevant to your data domain:
- **Text**: Vocabulary overlap, n-gram consistency, lexical diversity
- **Images**: Structural similarity, feature distribution, visual quality
- **Time Series**: Temporal patterns, seasonality, trend consistency
- **Graph Data**: Topological properties, degree distribution, motif analysis

### 2. Multiple Evaluation Aspects

Evaluate data quality from multiple perspectives:
- **Statistical**: Distribution similarity, correlation preservation
- **Adherence**: How well synthetic data follows real data patterns
- **Novelty**: Diversity and uniqueness of generated data
- **Utility**: How well synthetic data works for downstream tasks

### 3. Performance Considerations

For large datasets:
- Use sampling for computationally expensive metrics
- Implement efficient data structures (Counters, sets)
- Consider parallel processing for independent metrics

### 4. Interpretability

Make metrics meaningful:
- Use clear, descriptive titles
- Provide appropriate units of measurement
- Include context about what values mean (higher/lower is better)

### 5. Testing

Test your evaluator thoroughly:
```python
import unittest
import numpy as np

class TestTextEvaluator(unittest.TestCase):
    def setUp(self):
        self.real_texts = ["test text one", "test text two"]
        self.synth_texts = ["generated text one", "generated text two"]
        
        self.real_dataset = TextDataset(self.real_texts, vocab={})
        self.synth_dataset = TextDataset(self.synth_texts, vocab={})
    
    def test_compute_metrics(self):
        evaluator = TextEvaluator(self.real_dataset, self.synth_dataset)
        results = evaluator.compute()
        
        self.assertIn("statistical_metrics", results)
        self.assertIn("adherence_metrics", results)
        self.assertIn("novelty_metrics", results)
    
    def test_empty_data(self):
        empty_dataset = TextDataset([], vocab={})
        evaluator = TextEvaluator(self.real_dataset, empty_dataset)
        results = evaluator.compute()
        
        self.assertEqual(results, {"available": "false"})
```




This comprehensive guide provides the foundation for creating custom evaluators that integrate seamlessly with the GENESIS system while providing domain-specific evaluation capabilities for your data types.
