# Evaluation API Reference

This document provides detailed API documentation for quality evaluation and metrics used in synthetic data generation.

## Table of Contents

1. [Evaluator Classes](#evaluator-classes)
2. [Quality Metrics](#quality-metrics)
3. [Usage Examples](#usage-examples)
4. [Related APIs](#related-apis)

---

## Evaluator Classes

### Base Evaluator

Abstract base class for all quality evaluators.

#### Methods

##### compute()
```python
compute() -> dict
```
Compute quality metrics between real and synthetic data.

**Returns:**
- `dict`: Dictionary containing quality metrics

### TableEvaluator

Evaluator for tabular datasets.

#### Constructor

```python
TableEvaluator(real_data: Table, synthetic_data: Table)
```

**Parameters:**
- `real_data` (Table): Original real dataset
- `synthetic_data` (Table): Generated synthetic dataset

#### Methods

##### compute()
```python
compute() -> dict
```

Computes quality metrics for tabular data.

**Returns:**
- `dict`: Quality metrics including:
  - `statistical_similarity`: Overall statistical similarity score
  - `correlation_preservation`: Feature correlation preservation
  - `distribution_similarity`: Distribution similarity metrics
  - `privacy_score`: Privacy preservation metrics

### TimeSeriesEvaluator

Evaluator for time series datasets.

#### Constructor

```python
TimeSeriesEvaluator(real_data: TimeSeries, synthetic_data: TimeSeries)
```

**Parameters:**
- `real_data` (TimeSeries): Original real time series
- `synthetic_data` (TimeSeries): Generated synthetic time series

#### Methods

##### compute()
```python
compute() -> dict
```

Computes quality metrics for time series data.

**Returns:**
- `dict`: Quality metrics including:
  - `temporal_similarity`: Temporal pattern preservation
  - `autocorrelation_preservation`: Autocorrelation similarity
  - `seasonality_preservation`: Seasonal pattern preservation
  - `trend_preservation`: Trend similarity
  - `statistical_similarity`: Overall statistical similarity

---

## Quality Metrics

### Statistical Similarity

Measures how well synthetic data preserves statistical properties of real data.

#### Components
- **Mean Absolute Error**: Difference in means between real and synthetic data
- **Standard Deviation Error**: Difference in variability
- **Distribution Distance**: Statistical distance between distributions
- **Correlation Matrix Similarity**: Preservation of feature correlations

#### Interpretation
- **0.0-0.3**: Poor similarity
- **0.3-0.6**: Moderate similarity
- **0.6-0.8**: Good similarity
- **0.8-1.0**: Excellent similarity

### Privacy Preservation

Measures how well synthetic data protects privacy of real data.

#### Components
- **Distance to Closest Record**: Average distance to nearest real record
- **Uniqueness Ratio**: Proportion of unique synthetic records
- **Membership Disclosure Risk**: Risk of identifying real records
- **Attribute Disclosure Risk**: Risk of revealing sensitive attributes

#### Interpretation
- **0.0-0.3**: Poor privacy protection
- **0.3-0.6**: Moderate privacy protection
- **0.6-0.8**: Good privacy protection
- **0.8-1.0**: Excellent privacy protection

### Temporal Similarity (Time Series)

Measures preservation of temporal patterns in time series data.

#### Components
- **Autocorrelation Similarity**: Preservation of temporal dependencies
- **Seasonality Similarity**: Preservation of seasonal patterns
- **Trend Similarity**: Preservation of long-term trends
- **Volatility Similarity**: Preservation of volatility patterns

#### Interpretation
- **0.0-0.3**: Poor temporal preservation
- **0.3-0.6**: Moderate temporal preservation
- **0.6-0.8**: Good temporal preservation
- **0.8-1.0**: Excellent temporal preservation

---

## Usage Examples

### Basic Evaluation

```python
from sdg_core_lib.dataset.evaluators import TableEvaluator
from sdg_core_lib.dataset.datasets import Table

# Load real and synthetic data
real_data = Table.from_json(real_data_config)
synthetic_data = Table.from_json(synthetic_data_config)

# Create evaluator
evaluator = TableEvaluator(real_data, synthetic_data)

# Compute metrics
metrics = evaluator.compute()

print(f"Statistical Similarity: {metrics['statistical_similarity']:.3f}")
print(f"Privacy Score: {metrics['privacy_score']:.3f}")
```

### Time Series Evaluation

```python
from sdg_core_lib.dataset.evaluators import TimeSeriesEvaluator
from sdg_core_lib.dataset.datasets import TimeSeries

# Load time series data
real_ts = TimeSeries.from_json(real_ts_config)
synthetic_ts = TimeSeries.from_json(synthetic_ts_config)

# Create evaluator
evaluator = TimeSeriesEvaluator(real_ts, synthetic_ts)

# Compute metrics
metrics = evaluator.compute()

print(f"Temporal Similarity: {metrics['temporal_similarity']:.3f}")
print(f"Autocorrelation Preservation: {metrics['autocorrelation_preservation']:.3f}")
```

### Custom Evaluation Metrics

```python
from sdg_core_lib.dataset.evaluators.base import BaseEvaluator

class CustomEvaluator(BaseEvaluator):
    def __init__(self, real_data, synthetic_data):
        super().__init__(real_data, synthetic_data)
    
    def compute(self):
        # Implement custom metric calculation
        custom_metric = self.calculate_custom_metric()
        
        return {
            "custom_metric": custom_metric,
            "additional_info": self.get_additional_info()
        }
    
    def calculate_custom_metric(self):
        # Custom calculation logic
        pass
    
    def get_additional_info(self):
        # Additional evaluation information
        pass

# Use custom evaluator
evaluator = CustomEvaluator(real_data, synthetic_data)
metrics = evaluator.compute()
```

### Integration with Job

```python
from sdg_core_lib import Job

# Job automatically computes evaluation metrics
job = Job(
    n_rows=1000,
    model_info=model_config,
    dataset=dataset_config
)

# Train and evaluate
synthetic_data, metrics, model, schema = job.train()

# Metrics are automatically computed and returned
print("Quality Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value}")
```

### Evaluation Report Analysis

```python
def analyze_evaluation_report(metrics):
    """Analyze evaluation metrics and provide insights."""
    
    print("=== Quality Evaluation Report ===")
    
    # Statistical similarity analysis
    if 'statistical_similarity' in metrics:
        sim_score = metrics['statistical_similarity']
        print(f"\nStatistical Similarity: {sim_score:.3f}")
        if sim_score >= 0.8:
            print("✓ Excellent statistical preservation")
        elif sim_score >= 0.6:
            print("✓ Good statistical preservation")
        elif sim_score >= 0.3:
            print("⚠ Moderate statistical preservation")
        else:
            print("✗ Poor statistical preservation")
    
    # Privacy analysis
    if 'privacy_score' in metrics:
        privacy_score = metrics['privacy_score']
        print(f"\nPrivacy Score: {privacy_score:.3f}")
        if privacy_score >= 0.8:
            print("✓ Excellent privacy protection")
        elif privacy_score >= 0.6:
            print("✓ Good privacy protection")
        elif privacy_score >= 0.3:
            print("⚠ Moderate privacy protection")
        else:
            print("✗ Poor privacy protection")
    
    # Time series specific analysis
    if 'temporal_similarity' in metrics:
        temp_score = metrics['temporal_similarity']
        print(f"\nTemporal Similarity: {temp_score:.3f}")
        if temp_score >= 0.8:
            print("✓ Excellent temporal pattern preservation")
        elif temp_score >= 0.6:
            print("✓ Good temporal pattern preservation")
        elif temp_score >= 0.3:
            print("⚠ Moderate temporal pattern preservation")
        else:
            print("✗ Poor temporal pattern preservation")
    
    # Overall assessment
    print(f"\n=== Overall Assessment ===")
    avg_score = sum(metrics.values()) / len(metrics)
    if avg_score >= 0.7:
        print("🎯 Overall: High quality synthetic data")
    elif avg_score >= 0.5:
        print("👍 Overall: Good quality synthetic data")
    else:
        print("⚠️ Overall: Needs improvement")

# Analyze metrics
analyze_evaluation_report(metrics)
```

### Quality Thresholds

You can set quality thresholds for automated decision making:

```python
class QualityThresholds:
    def __init__(self):
        self.min_statistical_similarity = 0.6
        self.min_privacy_score = 0.7
        self.min_temporal_similarity = 0.5
    
    def meets_quality_standards(self, metrics):
        """Check if metrics meet quality standards."""
        
        if 'statistical_similarity' in metrics:
            if metrics['statistical_similarity'] < self.min_statistical_similarity:
                return False, "Statistical similarity too low"
        
        if 'privacy_score' in metrics:
            if metrics['privacy_score'] < self.min_privacy_score:
                return False, "Privacy protection insufficient"
        
        if 'temporal_similarity' in metrics:
            if metrics['temporal_similarity'] < self.min_temporal_similarity:
                return False, "Temporal similarity too low"
        
        return True, "All quality standards met"

# Use thresholds
thresholds = QualityThresholds()
passes, reason = thresholds.meets_quality_standards(metrics)

if passes:
    print("✓ Quality standards met")
else:
    print(f"✗ Quality standards not met: {reason}")
```

---

## Related APIs

For complete API documentation, see:

- **[Job API Reference](./job-API-reference.md)** - Core job management and orchestration
- **[Dataset API Reference](./dataset-API-reference.md)** - Data input/output and skeleton operations
- **[Model API Reference](./model-API-reference.md)** - Machine learning model interfaces
- **[Functions API Reference](./functions-API-reference.md)** - Mathematical functions for data generation
- **[Processor API Reference](./processor-API-reference.md)** - Data preprocessing and postprocessing
