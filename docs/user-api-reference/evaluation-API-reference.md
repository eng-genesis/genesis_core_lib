# Evaluation API Reference

This document provides comprehensive API documentation for the quality evaluation system, which assesses the similarity and quality of synthetic data compared to real data.

## Architecture Overview

The evaluation system consists of three main components:

1. **BaseEvaluator**: Abstract base class that defines the evaluation interface
2. **TabularComparisonEvaluator**: Specialized evaluator for tabular datasets
3. **TimeSeriesComparisonEvaluator**: Extended evaluator for time series data
4. **MetricReport**: Structured reporting system for evaluation results

```
+-------------------------------------------+
|              BaseEvaluator                |
|  +-------------------------------------+  |
|  | Abstract Interface    | Report Mgmt |  |
|  | Dataset Validation    | Metric Org |  |
|  +-------------------------------------+  |
+-------------------------------------------+
                    |
                    | (inherits)
                    |
+-------------------------------------------+
|        TabularComparisonEvaluator         |
|  +-------------------------------------+  |
|  | Statistical Metrics  | Wasserstein |  |
|  | Adherence Metrics   | Cramer's V  |  |
|  | Novelty Metrics     | Frequency   |  |
|  +-------------------------------------+  |
+-------------------------------------------+
                    |
                    | (inherits)
                    |
+-------------------------------------------+
|      TimeSeriesComparisonEvaluator       |
|  +-------------------------------------+  |
|  | All Tabular Metrics  | DTW Analysis |  |
|  | Temporal Similarity  | Time Patterns|  |
|  +-------------------------------------+  |
+-------------------------------------------+
```

## Table of Contents

1. [Base Evaluator](#base-evaluator)
2. [TabularComparisonEvaluator](#tabularcomparisonevaluator)
3. [TimeSeriesComparisonEvaluator](#timeseriescomparisonevaluator)
4. [Metric Classes](#metric-classes)
5. [Complete Usage Examples](#complete-usage-examples)
6. [Best Practices](#best-practices)
7. [Related APIs](#related-apis)

---

## Base Evaluator

Abstract base class that defines the evaluation interface and manages metric reporting.

### Core Architecture

The BaseEvaluator maintains:
- `_real_data`: Reference to the original real dataset
- `_synth_data`: Reference to the synthetic dataset
- `report`: MetricReport instance for structured results

### Constructor

```python
BaseEvaluator(real_data: Dataset, synthetic_data: Dataset)
```

**Parameters:**
- `real_data` (Dataset): Original real dataset
- `synthetic_data` (Dataset): Generated synthetic dataset

### Methods

#### compute()
```python
compute() -> dict
```
Abstract method that computes quality metrics between real and synthetic data.

**Returns:**
- `dict`: Dictionary containing structured quality metrics

**Note:** Must be implemented by concrete evaluator classes

---

## TabularComparisonEvaluator

Comprehensive evaluator for tabular datasets that computes statistical, adherence, and novelty metrics.

### Evaluation Categories

The TabularComparisonEvaluator computes three main categories of metrics:

1. **Statistical Metrics**: Distribution similarity and feature relationships
2. **Adherence Metrics**: Boundary and category compliance
3. **Novelty Metrics**: Uniqueness and new value generation

### Constructor

```python
TabularComparisonEvaluator(real_data: Table, synthetic_data: Table)
```

**Parameters:**
- `real_data` (Table): Original real tabular dataset
- `synthetic_data` (Table): Generated synthetic tabular dataset

**Validation:**
- Raises `TypeError` if inputs are not Table instances

### Methods

#### compute()
```python
compute() -> dict
```
Computes comprehensive quality metrics for tabular data.

**Returns:**
- `dict`: Structured metrics with keys:
  - `statistical_metrics`: Statistical similarity measures
  - `adherence_metrics`: Boundary and category adherence
  - `novelty_metrics`: Uniqueness and novelty scores

**Functioning:**
1. Extracts numeric and categorical columns from both datasets
2. Computes Metrics
3. Returns metrics in JSON format

### Statistical Metrics Computed

#### Wasserstein Distance
- **Title**: "Continuous Features Statical Distance (Wasserstein Distance)"
- **Range**: 0-100% (0 = identical distribution, 100 = completely different)
- **Interpretation**: Lower values indicate better distribution matching

#### Cramer's V Association Distance
- **Title**: "Association Distance Index (Cramer's V, Real vs Synthetic)"
- **Range**: 0-100% (0 = perfect association preservation, 100 = no association preservation)
- **Interpretation**: Measures preservation of categorical feature relationships

#### Categorical Frequency Difference
- **Title**: "Categorical Frequency Difference"
- **Range**: -100% to 100% (0 = perfect frequency matching)
- **Interpretation**: Negative values = overrepresentation, positive = underrepresentation

### Adherence Metrics Computed

#### Numerical Boundary Adherence
- **Title**: "Synthetic Numerical Min-Max Boundaries Adherence"
- **Range**: 0-100% (100% = all values within real data boundaries)
- **Interpretation**: Percentage of synthetic values within real data min-max ranges

#### Categorical Adherence
- **Title**: "Synthetic Categories Adherence to Real Categories"
- **Range**: 0-100% (100% = no new categories introduced)
- **Interpretation**: Percentage of synthetic values using only real data categories

### Novelty Metrics Computed

#### Uniqueness Score
- **Title**: "Synthetic Data Uniqueness Score (Unique Synthetic Rows / Total Synthetic Rows)"
- **Range**: 0-100% (higher = more unique synthetic records)
- **Interpretation**: Measures diversity within synthetic data

#### Novelty Score
- **Title**: "Synthetic Data Novelty Score (Synthetic Rows not in Original Data / Total Synthetic Rows)"
- **Range**: 0-100% (higher = more new combinations)
- **Interpretation**: Measures generation of unseen data patterns

---

## TimeSeriesComparisonEvaluator

Extended evaluator for time series datasets that includes all tabular metrics plus temporal similarity measures.

### Enhanced Features
- Inherits all tabular evaluation capabilities
- Adds Dynamic Time Warping (DTW) for temporal pattern comparison
- Handles multivariate time series data

### Constructor

```python
TimeSeriesComparisonEvaluator(real_data: TimeSeries, synthetic_data: TimeSeries)
```

**Parameters:**
- `real_data` (TimeSeries): Original real time series dataset
- `synthetic_data` (TimeSeries): Generated synthetic time series dataset

**Validation:**
- Raises `ValueError` if inputs are not TimeSeries instances

### Methods

#### compute()
```python
compute() -> dict
```
Computes comprehensive metrics including temporal analysis.

**Returns:**
- `dict`: All tabular metrics plus:
  - `time_series_metrics`: Temporal pattern similarity measures

**Functioning:**
1. Executes all tabular evaluation metrics
2. Performs multivariate DTW analysis on numeric time series
3. Samples 30 records from each dataset for efficiency
4. Computes average DTW distance between all sample pairs

### Time Series Metrics Computed

#### Multivariate DTW Similarity
- **Title**: "Sample Mean Time Series Evolution Similarity (Multivariate Dependent Dynamic Time Warping)"
- **Range**: 0 to infinity (lower = better temporal similarity)
- **Interpretation**: Measures similarity of temporal evolution patterns across all features

**Requirements:**
- Minimum 30 samples in each dataset
- At least one numeric column
- Uses random sampling for computational efficiency

---

## Metric Classes

The evaluation system uses a structured metric hierarchy for organized reporting.

### Base Metric

Abstract base class for all metrics.

```python
class Metric:
    def __init__(self, title: str, unit_measure: str, value: float | int | dict)
```

**Attributes:**
- `title` (str): Human-readable metric name
- `unit_measure` (str): Description of measurement units and range
- `value` (float|int|dict): Computed metric value(s)
- `type` (str): Metric type category (set by subclasses)

#### Methods

##### to_json()
```python
to_json() -> dict
```
Converts metric to JSON-serializable dictionary.

### StatisticalMetric

Metrics related to statistical properties and distributions.

```python
class StatisticalMetric(Metric):
    type = "statistical_metrics"
```

**Used For:**
- Wasserstein distance
- Cramer's V association
- Frequency differences

### AdherenceMetric

Metrics related to boundary and category compliance.

```python
class AdherenceMetric(Metric):
    type = "adherence_metrics"
```

**Used For:**
- Numerical boundary adherence
- Categorical category adherence

### NoveltyMetric

Metrics related to uniqueness and new value generation.

```python
class NoveltyMetric(Metric):
    type = "novelty_metrics"
```

**Used For:**
- Data uniqueness scores
- Novelty generation measures

### TimeSeriesSpecificMetric

Metrics specific to time series data analysis.

```python
class TimeSeriesSpecificMetric(Metric):
    type = "time_series_metrics"
```

**Used For:**
- Dynamic Time Warping similarity
- Temporal pattern preservation

### MetricReport

Container for organizing and exporting evaluation results.

```python
class MetricReport:
    def __init__(self)
```

#### Methods

##### add_metric()
```python
add_metric(metric: Metric) -> None
```
Adds a metric to the appropriate category in the report.

**Functioning:**
- Groups metrics by their `type` attribute
- Maintains list of metrics per category

##### to_json()
```python
to_json() -> dict
```
Exports the complete report as structured JSON.

**Returns:**
- `dict`: Nested structure with metric categories as keys

---

## Complete Usage Examples

### Basic Tabular Evaluation

```python
from sdg_core_lib.evaluate.tables import TabularComparisonEvaluator
from sdg_core_lib.dataset.datasets import Table

# Create sample data
real_data_json = [
    {
        "column_name": "age",
        "column_type": "continuous",
        "column_data": [25, 30, 35, 40, 45, 50],
        "column_datatype": "int32"
    },
    {
        "column_name": "income",
        "column_type": "continuous", 
        "column_data": [50000, 60000, 70000, 80000, 90000, 100000],
        "column_datatype": "float32"
    },
    {
        "column_name": "education",
        "column_type": "categorical",
        "column_data": ["HS", "Bachelor", "Master", "PhD", "Bachelor", "Master"],
        "column_datatype": "str"
    }
]

# Create datasets
real_data = Table.from_json(real_data_json)
synthetic_data = Table.from_json(real_data_json)  # Using same data for demo

# Create evaluator
evaluator = TabularComparisonEvaluator(real_data, synthetic_data)

# Compute evaluation metrics
metrics = evaluator.compute()

# Print results
print("=== Tabular Evaluation Results ===")
for category, metric_list in metrics.items():
    print(f"\n{category.upper()}:")
    for metric in metric_list:
        print(f"  {metric['title']}: {metric['value']} {metric['unit_measure']}")
```

### Time Series Evaluation

```python
from sdg_core_lib.evaluate.time_series import TimeSeriesComparisonEvaluator
from sdg_core_lib.dataset.datasets import TimeSeries

# Create time series data (assuming TimeSeries.from_json works similarly)
real_ts_data = TimeSeries.from_json(real_time_series_json)
synthetic_ts_data = TimeSeries.from_json(synthetic_time_series_json)

# Create time series evaluator
ts_evaluator = TimeSeriesComparisonEvaluator(real_ts_data, synthetic_ts_data)

# Compute metrics including temporal analysis
ts_metrics = ts_evaluator.compute()

# Analyze temporal similarity
if 'time_series_metrics' in ts_metrics:
    print("\n=== Time Series Analysis ===")
    for metric in ts_metrics['time_series_metrics']:
        print(f"{metric['title']}: {metric['value']} {metric['unit_measure']}")
```


### Integration with Data Generation Pipeline

```python
from sdg_core_lib.dataset.datasets import Table
from sdg_core_lib.evaluate.tables import TabularComparisonEvaluator
from sdg_core_lib.preprocess.table_processor import TableProcessor
from sdg_core_lib.preprocess.strategies.vae_strategy import TabularVAEPreprocessingStrategy
import numpy as np
    
# Step 1: Load and prepare real data (Table)

# Step 2: Set up preprocessing of real data

# Step 3: Train and generate synthetic data

# Create synthetic dataset
synthetic_data = Table(synthetic_columns) # Generated by the model

# Step 4: Postprocess synthetic data
postprocessed_synthetic = processor.inverse_process(synthetic_columns)

    
# Step 5: Evaluate quality
evaluator = TabularComparisonEvaluator(real_data, synthetic_data)
evaluation_metrics = evaluator.compute()
print(evaluation_metrics)

```

---

## Best Practices

### Data Quality Requirements

1. **Minimum Data Size**: Ensure sufficient data for reliable evaluation
   - Tabular data: At least 100 rows recommended
   - Time series: Minimum 30 samples for DTW analysis

2. **Column Consistency**: Real and synthetic datasets must have:
   - Same column names
   - Same column order
   - Compatible data types

3. **Data Validation**: Check for common issues:
   - NaN values in numeric columns
   - Empty categorical columns
   - Single-value columns

### Metric Interpretation Guidelines

1. **Statistical Metrics**:
   - Wasserstein < 20%: Excellent distribution matching
   - Cramer's V < 10%: Good association preservation
   - Frequency differences < 5%: Balanced categorical representation

2. **Adherence Metrics**:
   - Boundary adherence > 95%: Proper range constraints
   - Category adherence > 90%: No category leakage

3. **Novelty Metrics**:
   - Uniqueness > 95%: New Samples generated correctly without collapsing
   - Novelty > 95%: Good new pattern generation

### Performance Considerations

1. **Large Datasets**: For datasets > 10,000 rows:
   - Consider sampling for efficiency
   - Monitor memory usage
   - Use batch processing if needed

2. **Time Series**: Optimize DTW computation:
   - System uses 30-sample random sampling
   - Ensure adequate numeric columns
   - Consider dimensionality reduction for high-dimensional data

### Common Pitfalls to Avoid

1. **Data Mismatch**: Never compare datasets with different schemas
2. **Small Sample Sizes**: Avoid evaluation with < 30 samples
3. **Single-Value Columns**: Metrics may be undefined for constant columns
4. **Interpretation Errors**: Remember that lower Wasserstein scores are better

### Advanced Usage Tips

1. **Custom Metrics**: Extend BaseEvaluator for domain-specific needs
2. **Batch Evaluation**: Process multiple synthetic datasets systematically
3. **Threshold Setting**: Establish quality thresholds for your specific use case
4. **Trend Analysis**: Track metrics over multiple generation runs


---

## Related APIs

For complete API documentation, see:

- **[Job API Reference](./job-API-reference.md)** - Core job management and orchestration
- **[Dataset API Reference](./dataset-API-reference.md)** - Data input/output and skeleton operations
- **[Model API Reference](./model-API-reference.md)** - Machine learning model interfaces
- **[Functions API Reference](./functions-API-reference.md)** - Mathematical functions for data generation
- **[Processor API Reference](./processor-API-reference.md)** - Data preprocessing and transformation
