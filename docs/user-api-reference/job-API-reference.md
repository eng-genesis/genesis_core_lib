# Job API Reference

This document provides detailed API documentation for the Job class, the main orchestrator for synthetic data generation workflows.

## Table of Contents

1. [Job Class](#job-class)
2. [Related APIs](#related-apis)

---

## Job Class

The `Job` class is the main orchestrator for synthetic data generation workflows.

### Constructor

```python
Job(
    n_rows: int,
    model_info: Optional[dict] = None,
    dataset: Optional[dict] = None,
    save_filepath: Optional[str] = None,
    functions: Optional[list[dict]] = None
)
```

**Parameters:**
- `n_rows` (int): Number of synthetic rows to generate
- `model_info` (Optional[dict]): Model configuration dictionary containing:
  - `algorithm_name` (str): Full class path of the model (e.g., `"sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE.TabularVAE"`)
  - `model_name` (str): Name identifier for the model
  - `input_shape` (Optional[tuple]): Model input shape (auto-inferred if not provided)
  - `training_data_info` (Optional[dict]): Training data schema for inference mode
- `dataset` (Optional[dict]): Dataset configuration dictionary containing:
  - `dataset_type` (str): Type of dataset (`"table"` or `"time_series"`)
  - `data` (list[dict]): List of column definitions with:
    - `column_name` (str): Name of column
    - `column_data` (list): Raw data values
    - `column_type` (str): Type (`"continuous"`, `"categorical"`, `"primary_key"`, `"group_index"`)
    - `column_datatype` (str): Data type (`"float64"`, `"int"`, `"str"`, etc.)
- `save_filepath` (Optional[str]): Path to save trained models
- `functions` (Optional[list[dict]]): List of function configurations for data generation/modification, each containing:
  - `feature` (str): Target column/feature name
  - `function_name` (str): Name of the function class (e.g., `"LinearFunction"`, `"QuadraticFunction"`)
  - `parameters` (dict): Function-specific parameters (varies by function type)

**Example:**
```python
from sdg_core_lib import Job

job = Job(
    n_rows=1000,
    model_info={
        "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE.TabularVAE",
        "model_name": "my_model"
    },
    dataset={
        "dataset_type": "table",
        "data": [...]  # Your data here
    },
    save_filepath="./models"
)
```

### Methods

#### train()

```python
train() -> tuple[list[dict], dict, UnspecializedModel, list[dict]]
```

Trains a model and generates synthetic data.

**Returns:**
- `results` (list[dict]): Generated synthetic data
- `metrics` (dict): Quality evaluation metrics
- `model` (UnspecializedModel): Trained model instance
- `schema` (list[dict]): Data schema information

**Example:**
```python
synthetic_data, metrics, model, schema = job.train()
print(f"Generated {len(synthetic_data)} rows")
print(f"Quality metrics: {metrics}")
```

#### infer()

```python
infer() -> tuple[list[dict], dict]
```

Generates data using a pre-trained model.

**Returns:**
- `results` (list[dict]): Generated synthetic data
- `metrics` (dict): Quality metrics (if real data available)

**Example:**
```python
# Assuming model is already trained
synthetic_data, metrics = job.infer()
```

#### generate_from_functions()

```python
generate_from_functions(dataset: Optional[Dataset] = None) -> list[dict]
```

Generates data using mathematical functions.

**Parameters:**
- `dataset` (Optional[Dataset]): Optional existing dataset to modify

**Returns:**
- `results` (list[dict]): Generated synthetic data

**Example:**
```python
functions = [
    {
        "feature": "x",
        "function_name": "LinearFunction",
        "parameters": {"m": 1.0, "q": 0.0, "min_value": 0.0, "max_value": 10.0}
    }
]

job = Job(n_rows=100, functions=functions)
data = job.generate_from_functions()
```

---

## Related APIs

For complete API documentation, see:

- **[Dataset API Reference](./dataset-API-reference.md)** - Data input/output and skeleton operations
- **[Model API Reference](./model-API-reference.md)** - Machine learning model interfaces
- **[Functions API Reference](./functions-API-reference.md)** - Mathematical functions for data generation
- **[Processor API Reference](./processor-API-reference.md)** - Data preprocessing and postprocessing
- **[Evaluation API Reference](./evaluation-API-reference.md)** - Quality evaluation and metrics
