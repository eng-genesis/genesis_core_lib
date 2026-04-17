# Dataset API Reference

This document provides detailed API documentation for Dataset classes and data handling operations.

## Table of Contents

1. [Dataset Base Class](#dataset-base-class)
2. [Table Class](#table-class)
3. [TimeSeries Class](#timeseries-class)
4. [Related APIs](#related-apis)

---

## Dataset Base Class

Abstract base class for all data types.

### Methods

#### from_json()
```python
from_json(data: list[dict]) -> Dataset
```
Load data from JSON format.

**Parameters:**
- `data` (list[dict]): JSON data configuration

**Returns:**
- `Dataset`: Dataset instance

**Example Structure:**
```python
# Tabular data example
tabular_data = [
    {
        "column_name": "age",
        "column_type": "continuous",
        "column_datatype": "float64",
        "column_data": [25.0, 30.0, 35.0, 40.0, 45.0]
    },
    {
        "column_name": "category",
        "column_type": "categorical",
        "column_datatype": "str",
        "column_data": ["A", "B", "A", "C", "B"]
    }
]

# Time series data example
time_series_data = [
    {
        "column_name": "experiment_id",
        "column_type": "group_index",
        "column_datatype": "int",
        "column_data": [1, 1, 1, 2, 2, 2]
    },
    {
        "column_name": "time",
        "column_type": "primary_key",
        "column_datatype": "int",
        "column_data": [0, 1, 2, 0, 1, 2]
    },
    {
        "column_name": "value",
        "column_type": "continuous",
        "column_datatype": "float",
        "column_data": [1.1, 1.2, 1.3, 2.1, 2.2, 2.3]
    }
]

dataset = Dataset.from_json(tabular_data)
```

#### from_skeleton()
```python
from_skeleton(skeleton: list[dict]) -> Dataset
```
Create dataset from schema skeleton.

**Parameters:**
- `skeleton` (list[dict]): Schema skeleton

**Returns:**
- `Dataset`: Dataset instance

## What is a Data Skeleton?

A **Data Skeleton** is a schema representation of a Pre-Processed dataset that contains structural metadata without actual data values. It describes the dataset's architecture, including column names, types, positions, and sizes, which is used for:

- **Model Configuration**: Providing input shape and feature information to ML models
- **Data Generation**: Creating new datasets from schema definitions
- **Inference Mode**: Enabling model inference without requiring original training data
- **Schema Validation**: Ensuring data consistency across operations

**Example Structure:**
```python
# Tabular skeleton example (actual structure from implementation)
tabular_skeleton = [
    {
        "feature_name": "age",
        "feature_position": 0,
        "feature_type": "continuous",
        "type": "float64",
        "is_categorical": false,
        "feature_size": "1"
    },
    {
        "feature_name": "category",
        "feature_position": 1,
        "feature_type": "categorical",
        "type": "str",
        "is_categorical": true,
        "feature_size": "1"
    }
]

# Time series skeleton example
time_series_skeleton = [
    {
        "feature_name": "time",
        "feature_position": 1,
        "feature_type": "primary_key",
        "type": "int",
        "is_categorical": false,
        "feature_size": "1"
    },
    {
        "feature_name": "value",
        "feature_position": 2,
        "feature_type": "continuous",
        "type": "float",
        "is_categorical": false,
        "feature_size": "1"
    }
]

# Create dataset from skeleton
dataset = Dataset.from_skeleton(tabular_skeleton)
```

**Skeleton Fields Explained:**
- `feature_name`: Name of column/feature
- `feature_position`: Zero-based position index in dataset
- `feature_type`: Type of feature (`"continuous"`, `"categorical"`, `"primary_key"`, `"group_index"`)
- `type`: Data type (`"float64"`, `"int"`, `"str"`, etc.)
- `is_categorical`: Boolean indicating if feature is categorical
- `feature_size`: Size of feature (number of dimensions, typically "1" for single columns)

**Note**: The skeleton structure is a **list of dictionaries**, not a nested dictionary with `"columns"` key. This is the actual structure returned by the `to_skeleton()` method and expected by `from_skeleton()`.

#### preprocess()
```python
preprocess(processor: Processor) -> Dataset
```
Apply preprocessing transformations using a processor. 

**Parameters:**
- `processor` (Processor): Processor instance containing preprocessing steps and strategy

*See [Processor API Reference](./processor-API-reference.md) for detailed processor documentation.*

**Returns:**
- `Dataset`: Preprocessed dataset with transformed columns

**Functioning:**
- Applies all preprocessing steps defined in the processor to each column
- Saves preprocessing artifacts (scalers, encoders, etc.) to disk
- Returns new dataset instance with preprocessed columns
- For Table: Returns new Table with same primary key indexes
- For TimeSeries: Returns new TimeSeries with same group index

#### postprocess()
```python
postprocess(processor: Processor) -> Dataset
```
Apply postprocessing transformations to reverse preprocessing.

**Parameters:**
- `processor` (Processor): Processor instance containing preprocessing steps

**Returns:**
- `Dataset`: Postprocessed dataset with original data types restored

**Functioning:**
- Loads saved preprocessing artifacts from disk
- Applies inverse transformations in reverse order to each column
- Restores original data types (e.g., converts float back to int/str)
- Returns new dataset instance with postprocessed columns
- For Table: Returns new Table with same primary key indexes
- For TimeSeries: Returns new TimeSeries with same group index

#### to_json()
```python
to_json() -> list[dict]
```
Convert dataset to JSON format.

**Returns:**
- `list[dict]`: JSON representation

#### to_skeleton()
```python
to_skeleton() -> list[dict]
```
Extract schema skeleton.

**Returns:**
- `list[dict]`: Schema skeleton representation

#### clone()
```python
clone(new_data: np.ndarray) -> Dataset
```
Create new dataset with different data.

**Parameters:**
- `new_data` (np.ndarray): New data array

**Returns:**
- `Dataset`: New dataset instance

#### get_computing_data()
```python
get_computing_data() -> np.ndarray
```
Get data in computing format.

**Returns:**
- `np.ndarray`: Data array for ML operations

#### get_computing_shape()
```python
get_computing_shape() -> tuple[int, ...]
```
Get shape of computing data.

**Returns:**
- `tuple[int, ...]`: Data shape tuple

---

## Table Class

Specialized dataset for tabular data.

### Constructor

```python
Table(columns: list[Column], pk_indexes: list[int] = None)
```

**Parameters:**
- `columns` (list[Column]): List of column objects
- `pk_indexes` (list[int]): Primary key column indexes

### Methods

#### get_primary_keys()
```python
get_primary_keys() -> list[Column]
```
Get primary key columns.

**Returns:**
- `list[Column]`: Primary key column objects

#### get_numeric_columns()
```python
get_numeric_columns() -> list[Numeric]
```
Get numeric columns.

**Returns:**
- `list[Numeric]`: Numeric column objects

#### get_categorical_columns()
```python
get_categorical_columns() -> list[Categorical]
```
Get categorical columns.

**Returns:**
- `list[Categorical]`: Categorical column objects

#### all_to_numeric()
```python
all_to_numeric() -> Table
```
Convert all columns to numeric.

**Returns:**
- `Table`: New table with numeric columns

#### all_to_categorical()
```python
all_to_categorical() -> Table
```
Convert all columns to categorical.

**Returns:**
- `Table`: New table with categorical columns

---

## TimeSeries Class

Specialized dataset for time series data, extends Table.

### Constructor

```python
TimeSeries(inner_table: Table, group_index: int = None)
```

**Parameters:**
- `inner_table` (Table): Inner table containing time series data
- `group_index` (int): Index of group index column

### Methods

#### get_experiment_length()
```python
get_experiment_length() -> int
```
Get length of time series experiments.

**Returns:**
- `int`: Length of each experiment

#### get_computing_data()
```python
get_computing_data() -> np.ndarray
```
Get data in time series computing format.

**Returns:**
- `np.ndarray`: Data in shape (batch, features, time_steps)

---

## Related APIs

For complete API documentation, see:

- **[Job API Reference](./job-API-reference.md)** - Core job management and orchestration
- **[Model API Reference](./model-API-reference.md)** - Machine learning model interfaces
- **[Functions API Reference](./functions-API-reference.md)** - Mathematical functions for data generation
- **[Processor API Reference](./processor-API-reference.md)** - Data preprocessing and postprocessing
- **[Evaluation API Reference](./evaluation-API-reference.md)** - Quality evaluation and metrics
