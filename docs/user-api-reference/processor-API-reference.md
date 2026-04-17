# Processor API Reference

This document provides detailed API documentation for data preprocessing and postprocessing operations.

## Table of Contents

1. [Processor Base Class](#processor-base-class)
2. [TableProcessor](#tableprocessor)
3. [Preprocessing Steps](#preprocessing-steps)
4. [Usage Examples](#usage-examples)
5. [Related APIs](#related-apis)

---

## Processor Base Class

Abstract base class for all data processors.

### Constructor

```python
Processor(dir_path: str)
```

**Parameters:**
- `dir_path` (str): Directory path to save/load preprocessing artifacts

### Methods

#### set_strategy()
```python
set_strategy(strategy: BasePreprocessingStrategy) -> Processor
```
Set preprocessing strategy for processor.

**Parameters:**
- `strategy` (BasePreprocessingStrategy): Strategy instance defining preprocessing approach

**Returns:**
- `Processor`: Self instance for method chaining

#### add_steps()
```python
add_steps(steps: list[Step], col_position: int, data_position: int) -> Processor
```
Add preprocessing steps for a specific column.

**Parameters:**
- `steps` (list[Step]): List of preprocessing steps to apply
- `col_position` (int): Column position in dataset
- `data_position` (int): Data position in array

**Returns:**
- `Processor`: Self instance for method chaining

#### process()
```python
process(data: list) -> dict[int, np.ndarray]
```
Apply preprocessing transformations to data.

**Parameters:**
- `data` (list): List of column data to preprocess

**Returns:**
- `dict[int, np.ndarray]`: Dictionary mapping column positions to preprocessed arrays

**Functioning:**
- Applies all preprocessing steps to each column
- Fits and transforms data using defined steps
- Saves preprocessing artifacts to disk
- Returns transformed data as numpy arrays

#### inverse_process()
```python
inverse_process(data: list) -> dict[int, np.ndarray]
```
Apply inverse transformations to reverse preprocessing.

**Parameters:**
- `data` (list): List of preprocessed column data

**Returns:**
- `dict[int, np.ndarray]`: Dictionary mapping column positions to original-format arrays

**Functioning:**
- Loads saved preprocessing artifacts from disk
- Applies inverse transformations in reverse order
- Restores data to original scale/format
- Returns data suitable for postprocessing

#### save_all()
```python
save_all() -> None
```
Save all preprocessing artifacts to disk.

**Functioning:**
- Saves scalers, encoders, and other preprocessing artifacts
- Uses processor's directory path for storage
- Ensures artifacts are available for later use

#### load_all()
```python
load_all() -> Processor
```
Load all preprocessing artifacts from disk.

**Returns:**
- `Processor`: Self instance with loaded artifacts

**Functioning:**
- Loads previously saved preprocessing artifacts
- Enables consistent preprocessing across different sessions
- Essential for inference mode operations

---

## TableProcessor

Specialized processor for tabular data preprocessing.

### Features
- Handles mixed data types (numeric, categorical)
- Supports scaling, encoding, and normalization
- Maintains column relationships
- Optimized for tabular datasets

### Constructor

```python
TableProcessor(dir_path: str)
```

**Parameters:**
- `dir_path` (str): Directory path for saving preprocessing artifacts

---

## Preprocessing Steps

### Numeric Scaling Steps

#### StandardScalerStep
Standardizes features by removing mean and scaling to unit variance.

**Parameters:**
- None (automatic)

**Effect:**
- Mean = 0, Standard Deviation = 1
- Preserves Gaussian distribution shape

#### MinMaxScalerStep
Scales features to a given range.

**Parameters:**
- `feature_range` (tuple): Target range (default: (0, 1))

**Effect:**
- All values in specified range
- Preserves original distribution shape

#### RobustScalerStep
Scales features using statistics robust to outliers.

**Parameters:**
- `quantile_range` (tuple): Quantile range (default: (25.0, 75.0))

**Effect:**
- Uses median and IQR for scaling
- Robust to outliers

### Categorical Encoding Steps

#### OneHotEncoderStep
Encodes categorical features as one-hot numeric array.

**Parameters:**
- `handle_unknown` (str): How to handle unknown categories ('error', 'ignore')
- `drop` (str): Whether to drop one category to avoid multicollinearity

**Effect:**
- Binary columns for each category
- Suitable for linear models

#### LabelEncoderStep
Encodes categorical features as integer labels.

**Parameters:**
- None (automatic)

**Effect:**
- Integer labels for categories
- Preserves ordinal information

### Missing Value Handling Steps

#### SimpleImputerStep
Imputes missing values using simple strategies.

**Parameters:**
- `strategy` (str): Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
- `fill_value` (any): Value to use for 'constant' strategy

**Effect:**
- Replaces NaN/None values
- Maintains data completeness

### Feature Engineering Steps

#### PolynomialFeaturesStep
Generates polynomial and interaction features.

**Parameters:**
- `degree` (int): Polynomial degree
- `include_bias` (bool): Whether to include bias term
- `interaction_only` (bool): Whether to include only interaction features

**Effect:**
- Captures non-linear relationships
- Increases feature dimensionality

---

## Usage Examples

### Basic Preprocessing

```python
from sdg_core_lib.preprocess.table_processor import TableProcessor
from sdg_core_lib.preprocess.strategies.table_strategy import TableStrategy

# Create processor
processor = TableProcessor("./preprocessing_artifacts")

# Set strategy
strategy = TableStrategy()
processor.set_strategy(strategy)

# Add preprocessing steps
steps = [
    StandardScalerStep(),
    OneHotEncoderStep()
]
processor.add_steps(steps, col_position=0, data_position=0)

# Apply preprocessing
preprocessed_data = processor.process(dataset.columns)

# Apply inverse preprocessing
original_data = processor.inverse_process(preprocessed_data)
```

### Advanced Preprocessing Pipeline

```python
# Complex preprocessing for mixed data
processor = TableProcessor("./artifacts")
processor.set_strategy(TableStrategy())

# Numeric column preprocessing
numeric_steps = [
    SimpleImputerStep(strategy="median"),
    RobustScalerStep(),
    PolynomialFeaturesStep(degree=2)
]
processor.add_steps(numeric_steps, col_position=0, data_position=0)

# Categorical column preprocessing
categorical_steps = [
    SimpleImputerStep(strategy="most_frequent"),
    OneHotEncoderStep(handle_unknown="ignore", drop="first")
]
processor.add_steps(categorical_steps, col_position=1, data_position=1)

# Process dataset
preprocessed = processor.process(dataset.columns)
```

### Custom Preprocessing Steps

```python
from sdg_core_lib.preprocess.steps.base import Step

class CustomStep(Step):
    def fit(self, data):
        # Custom fitting logic
        return self
    
    def transform(self, data):
        # Custom transformation logic
        return transformed_data
    
    def inverse_transform(self, data):
        # Custom inverse logic
        return original_data
    
    def save(self, path):
        # Save custom artifacts
        pass
    
    def load(self, path):
        # Load custom artifacts
        pass

# Use in processor
processor.add_steps([CustomStep()], col_position=2, data_position=2)
```

### Integration with Dataset

```python
# Preprocess dataset
preprocessed_dataset = dataset.preprocess(processor)

# Train model on preprocessed data
model.train(preprocessed_dataset.get_computing_data())

# Generate synthetic data
synthetic_data = model.infer(n_rows=1000)

# Postprocess to original format
final_dataset = preprocessed_dataset.clone(synthetic_data)
final_dataset = final_dataset.postprocess(processor)
```

---

## Related APIs

For complete API documentation, see:

- **[Job API Reference](./job-API-reference.md)** - Core job management and orchestration
- **[Dataset API Reference](./dataset-API-reference.md)** - Data input/output and skeleton operations
- **[Model API Reference](./model-API-reference.md)** - Machine learning model interfaces
- **[Functions API Reference](./functions-API-reference.md)** - Mathematical functions for data generation
- **[Evaluation API Reference](./evaluation-API-reference.md)** - Quality evaluation and metrics
