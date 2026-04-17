# Processor API Reference

This document provides comprehensive API documentation for the preprocessing system, which follows a Strategy-Step-Processor pattern for flexible data transformation.

## Architecture Overview

The preprocessing system consists of three main components:

1. **Processor**: Orchestrates the preprocessing workflow and manages step execution
2. **Strategy**: Determines which preprocessing steps to apply to different feature types
3. **Step**: Individual transformation operations that can be chained together

```
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Processor │───▶│    Strategy     │───▶│      Step       │
│             │    │                 │    │                 │
│ • Orchestrates│    │ • Determines    │    │ • Transforms    │
│ • Manages    │    │   steps per     │    │   data          │
│   execution  │    │   feature type  │    │ • Saves/Loads   │
│ • Persistence│    │ • Encapsulates  │    │   artifacts     │
└─────────────┘    │   preprocessing │    └─────────────────┘
                   │   logic         │
                   └─────────────────┘
```

## Table of Contents

1. [Processor Base Class](#processor-base-class)
2. [TableProcessor](#tableprocessor)
3. [Strategies](#strategies)
4. [Steps](#steps)
6. [Complete Usage Examples](#complete-usage-examples)
7. [Related APIs](#related-apis)

---

## Processor Base Class

Abstract base class that orchestrates preprocessing workflows.

### Core Architecture

The Processor maintains:
- `steps`: Dictionary mapping column positions to lists of preprocessing steps
- `idx_to_data`: Mapping between column positions and data array indices (for data shape transformation)
- `strategy`: Strategy instance that determines appropriate steps per feature
- `dir_path`: Directory for persisting preprocessing artifacts

### Constructor

```python
Processor(dir_path: str)
```

**Parameters:**
- `dir_path` (str): Directory path where preprocessing artifacts are saved/loaded

### Methods

#### set_strategy()
```python
set_strategy(strategy: BasePreprocessingStrategy) -> Processor
```
Sets the preprocessing strategy that determines which steps to apply to each feature type.

**Parameters:**
- `strategy` (BasePreprocessingStrategy): Strategy instance defining preprocessing logic

**Returns:**
- `Processor`: Self instance for method chaining

**Example:**
```python
from sdg_core_lib.preprocess.strategies.vae_strategy import TabularVAEPreprocessingStrategy

processor = TableProcessor("./artifacts")
processor.set_strategy(TabularVAEPreprocessingStrategy())
```

#### add_steps()
```python
add_steps(steps: list[Step], col_position: int, data_position: int) -> Processor
```
Manually adds preprocessing steps for a specific column. Usually called automatically by strategy.

**Parameters:**
- `steps` (list[Step]): List of preprocessing steps to apply in sequence
- `col_position` (int): Column position in the original dataset
- `data_position` (int): Position of this column's data in the input array

**Returns:**
- `Processor`: Self instance for method chaining

#### process()
```python
process(data: list) -> dict[int, np.ndarray]
```
Applies preprocessing transformations to input data.

**Parameters:**
- `data` (list): List of column data arrays to preprocess

**Returns:**
- `dict[int, np.ndarray]`: Dictionary mapping column positions to preprocessed arrays

**Functioning:**
1. For each column position, applies the configured steps in sequence
2. Each step calls `fit_transform()` to learn parameters and transform data
3. Automatically saves all preprocessing artifacts to disk
4. Returns transformed data ready for model training

#### inverse_process()
```python
inverse_process(data: list) -> dict[int, np.ndarray]
```
Applies inverse transformations to reverse preprocessing (post-processing).

**Parameters:**
- `data` (list): List of preprocessed column data arrays

**Returns:**
- `dict[int, np.ndarray]`: Dictionary mapping column positions to original-format arrays

**Functioning:**
1. Loads saved preprocessing artifacts from disk
2. Applies inverse transformations in reverse order
3. Restores data to original scale and format
4. Returns data suitable for downstream consumption

#### save_all()
```python
save_all() -> None
```
Saves all preprocessing artifacts to disk for later reuse.

**Functioning:**
- Iterates through all steps and calls their `save_if_not_exist()` methods
- Uses skops format for sklearn-based transformers
- Essential for consistent preprocessing across training and inference

#### load_all()
```python
load_all() -> Processor
```
Loads all preprocessing artifacts from disk.

**Returns:**
- `Processor`: Self instance with loaded artifacts

**Functioning:**
- Restores sklearn transformers and custom operators
- Enables consistent preprocessing in inference mode
- Called automatically by `inverse_process()`

---

## TableProcessor

Specialized processor for both tabular and time-series data that works with Column objects.

### Enhanced Features
- Automatic step initialization based on column types
- Column object preservation through preprocessing pipeline
- Seamless integration with dataset preprocessing workflows

### Constructor

```python
TableProcessor(dir_path: str)
```

**Parameters:**
- `dir_path` (str): Directory path for saving/loading preprocessing artifacts

### Enhanced Methods

#### process()
```python
process(columns: list[Column]) -> list[Column]
```
Processes a list of Column objects and returns preprocessed Column objects.

**Parameters:**
- `columns` (list[Column]): List of Column objects containing raw data

**Returns:**
- `list[Column]`: List of Column objects with preprocessed data

**Functioning:**
1. Calls `_init_steps()` to automatically configure steps based on strategy
2. Extracts data from Column objects
3. Applies preprocessing using base Processor logic
4. Creates new Column objects with transformed data
5. Preserves column metadata (name, type, position)

#### inverse_process()
```python
inverse_process(preprocessed_columns: list[Column]) -> list[Column]
```
Reverses preprocessing on Column objects.

**Parameters:**
- `preprocessed_columns` (list[Column]): Column objects with preprocessed data

**Returns:**
- `list[Column]`: Column objects with data restored to original format

---

## Strategies

Strategies encapsulate the logic for determining which preprocessing steps to apply to different feature types.

### BasePreprocessingStrategy

Abstract base class for all preprocessing strategies.

```python
class BasePreprocessingStrategy:
    @staticmethod
    def get_steps_per_feature(feature: Column) -> list[Step]:
        # Returns empty list - should be overridden
        return []
```

### Available Strategies

#### TabularVAEPreprocessingStrategy
Optimized for VAE-based tabular data generation.

**Step Selection:**
- **Numeric columns**: StandardScaler (mean=0, std=1)
- **Categorical columns**: OneHotEncoder
- **Generic columns**: NoneStep (no transformation)

```python
from sdg_core_lib.preprocess.strategies.vae_strategy import TabularVAEPreprocessingStrategy

strategy = TabularVAEPreprocessingStrategy()
processor.set_strategy(strategy)
```

#### TimeSeriesVAEPreprocessingStrategy
Optimized for time series VAE models.

**Step Selection:**
- **Numeric columns**: MinMaxScaler (range [0, 1])
- **Categorical columns**: OneHotEncoder
- **Generic columns**: NoneStep

```python
from sdg_core_lib.preprocess.strategies.vae_strategy import TimeSeriesVAEPreprocessingStrategy

strategy = TimeSeriesVAEPreprocessingStrategy()
processor.set_strategy(strategy)
```

#### CTGANPreprocessingStrategy
Specialized for CTGAN models with mode-based normalization.

**Step Selection:**
- **Numeric columns**: PerModeNormalization (Gaussian mixture-based)
- **Categorical columns**: OneHotEncoder
- **Generic columns**: NoneStep

```python
from sdg_core_lib.preprocess.strategies.ctgan_strategy import CTGANPreprocessingStrategy

strategy = CTGANPreprocessingStrategy()
processor.set_strategy(strategy)
```

---

## Steps

Steps are individual transformation operations that implement the preprocessing logic.

### Base Step Architecture

All steps inherit from the abstract `Step` class:

```python
class Step(ABC):
    def __init__(self, type_name: str, position: int, col_name: str, mode: str)
    
    @abstractmethod
    def _set_operator(self)  # Sets the sklearn transformer
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray
    def transform(self, data: np.ndarray) -> np.ndarray
    def inverse_transform(self, data: np.ndarray) -> np.ndarray
    def save_if_not_exist(self, directory_path: str)
    def load(self, directory_path: str)
```

### Available Step Types

#### NoneStep
No-op step that passes data through unchanged.

**Use Case:**
- Columns that don't require preprocessing
- Placeholder for future preprocessing steps

```python
from sdg_core_lib.preprocess.strategies.steps import NoneStep

step = NoneStep(position=0)
```

#### ScalerWrapper
Wraps sklearn scalers for numeric data normalization.

**Modes:**
- `"standard"`: StandardScaler (mean=0, std=1)
- `"minmax"`: MinMaxScaler (default range [0, 1])

```python
from sdg_core_lib.preprocess.strategies.steps import ScalerWrapper

# Standard scaling
standard_step = ScalerWrapper(position=0, col_name="age", mode="standard")

# Min-max scaling
minmax_step = ScalerWrapper(position=1, col_name="income", mode="minmax")
```

#### OneHotEncoderWrapper
Wraps sklearn OneHotEncoder for categorical data.

**Features:**
- Handles unknown categories (error mode)
- Returns dense numpy arrays
- Numerical stability in inverse transform

```python
from sdg_core_lib.preprocess.strategies.steps import OneHotEncoderWrapper

step = OneHotEncoderWrapper(position=2, col_name="education")
```

#### OrdinalEncoderWrapper
Wraps sklearn OrdinalEncoder for categorical data.

**Features:**
- Handles unknown categories with NaN values
- Preserves ordinal relationships

```python
from sdg_core_lib.preprocess.strategies.steps import OrdinalEncoderWrapper

step = OrdinalEncoderWrapper(position=3, col_name="rating")
```

#### PerModeNormalization
Advanced step for CTGAN models using Gaussian mixture models.

**Features:**
- Automatically detects data modes using Bayesian Gaussian Mixture
- Performs mode-specific normalization
- Essential for CTGAN's conditional generation

**Parameters:**
- `n_components` (int, default=10): Maximum number of mixture components
- `max_iter` (int, default=1000): Maximum iterations for GMM fitting
- `random_state` (int, default=42): Random seed for reproducibility

```python
from sdg_core_lib.preprocess.strategies.steps import PerModeNormalization

step = PerModeNormalization(
    position=0, 
    col_name="income",
    n_components=15,
    max_iter=2000
)
```

**Internal Functioning:**
1. Fits Bayesian Gaussian Mixture to detect data modes
2. Assigns each data point to most likely mode
3. Normalizes values within each mode: `(x - mean) / (4 * std)`
4. Returns concatenated normalized values + mode assignments

---

## Complete Usage Examples

### Basic TableProcessor Usage

```python
from sdg_core_lib.preprocess.table_processor import TableProcessor
from sdg_core_lib.preprocess.strategies.vae_strategy import TabularVAEPreprocessingStrategy
from sdg_core_lib.dataset.columns import Numeric, Categorical

# Create sample columns
columns = [
    Numeric("age", "float32", 0, [25, 30, 35, 40, 45]),
    Numeric("income", "float32", 1, [50000, 60000, 70000, 80000, 90000]),
    Categorical("education", "string", 2, ["HS", "Bachelor", "Master", "PhD"]),
    Categorical("city", "string", 3, ["NYC", "LA", "Chicago", "Houston"])
]

# Create processor with VAE strategy
processor = TableProcessor("./preprocessing_artifacts")
processor.set_strategy(TabularVAEPreprocessingStrategy())

# Apply preprocessing
preprocessed_columns = processor.process(columns)

# Access preprocessed data
for col in preprocessed_columns:
    print(f"{col.name}: {col.get_data().shape}")
    print(f"Sample data: {col.get_data()[:3]}")

# Apply inverse preprocessing
original_columns = processor.inverse_process(preprocessed_columns)
```

### Manual Step Configuration

```python
from sdg_core_lib.preprocess.table_processor import TableProcessor
from sdg_core_lib.preprocess.strategies.steps import (
    ScalerWrapper, OneHotEncoderWrapper, PerModeNormalization
)

# Create processor
processor = TableProcessor("./manual_artifacts")

# Manually configure steps for each column
processor.add_steps([
    ScalerWrapper(position=0, col_name="age", mode="standard")
], col_position=0, data_position=0)

processor.add_steps([
    PerModeNormalization(position=1, col_name="income")
], col_position=1, data_position=1)

processor.add_steps([
    OneHotEncoderWrapper(position=2, col_name="education")
], col_position=2, data_position=2)

# Process data
preprocessed_data = processor.process([
    [25, 30, 35, 40, 45],  # age
    [50000, 60000, 70000, 80000, 90000],  # income
    ["HS", "Bachelor", "Master", "PhD", "Bachelor"]  # education
])

print("Preprocessed data keys:", preprocessed_data.keys())
for pos, data in preprocessed_data.items():
    print(f"Position {pos}: shape {data.shape}")
```

### Integration with Dataset

```python
from sdg_core_lib import Dataset
from sdg_core_lib.preprocess.table_processor import TableProcessor
from sdg_core_lib.preprocess.strategies.ctgan_strategy import CTGANPreprocessingStrategy

# Load dataset
dataset = Dataset.from_csv("data.csv")

# Create processor with CTGAN strategy
processor = TableProcessor("./ctgan_preprocessing")
processor.set_strategy(CTGANPreprocessingStrategy())

# Preprocess dataset
preprocessed_dataset = dataset.preprocess(processor)

# Train CTGAN model
ctgan = CTGAN(metadata=preprocessed_dataset.get_skeleton(), model_name="my_ctgan")
ctgan.train(preprocessed_dataset.get_computing_data())

# Generate synthetic data
synthetic_data = ctgan.infer(n_rows=1000)

# Postprocess to original format
synthetic_dataset = preprocessed_dataset.clone(synthetic_data)
synthetic_dataset = synthetic_dataset.postprocess(processor)

# Save results
synthetic_dataset.to_csv("synthetic_data.csv")
```


---

## Best Practices

1. **Strategy Selection**: Choose strategies based on your target model:
   - VAE models: `TabularVAEPreprocessingStrategy` or `TimeSeriesVAEPreprocessingStrategy`
   - CTGAN models: `CTGANPreprocessingStrategy`

2. **Artifact Management**: Always use consistent directory paths for preprocessing artifacts to ensure reproducibility.

3. **Custom Steps**: When creating custom steps, ensure proper implementation of `save_if_not_exist()` and `load()` methods for persistence.

4. **Column Positioning**: Maintain consistent column positions between preprocessing and postprocessing to avoid data misalignment.

5. **Memory Management**: For large datasets, consider processing in batches to manage memory usage effectively.

---

## Related APIs

For complete API documentation, see:

- **[Job API Reference](./job-API-reference.md)** - Core job management and orchestration
- **[Dataset API Reference](./dataset-API-reference.md)** - Data input/output and skeleton operations  
- **[Model API Reference](./model-API-reference.md)** - Machine learning model interfaces
- **[Functions API Reference](./functions-API-reference.md)** - Mathematical functions for data generationn
- **[Evaluation API Reference](./evaluation-API-reference.md)** - Quality evaluation and metrics
