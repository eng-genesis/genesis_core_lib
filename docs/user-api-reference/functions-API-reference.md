# Functions API Reference

This document provides comprehensive API documentation for the Functions system, which follows a FunctionApplier-Function-Parameter pattern for flexible data generation and modification.

## Architecture Overview

The Functions system consists of three main components:

1. **FunctionApplier**: Orchestrates function application and manages data generation/modification workflows
2. **Function**: Individual mathematical operations that generate or modify data
3. **Parameter**: Type-safe parameter handling with validation and conversion



## Table of Contents

1. [FunctionApplier](#functionapplier)
2. [Function Base Class](#function-base-class)
3. [Parameter Class](#parameter-class)
4. [Available Functions](#available-functions)
5. [Complete Usage Examples](#complete-usage-examples)
6. [Related APIs](#related-apis)

---

## FunctionApplier

The main orchestrator for applying functions to datasets, supporting both generation from scratch and modification of existing data.

### Core Architecture

The TabularFunctionApplier maintains:
- `function_feature_dict`: List of function configurations
- `n_rows`: Number of rows to generate/modify
- `from_scratch`: Whether to generate new data or modify existing
- `function_feature_mapping`: Processed mapping of features to function instances

### Constructor

```python
TabularFunctionApplier(
    function_feature_dict: list[dict], 
    n_rows: int, 
    from_scratch: bool = False
)
```

**Parameters:**
- `function_feature_dict` (list[dict]): List of function configurations
- `n_rows` (int): Number of rows to generate/modify
- `from_scratch` (bool): True for generation, False for modification

### Methods

#### apply_all()
```python
apply_all(dataset: Optional[Dataset] = None) -> Dataset
```
Applies all configured functions to generate or modify data.

**Parameters:**
- `dataset` (Optional[Dataset]): Existing dataset to modify (required if from_scratch=False)

**Returns:**
- `Dataset`: Dataset with functions applied

**Functioning:**
- Validates input parameters and function sequences
- Routes to `_generate_from_scratch()` or `_modify_existing_dataset()`
- Handles data cleaning and NaN removal
- Returns processed dataset

#### _generate_from_scratch()
```python
_generate_from_scratch() -> Dataset
```
Generates new dataset from scratch using configured functions.

**Returns:**
- `Dataset`: New Table dataset with generated data

**Functioning:**
1. Validates that first function for each feature is generative
2. Creates empty data arrays with appropriate dtypes
3. Applies functions in priority order (highest to lowest)
4. Removes NaN rows and creates final dataset
5. Returns new Table dataset

#### _modify_existing_dataset()
```python
_modify_existing_dataset(dataset: Dataset) -> Dataset
```
Modifies existing dataset by applying non-generative functions.

**Parameters:**
- `dataset` (Dataset): Existing dataset to modify

**Returns:**
- `Dataset`: Modified dataset

**Functioning:**
1. Validates that only non-generative functions are applied
2. Preserves unmapped features unchanged
3. Applies functions in priority order
4. Handles shape compatibility and data cleaning
5. Returns modified dataset clone

---

## Function Base Class

Abstract base class for all mathematical functions.

### Core Architecture

All functions inherit from `UnspecializedFunction` and define:

```python
class UnspecializedFunction(ABC):
    parameters: list[Parameter] = None           # Class-level parameter definitions
    description: str = None                      # Function description
    priority: Priority = Priority.NONE           # Execution priority
    is_generative: bool = None                  # Whether function generates data
    allowed_data: list[AllowedData] = None       # Supported data types
```

### Priority System

Functions are executed in priority order (highest to lowest):

```python
class Priority(Enum):
    MAX = 5      # Highest priority (usually generative functions)
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1  # Lowest priority
    NONE = None  # Undefined priority
```

### Required Methods

#### _check_parameters()
```python
_check_parameters(self) -> None
```
Validates and maps parameters to instance attributes.

**Functioning:**
- Maps parameter values to instance attributes
- Performs parameter-specific validation
- Raises ValueError for invalid configurations

#### apply()
```python
apply(self, n_rows: int, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]
```
Core method that implements the function's logic.

**Parameters:**
- `n_rows` (int): Number of rows to process
- `data` (np.ndarray): Input data (may be empty for generative functions)

**Returns:**
- `tuple[np.ndarray, np.ndarray, bool]`: (new_data, affected_indexes, success_flag)

**Functioning:**
- Implements specific mathematical operation
- Returns modified data and affected row indices
- Indicates success/failure status

### Class Methods

#### from_json()
```python
from_json(cls, json_params: list[dict]) -> UnspecializedFunction
```
Creates function instance from JSON parameter list.

**Parameters:**
- `json_params` (list[dict]): List of parameter dictionaries

**Returns:**
- `UnspecializedFunction`: Configured function instance

#### self_describe()
```python
self_describe(cls) -> dict
```
Returns comprehensive function metadata.

**Returns:**
- `dict`: Function description, parameters, and supported data types


### Function Configuration

All functions follow this configuration structure:

```python
{
    "feature": "feature_name",           # Target column name
    "function_name": "FunctionName",      # Function class name
    "parameters": {                      # Function-specific parameters
        # Varies by function type
    }
}
```

### Function Types

#### Generative Functions
- Create new data from scratch
- Must be first function in sequence when generating from scratch
- Examples: LinearFunction, QuadraticFunction, NormalDistributionSample

#### Modification Functions
- Modify existing data
- Can be applied after generative functions
- Examples: WhiteNoiseAdder, ScalingFunction


---

## Parameter Class

Type-safe parameter handling with automatic validation and conversion.

### Constructor

```python
Parameter(name: str, value: str, parameter_type: str)
```

**Parameters:**
- `name` (str): Parameter name
- `value` (str): Parameter value as string
- `parameter_type` (str): Target Python type ("float", "int", "str", etc.)

### Methods

#### to_json()
```python
to_json() -> dict
```
Serializes parameter to JSON format.

**Returns:**
- `dict`: Parameter representation with name, value, and type

#### from_json()
```python
from_json(cls, json_data: dict) -> Parameter
```
Creates Parameter instance from JSON data.

**Parameters:**
- `json_data` (dict): JSON representation of parameter

**Returns:**
- `Parameter`: Configured parameter instance

**Functioning:**
- Converts string value to target type using `ast.literal_eval()`
- Validates final type matches expected type
- Handles type conversion for float and int

---

## Available Functions

### LinearFunction

Generates linear data following the equation: y = mx + q

#### Configuration

```python
{
    "feature": "linear_feature",
    "function_name": "LinearFunction",
    "parameters": {
        "m": 2.0,           # Slope
        "q": 10.0,          # Y-intercept
        "min_value": 0.0,   # Minimum x value
        "max_value": 100.0  # Maximum x value
    }
}
```

**Parameters:**
- `m` (float): Slope of the line
- `q` (float): Y-intercept
- `min_value` (float): Minimum input value
- `max_value` (float): Maximum input value

**Use Cases:**
- Linear relationships (age vs experience)
- Proportional data generation
- Simple trend simulation

### QuadraticFunction

Generates quadratic data following the equation: y = ax² + bx + c

#### Configuration

```python
{
    "feature": "quadratic_feature",
    "function_name": "QuadraticFunction",
    "parameters": {
        "a": 0.5,           # Quadratic coefficient
        "b": 2.0,           # Linear coefficient
        "c": 5.0,           # Constant term
        "min_value": 0.0,   # Minimum x value
        "max_value": 20.0   # Maximum x value
    }
}
```

**Parameters:**
- `a` (float): Quadratic coefficient
- `b` (float): Linear coefficient
- `c` (float): Constant term
- `min_value` (float): Minimum input value
- `max_value` (float): Maximum input value

**Use Cases:**
- Accelerating growth patterns
- Curved relationships (salary vs experience)
- Non-linear data generation

### SinusoidalFunction

Generates sinusoidal data following the equation: y = A·sin(ωx + φ)

#### Configuration

```python
{
    "feature": "sinusoidal_feature",
    "function_name": "SinusoidalFunction",
    "parameters": {
        "amplitude": 20.0,  # Amplitude (A)
        "frequency": 0.5,   # Frequency (ω)
        "phase": 0.0,       # Phase shift (φ)
        "min_value": 0.0,   # Minimum x value
        "max_value": 20.0   # Maximum x value
    }
}
```

**Parameters:**
- `amplitude` (float): Amplitude of the wave
- `frequency` (float): Frequency of the wave
- `phase` (float): Phase shift
- `min_value` (float): Minimum input value
- `max_value` (float): Maximum input value

**Use Cases:**
- Seasonal patterns
- Cyclical data (performance scores)
- Periodic behavior simulation

### ExponentialFunction

Generates exponential data following the equation: y = a·e^(bx)

#### Configuration

```python
{
    "feature": "exponential_feature",
    "function_name": "ExponentialFunction",
    "parameters": {
        "a": 1.0,           # Base coefficient
        "b": 0.1,           # Exponential coefficient
        "min_value": 0.0,   # Minimum x value
        "max_value": 10.0   # Maximum x value
    }
}
```

**Parameters:**
- `a` (float): Base coefficient
- `b` (float): Exponential coefficient
- `min_value` (float): Minimum input value
- `max_value` (float): Maximum input value

**Use Cases:**
- Growth patterns
- Compound interest calculations
- Exponential decay/growth

### LogarithmicFunction

Generates logarithmic data following the equation: y = a·log(bx) + c

#### Configuration

```python
{
    "feature": "logarithmic_feature",
    "function_name": "LogarithmicFunction",
    "parameters": {
        "a": 2.0,           # Coefficient
        "b": 1.0,           # Base multiplier
        "c": 0.0,           # Constant term
        "min_value": 1.0,   # Minimum x value (must be > 0)
        "max_value": 100.0  # Maximum x value
    }
}
```

**Parameters:**
- `a` (float): Coefficient
- `b` (float): Base multiplier
- `c` (float): Constant term
- `min_value` (float): Minimum input value (must be > 0)
- `max_value` (float): Maximum input value

**Use Cases:**
- Diminishing returns
- Log-scale data
- Compressing large ranges

### NormalDistributionSample

Generates data from normal distribution.

#### Configuration

```python
{
    "feature": "normal_feature",
    "function_name": "NormalDistributionSample",
    "parameters": {
        "mean": 50.0,        # Mean value
        "std": 10.0,         # Standard deviation
        "min_value": 0.0,   # Minimum value (clipping)
        "max_value": 100.0  # Maximum value (clipping)
    }
}
```

**Parameters:**
- `mean` (float): Mean of the distribution
- `std` (float): Standard deviation
- `min_value` (float): Minimum value for clipping
- `max_value` (float): Maximum value for clipping

**Use Cases:**
- Random variation around mean
- Natural phenomena simulation
- Noise generation

---

## Complete Usage Examples

### Basic Function Constructor Usage

```python
from sdg_core_lib.post_process.functions.generation.implementation.LinearFunction import LinearFunction
from sdg_core_lib.post_process.functions.generation.implementation.QuadraticFunction import QuadraticFunction
from sdg_core_lib.post_process.functions.Parameter import Parameter
import numpy as np

# Create parameters for LinearFunction
linear_params = [
    Parameter("m", "1.0", "float"),
    Parameter("q", "25.0", "float"),
    Parameter("min_value", "0.0", "float"),
    Parameter("max_value", "50.0", "float")
]

# Create LinearFunction instance
linear_func = LinearFunction(linear_params)

# Generate data using the function directly
age_data, age_indexes, age_success = linear_func.apply(n_rows=1000, data=np.array([]))

print(f"Generated age data shape: {age_data.shape}")
print(f"Success: {age_success}")

# Create parameters for QuadraticFunction
quadratic_params = [
    Parameter("a", "100.0", "float"),
    Parameter("b", "1000.0", "float"),
    Parameter("c", "30000.0", "float"),
    Parameter("min_value", "0.0", "float"),
    Parameter("max_value", "50.0", "float")
]

# Create QuadraticFunction instance
quadratic_func = QuadraticFunction(quadratic_params)

# Generate salary data
salary_data, salary_indexes, salary_success = quadratic_func.apply(n_rows=1000, data=np.array([]))

print(f"Generated salary data shape: {salary_data.shape}")
print(f"Success: {salary_success}")

# Combine data into a dataset structure
generated_data = {
    "age": age_data.flatten(),
    "salary": salary_data.flatten()
}
```

### Data Modification Example

```python
from sdg_core_lib.post_process.functions.modification.implementation.WhiteNoiseAdder import WhiteNoiseAdder
from sdg_core_lib.post_process.functions.Parameter import Parameter
import numpy as np

# Load or create existing data
existing_data = np.random.normal(50, 10, (1000, 1))  # Simulated existing measurements

# Create parameters for WhiteNoiseAdder
noise_params = [
    Parameter("mean", "0.0", "float"),
    Parameter("standard_deviation", "0.1", "float")
]

# Create WhiteNoiseAdder instance
noise_adder = WhiteNoiseAdder(noise_params)

# Apply noise to existing data
noisy_data, affected_indexes, success = noise_adder.apply(n_rows=1000, data=existing_data)

print(f"Original data shape: {existing_data.shape}")
print(f"Noisy data shape: {noisy_data.shape}")
print(f"Success: {success}")
print(f"Affected {len(affected_indexes)} rows")
```

### Complex Multi-Function Pipeline

```python
from sdg_core_lib.post_process.functions.generation.implementation.LinearFunction import LinearFunction
from sdg_core_lib.post_process.functions.generation.implementation.SinusoidalFunction import SinusoidalFunction
from sdg_core_lib.post_process.functions.modification.implementation.WhiteNoiseAdder import WhiteNoiseAdder
from sdg_core_lib.post_process.functions.Parameter import Parameter
import numpy as np

# Step 1: Generate base data with LinearFunction
base_params = [
    Parameter("m", "2.0", "float"),
    Parameter("q", "10.0", "float"),
    Parameter("min_value", "0.0", "float"),
    Parameter("max_value", "100.0", "float")
]
linear_func = LinearFunction(base_params)
base_data, _, _ = linear_func.apply(n_rows=365, data=np.array([]))

# Step 2: Add noise to base data
noise_params = [
    Parameter("mean", "0.0", "float"),
    Parameter("standard_deviation", "5.0", "float")
]
noise_adder = WhiteNoiseAdder(noise_params)
noisy_base_data, _, _ = noise_adder.apply(n_rows=365, data=base_data)

# Step 3: Generate seasonal pattern with SinusoidalFunction
seasonal_params = [
    Parameter("amplitude", "20.0", "float"),
    Parameter("frequency", "0.1", "float"),
    Parameter("phase", "0.0", "float"),
    Parameter("min_value", "0.0", "float"),
    Parameter("max_value", "365.0", "float")
]
sinusoidal_func = SinusoidalFunction(seasonal_params)
seasonal_data, _, _ = sinusoidal_func.apply(n_rows=365, data=np.array([]))

# Combine all data
complex_data = {
    "base_value": noisy_base_data.flatten(),
    "seasonal_pattern": seasonal_data.flatten()
}

print(f"Generated complex dataset with {len(complex_data)} features")
for feature, data in complex_data.items():
    print(f"  {feature}: shape {data.shape}, range [{data.min():.2f}, {data.max():.2f}]")
```

### Function Chaining and Composition

```python
from sdg_core_lib.post_process.functions.generation.implementation.NormalDistributionSample import NormalDistributionSample
from sdg_core_lib.post_process.functions.modification.implementation.WhiteNoiseAdder import WhiteNoiseAdder
from sdg_core_lib.post_process.functions.Parameter import Parameter
import numpy as np

# Create a base normal distribution
normal_params = [
    Parameter("mean", "50.0", "float"),
    Parameter("std", "10.0", "float"),
    Parameter("min_value", "0.0", "float"),
    Parameter("max_value", "100.0", "float")
]
normal_func = NormalDistributionSample(normal_params)

# Generate initial data
initial_data, _, _ = normal_func.apply(n_rows=1000, data=np.array([]))

# Apply multiple transformations in sequence
transformations = [
    (WhiteNoiseAdder, [Parameter("mean", "0.0", "float"), Parameter("standard_deviation", "2.0", "float")]),
    (WhiteNoiseAdder, [Parameter("mean", "1.0", "float"), Parameter("standard_deviation", "1.0", "float")])
]

current_data = initial_data
for i, (func_class, params) in enumerate(transformations):
    func = func_class(params)
    current_data, affected_indexes, success = func.apply(n_rows=1000, data=current_data)
    print(f"Transformation {i+1}: Success={success}, Affected={len(affected_indexes)} rows")

print(f"Final data shape: {current_data.shape}")
print(f"Data range: [{current_data.min():.2f}, {current_data.max():.2f}]")
```

---

## Best Practices

1. **Function Priority**: Use appropriate priorities:
   - Priority.MAX (5): Generative functions
   - Priority.LOW (2-3): Modification functions
   - Higher priority functions execute first

2. **Parameter Validation**: Always validate parameters in `_check_parameters()` method

3. **Type Safety**: Use the Parameter class for automatic type conversion and validation

4. **Data Compatibility**: Ensure functions work with specified `allowed_data` types

5. **Function Sequencing**: When generating from scratch, ensure first function is generative

---

## Related APIs

For complete API documentation, see:

- **[Job API Reference](./job-API-reference.md)** - Core job management and orchestration
- **[Dataset API Reference](./dataset-API-reference.md)** - Data input/output and skeleton operations
- **[Model API Reference](./model-API-reference.md)** - Machine learning model interfaces
- **[Processor API Reference](./processor-API-reference.md)** - Data preprocessing and postprocessing
- **[Evaluation API Reference](./evaluation-API-reference.md)** - Quality evaluation and metrics
