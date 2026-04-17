# Functions API Reference

This document provides detailed API documentation for mathematical functions used in data generation and modification.

## Table of Contents

1. [Available Functions](#available-functions)
2. [Function Configuration](#function-configuration)
3. [Usage Examples](#usage-examples)
4. [Related APIs](#related-apis)

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

## Function Configuration

### Standard Structure

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

## Usage Examples

### Single Function Generation

```python
from sdg_core_lib import Job

functions = [
    {
        "feature": "x",
        "function_name": "LinearFunction",
        "parameters": {
            "m": 1.0,
            "q": 0.0,
            "min_value": 0.0,
            "max_value": 10.0
        }
    }
]

job = Job(n_rows=100, functions=functions)
data = job.generate_from_functions()
```

### Multiple Function Generation

```python
functions = [
    {
        "feature": "experience_years",
        "function_name": "LinearFunction",
        "parameters": {
            "m": 1.0,
            "q": 0.0,
            "min_value": 0.0,
            "max_value": 20.0
        }
    },
    {
        "feature": "salary",
        "function_name": "QuadraticFunction",
        "parameters": {
            "a": 500.0,
            "b": 2000.0,
            "c": 30000.0,
            "min_value": 0.0,
            "max_value": 20.0
        }
    },
    {
        "feature": "performance_score",
        "function_name": "SinusoidalFunction",
        "parameters": {
            "amplitude": 20.0,
            "frequency": 0.5,
            "phase": 0.0,
            "min_value": 0.0,
            "max_value": 20.0
        }
    }
]

job = Job(n_rows=100, functions=functions)
data = job.generate_from_functions()
```

### Data Modification

```python
# Start with existing dataset
existing_dataset = load_existing_data()

# Add modification function
functions = [
    {
        "feature": "noisy_feature",
        "function_name": "WhiteNoiseAdder",
        "parameters": {
            "noise_level": 0.1
        }
    }
]

job = Job(n_rows=len(existing_dataset), functions=functions)
modified_data = job.generate_from_functions(dataset=existing_dataset)
```

### Custom Functions

You can extend the Functions API by creating custom function classes:

```python
from sdg_core_lib.post_process.functions.base import BaseFunction

class CustomFunction(BaseFunction):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        # Initialize custom parameters
    
    def generate(self, n_rows):
        # Implement custom generation logic
        return generated_values
    
    def get_parameters_schema(self):
        # Return parameter schema for validation
        return {
            "param1": {"type": "float", "required": True},
            "param2": {"type": "int", "default": 10}
        }
```

---

## Related APIs

For complete API documentation, see:

- **[Job API Reference](./job-API-reference.md)** - Core job management and orchestration
- **[Dataset API Reference](./dataset-API-reference.md)** - Data input/output and skeleton operations
- **[Model API Reference](./model-API-reference.md)** - Machine learning model interfaces
- **[Processor API Reference](./processor-API-reference.md)** - Data preprocessing and postprocessing
- **[Evaluation API Reference](./evaluation-API-reference.md)** - Quality evaluation and metrics
