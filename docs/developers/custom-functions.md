# Custom Functions

## Overview

This guide explains how to create custom functions for the GENESIS Core Lib. Custom functions allow you to implement specialized data transformations, mathematical operations, and post-processing logic beyond the built-in functions.

## Base Function Classes

### UnspecializedFunction Base Class

All custom functions must inherit from the `UnspecializedFunction` base class located in `src/sdg_core_lib/post_process/functions/UnspecializedFunction.py`:

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Optional
from sdg_core_lib.post_process.functions.Parameter import Parameter

class UnspecializedFunction(ABC):
    parameters: list[Parameter] = None # List of Parameters for a single function
    description: str = None # Description of the function
    priority: Priority = Priority.NONE # Priority System of function from higher to lower (NONE < LOW < MEDIUM < HIGH)
    is_generative: bool = None # Generative flag for generating from functions
    allowed_data: list[AllowedData] = None # List of data types the function is able to process

    def __init__(self, parameters: list[Parameter]):
        self.parameters = parameters
        self._check_parameters()

    @classmethod
    def from_json(cls, json_params):
        """
        This method is valid for each Function and should not be changed
        :param json_params: list of parameters in JSON format
        :return: instance of the function
        """
        return cls([Parameter.from_json(param) for param in json_params])

    @abstractmethod
    def _check_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def apply(
        self, n_rows: int, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Function application interface: class implementation defines how to apply the function to the data
        :param n_rows: how many rows have to be affected (optional for some functions)
        :param data: data on which function have to be applied (optional for some functions)
        :return:
            - new data: computational results
            - indexes: indexes of the rows that have been affected
            - success_flag: boolean flag indicating if the function has been applied correctly.
                For statistical tests, it is False if the test fails
        """
        raise NotImplementedError

    @classmethod
    def self_describe(cls):
        """
        This method is valid for each Function and should not be changed
        :return: dictionary containing function metadata
        """
        return {
            "function": {
                "name": f"{cls.__qualname__}",
                "description": cls.description,
                "function_reference": f"{cls.__module__}.{cls.__qualname__}",
                "priority": cls.priority.value,
                "is_generative": cls.is_generative,
            },
            "parameters": [param.to_json() for param in cls.parameters],
            "datatypes": [ad.to_json() for ad in cls.allowed_data],
        }
```

**Key Concepts:**
- Never change the `from_json()` method
- Never change the `self_describe()` method
- All functions must implement their `apply()` for data transformation
- All functions must implement their `check_parameters()` for validating parameters input
- All function must keep a Class dictionary for configuration, including:
  - parameters: a List of `Parameter` objects
  - description: a string description of the function
  - priority: a `Priority` enum value
  - is_generative: a boolean flag indicating if the function is generative
  - allowed_data: a List of `AllowedData` objects

## AllowedData Class

The `AllowedData` class (located in `src/sdg_core_lib/commons.py`) defines what data types a function can operate on.

### Constructor
```python
AllowedData(dtype: DataType, is_categorical: bool)
```

### Properties
- `dtype`: The data type (from DataType enum)
- `is_categorical`: Whether the data is categorical

### Methods
- `to_json()`: Serializes to JSON format

### DataType Enum
Defines supported data types:
- `int32`, `int64` - Integer types
- `float32`, `float64` - Float types  
- `string` - Text data
- `bool` - Boolean data

### Example Usage
```python
from sdg_core_lib.commons import AllowedData, DataType

allowed_data = [
    AllowedData(DataType.float64, is_categorical=False),
    AllowedData(DataType.string, is_categorical=True)
]
```

## Priority Class

The `Priority` enum (located in `src/sdg_core_lib/post_process/functions/UnspecializedFunction.py`) defines execution priority levels for functions.

The higher the priority, the sooner the function is executed.
### Priority Levels
- `MINIMAL = 1` - Lowest priority
- `LOW = 2`
- `MEDIUM = 3`
- `HIGH = 4` 
- `MAX = 5` - Highest priority -> Use this for generative functions

### Example Usage
```python
from sdg_core_lib.post_process.functions.UnspecializedFunction import Priority

class MyFunction(UnspecializedFunction):
    priority = Priority.HIGH  # This function should run with high priority
```

### Purpose
These classes enable the framework to automatically determine:
- Which functions can be applied to specific data types (via `allowed_data`)
- The order in which functions should be executed (via `priority`)


### Parameter Class

The `Parameter` class (in `src/sdg_core_lib/post_process/functions/Parameter.py`) defines function parameters.
This class can be consulted for understaning how to build a parameter.


## Folder Structure

Custom functions should be organized in the following structure:

```
src/sdg_core_lib/post_process/functions/
├── UnspecializedFunction.py            # Base class
├── Parameter.py                        # Parameter class
├── function_registry.py                # Function registry
└── category/                           # Function category (example: filter)
    ├── CategoryCommonClass.py          # If exists, a common class for implementation (must inherit from UnspecializedFunction)
    └── implementation/                 # Folder for function implementations 
        ├── filter_text.py              # Text processing functions
        ├── numeric_threshold.py        # Numeric/mathematical functions
        ...
```


**Important: ⚠️** The `implementation/` folder is where you should write your new functions. This separates custom implementations from the core framework.


## Example: Creating a Custom Function

The following example focus on creating a custom function from scratch following the library's structure.

### Step 1: Create the Function Class

```python
import numpy as np
from sdg_core_lib.commons import AllowedData, DataType
from sdg_core_lib.post_process.functions.UnspecializedFunction import (
    UnspecializedFunction,
    Priority,
)
from sdg_core_lib.post_process.functions.Parameter import Parameter
from sdg_core_lib.post_process.function_utils import check_min_max_boundary


class ExponentialFunction(UnspecializedFunction):
    # Define the parameters this function accepts
    parameters = [
        Parameter("base", "2.0", "float"),           # Base of the exponential
        Parameter("min_value", "0.0", "float"),      # Minimum x value
        Parameter("max_value", "1.0", "float"),      # Maximum x value
        Parameter("scale_factor", "1.0", "float"),    # Scale factor for output
    ]
    
    # Class-level configuration
    description = "Generates exponential data using y = base^x * scale_factor"
    priority = Priority.MAX                         # Highest priority for generative functions
    is_generative = True                            # This function generates new data
    allowed_data = [
        AllowedData(DataType.float32, False),       # Works with 32-bit floats
        AllowedData(DataType.float64, False),       # Also works with 64-bit floats
    ]

    def __init__(self, parameters: list[Parameter]):
        # Initialize instance variables
        self.base = None
        self.min_value = None
        self.max_value = None
        self.scale_factor = None
        super().__init__(parameters)

    def _check_parameters(self):
        """Validate and set parameter values"""
        # Get allowed parameter names
        allowed_parameters = [param.name for param in type(self).parameters]
        
        # Create mapping of provided parameters
        param_mapping = {
            param.name: param
            for param in self.parameters
            if param.name in allowed_parameters
        }
        
        # Set instance attributes from parameters
        for name, param in param_mapping.items():
            setattr(self, name, param.value)
        
        # Validate parameter constraints
        if self.base <= 0:
            raise ValueError("Base must be greater than 0")
        if self.base == 1:
            raise ValueError("Base cannot be 1 (would produce constant output)")
        check_min_max_boundary(self.min_value, self.max_value)
        if self.scale_factor < 0:
            raise ValueError("Scale factor cannot be negative")

    def apply(
        self, n_rows: int, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Generate exponential data points
        
        Args:
            n_rows: Number of data points to generate
            data: Input data (ignored for generative functions)
            
        Returns:
            tuple: (generated_data, empty_indices, success_flag)
        """
        # Generate x values evenly spaced in the specified range
        x_values = np.linspace(self.min_value, self.max_value, n_rows)
        
        # Apply exponential function: y = base^x * scale_factor
        y_values = (self.base ** x_values) * self.scale_factor
        
        # Return data in expected format
        return y_values.reshape(-1, 1), np.empty((n_rows, 1)), True
```

###  Key Implementation Details

#### Class Attributes
- **`parameters`**: List of `Parameter` objects defining function inputs
- **`description`**: Human-readable description of what the function does
- **`priority`**: Execution priority (use `Priority.MAX` for generative functions)
- **`is_generative`**: Set to `True` for functions that create new data
- **`allowed_data`**: List of compatible data types

#### Required Methods
- **`__init__`**: Initialize instance variables and call parent constructor
- **`_check_parameters`**: Validate parameter values and set instance attributes
- **`apply`**: Core function logic that transforms/generates data

#### Return Format
The `apply` method must return:
- `np.ndarray`: Generated/transformed data (reshaped to column vector)
- `np.ndarray`: Indices of affected rows (empty for generative functions)
- `bool`: Success flag (always `True` for successful operations)

### Step 3: Usage Examples

#### Creating the Function
```python
from sdg_core_lib.post_process.functions.Parameter import Parameter

# Create parameters
params = [
    Parameter("base", "2.718", "float"),      # Using e as base
    Parameter("min_value", "0.0", "float"),
    Parameter("max_value", "3.0", "float"),
    Parameter("scale_factor", "1.5", "float"),
]

# Create function instance
exp_function = ExponentialFunction(params)
```

#### Applying the Function
```python
import numpy as np

# Generate 100 exponential data points
result_data, indices, success = exp_function.apply(100, np.array([]))

if success:
    print(f"Generated {len(result_data)} data points")
    print(f"Data range: {result_data.min():.3f} to {result_data.max():.3f}")
```

#### JSON Configuration
```python
# Create from JSON configuration
json_params = [
    {"name": "base", "value": "2.0", "parameter_type": "float"},
    {"name": "min_value", "value": "0.0", "parameter_type": "float"},
    {"name": "max_value", "value": "2.0", "parameter_type": "float"},
    {"name": "scale_factor", "value": "1.0", "parameter_type": "float"}
]

exp_function = ExponentialFunction.from_json(json_params)
```

**Note**: Each parameter must include the `parameter_type` field matching the type specified in the function's `parameters` list.

### Step 4: Integration

To make your function available to the framework:

1. **File Location**: Place in appropriate subdirectory of `post_process/functions/`
2. **Import**: Add to the module's `__init__.py` if needed
3. **Registration**: The framework will automatically discover functions that inherit from `UnspecializedFunction`

### Best Practices

- **Parameter Validation**: Always validate parameter constraints in `_check_parameters()`
- **Error Handling**: Raise descriptive `ValueError` exceptions for invalid inputs
- **Data Types**: Specify all compatible data types in `allowed_data`
- **Documentation**: Include clear docstrings for the `apply` method
- **Testing**: Test with various parameter combinations and edge cases
