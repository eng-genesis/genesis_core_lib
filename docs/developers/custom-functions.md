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
    parameters: list[Parameter] = None 
    description: str = None
    priority: Priority = Priority.NONE
    is_generative: bool = None
    allowed_data: list[AllowedData] = None

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

**Key Requirements:**
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

# WORK IN PROGRESS