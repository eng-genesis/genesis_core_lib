# Developer Documentation

## Overview

This comprehensive developer documentation covers how to extend the GENESIS Core Lib library by creating custom components. Whether you want to add new data types, implement custom models, or create specialized processors, this guide will show you how.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Core Components](#core-components)
3. [Extension Guides](#extension-guides)

## Project Structure

GENESIS Core Lib follows a modular architecture that makes it easy to extend and customize. The main components are organized as follows:

```
src/sdg_core_lib/
├── dataset/                            # Data structures and types
│   ├── datasets.py                     # Base Dataset, Table, TimeSeries classes
│   ├── columns.py                      # Column types (Numeric, Categorical, etc.)
│   └── validation_schema.py            # Data validation schemas
├── data_generator/                     # Machine learning models
│   └── models/                         # Model-related files
│       └─── /.../implementation        # Model implementations (VAEs, GANs, etc.)
├── preprocess/                         # Data preprocessing
│   ├── base_processor.py               # Base processor class
    │   ├── table_processor.py          # Table-specific processor
    │   └── strategies/                 # Preprocessing strategies
├── post_process/                       # Data post-processing and functions
│   ├── function_factory.py             # Function factory pattern
│   └── functions/                      # Custom function implementations
├── evaluate/                           # Quality evaluation
│   ├── base_evaluator.py               # Base evaluator class
│   ├── metrics.py                      # Evaluation metrics
│   ├── tables.py                       # Table-specific evaluators
│   └── time_series.py                  # Time series evaluators
└── job.py                              # Main job orchestrator
```

### Key Design Patterns

The library uses several design patterns to ensure extensibility:

- **Template Method Pattern**: All major components inherit from abstract base classes, ensuring consistent interfaces
- **Factory Pattern**: Models and functions use factory patterns for dynamic instantiation
- **Strategy Pattern**: Processors use strategy patterns for different preprocessing approaches
- **Registry Pattern**: Components can be registered and discovered dynamically

## Core Components

### 1. Datasets (`dataset/`)

The dataset module provides the fundamental data structures:

- **`Dataset`**: Abstract base class for all data types
  - **Specialized Datasets**: Dataset that extends the common dataset
- **`Column`**: Base column class with metadata 
  - **Specialized Columns**: Encapsulate essence of Data Types

### 2. Models (`data_generator/models/`)

The model system supports various ML approaches:

- **`UnspecializedModel`**: Abstract base class for all models
- **ModelInfo**: Model metadata and configuration
- **TrainingInfo**: Training configuration and tracking
- **`implementation` folder**: contain ready-to-use architectures

### 3. Processors (`preprocess/`)

Data preprocessing components:

- **`Processor`**: Abstract base processor class
- **`TableProcessor`**: Table-specific preprocessing
- **`BasePreprocessingStrategy`**: Strategy pattern for different approaches
- **`Step`**: Individual preprocessing steps

### 4. Functions (`post_process/functions/`)

Post-processing and transformation functions:

- **`UnspecializedFunction`**: Base function class
- **`Parameter`**: Function parameter handling
- **`implementation` folder**: contain ready-to-use functions

### 5. Evaluators (`evaluate/`)

Quality assessment components:

- **`BaseEvaluator`**: Abstract evaluator base class
- **`MetricReport`**: Evaluation result reporting
- **Specialized Evaluators**: Table and time series specific evaluators

## Extension Guides

The following guides provide detailed instructions for creating custom components:
⚠️ **IMPORTANT WARNING:** Most content in the following docs is AI-generated and may contain errors. Use this as a basis for understanding the base mechanisms of the library. You can also try and import the code, but always verify the code and logic before using it. 
The reason why this code is not included in the library is that it is AI-generated and needs to be verified. You probably will see the cited examples in following releases. ⚠️

- **[Custom Datasets](./developers/custom-datasets.md)** - Creating new data types and structures
- **[Custom Data Types](./developers/custom-data-types.md)** - Implementing new column types
- **[Custom Processors](./developers/custom-processors.md)** - Building data preprocessing components
- **[Custom Functions](./developers/custom-functions.md)** - Creating transformation and utility functions
- **[Custom Models](./developers/custom-models.md)** - Implementing new machine learning models
- **[Custom Evaluators](./developers/custom-evaluators.md)** - Building quality assessment tools

## Integration with Job System

After creating custom components, you need to register them in the `mappings.py` file to make them available to the Job orchestrator. The mappings system uses a hierarchical approach to associate dataset types with their corresponding evaluators, processors, and models.

### Understanding the Mapping Structure

The `mappings.py` file contains three main mapping classes:

1. **`DatasetMapping`** (Abstract base class)
   - Defines the interface for all dataset type mappings
   - Contains default fallback mappings

2. **`DatasetTypeMapping`** (Concrete implementations like `TableMapping`, `TimeSeriesMapping`)
   - Maps specific dataset types to their components
   - Associates: `dataset` → `evaluator` → `processor`

3. **`ModelStrategyMapping`**
   - Maps model classes to their preprocessing strategies
   - Ensures compatible model-strategy pairings

### Adding New Dataset Types

To integrate a new dataset type (e.g., `TextDataset`):

1. **Create a new mapping class** that inherits from `DatasetMapping`:

```python
class TextMapping(DatasetMapping):
    mapping = {
        "dataset": TextDataset,
        "evaluator": TextComparisonEvaluator,
        "processor": TextProcessor,
    }
```

2. **Import your custom components** at the top of `mappings.py`:

```python
from sdg_core_lib.dataset.datasets import TextDataset
from sdg_core_lib.evaluate.text import TextComparisonEvaluator
from sdg_core_lib.preprocess.text_processor import TextProcessor
```

### Adding New Models and Strategies

To integrate a new model with its preprocessing strategy:

1. **Add to ModelStrategyMapping**:

```python
class ModelStrategyMapping:
    mapping = {
        # ... existing mappings ...
        TextGenerationModel: TextGenerationModelPreprocessingStrategy,
    }
```

2. **Import the components**:

```python
from sdg_core_lib.data_generator.models.text.implementation.TextGenerationModel import TextGenerationModel
from sdg_core_lib.preprocess.strategies.textgeneration_strategy import TextGenerationModelPreprocessingStrategy
```

### Registration Process

The Job system automatically discovers and uses these mappings:

1. **Dataset Type Resolution**: When a Job is created with a dataset type, it looks up the corresponding mapping class
2. **Component Instantiation**: The Job uses the mapping to instantiate the correct evaluator and processor
3. **Model-Strategy Matching**: When a model is specified, the `ModelStrategyMapping` provides the compatible preprocessing strategy

### Example Integration

Here's a complete example for adding a custom `TextDataset`:

```python
# In mappings.py
from sdg_core_lib.dataset.datasets import TextDataset
from sdg_core_lib.evaluate.text import TextComparisonEvaluator
from sdg_core_lib.preprocess.text_processor import TextProcessor
from sdg_core_lib.data_generator.models.text.implementation.TextGenerationModel import TextGenerationModel
from sdg_core_lib.preprocess.strategies.textgeneration_strategy import TextGenerationModelPreprocessingStrategy

class TextMapping(DatasetMapping):
    mapping = {
        "dataset": TextDataset,
        "evaluator": TextComparisonEvaluator,
        "processor": TextProcessor,
    }

class ModelStrategyMapping:
    mapping = {
        # ... existing mappings ...
        TextGenerationModel: TextGenerationModelPreprocessingStrategy,
    }
```

### Best Practices for Mapping Updates

1. **Consistent Naming**: Use descriptive names that clearly indicate the dataset type
2. **Import Organization**: Group imports by module type (dataset, evaluate, preprocess, models)
3. **Documentation**: Add comments explaining the purpose of new mappings
4. **Testing**: Verify that the Job system can correctly instantiate your components
5. **Backward Compatibility**: Ensure existing mappings remain functional when adding new ones


This approach allows external modules to register their mappings without modifying the core `mappings.py` file.

## Development Best Practices

When extending the library, follow these guidelines:

1. **Inherit from Base Classes**: Always extend the appropriate abstract base class
2. **Implement Required Methods**: Ensure all abstract methods are properly implemented
3. **Follow Naming Conventions**: Use consistent naming patterns across the codebase
4. **Add Type Hints**: Include proper type annotations for better code documentation
5. **Write Tests**: Create comprehensive tests for new components
6. **Document Your Code**: Add docstrings and comments explaining your implementation
7. **Handle Errors Gracefully**: Include proper error handling and validation
8. **Consider Performance**: Optimize for memory usage and computational efficiency

## Contributing

When contributing to the library:

1. Fork the repository and create a feature branch
2. Follow the existing code style and patterns
3. Add tests for your new functionality
4. Update documentation as needed
5. Submit a pull request with a clear description of changes

For detailed contribution guidelines, see the project's README.md file.