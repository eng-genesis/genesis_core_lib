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
  - **`Table`**: Tabular data implementation with columns
  - **`TimeSeries`**: Time series data extending Table
- **`Column`**: Base column class with metadata 
  - **`Numeric`**: Numeric column implementation
  - **`Categorical`**: Categorical column implementation

### 2. Models (`data_generator/models/`)

The model system supports various ML approaches:

- **`UnspecializedModel`**: Abstract base class for all models
- **VAEs**: Variational Autoencoders (TabularVAE, TimeSeriesVAE, AutoTabularVAE)
- **GANs**: Generative Adversarial Networks (CTGAN)
- **ModelInfo**: Model metadata and configuration
- **TrainingInfo**: Training configuration and tracking

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
- **Function Factory**: Dynamic function instantiation
- **Various Function Types**: Distribution evaluators, filters, transformations

### 5. Evaluators (`evaluate/`)

Quality assessment components:

- **`BaseEvaluator`**: Abstract evaluator base class
- **`MetricReport`**: Evaluation result reporting
- **Specialized Evaluators**: Table and time series specific evaluators

## Extension Guides

The following guides provide detailed instructions for creating custom components:

- **[Custom Datasets](./developers/custom-datasets.md)** - Creating new data types and structures
- **[Custom Data Types](./developers/custom-data-types.md)** - Implementing new column types
- **[Custom Processors](./developers/custom-processors.md)** - Building data preprocessing components
- **[Custom Functions](./developers/custom-functions.md)** - Creating transformation and utility functions
- **[Custom Models](./developers/custom-models.md)** - Implementing new machine learning models
- **[Custom Evaluators](./developers/custom-evaluators.md)** - Building quality assessment tools

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