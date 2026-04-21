# User Documentation

## Overview

This comprehensive user documentation covers all aspects of using GENESIS Core Lib, from basic concepts to advanced techniques. Whether you're a beginner or an experienced data scientist, this guide will help you master synthetic data generation.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Data Types](#data-types)
3. [Model Types](#model-types)
4. [Configuration](#configuration)
5. [API Reference](#api-reference)

## Core Concepts

### The Job System

The `Job` class is the central orchestrator in GENESIS Core Lib. It manages the entire synthetic data generation pipeline from data loading to model training and inference.

#### Job Lifecycle

<img width="4273" height="1255" alt="image" src="https://github.com/user-attachments/assets/e2297190-a83d-4cc7-b831-141b23018dc8" />

- **Red**: training-specific flow
- **Blue**: infer-only flow
- **Purple**: training and infer shared flow
- **Green**: function-generation flow
- **Dashed**: optional steps


#### Key Components

1. **Dataset Configuration**: Defines the input data and its type
2. **Model Configuration**: Specifies the ML model to use
3. **Functions**: Mathematical functions for data generation/modification
4. **Output Settings**: Where to save models and results

### Data Processing Pipeline

GENESIS Core Lib follows a structured approach to data processing:

#### 1. Data Input
- Load real data or define generation functions
- Validate data format and structure
- Extract metadata and schema information

#### 2. Preprocessing
- Normalize and scale features
- Encode categorical variables
- Handle missing values
- Prepare data for model training

#### 3. Model Training
- Initialize the specified ML model
- Train on preprocessed data
- Optimize hyperparameters
- Save trained model

#### 4. Data Generation
- Use trained model for inference
- Generate specified number of rows
- Apply post-processing functions

#### 5. Postprocessing
- Reverse preprocessing transformations
- Apply custom functions
- Format output data

#### 6. Evaluation
- Compare synthetic vs real data
- Calculate quality metrics
- Generate evaluation report

## Data Types

### Tabular Data

Tabular data is structured data organized in rows and columns, similar to a spreadsheet or database table.

#### Characteristics
- Fixed schema with defined columns
- Mixed data types (numeric, categorical, text)
- Independent observations
- Suitable for most business datasets

#### Configuration Example

```python
dataset_config = {
    "dataset_type": "table",
    "data": [
        {
            "column_data": [13.71, 13.4, 13.27, 13.17, 14.13],
            "column_name": "alcohol",
            "column_type": "continuous",
            "column_datatype": "float64"
        },
        {
            "column_data": [5.65, 3.91, 4.28, 2.59, 4.1],
            "column_name": "malic_acid",
            "column_type": "continuous",
            "column_datatype": "float64"
        }
    ]
}
```

#### Supported Feature Types
- **Numeric**: Integers and floats
- **Categorical**: Discrete categories

#### Best Practices
- Ensure consistent data types across columns
- Handle missing values appropriately
- Limit categorical cardinality (<100 categories recommended)
- Normalize numeric features when possible

### Time Series Data

Time series data consists of observations collected sequentially over time.

#### Characteristics
- Temporal ordering is significant
- May have trends, seasonality, and patterns
- Can be univariate or multivariate
- Requires special handling for temporal dependencies

#### Configuration Example

```python
dataset_config = {
    "dataset_type": "time_series",
    "data": [
        {
            "column_name": "experiment_id",
            "column_type": "group_index",
            "column_data": [1, 1, 1, 1, 2, 2, 2, 2],
            "column_datatype": "int"
        },
        {
            "column_name": "time",
            "column_type": "primary_key",
            "column_data": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "column_datatype": "int"
        },
        {
            "column_name": "value1",
            "column_type": "continuous",
            "column_data": [1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3],
            "column_datatype": "float"
        },
        {
            "column_name": "category",
            "column_type": "categorical",
            "column_data": ["a", "b", "c", "a", "b", "c", "a", "b", "c"],
            "column_datatype": "str"
        }
    ]
}
```

#### Required Columns
- **group_index**: Identifies different experiments/time series groups
- **continuous**: Numeric measurement values
- **categorical**: Discrete category values

#### Best Practices
- Ensure consistent experiment lengths
- Ensure at least **12** samples for experiment
- Provide proper group_index for experiment identification
- Use primary_key for temporal ordering
- Handle missing time points appropriately
- Ensure regular time intervals
- Use appropriate frequency settings


## Model Types

### VAEs (Variational Autoencoders)

VAEs learn a compressed latent representation of data and can generate new samples by sampling from the latent space.

#### When to Use VAEs
- When smooth interpolation is desired
- For structured latent space
- When training stability is a concern
- For datasets with clear patterns

#### Available VAE Models

##### TabularVAE
```python
model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE.TabularVAE",
    "model_name": "tabular_vae_model"
}
```

- Stable training process
- Interpretable latent space
- Good for feature analysis
- Handles missing values well

##### AutoTabularVAE
```python
model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.AutoTabularVAE.AutoTabularVAE",
    "model_name": "auto_tabular_vae_model"
}
```
- Longer Training Times (10x)
- Same as Tabular VAE
- Grid Searches Hyperparameters

##### TimeSeriesVAE
```python
model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TimeSeriesVAE.TimeSeriesVAE",
    "model_name": "time_series_vae_model"
}
```

- Captures temporal dependencies
- Handles variable-length sequences
- Preserves temporal patterns

#### VAE Best Practices
- Use appropriate latent dimensionality
- Monitor reconstruction loss
- Consider the beta-VAE variant for disentangled representations
- Validate latent space interpretability

### CTGAN (Conditional Tabular GAN)

CTGAN is a specialized GAN for tabular data that handles mixed data types effectively.

#### When to Use CTGAN
- Complex data distributions
- High-dimensional data
- When generating highly realistic data is critical
- Tabular data with complex feature interactions

#### Available CTGAN Models

##### CTGAN
```python
model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.GANs.implementation.CTGAN.CTGAN",
    "model_name": "ctgan_model"
}
```

- Handles mixed data types
- Preserves feature correlations
- Good for medium-sized datasets (1K-100K rows)

#### CTGAN Best Practices
- Use sufficient training data (1000+ rows recommended)
- Monitor training stability
- Adjust learning rates if training fails
- Consider data preprocessing for better convergence

### Model Selection Guide

| Dataset Size | Data Complexity | Recommended Model | Reason |
|-------------|-----------------|-------------------|---------|
| < 1,000 rows | Low | TabularVAE | More stable with small data |
| 1,000-10,000 rows | Medium | TabularVAE or CTGAN | Both work well |
| > 10,000 rows | High | CTGAN | Can capture complex distributions |
| Any size | Time series | TimeSeriesVAE | Specialized for temporal data |
| Any size | Very high dimensional | TabularVAE | More stable training |


## Build a Job

See [This](examples/user_example.py) as an example on how to use the library classes for generating data

1. **Dataset Creation**: `Table.from_json()` creates a structured dataset from column definitions
2. **Processor Setup**: `TableProcessor` with `VAEStrategy` handles data preprocessing
3. **Metadata Extraction**: `preprocessed_data.to_skeleton()` extracts metadata for model
4. **Model Instantiation and Training**: `TabularVAE()` creates the model with specific hyperparameters
5. **Inference**: `model.infer()` generates new synthetic data
6. **Post-processing**: Data is transformed back to original format with `postprocess()`
7. **Evaluation**: `TabularComparisonEvaluator` assesses data quality
8**Results**: Final data is available in JSON format for downstream use

This approach gives you full control over each component while maintaining the same functionality as the configuration-based approach.


## API Reference

For detailed API documentation, see the following specialized reference files:

- **[Job API Reference](./user-api-reference/job-API-reference.md)** - Core job management and orchestration
- **[Dataset API Reference](./user-api-reference/dataset-API-reference.md)** - Data input/output and skeleton operations
- **[Model API Reference](./user-api-reference/model-API-reference.md)** - Machine learning model interfaces
- **[Functions API Reference](./user-api-reference/functions-API-reference.md)** - Mathematical functions for data generation
- **[Processor API Reference](./user-api-reference/processor-API-reference.md)** - Data preprocessing and postprocessing
- **[Evaluation API Reference](./user-api-reference/evaluation-API-reference.md)** - Quality evaluation and metrics 

