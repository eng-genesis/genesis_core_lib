# Quick Start Guide

## Overview

This guide will get you up and running with GENESIS Core Lib in minutes. You'll learn how to install the library, generate your first synthetic dataset, and understand the basic concepts.

## Installation

### Prerequisites
- Python 3.12 or higher
- pip or uv (https://pypi.org/project/uv/) package manager.

### Option 1: Standard Installation (Recommended)
```bash
pip install sdg-core-lib
```

### Option 2: Using UV (Faster)
```bash
uv add sdg-core-lib
```


### Verify Installation
```python
from sdg_core_lib import Job
print("GENESIS Core Lib installed successfully!")
```

## Your First Synthetic Dataset

### Basic Example: Tabular Data Generation

```python
from sdg_core_lib import Job
import json

# Load configuration from JSON file 
with open('your_config.json', 'r') as f:
    config = json.load(f)

# Create and run the job
job = Job(
    n_rows=config["n_rows"],
    model_info=config["model"],
    dataset=config["dataset"],
    save_filepath=config.get("save_filepath", "./models")
)

# Train the model and generate synthetic data
results, metrics, model, schema = job.train()

# View your results
print(f"Generated {len(results)} synthetic rows")
print(f"First few rows: {results[:3]}")
print(f"Quality metrics: {metrics}")
```

### Expected Output
```
Generated 1000 synthetic rows
First few rows: [{'column_data': [...], ...}, ...]
Quality metrics: {'statistical_metrics': {...}, 'adherence_metrics': {...}}
```

## Function-Based Generation

### Generate Data Using Mathematical Functions

```python
from sdg_core_lib import Job

# Define mathematical functions for data generation
functions = [
    {
        "feature": "linear_data",
        "function_name": "LinearFunction",
        "parameters": {
            "m": 2.0,      # slope
            "q": 1.0,      # y-intercept
            "min_value": 0.0,
            "max_value": 100.0
        }
    },
    {
        "feature": "quadratic_data",
        "function_name": "QuadraticFunction",
        "parameters": {
            "a": 1.0,      # quadratic coefficient
            "b": -2.0,     # linear coefficient
            "c": 1.0,      # constant term
            "min_value": 0.0,
            "max_value": 10.0
        }
    },
    {
        "feature": "sinusoidal_data",
        "function_name": "SinusoidalFunction",
        "parameters": {
            "amplitude": 1.0,
            "frequency": 0.1,
            "phase": 0.0,
            "min_value": 0.0,
            "max_value": 100.0
        }
    }
]

# Create job with function-based generation
job = Job(n_rows=50, functions=functions)

# Generate synthetic data
synthetic_data = job.generate_from_functions()

# Display results
print("Generated data:")
for i, row in enumerate(synthetic_data[:5]):
    print(f"Row {i+1}: {row}")
print(f"Total rows generated: {len(synthetic_data)}")
```

### Expected Output
```
Generated data:
[
  {
    "column_data": [3.0, 5.0, 7.0, 9.0, 11.0, ...],
    "column_name": "linear_data",
    "column_type": "continuous",
    "column_datatype": "float64"
  },
  {
    "column_data": [0.0, -3.0, -4.0, -3.0, 0.0, ...],
    "column_name": "quadratic_data",
    "column_type": "continuous",
    "column_datatype": "float64"
  },
  {
    "column_data": [0.0, 0.0998, 0.1987, 0.2955, 0.3894, ...],
    "column_name": "sinusoidal_data",
    "column_type": "continuous",
    "column_datatype": "float64"
  }
]
Total rows generated: 50
```

## Working with Tabular Data


Tabular data is structured as a list of column dictionaries, where each dictionary represents a column in the table. Each dictionary contains the column data, name, type, and datatype. For example:

### Load and Process Data

```python
from sdg_core_lib import Job
import json

# Your data should be in column-based format
# Example structure (similar to test files):
data_config = {
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

# Configure the generation job using available models
model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE.TabularVAE",
    "model_name": "customer_synthetic_model"
}

# Create and run the job
job = Job(
    n_rows=1000,  # Generate 1000 synthetic rows
    model_info=model_config,
    dataset=data_config,
    save_filepath="./customer_models"
)

# Train and generate
synthetic_data, metrics, model, schema = job.train()

print(f"Generated {len(synthetic_data)} synthetic rows")
print(f"Quality metrics: {metrics}")
```

## Time Series Generation

### Generate Sequential Data

To generate sequential data, you need to provide a time series structure. The TimeSeries dataset requires a column of type `group_index` to identify isolated experiments. The group index divides the data into groups, where each group represents an independent time series experiment. All groups must have the same size for proper time series processing.

```python
from sdg_core_lib import Job
import json

# Time series data uses the same column-based format
time_series_config = {
    "dataset_type": "time_series",
    "data": [
        {
            "column_data": [1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.1, 2.2, 2.3, 2.4, 3.0, 3.1, 3.2, 3.3, 3.4],
            "column_name": "value",
            "column_type": "continuous",
            "column_datatype": "float64"
        },
        {
            "column_data": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "column_name": "group_index",
            "column_type": "group_index",
            "column_datatype": "int64"
        }
        
    ]
}

# In this example:
# - Group 0 (index 0): values [1.0, 1.1, 1.2, 1.3, 1.4] - 5 time steps
# - Group 1 (index 1): values [2.0, 2.1, 2.2, 2.3, 2.4] - 5 time steps  
# - Group 2 (index 2): values [3.0, 3.1, 3.2, 3.3, 3.4] - 5 time steps
# Each group has the same size (5 time steps) which is required for TimeSeries datasets

# Use TimeSeriesVAE for time series data
time_series_model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TimeSeriesVAE.TimeSeriesVAE",
    "model_name": "time_series_model"
}

# Generate synthetic time series
job = Job(
    n_rows=200,
    model_info=time_series_model_config,
    dataset=time_series_config,
    save_filepath="./time_series_models"
)

results, metrics, model, schema = job.train()

print(f"Generated {len(results)} time series points")
print(f"First 5 points: {results[:5]}")
```

## Advanced Configuration

### Custom Hyperparameters

```python
import os
from sdg_core_lib import Job

# Set custom hyperparameters
os.environ["EPOCHS"] = "100"
os.environ["LEARNING_RATE"] = "0.001"
os.environ["BATCH_SIZE"] = "32"

# Your job will automatically use these hyperparameters
job = Job(
    n_rows=500,
    model_info=model_config,
    dataset=dataset_config,
    save_filepath="./custom_models"
)

results, metrics, model, schema = job.train()
```

### Model Inference Only

```python
# Use a pre-trained model for inference only
job = Job(
    n_rows=1000,
    model_info={
        "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE.TabularVAE",
        "model_name": "pretrained_model",
        "image": "./models/pretrained_model",  # Path to saved model
        "training_data_info": schema  # Schema from previous training
    }
)

# Example data skeleton using Dataset validation
# Skeleton defines the structure without actual data values
data_skeleton = [
    {
        "column_name": "feature_1",
        "column_datatype": "float64",
        "column_type": "continuous",
        "column_position": 0,
        "column_size": 1
    },
    {
        "column_name": "category_feature", 
        "column_datatype": "int64",
        "column_type": "categorical",
        "column_position": 1,
        "column_size": 1
    }
]

# Skeleton explanation:
# - column_name: Name of the feature/column
# - column_datatype: Data type (float64, int64, etc.)
# - column_type: Feature type (continuous, categorical, primary_key, group_index)
# - column_position: Position/order of the column (0-based)
# - column_size: Size of the column (usually 1 for single values)
# 
# Skeleton is used to define the dataset structure before data generation
# The actual data will be filled during the generation process

# Generate data without retraining
results, metrics = job.infer()
print(f"Generated {len(results)} rows using pre-trained model")
```

## Common Use Cases

### 1. Data Augmentation for Machine Learning

```python
# Load your existing data configuration
with open('your_data_config.json', 'r') as f:
    dataset_config = json.load(f)

# Generate additional training data
job = Job(
    n_rows=len(dataset_config["data"]) * 5,  # 5x augmentation
    model_info={
        "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE.TabularVAE",
        "model_name": "augmentation_model"
    },
    dataset=dataset_config,
    save_filepath="./augmentation_models"
)

augmented_data, _, _, _ = job.train()

print(f"Original data size: {len(dataset_config['data'])}")
print(f"Augmented data size: {len(augmented_data)}")
```

### 2. Privacy-Preserving Data Sharing

```python
# Load sensitive data configuration
with open('sensitive_data_config.json', 'r') as f:
    sensitive_config = json.load(f)

# Create synthetic version for sharing
job = Job(
    n_rows=len(sensitive_config["data"]),
    model_info={
        "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE.TabularVAE",
        "model_name": "privacy_model"
    },
    dataset=sensitive_config,
    save_filepath="./privacy_models"
)

synthetic_data, metrics, _, _ = job.train()

# Check privacy metrics
print(f"Privacy preservation metrics: {metrics}")

# Save synthetic data for sharing
with open("synthetic_data.json", "w") as f:
    json.dump(synthetic_data, f)

print("Synthetic data ready for safe sharing")
```

### 3. Load Testing Data Generation

```python
# Generate large dataset for performance testing
job = Job(
    n_rows=100000,  # 100K rows for load testing
    model_info={
        "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE.TabularVAE",
        "model_name": "load_test_model"
    },
    dataset=dataset_config,  # Use existing data structure
    save_filepath="./load_test_models"
)

load_test_data, _, _, _ = job.train()
print(f"Generated {len(load_test_data)} rows for load testing")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Installation Problems
```bash
# If you encounter dependency conflicts, try:
pip install --upgrade pip
pip install sdg-core-lib --no-cache-dir
```

#### 2. Memory Issues
```python
# For large datasets, reduce batch size or use smaller n_rows
job = Job(
    n_rows=1000000,  # Reduce if memory is limited
    model_info=model_config,
    dataset=dataset_config
)
```

#### 3. Model Training Fails
```python
# Ensure your data format is correct
def validate_data_format(data):
    if not isinstance(data, list):
        raise ValueError("Data must be a list of dictionaries")
    if len(data) == 0:
        raise ValueError("Data cannot be empty")
    if not all(isinstance(row, dict) for row in data):
        raise ValueError("Each row must be a dictionary")
    return True

# Validate before creating job
validate_data_format(dataset_config["data"])
```

#### 4. Poor Quality Results
```python
# Try different models or adjust hyperparameters
models_to_try = [
    "sdg_core_lib.data_generator.models.GANs.TabularGAN",
    "sdg_core_lib.data_generator.models.VAEs.TabularVAE"
]

for model_name in models_to_try:
    model_config["algorithm_name"] = model_name
    job = Job(n_rows=500, model_info=model_config, dataset=dataset_config)
    results, metrics, _, _ = job.train()
    print(f"Model: {model_name}, Quality: {metrics.get('statistical_similarity', 'N/A')}")
```

## Next Steps

Now that you've completed the quick start, you can:

1. **Explore Advanced Features**: Read the [User Documentation](user-documentation.md)
2. **Learn Architecture**: Understand the system in [Developer Documentation](developer-documentation.md)
3. **Follow Tutorial**: Complete the [Step-by-Step Tutorial](step-by-step-tutorial.md)
4. **View Examples**: Check out the examples directory for more use cases
5. **Join Community**: Participate in discussions and contribute to the project

## Need Help?

- 📖 [Full Documentation](../README.md)
- 🐛 [Report Issues](https://github.com/emiliocimino/generator_core_lib/issues)
- 💬 [Community Forum](https://github.com/emiliocimino/generator_core_lib/discussions)
- 📧 [Email Support](mailto:emilio.cimino@outlook.it)

Happy synthetic data generation! 🚀
