# Model API Reference

This document provides detailed API documentation for machine learning models used in synthetic data generation.

## Table of Contents

1. [UnspecializedModel Base Class](#unspecializedmodel-base-class)
2. [Available Models](#available-models)
3. [Related APIs](#related-apis)

---

## UnspecializedModel Base Class

Abstract base class for all machine learning models.

### Methods

#### train()
```python
train(data: np.ndarray) -> UnspecializedModel
```
Train the model on provided dataset.

**Parameters:**
- `data` (np.ndarray): Training data array

**Returns:**
- `UnspecializedModel`: Trained model instance

#### infer()
```python
infer(n_rows: int) -> np.ndarray
```
Generate synthetic data.

**Parameters:**
- `n_rows` (int): Number of rows to generate

**Returns:**
- `np.ndarray`: Generated synthetic data

#### save()
```python
save(filepath: str) -> None
```
Save model to disk.

**Parameters:**
- `filepath` (str): Path to save model

#### load()
```python
load(filepath: str) -> UnspecializedModel
```
Load model from disk.

**Parameters:**
- `filepath` (str): Path to saved model

**Returns:**
- `UnspecializedModel`: Loaded model instance

#### set_hyperparameters()
```python
set_hyperparameters(**hyperparams) -> None
```
Set model hyperparameters.

**Parameters:**
- `**hyperparams`: Hyperparameter keyword arguments

---

## Available Models

### TabularVAE

Variational Autoencoder for tabular data.

#### Configuration

```python
model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE.TabularVAE",
    "model_name": "tabular_vae_model",
    "hyperparameters": {
        "latent_dim": 32,
        "hidden_dims": [128, 64],
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    }
}
```

**Hyperparameters:**
- `latent_dim` (int): Dimensionality of latent space
- `hidden_dims` (list[int]): Hidden layer dimensions
- `learning_rate` (float): Learning rate for training
- `batch_size` (int): Batch size for training
- `epochs` (int): Number of training epochs

**Features:**
- Stable training process
- Interpretable latent space
- Good for feature analysis
- Handles missing values well

**Best For:**
- Small to medium datasets (< 10K rows)
- When training stability is important
- Feature analysis and interpretation

### TimeSeriesVAE

Variational Autoencoder for time series data.

#### Configuration

```python
model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TimeSeriesVAE.TimeSeriesVAE",
    "model_name": "time_series_vae_model",
    "hyperparameters": {
        "latent_dim": 16,
        "hidden_dims": [64, 32],
        "sequence_length": 12,
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 150
    }
}
```

**Hyperparameters:**
- `latent_dim` (int): Dimensionality of latent space
- `hidden_dims` (list[int]): Hidden layer dimensions
- `sequence_length` (int): Length of time series sequences
- `learning_rate` (float): Learning rate for training
- `batch_size` (int): Batch size for training
- `epochs` (int): Number of training epochs

**Features:**
- Captures temporal dependencies
- Handles variable-length sequences
- Preserves temporal patterns
- Specialized for time series data

**Best For:**
- Any time series data
- When temporal patterns are important
- Sequential data generation

### CTGAN

Conditional Tabular GAN for complex tabular data.

#### Configuration

```python
model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.GANs.implementation.CTGAN.CTGAN",
    "model_name": "ctgan_model",
    "hyperparameters": {
        "embedding_dim": 128,
        "generator_dim": [256, 256],
        "discriminator_dim": [256, 256],
        "learning_rate": 0.0002,
        "batch_size": 500,
        "epochs": 300
    }
}
```

**Hyperparameters:**
- `embedding_dim` (int): Embedding dimension
- `generator_dim` (list[int]): Generator network dimensions
- `discriminator_dim` (list[int]): Discriminator network dimensions
- `learning_rate` (float): Learning rate for training
- `batch_size` (int): Batch size for training
- `epochs` (int): Number of training epochs

**Features:**
- Handles mixed data types
- Preserves feature correlations
- Good for medium-sized datasets (1K-100K rows)
- Captures complex distributions

**Best For:**
- Medium to large datasets (> 1K rows)
- Complex data distributions
- When high realism is required
- Tabular data with complex feature interactions

### Model Selection Guide

| Dataset Size | Data Complexity | Recommended Model | Reason |
|-------------|-----------------|-------------------|---------|
| < 1,000 rows | Low | TabularVAE | More stable with small data |
| 1,000-10,000 rows | Medium | TabularVAE or CTGAN | Both work well |
| > 10,000 rows | High | CTGAN | Can capture complex distributions |
| Any size | Time series | TimeSeriesVAE | Specialized for temporal data |
| Any size | Very high dimensional | TabularVAE | More stable training |

### Usage Example

```python
from sdg_core_lib import Job

# Configure model
model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE.TabularVAE",
    "model_name": "my_vae_model"
}

# Create job with model
job = Job(
    n_rows=1000,
    model_info=model_config,
    dataset=dataset_config,
    save_filepath="./models"
)

# Train and generate
synthetic_data, metrics, model, schema = job.train()
```

### Environment Variables

Models can be configured using environment variables:

```python
import os

# Set hyperparameters
os.environ["EPOCHS"] = "200"
os.environ["LEARNING_RATE"] = "0.0001"
os.environ["BATCH_SIZE"] = "64"

# These will be automatically picked up by models
```

---

## Related APIs

For complete API documentation, see:

- **[Job API Reference](./job-API-reference.md)** - Core job management and orchestration
- **[Dataset API Reference](./dataset-API-reference.md)** - Data input/output and skeleton operations
- **[Functions API Reference](./functions-API-reference.md)** - Mathematical functions for data generation
- **[Processor API Reference](./processor-API-reference.md)** - Data preprocessing and postprocessing
- **[Evaluation API Reference](./evaluation-API-reference.md)** - Quality evaluation and metrics
