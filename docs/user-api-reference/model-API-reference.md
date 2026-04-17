# Model API Reference

This document provides detailed API documentation for machine learning models used in synthetic data generation.

## Table of Contents

1. [UnspecializedModel Base Class](#unspecializedmodel-base-class)
2. [Available Models](#available-models)
3. [Related APIs](#related-apis)

---

## UnspecializedModel Base Class

Abstract base class that provides the foundation for all machine learning models in GENESIS Core Lib. This class implements common functionalities and defines abstract methods that must be implemented by all subclasses, ensuring consistent behavior across different model architectures.

### Class Design and Architecture

The UnspecializedModel serves as a template for model implementations by:

- **Providing Common Infrastructure**: Shared attributes and utility methods
- **Defining Required Interface**: Abstract methods that all models must implement
- **Managing Model Lifecycle**: Handles initialization, loading, and building processes
- **Standardizing Metadata**: Ensures consistent data structure handling

### Constructor Parameters

```python
def __init__(
    self,
    metadata: list[dict],
    model_name: str,
    input_shape: str = None,
    load_path: str = None,
):
```

**Parameters:**
- `metadata` (list[dict]): Dataset metadata containing column information and schema. [What is a data skeleton?](./dataset-API-reference.md#what-is-a-data-skeleton)
- `model_name` (str): Identifier for the model, used for logging and saving
- `input_shape` (str): String representation of input dimensions (e.g., "(1000,10)")
- `load_path` (str): Path to load pre-trained model from (optional)

**Key Attributes:**
- `_metadata`: Stores dataset schema and column information
- `model_name`: Model identifier for file naming and logging
- `input_shape`: Parsed tuple of input dimensions
- `_load_path`: Path for loading pre-trained models
- `_model`: Placeholder for the actual model instance
- `training_info`: Placeholder for training configuration

### How It Serves as a Model Basis

#### 1. Metadata-Driven Architecture
```python
# Models receive dataset metadata (skeleton) at initialization
metadata = [
    {
        "feature_name": "age",
        "feature_position": 0,
        "feature_type": "continuous",
        "type": "float64",
        "is_categorical": false,
        "feature_size": "1"
    },
    {
        "feature_name": "category",
        "feature_position": 1,
        "feature_type": "categorical",
        "type": "str",
        "is_categorical": true,
        "feature_size": "1"
    }
]

model = TabularVAE(
    metadata=metadata,
    model_name="my_vae",
    input_shape="(1000,2)"  # 1000 rows, 2 features
)
```

#### 2. Abstract Method Implementation
Subclasses must implement these core methods:

```python
class CustomModel(UnspecializedModel):
    def _build(self, input_shape: tuple[int, ...]):
        """Build the model architecture with given input shape."""
        # Implement neural network construction
        pass
    
    def _load(self, model_filepath: str):
        """Load pre-trained model weights."""
        # Implement model loading logic
        pass
    
    def train(self, data: np.ndarray):
        """Train the model on provided data."""
        # Implement training algorithm
        pass
    
    def fine_tune(self, data: np.ndarray, **kwargs):
        """Fine-tune the model."""
        # Implement fine-tuning logic
        pass
    
    def infer(self, n_rows: int, **kwargs):
        """Generate synthetic data."""
        # Implement inference logic
        pass
    
    def save(self, folder_path):
        """Save model artifacts."""
        # Implement saving logic
        pass
    
    def set_hyperparameters(self, **kwargs):
        """Configure model hyperparameters."""
        # Implement hyperparameter setting
        pass
```

#### 3. Automatic Model Instantiation
The base class handles model creation through the `_instantiate()` method:

```python
# Automatic instantiation workflow:
# 1. If load_path provided -> Load pre-trained model
# 2. If input_shape available -> Build new model
# 3. Store model instance in self._model
```

#### 4. Input Shape Handling
The base class provides utility for parsing string input shapes:

```python
# Converts string format to tuple
input_shape = "(1000,10)"  # String input
parsed_shape = (1000, 10)  # Parsed tuple
```


### Methods

#### train()
```python
train(data: np.ndarray) -> UnspecializedModel
```
Train the model on the provided dataset using the model's specific learning algorithm. This method automatically handles model instantiation if not already built.

**Parameters:**
- `data` (np.ndarray): Training data array in computing format (preprocessed numerical data)

**Returns:**
- `UnspecializedModel`: Self instance with trained weights and updated model state

**Process:**
1. Calls `_instantiate()` to ensure model is built or loaded
2. Executes model-specific training algorithm
3. Updates internal model state and training info
4. Returns self for method chaining

**Note:** The input data should be in computing format (numeric array) as obtained from `dataset.get_computing_data()`.

#### infer()
```python
infer(n_rows: int, **kwargs) -> np.ndarray
```
Generate synthetic data using the trained model. The model must be trained before calling this method.

**Parameters:**
- `n_rows` (int): Number of synthetic rows to generate
- `**kwargs`: Additional model-specific parameters (e.g., random_seed, temperature)

**Returns:**
- `np.ndarray`: Generated synthetic data in computing format

**Process:**
1. Validates that model is trained and ready
2. Generates latent representations or random inputs
3. Performs forward pass through the model
4. Returns synthetic data matching the training data format

**Note:** Output data will be in computing format and may need postprocessing to convert back to original data types.

#### save()
```python
save(folder_path: str) -> None
```
Persist the trained model and all associated artifacts to disk for later use.

**Parameters:**
- `folder_path` (str): Directory path where model artifacts will be saved

**Saved Components:**
- Model weights and architecture
- Training configuration and hyperparameters
- Metadata and schema information
- Training history and performance metrics

**Note:** Creates the specified directory if it doesn't exist. All files are saved within the provided folder path.

#### load()
```python
load(filepath: str) -> UnspecializedModel
```
Load a previously saved model from disk and restore it to a ready-to-use state.

**Parameters:**
- `filepath` (str): Directory path containing the saved model artifacts

**Returns:**
- `UnspecializedModel`: Self instance with loaded model ready for inference

**Process:**
1. Validates the existence and compatibility of saved files
2. Reconstructs the model architecture
3. Loads trained weights and parameters
4. Restores training metadata and configuration
5. Sets model to inference-ready state

**Note:** This method is typically called automatically during initialization if `load_path` is provided.

#### set_hyperparameters()
```python
set_hyperparameters(**kwargs) -> None
```
Configure model-specific hyperparameters for training and inference.

**Parameters:**
- `**kwargs`: Model-specific hyperparameter keyword arguments

**Common Hyperparameters:**
- `learning_rate`: Optimization step size
- `batch_size`: Training batch size
- `epochs`: Maximum training iterations
- `latent_dim`: Latent space dimension (VAEs)
- `hidden_dims`: Hidden layer dimensions

**Note:** Hyperparameters take effect on the next training session. Some models may require re-initialization after changing certain parameters.

---

### Training Info

The `training_info` attribute stores basic training information about the model's training process. This information is automatically populated during training and provides essential metrics for model evaluation.

#### TrainingInfo Class Structure

The TrainingInfo class stores fundamental training metrics:

```python
class TrainingInfo:
    def __init__(
        self,
        loss_fn: str,              # Loss function used
        train_samples: int,        # Number of training samples
        train_loss: float,         # Final training loss
        validation_samples: int = None,   # Number of validation samples
        validation_loss: float = None,    # Final validation loss
    )
```

#### Training Info Dictionary Format

When converted to dictionary format (via `to_dict()` method), the structure is:

```python
training_info_dict = {
    "loss_function": str,          # Name of loss function
    "train_samples": int,          # Number of training samples
    "train_loss": float,            # Final training loss
    "val_samples": int,            # Number of validation samples (optional)
    "val_loss": float,              # Final validation loss (optional)
}
```

#### Accessing Training Information

```python
# After training
model.train(training_data)

# Access training info object
info = model.training_info

# Convert to dictionary for easier access
info_dict = info.to_dict()

# Check basic training metrics
print(f"Loss function: {info_dict['loss_function']}")
print(f"Training samples: {info_dict['train_samples']}")
print(f"Training loss: {info_dict['train_loss']:.4f}")

# Check validation metrics if available
if info_dict['val_samples'] is not None:
    print(f"Validation samples: {info_dict['val_samples']}")
    print(f"Validation loss: {info_dict['val_loss']:.4f}")
```

#### Training Info Persistence

Training information is automatically saved and loaded with the model:

```python
# Save model (includes training info)
model.save("./my_model")

# Load model (restores training info)
loaded_model = TabularVAE(metadata, "loaded_model", load_path="./my_model")

# Training info is available after loading
info = loaded_model.training_info.to_dict()
print(f"Training loss from loaded model: {info['train_loss']}")
```

#### JSON Export

Training info can be exported to JSON format for external analysis:

```python
# Convert to JSON
json_info = model.training_info.to_json()
print(json_info)

# Save to file for analysis
with open("training_info.json", "w") as f:
    f.write(json_info)
```

#### Basic Model Comparison

```python
# Compare basic metrics across models
models = [vae_model, gan_model]

for model in models:
    info = model.training_info.to_dict()
    print(f"\n{model.model_name}:")
    print(f"  Loss function: {info['loss_function']}")
    print(f"  Training samples: {info['train_samples']}")
    print(f"  Training loss: {info['train_loss']:.4f}")
    if info['val_loss']:
        print(f"  Validation loss: {info['val_loss']:.4f}")
```


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

Conditional Tabular GAN for complex tabular data with mixed data types.

#### Creating a CTGAN Instance

CTGAN requires a specific data schema metadata to understand your data structure. Here's how to create and configure a CTGAN instance:

```python
from sdg_core_lib.data_generator.models.GANs.implementation.CTGAN import CTGAN
import numpy as np

# Step 1: Define your data schema metadata using DataSkeleton model
# This tells CTGAN about each column in your dataset
metadata = [
    {
        "column_name": "age",
        "column_type": "continuous",
        "column_datatype": "float32",
        "column_position": 0,
        "column_size": 2  # 1 for normalized value + 1 for mode indicator
    },
    {
        "column_name": "income",
        "column_type": "continuous",
        "column_datatype": "float32", 
        "column_position": 1,
        "column_size": 3  # 1 for normalized value + 2 for mode indicators
    },
    {
        "column_name": "education",
        "column_type": "categorical",
        "column_datatype": "string",
        "column_position": 2,
        "column_size": 4  # Number of categories (high school, bachelor, master, phd)
    },
    {
        "column_name": "city",
        "column_type": "categorical",
        "column_datatype": "string",
        "column_position": 3,
        "column_size": 5  # Number of cities
    }
]

# Step 2: Create CTGAN instance with hyperparameters
ctgan = CTGAN(
    metadata=metadata,
    model_name="my_ctgan_model",
    # Network architecture parameters
    gen_hidden=256,          # Generator hidden layer size
    critic_hidden=256,       # Critic (discriminator) hidden layer size
    critic_dropout=0.2,      # Dropout rate for critic to prevent overfitting
    
    # Training parameters
    learning_rate=0.001,     # Learning rate for both generator and critic
    batch_size=100,          # Batch size for training
    epochs=50,               # Number of training epochs
    gen_steps=4,             # Generator training steps per critic step
    
    # CTGAN-specific parameters
    pac_size=10              # Size of pac groups for critic training
)
```

#### Initialization Parameters Explained

**Required Parameters:**
- `metadata` (list[dict]): Data schema defining each column's type and size
- `model_name` (str): Unique identifier for your model instance

**Network Architecture:**
- `gen_hidden` (int, default=256): Hidden layer size for the generator network
- `critic_hidden` (int, default=256): Hidden layer size for the critic (discriminator)
- `critic_dropout` (float, default=0.2): Dropout rate applied to critic layers

**Training Configuration:**
- `learning_rate` (float, default=0.001): Learning rate for Adam optimizers (β₁=0.5, β₂=0.9)
- `batch_size` (int, default=100): Number of samples per training batch
- `epochs` (int, default=50): Total number of training epochs
- `gen_steps` (int, default=4): Generator updates per single critic update

**CTGAN-Specific:**
- `pac_size` (int, default=10): Size of pac (Pac) groups for critic training - groups samples together for more stable discrimination

#### Data Schema Requirements

Your skeleton (metadata) must follow the DataSkeleton model structure:

```python
# For continuous columns:
{
    "column_name": "column_name",
    "column_type": "continuous",
    "column_datatype": "float32",  # or "int32"
    "column_position": 0,          # Zero-based position in dataset
    "column_size": N                # N-1 modes + 1 normalized value
}

# For categorical columns:
{
    "column_name": "column_name", 
    "column_type": "categorical",
    "column_datatype": "string",    # or "int32" for encoded categories
    "column_position": 1,           # Zero-based position in dataset
    "column_size": N                # Number of categories
}
```

**Important Notes:**
- At least one categorical column is required for CTGAN to work
- Continuous columns need `column_size >= 2` (normalized value + mode indicators)
- Categorical columns use one-hot encoding internally
- `column_position` must be zero-based and unique for each column
- `column_datatype` should match your actual data type ("float32", "int32", or "string")

#### Training and Usage

```python
# Prepare your data (numpy array, preprocessed according to schema)
train_data = np.random.rand(1000, total_feature_size)  # Your actual data. This will not work!

# Train the model
ctgan.train(train_data)

# Generate synthetic data
synthetic_data = ctgan.infer(n_rows=500)

# Save model
ctgan.save("/path/to/save/model")

# Load model later
loaded_ctgan = CTGAN(
    metadata=metadata,
    model_name="my_ctgan_model", 
    load_path="/path/to/save/model"
)
```

#### Advanced Configuration

You can modify hyperparameters after initialization:

```python
# Update training parameters
ctgan.set_hyperparameters(
    learning_rate=0.0005,
    batch_size=200,
    epochs=100
)
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
