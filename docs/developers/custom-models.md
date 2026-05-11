# Custom Models

This guide explains how to create custom models for the Genesis Core Library by extending the `UnspecializedModel` class.

## Understanding the UnspecializedModel Class

The `UnspecializedModel` is the abstract base class that all models in the Genesis Core Library must inherit from. It provides a common interface and defines the essential methods that every model implementation must provide.

### Class Structure

```python
class UnspecializedModel(ABC):
    """
    Abstract class for all models. Implements common functionalities and defines abstract methods that must be implemented
    by all subclasses.

    Attributes:
        _metadata (dict): A dictionary containing miscellaneous information about the data structure used by a model.
        model_name (str): The model name, used to identify the model itself.
        input_shape (tuple): A tuple containing the input shape of the model.
        _load_path (str): A string containing the path where to load the model from.
        _model (keras.Model): The model instance.
        training_info (TrainingInfo): The training info instance.
    """
```

### Methods Overview

#### Abstract Methods (Must Implement)

These methods are marked as `@abstractmethod` and **must be implemented** by your custom model:

1. **`_build(self, input_shape: tuple[int, ...])`**
   - Build and return the model architecture
   - Called during initialization if no load path is provided
   - Should return the compiled model instance

2. **`_load(self, model_filepath: str)`**
   - Load a pre-trained model from the specified file path
   - Should set the `self._model` attribute
   - Does not return the model

3. **`train(self, data: np.ndarray)`**
   - Train the model on the provided data
   - Should set `self.training_info` with training metrics
   - Accepts numpy array as input data

4. **⚠️ `fine_tune(self, data: np.ndarray, **kwargs)` ⚠️**
   - ⚠️ Currently unsupported ⚠️
   - Fine-tune the model on new data ️
   - Optional additional parameters via kwargs

5. **`infer(self, n_rows: int, **kwargs)`**
   - Generate/infer data using the trained model
   - `n_rows`: Number of samples to generate
   - Should return generated data as numpy array

6. **`save(self, folder_path)`**
   - Save the model to the specified folder
   - Should save all necessary components for later loading

7. **`set_hyperparameters(self, **kwargs)`**
   - Set model hyperparameters dynamically
   - Accepts parameters as keyword arguments

8. **`self_describe(cls)`**
   - Class method that returns model metadata
   - Should return a `ModelInfo` dictionary

#### Concrete Methods (Keep Intact)

These methods are already implemented and **should not be overridden** unless you have specific requirements:

1. **`__init__(self, metadata, model_name, input_shape=None, load_path=None)`**
   - Initializes common attributes
   - Calls `_instantiate()` automatically

2. **`_instantiate(self)`**
   - Handles model instantiation logic
   - Calls `_load()` if load_path exists, otherwise calls `_build()`

3. **`_parse_stringed_input_shape(stringed_shape: str)`**
   - Static utility method for parsing input shapes
   - Converts string format like "[x,y,z]" to tuple (x, y, z)

## Supporting Classes

### ModelInfo Class

The `ModelInfo` class is used to provide metadata about your model to the upper layers of the GENESIS system. It helps the system understand your model's capabilities and requirements.

```python
class ModelInfo:
    def __init__(
        self,
        name: str,                    # Full model class path
        default_loss_function: str,    # Default loss function name
        description: str,              # Model description
        allowed_data: list[AllowedData], # Supported data types
    )
```

**Purpose in GENESIS System:**
- **Model Discovery**: Allows the system to automatically discover and categorize available models
- **Data Validation**: Ensures compatible data types are used with your model

### TrainingInfo Class

The `TrainingInfo` class captures training metrics and metadata that are used by the GENESIS system for tracking, comparison, and reporting purposes.

```python
class TrainingInfo:
    def __init__(
        self,
        loss_fn: str,              # Loss function used
        train_samples: int,       # Number of training samples
        train_loss: float,         # Final training loss
        validation_samples: int = None,  # Number of validation samples
        validation_loss: float = None,   # Final validation loss
    )
```

**Purpose in GENESIS System:**
- **Performance Tracking**: Records training metrics for model comparison
- **Experiment Logging**: Maintains training history for reproducibility
- **Model Selection**: Provides metrics for automated model selection
- **Reporting**: Generates training reports and analytics
- **Quality Assurance**: Validates training completion and performance standards

Both classes are essential components that enable your custom model to integrate seamlessly with the GENESIS ecosystem's higher-level functionalities like automated model selection, performance monitoring, and experiment management.



## Creating a Custom Model Example

Let's create a custom text generation model as an example. This model will use a simple LSTM-based architecture for generating text sequences.

### Step 1: Create the Custom Model Class

```python
import numpy as np
import keras
from keras import layers
from typing import List, Dict

from sdg_core_lib.data_generator.models.UnspecializedModel import UnspecializedModel
from sdg_core_lib.data_generator.models.ModelInfo import ModelInfo
from sdg_core_lib.data_generator.models.TrainingInfo import TrainingInfo
from sdg_core_lib.commons import AllowedData, DataType


class TextGenerationModel(UnspecializedModel):
    """
    A custom LSTM-based sequence generation model.
    
    This model generates integer sequences using an LSTM architecture with embedding
    and dense layers. It's designed to work with preprocessed integer data where
    each token/character is already encoded as an integer. Text conversion is handled
    separately by the preprocessing system (see [Processor API Reference](../user-api-reference/processor-API-reference.md)).
    
    Attributes:
        _vocab_size (int): Size of the vocabulary (max integer value + 1)
        _embedding_dim (int): Dimension of word embeddings
        _lstm_units (int): Number of LSTM units
        _sequence_length (int): Length of input sequences
        _temperature (float): Sampling temperature for generation
    """
    
    def __init__(
        self,
        metadata: List[Dict],  # DataSkeleton-compliant metadata
        model_name: str,
        input_shape: str = None,
        load_path: str = None,
        vocab_size: int = 1000,
        embedding_dim: int = 128,
        lstm_units: int = 256,
        sequence_length: int = 50,
        temperature: float = 1.0,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
    ):
        super().__init__(metadata, model_name, input_shape, load_path)
        
        # Model hyperparameters
        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._lstm_units = lstm_units
        self._sequence_length = sequence_length
        self._temperature = temperature
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._epochs = epochs
        
        # Instantiate the model
        self._instantiate()
    
    def _build(self, input_shape: tuple[int, ...]):
        """
        Build the LSTM-based text generation model.
        
        Args:
            input_shape: Tuple containing (sequence_length, vocab_size)
            
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = keras.Input(shape=(self._sequence_length,), dtype='int32')
        
        # Embedding layer
        x = layers.Embedding(
            input_dim=self._vocab_size,
            output_dim=self._embedding_dim,
            input_length=self._sequence_length
        )(inputs)
        
        # LSTM layers
        x = layers.LSTM(self._lstm_units, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(self._lstm_units)(x)
        x = layers.Dropout(0.2)(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(self._vocab_size, activation='softmax')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name="TextGenerationModel")
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self._learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _load(self, model_filepath: str):
        """
        Load a pre-trained model from the specified path.
        
        Args:
            model_filepath: Path to the saved model file
        """
        self._model = keras.saving.load_model(model_filepath)
    
    def train(self, data: np.ndarray):
        """
        Train the sequence generation model.
        
        Args:
            data: Training data as numpy array of shape (n_samples, sequence_length + 1)
                  containing integer-encoded sequences. The last column contains the target token.
                  Text conversion should be handled by preprocessing steps before training.
        """
        # Validate data is integer-encoded
        if not np.issubdtype(data.dtype, np.integer):
            raise ValueError("Training data must be integer-encoded. Use preprocessing steps for text conversion.")
        
        # Prepare training data
        X = data[:, :-1]  # Input sequences (integer-encoded)
        y = data[:, -1]   # Target tokens (integer-encoded)
        
        # Train the model
        history = self._model.fit(
            X, y,
            batch_size=self._batch_size,
            epochs=self._epochs,
            validation_split=0.2,
            verbose=1
        )
        
        # Store training information
        self.training_info = TrainingInfo(
            loss_fn="sparse_categorical_crossentropy",
            train_samples=len(X),
            train_loss=history.history["loss"][-1],
            validation_samples=0,
            validation_loss=history.history["val_loss"][-1]
        )
    
    
    def infer(self, n_rows: int, **kwargs):
        """
        Generate integer sequences.
        
        Args:
            n_rows: Number of sequences to generate
            
        Returns:
            Generated sequences as numpy array of integers
            Use preprocessing steps for text conversion if needed.
        """
        generated_sequences = []
        
        for _ in range(n_rows):
            # Start with random integer token
            current_sequence = [np.random.randint(0, self._vocab_size)]
            
            # Generate sequence
            for _ in range(self._sequence_length):
                # Pad sequence if needed
                if len(current_sequence) < self._sequence_length:
                    padded_sequence = current_sequence + [0] * (self._sequence_length - len(current_sequence))
                else:
                    padded_sequence = current_sequence[-self._sequence_length:]
                
                # Predict next token
                prediction = self._model.predict(
                    np.array([padded_sequence]), 
                    verbose=0
                )[0]
                
                # Apply temperature and sample
                prediction = np.log(prediction + 1e-10) / self._temperature
                exp_pred = np.exp(prediction)
                prediction = exp_pred / np.sum(exp_pred)
                
                next_token = np.random.choice(self._vocab_size, p=prediction)
                current_sequence.append(next_token)
            
            generated_sequences.append(current_sequence[-self._sequence_length:])
        
        return np.array(generated_sequences)
    
    def save(self, folder_path):
        """
        Save the model to the specified folder.
        
        Args:
            folder_path: Path to save the model
        """
        self._model.save(f"{folder_path}/text_generation_model.keras")
    
    def set_hyperparameters(self, **kwargs):
        """
        Set model hyperparameters dynamically.
        
        Args:
            **kwargs: Hyperparameters to set
        """
        self._vocab_size = kwargs.get("vocab_size", self._vocab_size)
        self._embedding_dim = kwargs.get("embedding_dim", self._embedding_dim)
        self._lstm_units = kwargs.get("lstm_units", self._lstm_units)
        self._sequence_length = kwargs.get("sequence_length", self._sequence_length)
        self._temperature = kwargs.get("temperature", self._temperature)
        self._learning_rate = kwargs.get("learning_rate", self._learning_rate)
        self._batch_size = kwargs.get("batch_size", self._batch_size)
        self._epochs = kwargs.get("epochs", self._epochs)
    
    @classmethod
    def self_describe(cls):
        """
        Return model metadata and capabilities.
        
        Returns:
            Dictionary containing model information
        """
        return ModelInfo(
            name=f"{cls.__module__}.{cls.__qualname__}",
            default_loss_function="sparse_categorical_crossentropy",
            description="An LSTM-based text generation model for sequential data",
            allowed_data=[
                AllowedData(DataType.str, False),  # The model requires text input (even if it processes integers)
            ],
        ).get_model_info()
```

⚠️ As already explained, self_describe method is used in different layers of the GENESIS System. Self description should help users in choosing the adapt model for their data, so, even if the model treats uniquely integer values (encoded text), its description should tell the user which data is acceptable by the whole "Text Processing" Pipeline, including the Text Processor.


### Step 2: Using the Custom Model


```python
import numpy as np
from your_module import TextGenerationModel

# Prepare metadata (DataSkeleton compliant)
from sdg_core_lib.commons import DataType

# Metadata comes from the Text Processor -> You will see "integers" because the model treats integers
metadata = [
    {
        "feature_name": "sequence_tokens",
        "feature_position": 0,
        "is_categorical": True,
        "type": DataType.int32,  # Integer-encoded tokens
        "feature_type": "categorical",
        "feature_size": "(50,)" # Processing shape
    }
]

# Create model instance
model = TextGenerationModel(
    metadata=metadata,
    model_name="my_sequence_generator",
    input_shape="(50,)",  # sequence_length
    vocab_size=1000,
    embedding_dim=128,
    lstm_units=256,
    sequence_length=50,
    learning_rate=0.001,
    batch_size=32,
    epochs=50
)

# Prepare training data (integer-encoded sequences)
# data should be shape (n_samples, sequence_length + 1)
# Use preprocessing steps to convert text to integers before this step
training_data = np.random.randint(0, 1000, size=(1000, 51))

# Train the model
model.train(training_data)

# Generate integer sequences
generated_sequences = model.infer(n_rows=100)

# For text conversion, use preprocessing inverse_transform:
# processor.inverse_transform(generated_sequences)

# Save the model
model.save("/path/to/save/model")

# Load a saved model
loaded_model = TextGenerationModel(
    metadata=metadata,
    model_name="loaded_text_generator",
    load_path="/path/to/save/model/text_generation_model.keras"
)
```

## Best Practices

### 1. Standardized Method Signatures

While the abstract `UnspecializedModel` class allows `**kwargs` in some methods, it's recommended to use standardized parameters for consistency and better integration with the GENESIS system:

```python
# Recommended - Standardized parameters
def train(self, data: np.ndarray):
    # Use hyperparameters set via set_hyperparameters() or constructor

def infer(self, n_rows: int):
    # Use model's configured parameters for generation

# Avoid - Unstructured kwargs
def train(self, data: np.ndarray, **kwargs):
    # Harder to standardize across models

def infer(self, n_rows: int, **kwargs):
    # Inconsistent parameter handling
```

### 2. Metadata Structure
Ensure your metadata follows the DataSkeleton format used by the GENESIS system. For detailed user documentation on DataSkeleton requirements, see the [What is a Data Skeleton?](../user-api-reference/dataset-API-reference.md#what-is-a-data-skeleton) documentation.

```python
metadata = [
    {
        "feature_name": "feature_name",
        "feature_position": int,
        "is_categorical": bool,
        "type": DataType,  # From sdg_core_lib.commons.DataType
        "feature_type": "continuous|categorical|primary_key|group_index",
        "feature_size": str  # String representation of size
    }
]
```


### 3. Training Information
Always set `self.training_info` after training:
```python
self.training_info = TrainingInfo(
    loss_fn="your_loss_function",
    train_samples=len(X),
    train_loss=final_loss,
    validation_samples=len(X_val) if validation_split else 0,
    validation_loss=final_val_loss if validation_split else None
)
```

### 4. Model Description
Implement `self_describe()` to provide model metadata:
```python
@classmethod
def self_describe(cls):
    return ModelInfo(
        name=f"{cls.__module__}.{cls.__qualname__}",
        default_loss_function="your_loss_function",
        description="Brief description of your model",
        allowed_data=[
            AllowedData(DataType.float32, False),
            AllowedData(DataType.int32, True),
            # Add other supported data types
        ],
    ).get_model_info()
```

### 5. Error Handling
Include proper error handling and validation:
```python
def train(self, data: np.ndarray):
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.shape[1] != self._sequence_length + 1:
        raise ValueError(f"Data shape mismatch. Expected (_, {self._sequence_length + 1})")
    # ... rest of training logic
```

### 6. Documentation
Document your model class and methods thoroughly, including:
- Purpose and capabilities
- Expected input/output formats
- Hyperparameter descriptions
- Usage examples

## Integration with the Library

Your custom model will automatically integrate with the Genesis Core Library's:

- **Model Registry**: Models are automatically discovered and registered
- **Training Pipeline**: Compatible with the standard training workflow
- **Inference Engine**: Can be used for data generation tasks
- **Evaluation System**: Works with the library's evaluation metrics
- **Persistence**: Models can be saved and loaded consistently

## Testing Your Custom Model

Always test your implementation:

```python
import unittest
import numpy as np

class TestTextGenerationModel(unittest.TestCase):
    def setUp(self):
        self.metadata = [{"feature_name": "text", "feature_type": "categorical", "feature_size": 100}]
        self.model = TextGenerationModel(self.metadata, "test_model", "(50,)")
    
    def test_model_creation(self):
        self.assertIsNotNone(self.model._model)
    
    def test_training(self):
        data = np.random.randint(0, 100, size=(100, 51))
        self.model.train(data)
        self.assertIsNotNone(self.model.training_info)
    
    def test_inference(self):
        result = self.model.infer(5)
        self.assertEqual(result.shape, (5, 50))

if __name__ == "__main__":
    unittest.main()
```

This comprehensive guide should help you create robust custom models that integrate seamlessly with the Genesis Core Library.
