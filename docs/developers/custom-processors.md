# Custom Processors

## Overview

This guide explains how to create custom data processors for the GENESIS Core Lib. Custom processors allow you to implement specialized data preprocessing and postprocessing logic beyond the built-in processors.

## Understanding the Base Classes

Before creating custom processors, it's essential to understand the three core base classes that form the foundation of the preprocessing system:

### 1. The Step Class

The `Step` class is the fundamental building block for individual preprocessing operations. Each step represents a single transformation that can be applied to data.

```python
from abc import ABC, abstractmethod
import numpy as np
import os
import skops.io as sio

class Step(ABC):
    """Base class for all preprocessing steps"""
    
    def __init__(self, type_name: str, position: int, col_name: str, mode: str):
        self.type_name = type_name
        self.mode = mode
        self.position = position
        self.col_name = col_name
        self.operator = None
        self.filename = f"{self.position}_{self.col_name}_{self.mode}_{self.type_name}.skops"

    @abstractmethod
    def _set_operator(self):
        """Create and return the sklearn-compatible operator"""
        raise NotImplementedError

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit the operator and transform data"""
        self.operator = self._set_operator()
        return self.operator.fit_transform(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted operator"""
        if self.operator is None:
            raise ValueError("Operator not initialized")
        return self.operator.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data back to original form"""
        if self.operator is None:
            raise ValueError("Operator not initialized")
        return self.operator.inverse_transform(data)

    def save_if_not_exist(self, directory_path: str):
        """Save the fitted operator to disk"""
        if self.operator is None:
            raise ValueError("Operator is not created")
        os.makedirs(directory_path, exist_ok=True)
        filename = os.path.join(directory_path, self.filename)
        if not os.path.exists(filename):
            sio.dump(self.operator, filename)

    def load(self, directory_path: str):
        """Load the fitted operator from disk"""
        filename = os.path.join(directory_path, self.filename)
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Operator file not found: {filename}")
        self.operator = sio.load(filename)
```

**Key Points:**
- Each step implementation must implement `_set_operator()` to return a sklearn-compatible transformer. A sklearn-compatible operator should implement their own `fit`, `transform`, `fit_transform`, `inverse_transform` methods, it does not mean it have to come from the sklearn-library.
- Steps support both forward and inverse transformations
- Steps are automatically saved/loaded using skops format
- Steps are identified by position, column name, mode, and type

### 2. The BasePreprocessingStrategy Class

The strategy class determines which steps should be applied to different types of data columns.

```python
from sdg_core_lib.dataset.columns import Column
from sdg_core_lib.preprocess.strategies.steps import Step
from loguru import logger

class BasePreprocessingStrategy:
    """Base strategy for determining preprocessing steps"""
    
    @staticmethod
    def get_steps_per_feature(feature: Column) -> list[Step]:
        """
        Return the list of steps to apply to a given feature/column.
        Override this method in custom strategies.
        """
        logger.warning(
            "You are processing a feature with the base strategy. "
            "This will lead to an empty processing pipeline"
        )
        return []
```

**Key Points:**
- Strategies map column types to appropriate preprocessing steps
- The base strategy returns an empty list (no processing)
- Custom strategies override `get_steps_per_feature()` to define processing logic
- Strategies are model-tied: different ML models may require different pre-processing strategies.

### 3. The Processor Class

The `Processor` class orchestrates the application of steps to data using a strategy.

```python
from abc import ABC, abstractmethod
import numpy as np
from sdg_core_lib.preprocess.strategies.base_strategy import BasePreprocessingStrategy
from sdg_core_lib.preprocess.strategies.steps import Step

class Processor(ABC):
    """Abstract base class for all processors"""
    
    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.steps: dict[int, list[Step]] = {}
        self.idx_to_data: dict[int, int] = {}
        self.strategy: BasePreprocessingStrategy = BasePreprocessingStrategy()

    @abstractmethod
    def _init_steps(self, data: list):
        """Initialize processing steps for the given data"""
        raise NotImplementedError

    def set_strategy(self, strategy: BasePreprocessingStrategy) -> "Processor":
        """Set the preprocessing strategy"""
        self.strategy = strategy
        return self

    def add_steps(self, steps: list[Step], col_position: int, data_position: int) -> "Processor":
        """Add processing steps for a specific column"""
        self.steps[col_position] = steps
        self.idx_to_data[col_position] = data_position
        return self

    def save_all(self):
        """Save all processing steps"""
        [
            step.save_if_not_exist(self.dir_path)
            for step_list in self.steps.values()
            for step in step_list
        ]

    def load_all(self) -> "Processor":
        """Load all processing steps"""
        [
            step.load(self.dir_path)
            for step_list in self.steps.values()
            for step in step_list
        ]
        return self

    def process(self, data: list) -> dict[int, np.ndarray]:
        """Apply preprocessing to data"""
        results = {
            idx: step.fit_transform(data[self.idx_to_data[idx]])
            for idx, step_list in self.steps.items()
            for step in step_list
        }
        self.save_all()
        return results

    def inverse_process(self, data: list) -> dict[int, np.ndarray]:
        """Apply inverse preprocessing to data"""
        self.load_all()
        return {
            idx: step.inverse_transform(data[self.idx_to_data[idx]])
            for idx, step_list in self.steps.items()
            for step in reversed(step_list)
        }
```

**Key Points:**
- Processors manage collections of steps for multiple columns
- They use strategies to determine which steps to apply
- They handle the orchestration of fitting, transforming, and inverse transforming
- Steps are applied in sequence, inverse transforms in reverse sequence

## Complete Example: Text Processor

Now let's create a complete text processing pipeline with custom steps, strategy, and processor that works with the Text Column from the custom data types documentation.

### 1. Text Tokenization Step

```python
import re
import pickle
from typing import List, Dict, Any
import numpy as np
from sdg_core_lib.preprocess.strategies.steps import Step
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

class TokenizerOperator(BaseEstimator, TransformerMixin):
    """Custom sklearn-compatible tokenizer operator with reversible transformations"""
    
    def __init__(self, remove_stopwords: bool = True, min_word_length: int = 2,
                 lowercase: bool = True, remove_punctuation: bool = True):
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        
        # Common English stopwords
        self.stopwords = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'
        ]) if remove_stopwords else set()
        
        # Store transformation mappings for reversibility
        self.original_texts = []
        self.processed_texts = []
        self.word_mappings = []  # Maps processed words back to original forms
        self.case_mappings = []  # Maps lowercase words back to original case
        self.punctuation_mappings = []  # Stores removed punctuation
    
    def fit(self, X, y=None):
        """Fit the tokenizer and build reversible mappings"""
        self.original_texts = [str(text) for text in X.flatten()]
        self.word_mappings = []
        self.case_mappings = []
        self.punctuation_mappings = []
        
        for text in self.original_texts:
            text_str = str(text)
            
            # Store original punctuation positions
            punctuation_map = {}
            if self.remove_punctuation:
                for i, char in enumerate(text_str):
                    if char in '.,!?;:()"\'()[]{}':
                        punctuation_map[i] = char
            self.punctuation_mappings.append(punctuation_map)
            
            # Store original case mapping
            case_map = {}
            if self.lowercase:
                words = text_str.split()
                for word in words:
                    case_map[word.lower()] = word
            self.case_mappings.append(case_map)
            
            # Store word mapping for stopwords and length filtering
            word_map = {}
            words = text_str.split()
            processed_words = []
            
            for word in words:
                processed_word = word.lower() if self.lowercase else word
                
                # Remove punctuation for processing
                if self.remove_punctuation:
                    processed_word = re.sub(r'[^\w\s]', '', processed_word)
                
                # Apply filters
                if (self.remove_stopwords and processed_word in self.stopwords) or \
                   (len(processed_word) < self.min_word_length):
                    word_map[word] = None  # Mark as removed
                else:
                    word_map[word] = processed_word  # Map to processed form
                    processed_words.append(processed_word)
            
            self.word_mappings.append(word_map)
        
        return self
    
    def transform(self, X):
        """Transform text data"""
        self.processed_texts = []
        
        for text in X.flatten():
            text_str = str(text)
            processed_words = []
            
            # Convert to lowercase
            if self.lowercase:
                text_str = text_str.lower()
            
            # Remove punctuation
            if self.remove_punctuation:
                text_str = re.sub(r'[^\w\s]', '', text_str)
            
            # Tokenize and filter
            words = text_str.split()
            if self.remove_stopwords:
                words = [word for word in words if word not in self.stopwords]
            words = [word for word in words if len(word) >= self.min_word_length]
            
            processed_text = ' '.join(words)
            self.processed_texts.append(processed_text)
        
        return np.array(self.processed_texts).reshape(-1, 1)
    
    def inverse_transform(self, X):
        """Reconstruct original text from processed text"""
        reconstructed_texts = []
        
        for i, processed_text in enumerate(X.flatten()):
            processed_text = str(processed_text)
            processed_words = processed_text.split()
            
            # Use stored mappings to reconstruct
            if i < len(self.word_mappings):
                word_map = self.word_mappings[i]
                case_map = self.case_mappings[i]
                punctuation_map = self.punctuation_mappings[i]
                
                # Reconstruct words with original case
                reconstructed_words = []
                for processed_word in processed_words:
                    # Find original word from mapping
                    original_word = None
                    for orig_word, proc_word in word_map.items():
                        if proc_word == processed_word:
                            original_word = orig_word
                            break
                    
                    if original_word:
                        # Restore original case
                        if self.lowercase and original_word.lower() in case_map:
                            restored_word = case_map[original_word.lower()]
                        else:
                            restored_word = original_word
                        reconstructed_words.append(restored_word)
                    else:
                        # Fallback: use processed word
                        reconstructed_words.append(processed_word)
                
                # Reconstruct text with original punctuation
                reconstructed_text = ' '.join(reconstructed_words)
                
                # Add back punctuation (approximate - we add at end of text)
                if self.remove_punctuation and punctuation_map:
                    # Simple approach: add punctuation at the end
                    punctuation_chars = list(punctuation_map.values())
                    if punctuation_chars:
                        reconstructed_text += ''.join(punctuation_chars[-3:])  # Add last 3 punctuation marks
                
                reconstructed_texts.append(reconstructed_text)
            else:
                # Fallback: return processed text
                reconstructed_texts.append(processed_text)
        
        return np.array(reconstructed_texts).reshape(-1, 1)
    
    def get_feature_names_out(self):
        """Get output feature names"""
        return ["processed_text"]
    

class TextTokenizationStep(Step):
    """Custom step for text tokenization and preprocessing"""
    
    def __init__(self, position: int, col_name: str, 
                 remove_stopwords: bool = True, min_word_length: int = 2,
                 lowercase: bool = True, remove_punctuation: bool = True):
        super().__init__(
            type_name="text_tokenization",
            position=position,
            col_name=col_name,
            mode="text"
        )
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

    def _set_operator(self):
        """Create the tokenizer operator"""
        return TokenizerOperator(
            remove_stopwords=self.remove_stopwords,
            min_word_length=self.min_word_length,
            lowercase=self.lowercase,
            remove_punctuation=self.remove_punctuation
        )
```

### 2. Text Embedding Step

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

class TextEmbeddingOperator(BaseEstimator, TransformerMixin):
    """Custom sklearn-compatible text embedding operator"""
    
    def __init__(self, method: str = "tfidf", max_features: int = 1000,
                 n_components: int = 100, embedding_dim: int = 128):
        self.method = method
        self.max_features = max_features
        self.n_components = n_components
        self.embedding_dim = embedding_dim
        
        # Initialize components based on method
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            self.dim_reducer = TruncatedSVD(n_components=embedding_dim)
            self.scaler = StandardScaler()
        elif method == "count":
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95,
                use_idf=False,
                norm=None
            )
            self.dim_reducer = TruncatedSVD(n_components=embedding_dim)
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown embedding method: {method}")
        
        self.original_texts = []
        self.feature_names = []
    
    def fit(self, X, y=None):
        """Fit the embedding operator"""
        self.original_texts = [str(text) for text in X.flatten()]
        
        # Fit vectorizer
        text_features = self.vectorizer.fit_transform(X.flatten())
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Fit dimensionality reduction
        reduced_features = self.dim_reducer.fit_transform(text_features)
        
        # Fit scaler
        self.scaler.fit(reduced_features)
        
        return self
    
    def transform(self, X):
        """Transform text to embeddings"""
        # Vectorize
        text_features = self.vectorizer.transform(X.flatten())
        
        # Reduce dimensionality
        reduced_features = self.dim_reducer.transform(text_features)
        
        # Scale
        scaled_features = self.scaler.transform(reduced_features)
        
        return scaled_features
    
    def inverse_transform(self, X):
        """Inverse transform embeddings back to text (approximate)"""
        # Inverse scaling
        unscaled_features = self.scaler.inverse_transform(X)
        
        # Inverse dimensionality reduction (approximate)
        reconstructed_tfidf = self.dim_reducer.inverse_transform(unscaled_features)
        
        # Convert TF-IDF back to text (very approximate)
        reconstructed_texts = []
        for doc_vector in reconstructed_tfidf:
            # Get top terms for each document
            top_indices = doc_vector.argsort()[-10:][::-1]  # Top 10 terms
            top_terms = [self.feature_names[i] for i in top_indices if doc_vector[i] > 0.01]
            reconstructed_texts.append(' '.join(top_terms))
        
        return np.array(reconstructed_texts).reshape(-1, 1)
    
    def get_feature_names_out(self):
        """Get output feature names"""
        return [f"embedding_{i}" for i in range(self.embedding_dim)]

class TextEmbeddingStep(Step):
    """Custom step for text embedding generation"""
    
    def __init__(self, position: int, col_name: str,
                 method: str = "tfidf", max_features: int = 1000,
                 embedding_dim: int = 128):
        super().__init__(
            type_name="text_embedding",
            position=position,
            col_name=col_name,
            mode="text"
        )
        self.method = method
        self.max_features = max_features
        self.embedding_dim = embedding_dim

    def _set_operator(self):
        """Create the embedding operator"""
        return TextEmbeddingOperator(
            method=self.method,
            max_features=self.max_features,
            embedding_dim=self.embedding_dim
        )
```

### 3. Text Transformer Strategy

Before starting, make sure you have read the [Custom Data Types Tutorial](custom-data-types.md) from which TextColumn is defined.

```python
from typing import List
from sdg_core_lib.dataset.columns import Column, Numeric, Categorical
from sdg_core_lib.preprocess.strategies.base_strategy import BasePreprocessingStrategy
from sdg_core_lib.preprocess.strategies.steps import (
    Step,
    TextTokenizationStep,
    TextEmbeddingStep,
    NoneStep
)

class TextTransformerStrategy(BasePreprocessingStrategy):
    """Strategy for processing text data with transformer-like models for NLP"""
    
    @staticmethod
    def get_steps_per_feature(feature: Column) -> list[Step]:
        """Return steps for text features based on transformer architecture principles"""
        step_list = []
        
        # Check if feature is a TextColumn (assuming TextColumn exists in custom data types)
        if type(feature) is TextColumn:
            # Step 1: Tokenization and preprocessing (like tokenizer in transformers)
            step_list.append(TextTokenizationStep(
                position=feature.position,
                col_name=feature.name,
                remove_stopwords=True,
                min_word_length=2,
                lowercase=True,
                remove_punctuation=True
            ))
            
            # Step 2: Embedding generation (like embedding layer in transformers)
            step_list.append(TextEmbeddingStep(
                position=feature.position,
                col_name=feature.name,
                method="tfidf",
                max_features=1000,
                embedding_dim=128
            ))
        elif type(feature) is Column:
            # Handle generic columns with no processing
            step_list.append(NoneStep(feature.position))
        else:
            raise NotImplementedError(f"Unsupported feature type: {type(feature)}")
        
        return step_list
    

```

### 4. Text Processor

```python
from sdg_core_lib.preprocess.base_processor import Processor
from sdg_core_lib.dataset.columns import Column
from typing import List, Dict, Any
import numpy as np

class TextProcessor(Processor):
    """Complete processor for text data using transformer-like pipeline"""
    
    def __init__(self, dir_path: str,
                 remove_stopwords: bool = True,
                 min_word_length: int = 2,
                 embedding_method: str = "tfidf",
                 max_features: int = 1000,
                 embedding_dim: int = 128):
        super().__init__(dir_path)
        
        # Create and set the text transformer strategy
        self.strategy = TextTransformerStrategy(
            remove_stopwords=remove_stopwords,
            min_word_length=min_word_length,
            embedding_method=embedding_method,
            max_features=max_features,
            embedding_dim=embedding_dim
        )
        
        # Store parameters for reference
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.embedding_method = embedding_method
        self.max_features = max_features
        self.embedding_dim = embedding_dim

    def _init_steps(self, data: List[Column]):
        """Initialize text processing steps for all text columns"""
        if len(data) == 0:
            raise ValueError("No columns provided for processing")
        
        for idx, col in enumerate(data):
            if col.column_type == "text":
                # Get steps from strategy
                step_list = self.strategy.get_steps_per_feature(col)
                
                # Add steps for this column
                self.add_steps(step_list, col_position=col.position, data_position=idx)

    def process(self, columns: List[Column]) -> List[Column]:
        """Process only text columns, leaving others unchanged"""
        # Initialize steps
        self._init_steps(columns)
        
        # Process all columns
        col_data = [col.get_data() for col in columns]
        results = super().process(col_data)
        
        # Create new columns with processed data
        processed_columns = []
        for col in columns:
            if col.position in results:
                processed_columns.append(
                    type(col)(
                        col.name,
                        col.value_type,
                        col.position,
                        results.get(col.position),
                        col.column_type,
                    )
                )
            else:
                # Column wasn't processed (not text), keep original
                processed_columns.append(col)
        
        return processed_columns

    def inverse_process(self, columns: List[Column]) -> List[Column]:
        """Inverse process only text columns"""
        # Initialize steps
        self._init_steps(columns)
        
        # Process all columns
        col_data = [col.get_data() for col in columns]
        results = super().inverse_process(col_data)
        
        # Create new columns with inverse processed data
        inverse_processed_columns = []
        for col in columns:
            if col.position in results:
                inverse_processed_columns.append(
                    type(col)(
                        col.name,
                        col.value_type,
                        col.position,
                        results.get(col.position),
                        col.column_type,
                    )
                )
            else:
                # Column wasn't processed, keep original
                inverse_processed_columns.append(col)
        
        return inverse_processed_columns

```

## Usage Example

```python
# Example usage with Text columns
from sdg_core_lib.dataset.columns import TextColumn
import numpy as np

# Create sample text data
texts = [
    "The quick brown fox jumps over the lazy dog",
    "Natural language processing is fascinating",
    "Machine learning models require good data",
    "Text preprocessing improves model performance"
]

# Create text column
text_column = TextColumn(
    name="description",
    value_type="string",
    position=0,
    values=np.array(texts)
)

# Create text processor
processor = TextProcessor(
    dir_path="./text_processor_cache",
    remove_stopwords=True,
    embedding_method="tfidf",
    max_features=1000,
    embedding_dim=64
)

# Process the text column
processed_columns = processor.process_text_columns([text_column])

# Get statistics
stats = processor.get_text_statistics([text_column])
print(f"Processing statistics: {stats}")

# Inverse process (reconstruct original text)
reconstructed_columns = processor.inverse_process_text_columns(processed_columns)

print(f"Original shape: {text_column.get_data().shape}")
print(f"Processed shape: {processed_columns[0].get_data().shape}")
print(f"Reconstructed shape: {reconstructed_columns[0].get_data().shape}")
```

## Integration with Existing Systems

### Using Custom Processors with Datasets

```python
# Extend existing dataset classes to use custom processors
from sdg_core_lib.dataset.datasets import Dataset
from sdg_core_lib.dataset.columns import Column, TextColumn
import numpy as np

class TextDataset(Dataset):
    """Dataset class for text data with integrated processing support"""
    
    def __init__(self, columns: List[TextColumn]):
        self.columns = columns
        
        # Validate that all columns are TextColumn instances
        for col in columns:
            if not isinstance(col, TextColumn):
                raise ValueError(f"All columns must be TextColumn instances, got {type(col)}")
    
    # Insert your Dataset Code Here -----
    
    def preprocess(self, processor: "TextProcessor") -> "TextDataset":
        """Preprocess text data using custom processor"""
        processed_columns = processor.process(self.columns)
        
        # Create new TextDataset with processed columns
        return TextDataset(
            columns=processed_columns,
        )
    
    def postprocess(self, processor: "TextProcessor") -> "TextDataset":
        """Postprocess/reconstruct text data using custom processor"""
        reconstructed_columns = processor.inverse_process(self.columns)
        
        # Create new TextDataset with reconstructed columns
        return TextDataset(
            columns=reconstructed_columns,
        )

```

## Testing Custom Processors

```python
import pytest
import numpy as np
from sdg_core_lib.dataset.columns import TextColumn

class TestTextProcessor:
    def setup_method(self):
        """Set up test data"""
        self.test_texts = [
            "Hello world this is a test",
            "Natural language processing with transformers",
            "Text preprocessing and tokenization",
            "Embedding generation for text data"
        ]
        
        self.text_column = TextColumn(
            name="test_text",
            value_type="string",
            position=0,
            values=np.array(self.test_texts)
        )
        
        self.processor = TextProcessor(
            dir_path="/tmp/test_text_processor",
            remove_stopwords=True,
            embedding_method="tfidf",
            max_features=100,
            embedding_dim=32
        )
    
    def test_tokenization_step(self):
        """Test text tokenization step"""
        from sdg_core_lib.preprocess.strategies.steps  import TextTokenizationStep
        
        step = TextTokenizationStep(
            position=0,
            col_name="test_text",
            remove_stopwords=True
        )
        
        # Fit and transform
        operator = step._set_operator()
        fitted_op = operator.fit(self.text_column.get_data())
        transformed = fitted_op.transform(self.text_column.get_data())
        
        # Check that stopwords are removed
        assert "this" not in str(transformed[0])  # "this" should be removed
        assert "hello" in str(transformed[0])     # "hello" should remain
    
    def test_embedding_step(self):
        """Test text embedding step"""
        from sdg_core_lib.preprocess.strategies.steps import TextEmbeddingStep
        
        # First tokenize
        tokenization_step = TextTokenizationStep(position=0, col_name="test_text")
        tokenized = tokenization_step.fit_transform(self.text_column.get_data())
        
        # Then embed
        embedding_step = TextEmbeddingStep(
            position=0,
            col_name="test_text",
            embedding_dim=16
        )
        
        embedded = embedding_step.fit_transform(tokenized)
        
        # Check embedding dimensions
        assert embedded.shape[0] == len(self.test_texts)  # Same number of samples
        assert embedded.shape[1] == 16                    # Correct embedding dimension
    
    def test_complete_processor(self):
        """Test complete text processor"""
        # Process
        processed_columns = self.processor.process_text_columns([self.text_column])
        
        # Check that processing occurred
        assert processed_columns[0].get_data().shape[1] == 32  # Embedding dimension
        assert processed_columns[0].get_data().shape[0] == len(self.test_texts)
        
        # Get statistics
        stats = self.processor.get_text_statistics([self.text_column])
        assert stats["total_text_columns"] == 1
        assert stats["processed_columns"] == 1
        assert stats["embedding_dim"] == 32
    
    def test_inverse_processing(self):
        """Test inverse processing"""
        # Process
        processed_columns = self.processor.process_text_columns([self.text_column])
        
        # Inverse process
        reconstructed_columns = self.processor.inverse_process_text_columns(processed_columns)
        
        # Check that we get text back (approximately)
        assert reconstructed_columns[0].get_data().shape == self.text_column.get_data().shape
        assert isinstance(reconstructed_columns[0], TextColumn)
    
    def test_processor_persistence(self):
        """Test saving and loading processor state"""
        # Process data (this saves the fitted operators)
        processed_columns = self.processor.process_text_columns([self.text_column])
        
        # Create new processor instance
        new_processor = TextProcessor(
            dir_path="/tmp/test_text_processor",
            remove_stopwords=True,
            embedding_method="tfidf",
            max_features=100,
            embedding_dim=32
        )
        
        # Load and process with new instance
        new_processed_columns = new_processor.process_text_columns([self.text_column])
        
        # Results should be identical
        np.testing.assert_array_almost_equal(
            processed_columns[0].get_data(),
            new_processed_columns[0].get_data()
        )
```

## Best Practices

1. **Modular Design**: Create separate, focused steps for each transformation
2. **Reversibility**: Implement meaningful inverse transforms where possible
3. **Persistence**: Ensure steps can be saved and loaded reliably
4. **Error Handling**: Handle edge cases and invalid data gracefully
5. **Performance**: Optimize for large datasets and memory efficiency
6. **Documentation**: Document parameters, behavior, and limitations
7. **Testing**: Test with various data types and edge cases
8. **Compatibility**: Ensure compatibility with existing datasets and models



This comprehensive guide provides the foundation for creating custom processors with detailed examples. The text processor demonstrates best practices for building modular, reversible, and persistent preprocessing pipelines that integrate seamlessly with the GENESIS Core Lib architecture.
