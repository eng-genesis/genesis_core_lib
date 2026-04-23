# Custom Processors

## Overview

This guide explains how to create custom data processors for the GENESIS Core Lib. Custom processors allow you to implement specialized data preprocessing and postprocessing logic beyond the built-in processors.

## Base Processor Classes

All custom processors should inherit from the base classes in `src/sdg_core_lib/preprocess/`:

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any, Optional
from sdg_core_lib.preprocess.strategies.base_strategy import BasePreprocessingStrategy
from sdg_core_lib.preprocess.strategies.steps import Step

class Processor(ABC):
    """Abstract base class for all processors"""
    
    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.steps: Dict[int, List[Step]] = {}
        self.idx_to_data: Dict[int, int] = {}
        self.strategy: BasePreprocessingStrategy = BasePreprocessingStrategy()

    @abstractmethod
    def _init_steps(self, data: List):
        """Initialize processing steps for the given data"""
        raise NotImplementedError

    def set_strategy(self, strategy: BasePreprocessingStrategy) -> "Processor":
        """Set the preprocessing strategy"""
        self.strategy = strategy
        return self

    def add_steps(self, steps: List[Step], col_position: int, data_position: int) -> "Processor":
        """Add processing steps for a specific column"""
        self.steps[col_position] = steps
        self.idx_to_data[col_position] = data_position
        return self

    def save_all(self):
        """Save all processing steps"""
        for step_list in self.steps.values():
            for step in step_list:
                step.save_if_not_exist(self.dir_path)

    def load_all(self) -> "Processor":
        """Load all processing steps"""
        for step_list in self.steps.values():
            for step in step_list:
                step.load(self.dir_path)
        return self

    def process(self, data: List) -> Dict[int, np.ndarray]:
        """Apply preprocessing to data"""
        results = {}
        for idx, step_list in self.steps.items():
            processed_data = data[self.idx_to_data[idx]]
            for step in step_list:
                processed_data = step.fit_transform(processed_data)
            results[idx] = processed_data
        self.save_all()
        return results

    def inverse_process(self, data: List) -> Dict[int, np.ndarray]:
        """Apply inverse preprocessing to data"""
        self.load_all()
        results = {}
        for idx, step_list in self.steps.items():
            processed_data = data[self.idx_to_data[idx]]
            for step in reversed(step_list):
                processed_data = step.inverse_transform(processed_data)
            results[idx] = processed_data
        return results
```

## Creating Custom Processors

### Example 1: Text Preprocessing Processor

```python
import re
import pickle
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sdg_core_lib.preprocess.base_processor import Processor
from sdg_core_lib.preprocess.strategies.steps import Step
from sdg_core_lib.dataset.columns import Column

class TextPreprocessingStep(Step):
    """Custom step for text preprocessing"""
    
    def __init__(self, remove_stopwords: bool = True, min_word_length: int = 2):
        super().__init__()
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']) if remove_stopwords else set()
        self.fitted = False

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform text data"""
        processed_texts = []
        
        for text in data.flatten():
            # Convert to string if needed
            text_str = str(text)
            
            # Remove special characters and convert to lowercase
            cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text_str.lower())
            
            # Tokenize and filter
            words = cleaned_text.split()
            if self.remove_stopwords:
                words = [word for word in words if word not in self.stopwords]
            words = [word for word in words if len(word) >= self.min_word_length]
            
            processed_text = ' '.join(words)
            processed_texts.append(processed_text)
        
        self.fitted = True
        return np.array(processed_texts).reshape(-1, 1)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform (returns processed text as-is)"""
        return data  # Text preprocessing is not easily reversible

    def save(self, filepath: str):
        """Save step parameters"""
        params = {
            'remove_stopwords': self.remove_stopwords,
            'min_word_length': self.min_word_length,
            'stopwords': list(self.stopwords),
            'fitted': self.fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)

    def load(self, filepath: str):
        """Load step parameters"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.remove_stopwords = params['remove_stopwords']
        self.min_word_length = params['min_word_length']
        self.stopwords = set(params['stopwords'])
        self.fitted = params['fitted']

class TFIDFVectorizationStep(Step):
    """Custom step for TF-IDF vectorization"""
    
    def __init__(self, max_features: int = 1000, ngram_range: tuple = (1, 2)):
        super().__init__()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.fitted = False

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit TF-IDF vectorizer and transform data"""
        texts = data.flatten().tolist()
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.fitted = True
        return tfidf_matrix.toarray()

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform TF-IDF (approximate)"""
        # Find the most important terms for each document
        feature_names = self.vectorizer.get_feature_names_out()
        reconstructed_texts = []
        
        for doc_vector in data:
            # Get top terms
            top_indices = doc_vector.argsort()[-5:][::-1]  # Top 5 terms
            top_terms = [feature_names[i] for i in top_indices if doc_vector[i] > 0]
            reconstructed_texts.append(' '.join(top_terms))
        
        return np.array(reconstructed_texts).reshape(-1, 1)

    def save(self, filepath: str):
        """Save vectorizer"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'max_features': self.max_features,
                'ngram_range': self.ngram_range,
                'fitted': self.fitted
            }, f)

    def load(self, filepath: str):
        """Load vectorizer"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.vectorizer = params['vectorizer']
        self.max_features = params['max_features']
        self.ngram_range = params['ngram_range']
        self.fitted = params['fitted']

class TextProcessor(Processor):
    """Custom processor for text data"""
    
    def __init__(self, dir_path: str, max_features: int = 1000, 
                 remove_stopwords: bool = True, min_word_length: int = 2):
        super().__init__(dir_path)
        self.max_features = max_features
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length

    def _init_steps(self, data: List[Column]):
        """Initialize text processing steps"""
        for i, column in enumerate(data):
            if column.column_type == "text":
                # Add text preprocessing steps
                steps = [
                    TextPreprocessingStep(
                        remove_stopwords=self.remove_stopwords,
                        min_word_length=self.min_word_length
                    ),
                    TFIDFVectorizationStep(max_features=self.max_features)
                ]
                self.add_steps(steps, i, i)

    def process_text_column(self, column: Column) -> np.ndarray:
        """Process a single text column"""
        steps = [
            TextPreprocessingStep(
                remove_stopwords=self.remove_stopwords,
                min_word_length=self.min_word_length
            ),
            TFIDFVectorizationStep(max_features=self.max_features)
        ]
        
        data = column.get_data()
        for step in steps:
            data = step.fit_transform(data)
        
        return data
```

### Example 2: Image Preprocessing Processor

```python
import cv2
import numpy as np
from PIL import Image
import pickle
from typing import Tuple, Optional
from sdg_core_lib.preprocess.base_processor import Processor
from sdg_core_lib.preprocess.strategies.steps import Step

class ImageResizeStep(Step):
    """Custom step for image resizing"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), 
                 interpolation: str = 'bilinear'):
        super().__init__()
        self.target_size = target_size
        self.interpolation = interpolation
        self.fitted = False

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Resize images to target size"""
        resized_images = []
        
        for img_data in data.flatten():
            # Convert bytes to PIL Image
            img = Image.open(io.BytesIO(img_data))
            
            # Resize image
            resized_img = img.resize(self.target_size, Image.Resampling.BILINEAR)
            
            # Convert back to bytes
            img_bytes = io.BytesIO()
            resized_img.save(img_bytes, format='JPEG')
            resized_images.append(img_bytes.getvalue())
        
        self.fitted = True
        return np.array(resized_images).reshape(-1, 1)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform (cannot restore original size)"""
        return data  # Size reduction is not reversible

    def save(self, filepath: str):
        """Save step parameters"""
        params = {
            'target_size': self.target_size,
            'interpolation': self.interpolation,
            'fitted': self.fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)

    def load(self, filepath: str):
        """Load step parameters"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.target_size = params['target_size']
        self.interpolation = params['interpolation']
        self.fitted = params['fitted']

class ImageNormalizationStep(Step):
    """Custom step for image normalization"""
    
    def __init__(self, method: str = 'standard', target_range: tuple = (0, 1)):
        super().__init__()
        self.method = method
        self.target_range = target_range
        self.mean = None
        self.std = None
        self.fitted = False

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Normalize image pixel values"""
        # Convert image bytes to numpy arrays
        img_arrays = []
        for img_data in data.flatten():
            img = Image.open(io.BytesIO(img_data))
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            img_arrays.append(img_array)
        
        img_arrays = np.array(img_arrays)
        
        if self.method == 'standard':
            # Standard normalization
            self.mean = np.mean(img_arrays)
            self.std = np.std(img_arrays)
            normalized = (img_arrays - self.mean) / (self.std + 1e-8)
        elif self.method == 'minmax':
            # Min-max normalization to target range
            min_val, max_val = self.target_range
            normalized = img_arrays * (max_val - min_val) + min_val
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        self.fitted = True
        return normalized.reshape(len(img_arrays), -1)  # Flatten for ML models

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse normalization"""
        if self.method == 'standard':
            denormalized = data * (self.std + 1e-8) + self.mean
        elif self.method == 'minmax':
            min_val, max_val = self.target_range
            denormalized = (data - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        # Clip values and convert back to image format
        denormalized = np.clip(denormalized, 0, 1) * 255
        return denormalized.astype(np.uint8)

    def save(self, filepath: str):
        """Save normalization parameters"""
        params = {
            'method': self.method,
            'target_range': self.target_range,
            'mean': self.mean,
            'std': self.std,
            'fitted': self.fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)

    def load(self, filepath: str):
        """Load normalization parameters"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.method = params['method']
        self.target_range = params['target_range']
        self.mean = params['mean']
        self.std = params['std']
        self.fitted = params['fitted']

class ImageProcessor(Processor):
    """Custom processor for image data"""
    
    def __init__(self, dir_path: str, target_size: Tuple[int, int] = (224, 224),
                 normalization: str = 'standard'):
        super().__init__(dir_path)
        self.target_size = target_size
        self.normalization = normalization

    def _init_steps(self, data: List[Column]):
        """Initialize image processing steps"""
        for i, column in enumerate(data):
            if column.column_type == "image":
                steps = [
                    ImageResizeStep(target_size=self.target_size),
                    ImageNormalizationStep(method=self.normalization)
                ]
                self.add_steps(steps, i, i)
```

### Example 3: Time Series Specialized Processor

```python
from scipy import signal
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import List, Dict, Optional
from sdg_core_lib.preprocess.base_processor import Processor
from sdg_core_lib.preprocess.strategies.steps import Step

class TimeSeriesDetrendingStep(Step):
    """Custom step for time series detrending"""
    
    def __init__(self, method: str = 'linear'):
        super().__init__()
        self.method = method
        self.trend_params = None
        self.fitted = False

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Remove trend from time series data"""
        detrended_series = []
        
        for series in data:
            if self.method == 'linear':
                # Linear detrending
                x = np.arange(len(series))
                coeffs = np.polyfit(x, series, 1)
                trend = np.polyval(coeffs, x)
                detrended = series - trend
                self.trend_params = coeffs
            elif self.method == 'mean':
                # Mean detrending
                mean_val = np.mean(series)
                detrended = series - mean_val
                self.trend_params = mean_val
            else:
                raise ValueError(f"Unknown detrending method: {self.method}")
            
            detrended_series.append(detrended)
        
        self.fitted = True
        return np.array(detrended_series)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Add trend back to detrended data"""
        if self.method == 'linear' and self.trend_params is not None:
            # Restore linear trend
            restored_series = []
            for i, series in enumerate(data):
                x = np.arange(len(series))
                trend = np.polyval(self.trend_params, x)
                restored = series + trend
                restored_series.append(restored)
            return np.array(restored_series)
        elif self.method == 'mean' and self.trend_params is not None:
            # Restore mean
            return data + self.trend_params
        else:
            return data

    def save(self, filepath: str):
        """Save detrending parameters"""
        params = {
            'method': self.method,
            'trend_params': self.trend_params,
            'fitted': self.fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)

    def load(self, filepath: str):
        """Load detrending parameters"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.method = params['method']
        self.trend_params = params['trend_params']
        self.fitted = params['fitted']

class TimeSeriesFilteringStep(Step):
    """Custom step for time series filtering"""
    
    def __init__(self, filter_type: str = 'lowpass', cutoff: float = 0.1, 
                 order: int = 4):
        super().__init__()
        self.filter_type = filter_type
        self.cutoff = cutoff
        self.order = order
        self.fitted = False

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply filter to time series data"""
        filtered_series = []
        
        for series in data:
            if self.filter_type == 'lowpass':
                # Low-pass Butterworth filter
                b, a = signal.butter(self.order, self.cutoff, btype='low')
                filtered = signal.filtfilt(b, a, series)
            elif self.filter_type == 'highpass':
                # High-pass Butterworth filter
                b, a = signal.butter(self.order, self.cutoff, btype='high')
                filtered = signal.filtfilt(b, a, series)
            elif self.filter_type == 'bandpass':
                # Band-pass filter
                b, a = signal.butter(self.order, [self.cutoff/2, self.cutoff*2], btype='band')
                filtered = signal.filtfilt(b, a, series)
            else:
                raise ValueError(f"Unknown filter type: {self.filter_type}")
            
            filtered_series.append(filtered)
        
        self.fitted = True
        return np.array(filtered_series)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Filtering is not easily reversible"""
        return data

    def save(self, filepath: str):
        """Save filter parameters"""
        params = {
            'filter_type': self.filter_type,
            'cutoff': self.cutoff,
            'order': self.order,
            'fitted': self.fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)

    def load(self, filepath: str):
        """Load filter parameters"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.filter_type = params['filter_type']
        self.cutoff = params['cutoff']
        self.order = params['order']
        self.fitted = params['fitted']

class TimeSeriesProcessor(Processor):
    """Custom processor for time series data"""
    
    def __init__(self, dir_path: str, detrending: str = 'linear',
                 filtering: Optional[Dict] = None):
        super().__init__(dir_path)
        self.detrending = detrending
        self.filtering = filtering or {}

    def _init_steps(self, data: List[Column]):
        """Initialize time series processing steps"""
        for i, column in enumerate(data):
            if column.column_type in ["continuous", "time_series"]:
                steps = []
                
                # Add detrending step
                if self.detrending:
                    steps.append(TimeSeriesDetrendingStep(method=self.detrending))
                
                # Add filtering step if specified
                if self.filtering:
                    steps.append(TimeSeriesFilteringStep(**self.filtering))
                
                if steps:
                    self.add_steps(steps, i, i)
```

## Custom Strategies

Create custom preprocessing strategies for different data types:

```python
from sdg_core_lib.preprocess.strategies.base_strategy import BasePreprocessingStrategy

class AdvancedPreprocessingStrategy(BasePreprocessingStrategy):
    """Custom strategy with advanced preprocessing options"""
    
    def __init__(self, handle_outliers: bool = True, 
                 feature_engineering: bool = True):
        super().__init__()
        self.handle_outliers = handle_outliers
        self.feature_engineering = feature_engineering
    
    def get_steps_for_column(self, column_type: str, column_data: np.ndarray) -> List[Step]:
        """Get appropriate steps for a column based on its type and data"""
        steps = []
        
        if column_type == "continuous":
            if self.handle_outliers:
                steps.append(OutlierRemovalStep(method='iqr'))
            
            steps.append(StandardScalerStep())
            
            if self.feature_engineering:
                steps.append(PolynomialFeaturesStep(degree=2))
                
        elif column_type == "categorical":
            steps.append(OneHotEncoderStep(handle_unknown='ignore'))
            
        elif column_type == "text":
            steps.extend([
                TextPreprocessingStep(remove_stopwords=True),
                TFIDFVectorizationStep(max_features=1000)
            ])
            
        elif column_type == "datetime":
            steps.extend([
                DateTimeFeatureExtractionStep(),
                StandardScalerStep()
            ])
        
        return steps
```

## Integration with Existing Systems

### Register Custom Processors

```python
# In your module's __init__.py
from .custom_processors import TextProcessor, ImageProcessor, TimeSeriesProcessor

PROCESSOR_REGISTRY = {
    "table": TableProcessor,
    "text": TextProcessor,
    "image": ImageProcessor,
    "time_series": TimeSeriesProcessor,
}
```

### Factory Pattern Implementation

```python
class ProcessorFactory:
    """Factory for creating processors"""
    
    @staticmethod
    def create_processor(processor_type: str, **kwargs) -> Processor:
        """Create processor based on type"""
        if processor_type == "text":
            return TextProcessor(**kwargs)
        elif processor_type == "image":
            return ImageProcessor(**kwargs)
        elif processor_type == "time_series":
            return TimeSeriesProcessor(**kwargs)
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")
```

## Testing Custom Processors

```python
import pytest
import numpy as np
from your_module import TextProcessor, ImageProcessor, TimeSeriesProcessor

class TestCustomProcessors:
    def test_text_processor(self):
        """Test text processing functionality"""
        texts = np.array(["Hello world", "This is a test", "Another example"])
        
        processor = TextProcessor("/tmp/test", max_features=100)
        
        # Test processing
        processed = processor.process_text_column(texts)
        assert processed.shape[1] <= 100  # TF-IDF features limit
        
        # Test inverse processing
        # Note: Text processing may not be perfectly reversible
    
    def test_image_processor(self):
        """Test image processing functionality"""
        # Create dummy image data
        image_data = np.array([b'\x89PNG\r\n\x1a\n...'] * 3)  # Simplified
        
        processor = ImageProcessor("/tmp/test", target_size=(64, 64))
        
        # Test processing steps
        # Implementation depends on actual image data format
    
    def test_time_series_processor(self):
        """Test time series processing functionality"""
        # Create sample time series data
        ts_data = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        ])
        
        processor = TimeSeriesProcessor("/tmp/test", detrending='linear')
        
        # Test processing
        # Implementation depends on specific step implementations
```

## Best Practices

1. **Modular Design**: Create separate steps for each transformation
2. **Reversibility**: Implement inverse transforms where possible
3. **Persistence**: Save and load processor state properly
4. **Error Handling**: Handle edge cases and invalid data gracefully
5. **Performance**: Optimize for large datasets
6. **Documentation**: Document parameters and behavior
7. **Testing**: Test with various data types and edge cases
8. **Compatibility**: Ensure compatibility with existing datasets and models

## Advanced Features

### 1. Conditional Processing

```python
class ConditionalStep(Step):
    """Step that applies processing based on conditions"""
    
    def __init__(self, condition_func, true_step, false_step=None):
        super().__init__()
        self.condition_func = condition_func
        self.true_step = true_step
        self.false_step = false_step
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        if self.condition_func(data):
            return self.true_step.fit_transform(data)
        elif self.false_step:
            return self.false_step.fit_transform(data)
        else:
            return data
```

### 2. Ensemble Processing

```python
class EnsembleProcessor(Processor):
    """Processor that applies multiple processing strategies"""
    
    def __init__(self, dir_path: str, processors: List[Processor]):
        super().__init__(dir_path)
        self.processors = processors
    
    def process(self, data: List) -> Dict[int, np.ndarray]:
        results = {}
        for processor in self.processors:
            processor_results = processor.process(data)
            results.update(processor_results)
        return results
```

### 3. Adaptive Processing

```python
class AdaptiveProcessor(Processor):
    """Processor that adapts based on data characteristics"""
    
    def _analyze_data(self, data: List[Column]) -> Dict[str, Any]:
        """Analyze data characteristics"""
        analysis = {}
        
        for column in data:
            col_analysis = {
                'data_type': column.column_type,
                'missing_ratio': self._calculate_missing_ratio(column),
                'outlier_ratio': self._calculate_outlier_ratio(column),
                'cardinality': self._calculate_cardinality(column)
            }
            analysis[column.name] = col_analysis
        
        return analysis
    
    def _init_steps(self, data: List[Column]):
        """Initialize steps based on data analysis"""
        analysis = self._analyze_data(data)
        
        for i, column in enumerate(data):
            steps = self._determine_steps(column, analysis[column.name])
            if steps:
                self.add_steps(steps, i, i)
```

This guide provides the foundation for creating custom processors. Adapt the examples to your specific preprocessing requirements and data types.
