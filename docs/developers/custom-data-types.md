# Custom Data Types

## Overview

This guide explains how to create custom column types for the GENESIS Core Lib. Custom data types allow you to support specialized data formats and validation beyond the built-in Numeric, Categorical, and basic Column types.

## Base Column Classes

All custom column types should inherit from the Column Class in `src/sdg_core_lib/dataset/columns.py`:

```python
import numpy as np
from typing import Any, Dict, List, Optional, Union

class Column:
    """Base column class for all data types"""
    
    def __init__(
        self,
        name: str,
        value_type: str,
        position: int,
        values: np.ndarray,
        column_type: str,
    ):
        self.name = name
        self.value_type = value_type
        self.position = position
        self.values = values
        self.column_type = column_type
        self.internal_shape = self.get_internal_shape()

    def get_internal_shape(self) -> tuple[int, ...]:
        """Get the internal shape of the column data"""
        return self.values.shape

    def get_metadata(self) -> dict:
        """Get column metadata"""
        return {
            "name": self.name,
            "value_type": self.value_type,
            "position": self.position,
            "internal_shape": self.internal_shape,
        }

    def get_data(self) -> np.ndarray:
        """Get column data as numpy array"""
        return self.values

```

## Creating Custom Data Types

### Example 1: DateTime Column

```python
from datetime import datetime, timezone
import pandas as pd
from sdg_core_lib.dataset.columns import Column

class DateTimeColumn(Column):
    """Custom column for datetime data"""
    
    def __init__(
        self,
        name: str,
        value_type: str,
        position: int,
        values: np.ndarray,
        column_type: str = "datetime",
        timezone_info: Optional[str] = None,
    ):
        super().__init__(name, value_type, position, values, column_type)
        self.timezone_info = timezone_info
        self._validate_datetime_values()

    def _validate_datetime_values(self):
        """Validate that all values are valid datetime representations"""
        try:
            # Convert to datetime objects for validation
            datetime_values = pd.to_datetime(self.values.flatten())
            if datetime_values.isna().any():
                raise ValueError("Invalid datetime values found")
        except Exception as e:
            raise ValueError(f"Invalid datetime data: {e}")

    # Helper methods ---------------------
    def get_min_date(self) -> datetime:
        """Get minimum datetime value"""
        return pd.to_datetime(self.values.flatten()).min()

    def get_max_date(self) -> datetime:
        """Get maximum datetime value"""
        return pd.to_datetime(self.values.flatten()).max()

    def get_date_range_days(self) -> int:
        """Get date range in days"""
        return (self.get_max_date() - self.get_min_date()).days

    def extract_features(self) -> Dict[str, np.ndarray]:
        """Extract datetime features for ML models"""
        datetime_series = pd.to_datetime(self.values.flatten())
        
        features = {
            "year": datetime_series.dt.year.values.reshape(-1, 1),
            "month": datetime_series.dt.month.values.reshape(-1, 1),
            "day": datetime_series.dt.day.values.reshape(-1, 1),
            "hour": datetime_series.dt.hour.values.reshape(-1, 1),
            "day_of_week": datetime_series.dt.dayofweek.values.reshape(-1, 1),
            "day_of_year": datetime_series.dt.dayofyear.values.reshape(-1, 1),
            "quarter": datetime_series.dt.quarter.values.reshape(-1, 1),
        }
        
        return features

    # Example Conversion --------------------------
    def to_timestamp(self) -> "Numeric":
        """Convert to Unix timestamp (numeric)"""
        datetime_series = pd.to_datetime(self.values.flatten())
        timestamps = datetime_series.astype(np.int64) // 10**9  # Convert to seconds
        
        return Numeric(
            f"{self.name}_timestamp",
            "int64",
            self.position,
            timestamps.reshape(self.values.shape),
            "numeric"
        )
    
    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        metadata.update({"column_type": "datetime"})
        return metadata

    def get_data(self) -> np.ndarray:
        # Already Defined in base class, do not override

    def get_internal_shape(self) -> tuple[int, ...]:
        # Already Defined in base class, do not override
```

### Example 2: Text Column

```python
from typing import List, Optional
import re
from collections import Counter
from sdg_core_lib.dataset.columns import Column

class TextColumn(Column):
    """Custom column for text data"""
    
    def __init__(
        self,
        name: str,
        value_type: str,
        position: int,
        values: np.ndarray,
        column_type: str = "text",
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        language: Optional[str] = None,
    ):
        super().__init__(name, value_type, position, values, column_type)
        self.max_length = max_length
        self.min_length = min_length
        self.language = language
        self._validate_text_values()

    def _validate_text_values(self):
        """Validate text values"""
        flat_values = self.values.flatten()
        
        for i, text in enumerate(flat_values):
            if not isinstance(text, str):
                raise ValueError(f"Row {i}: Expected string, got {type(text)}")
            
            if self.min_length and len(text) < self.min_length:
                raise ValueError(f"Row {i}: Text too short (min: {self.min_length})")
            
            if self.max_length and len(text) > self.max_length:
                raise ValueError(f"Row {i}: Text too long (max: {self.max_length})")

    def get_metadata(self) -> dict:
        """Get extended metadata for text column"""
        metadata = super().get_metadata()
        metadata.update({
            "column_type": "text"})
        return metadata

    # Helper methods ---------------------
    def get_average_length(self) -> float:
        """Get average text length"""
        lengths = [len(text) for text in self.values.flatten()]
        return sum(lengths) / len(lengths) if lengths else 0

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        all_text = " ".join(self.values.flatten())
        words = re.findall(r'\b\w+\b', all_text.lower())
        return len(set(words))

    def get_total_tokens(self) -> int:
        """Get total token count"""
        all_text = " ".join(self.values.flatten())
        return len(re.findall(r'\b\w+\b', all_text))

    def get_word_frequencies(self) -> Dict[str, int]:
        """Get word frequency distribution"""
        all_text = " ".join(self.values.flatten())
        words = re.findall(r'\b\w+\b', all_text.lower())
        return dict(Counter(words))

    def get_length_distribution(self) -> Dict[str, int]:
        """Get distribution of text lengths"""
        lengths = [len(text) for text in self.values.flatten()]
        return {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "median": sorted(lengths)[len(lengths) // 2],
        }

    def get_data(self) -> np.ndarray:
        # Already Defined in base class, do not override

    def get_internal_shape(self) -> tuple[int, ...]:
        # Already Defined in base class, do not override
```


## Integration with Dataset Classes

To use custom column types with existing datasets, update the column registry:

```python
# In Table class
class Table(Dataset):
    col_registry = {
        "continuous": Numeric,
        "categorical": Categorical,
        "primary_key": Column,
        "group_index": Column,
        "datetime": DateTimeColumn,      # Custom type
        "text": TextColumn,                # Custom type
    }
```

⚠️ Ensure methods are implemented in Dataset include column type for processing!

```python
# Example in Table Dataset Class
class Table(Dataset):
    
    def from_json(cls, json_data: list[dict]) -> "Table":
        pk_indexes = []
        group_index = None
        
        # ---Other Code Here---
        
        for idx, col_data in enumerate(json_data):
            ... 
            if col_type == "group_index":
                if group_index is not None:
                    raise ValueError("Group index already set")
                group_index = col_position
                pk_indexes.append(col_position)

            if col_type == "primary_key":
                pk_indexes.append(col_position)
                
        # ---Other Code Here---

        return Table(columns, pk_indexes)
    
    def get_primary_keys(self) -> list[Column]:
        return [col for col in self.columns if col.position in self.pk_col_indexes]

    def get_numeric_columns(self) -> list[Numeric]:
        return [col for col in self.columns if isinstance(col, Numeric)]

    def get_categorical_columns(self) -> list[Categorical]:
        return [col for col in self.columns if isinstance(col, Categorical)]

    def get_computing_data(self) -> np.ndarray:
        return np.hstack([col.get_data() for col in self._get_computing_columns()])

    def get_computing_shape(self) -> tuple[int, ...]:
        return self.get_computing_data().shape

    def _get_computing_columns(self):
        return [col for col in self.columns if col.position not in self.pk_col_indexes]
```

In such case, when column_type is both a group Index and a primary key, the column is added to both the group_index and pk_col_indexes.

⚠️ Ensure your addition don't break subclasses logic!

For instance, TimeSeries Datasets inherits from Table by adding custom logic. In such cases, be careful when inserting new columns in a dataset!



## Advanced Custom Column Features

### 1. Custom Validation

```python
class EmailColumn(Column):
    """Custom column for email addresses"""
    
    def _validate_email_format(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _validate_values(self):
        """Validate all email values"""
        for i, email in enumerate(self.values.flatten()):
            if not self._validate_email_format(email):
                raise ValueError(f"Row {i}: Invalid email format: {email}")
```

### 2. Custom Transformations

```python
class ImageColumn(Column):
    """Custom column for image data"""
    
    def resize_images(self, target_size: Tuple[int, int]) -> "ImageColumn":
        """Resize all images to target size"""
        from PIL import Image
        import io
        
        resized_images = []
        for img_data in self.values.flatten():
            img = Image.open(io.BytesIO(img_data))
            img_resized = img.resize(target_size)
            
            # Convert back to bytes
            img_bytes = io.BytesIO()
            img_resized.save(img_bytes, format='JPEG')
            resized_images.append(img_bytes.getvalue())
        
        return ImageColumn(
            self.name, self.value_type, self.position,
            np.array(resized_images), self.column_type
        )
    
    def extract_features(self, method: str = "histogram") -> np.ndarray:
        """Extract image features"""
        if method == "histogram":
            return self._extract_histogram_features()
        elif method == "size":
            return self._extract_size_features()
        else:
            raise ValueError(f"Unknown feature extraction method: {method}")
```

### 3. Custom Serialization

```python
class JSONColumn(Column):
    """Custom column for JSON data"""
    
    def to_json(self) -> list[dict]:
        """Convert to JSON format with proper serialization"""
        import json
        
        return [
            {
                "column_data": json.loads(item) if isinstance(item, str) else item,
                "column_name": self.name,
                "column_type": self.column_type,
                "column_datatype": self.value_type
            }
            for item in self.values.flatten()
        ]
    
    def flatten_json(self, separator: str = ".") -> "Table":
        """Flatten nested JSON into separate columns"""
        # Implementation to flatten JSON structure
        pass
```

## Testing Custom Data Types

Create comprehensive tests for your custom column types:

```python
import pytest
import numpy as np
from datetime import datetime
from sdg_core_lib.dataset.columns import DateTimeColumn, TextColumn

class TestCustomColumns:
    def test_datetime_column(self):
        """Test datetime column functionality"""
        dates = [
            "2023-01-01", 
            "2023-01-02", 
            "2023-01-03"
        ]
        
        col = DateTimeColumn("dates", "str", 0, np.array(dates))
        
        assert col.get_min_date() == datetime(2023, 1, 1)
        assert col.get_max_date() == datetime(2023, 1, 3)
        assert col.get_date_range_days() == 2
        
        # Test feature extraction
        features = col.extract_features()
        assert "year" in features
        assert "month" in features
    
    def test_text_column(self):
        """Test text column functionality"""
        texts = ["Hello world", "This is a test", "Another example"]
        
        col = TextColumn("texts", "str", 0, np.array(texts))
        
        assert col.get_average_length() > 0
        assert col.get_vocab_size() > 0
        assert len(col.get_word_frequencies()) > 0
    

```

## Best Practices

1. **Validation**: Always validate input data in the constructor
2. **Metadata**: Provide comprehensive metadata for better data understanding
3. **Type Safety**: Use proper type hints throughout your implementation
4. **Performance**: Consider memory efficiency for large datasets
5. **Compatibility**: Ensure compatibility with existing processors and models
6. **Documentation**: Add clear docstrings explaining the column's purpose
7. **Error Handling**: Provide meaningful error messages
8. **Testing**: Create comprehensive tests for all functionality

## Integration Considerations

When creating custom data types, consider:

- **Processor Compatibility**: Ensure your custom types work with existing processors
- **Model Integration**: Verify that models can handle your data format
- **Serialization**: Implement proper JSON serialization/deserialization

This guide provides the foundation for creating custom data types. Adapt the examples to your specific data requirements and use cases.
