# Custom Data Types

## Overview

This guide explains how to create custom column types for the GENESIS Core Lib. Custom data types allow you to support specialized data formats and validation beyond the built-in Numeric, Categorical, and basic Column types.

## Base Column Classes

All custom column types should inherit from the base classes in `src/sdg_core_lib/dataset/columns.py`:

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

class Numeric(Column):
    """Numeric column for continuous data"""
    
    def get_boundaries(self) -> tuple[float, float]:
        """Get min/max values"""
        return self.values.min(), self.values.max()

    def to_categorical(self, n_bins: int = 10) -> "Categorical":
        """Convert numeric to categorical via binning"""
        bins = np.linspace(self.values.min(), self.values.max(), n_bins)
        binned_values = np.digitize(self.values, bins)
        return Categorical(
            self.name, self.value_type, self.position, binned_values, "categorical"
        )

class Categorical(Column):
    """Categorical column for discrete data"""
    
    def get_categories(self) -> list[str]:
        """Get unique categories"""
        seen = {}
        unique = []
        for o in self.values.reshape(-1):
            if str(o) not in seen:
                seen[str(o)] = True
                unique.append(str(o))
        return unique

    def to_numeric(self) -> "Numeric":
        """Convert categorical to numeric via encoding"""
        all_categories = self.get_categories()
        category_mapping = {category: i for i, category in enumerate(all_categories)}
        return Numeric(
            self.name,
            self.value_type,
            self.position,
            np.array([category_mapping[str(category)] 
                     for category in self.values.reshape(-1)]).reshape(self.values.shape),
            "numeric",
        )
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

    def get_metadata(self) -> dict:
        """Get extended metadata for datetime column"""
        metadata = super().get_metadata()
        metadata.update({
            "column_type": "datetime",
            "timezone": self.timezone_info,
            "min_date": str(self.get_min_date()),
            "max_date": str(self.get_max_date()),
            "date_range_days": self.get_date_range_days(),
        })
        return metadata

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

    def to_timestamp(self) -> "Numeric":
        """Convert to Unix timestamp (numeric)"""
        datetime_series = pd.to_datetime(self.values.flatten())
        timestamps = datetime_series.astype(np.int64) // 10**9  # Convert to seconds
        
        return Numeric(
            f"{self.name}_timestamp",
            "int64",
            self.position,
            timestamps.reshape(self.values.shape),
            "timestamp"
        )

    def get_data(self) -> np.ndarray:
        """Get data in datetime format"""
        return self.values

    def get_internal_shape(self) -> tuple[int, ...]:
        """Get internal shape"""
        return self.values.shape
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
            "column_type": "text",
            "max_length": self.max_length,
            "min_length": self.min_length,
            "language": self.language,
            "avg_length": self.get_average_length(),
            "unique_vocab_size": self.get_vocab_size(),
            "total_tokens": self.get_total_tokens(),
        })
        return metadata

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

    def tokenize(self, method: str = "simple") -> np.ndarray:
        """Tokenize text data"""
        if method == "simple":
            tokens = [text.split() for text in self.values.flatten()]
        elif method == "character":
            tokens = [list(text) for text in self.values.flatten()]
        else:
            raise ValueError(f"Unknown tokenization method: {method}")
        
        return np.array(tokens, dtype=object)

    def get_length_distribution(self) -> Dict[str, int]:
        """Get distribution of text lengths"""
        lengths = [len(text) for text in self.values.flatten()]
        return {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "median": sorted(lengths)[len(lengths) // 2],
        }

    def filter_by_length(self, min_length: int, max_length: int) -> np.ndarray:
        """Filter texts by length range"""
        mask = np.array([
            min_length <= len(text) <= max_length 
            for text in self.values.flatten()
        ])
        return mask

    def get_data(self) -> np.ndarray:
        """Get text data"""
        return self.values
```

### Example 3: Geographic Coordinate Column

```python
import math
from typing import Tuple
from sdg_core_lib.dataset.columns import Column

class GeoCoordinateColumn(Column):
    """Custom column for geographic coordinates (latitude, longitude)"""
    
    def __init__(
        self,
        name: str,
        value_type: str,
        position: int,
        values: np.ndarray,
        column_type: str = "geo_coordinate",
        coordinate_system: str = "WGS84",
    ):
        super().__init__(name, value_type, position, values, column_type)
        self.coordinate_system = coordinate_system
        self._validate_coordinates()

    def _validate_coordinates(self):
        """Validate latitude/longitude ranges"""
        if self.values.shape[1] != 2:
            raise ValueError("Geo coordinates must have exactly 2 columns (lat, lon)")
        
        flat_coords = self.values.reshape(-1, 2)
        
        for i, (lat, lon) in enumerate(flat_coords):
            if not (-90 <= lat <= 90):
                raise ValueError(f"Row {i}: Invalid latitude {lat} (must be -90 to 90)")
            if not (-180 <= lon <= 180):
                raise ValueError(f"Row {i}: Invalid longitude {lon} (must be -180 to 180)")

    def get_metadata(self) -> dict:
        """Get extended metadata for geo coordinates"""
        metadata = super().get_metadata()
        metadata.update({
            "column_type": "geo_coordinate",
            "coordinate_system": self.coordinate_system,
            "bounds": self.get_bounds(),
            "center_point": self.get_center_point(),
            "total_area_km2": self.estimate_area(),
        })
        return metadata

    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get bounding box of coordinates"""
        flat_coords = self.values.reshape(-1, 2)
        lats = flat_coords[:, 0]
        lons = flat_coords[:, 1]
        
        return {
            "lat_range": (float(lats.min()), float(lats.max())),
            "lon_range": (float(lons.min()), float(lons.max())),
        }

    def get_center_point(self) -> Tuple[float, float]:
        """Get center point of all coordinates"""
        flat_coords = self.values.reshape(-1, 2)
        return (
            float(flat_coords[:, 0].mean()),
            float(flat_coords[:, 1].mean())
        )

    def estimate_area(self) -> float:
        """Estimate area covered by coordinates (simplified)"""
        bounds = self.get_bounds()
        lat_range = bounds["lat_range"][1] - bounds["lat_range"][0]
        lon_range = bounds["lon_range"][1] - bounds["lon_range"][0]
        
        # Very rough estimation (not accounting for Earth's curvature)
        return lat_range * lon_range * 111 * 111  # Approximate km² per degree

    def calculate_distances(self, reference_point: Tuple[float, float]) -> np.ndarray:
        """Calculate distances from a reference point using Haversine formula"""
        flat_coords = self.values.reshape(-1, 2)
        ref_lat, ref_lon = reference_point
        
        distances = []
        for lat, lon in flat_coords:
            dist = self._haversine_distance(ref_lat, ref_lon, lat, lon)
            distances.append(dist)
        
        return np.array(distances).reshape(-1, 1)

    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate Haversine distance between two points"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c

    def cluster_by_region(self, grid_size: float = 0.1) -> np.ndarray:
        """Cluster coordinates by geographic grid"""
        flat_coords = self.values.reshape(-1, 2)
        
        # Create grid cells
        lat_grid = (flat_coords[:, 0] // grid_size).astype(int)
        lon_grid = (flat_coords[:, 1] // grid_size).astype(int)
        
        # Combine lat and lon grid indices
        grid_ids = lat_grid * 10000 + lon_grid  # Unique grid identifier
        
        return grid_ids.reshape(-1, 1)

    def get_data(self) -> np.ndarray:
        """Get coordinate data"""
        return self.values
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
        "geo_coordinate": GeoCoordinateColumn,  # Custom type
    }
```

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
from your_module import DateTimeColumn, TextColumn, GeoCoordinateColumn

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
    
    def test_geo_coordinate_column(self):
        """Test geo coordinate column functionality"""
        coords = np.array([
            [40.7128, -74.0060],  # New York
            [34.0522, -118.2437], # Los Angeles
            [51.5074, -0.1278]    # London
        ])
        
        col = GeoCoordinateColumn("coords", "float64", 0, coords)
        
        bounds = col.get_bounds()
        assert "lat_range" in bounds
        assert "lon_range" in bounds
        
        # Test distance calculation
        distances = col.calculate_distances((40.7128, -74.0060))
        assert distances[0] == 0  # Distance to self is 0
        assert distances[1] > 0  # Distance to other points > 0
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
- **Visualization**: Consider how your data will be visualized
- **Export Formats**: Support various export formats (CSV, JSON, etc.)

This guide provides the foundation for creating custom data types. Adapt the examples to your specific data requirements and use cases.
