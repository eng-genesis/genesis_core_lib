# Custom Datasets

## Overview

This guide explains how to create custom dataset types for the GENESIS Core Lib. Custom datasets allow you to support new data structures and formats beyond the built-in Table and TimeSeries datasets.

## Base Dataset Class

All custom datasets must inherit from the `Dataset` abstract base class located in `src/sdg_core_lib/dataset/datasets.py`:

```python
from abc import ABC, abstractmethod
import numpy as np

class Dataset(ABC):
    @classmethod
    @abstractmethod
    def from_json(cls, json_data: list[dict]) -> "Dataset":
        """Create dataset from JSON configuration"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_skeleton(cls, skeleton: list[dict]):
        """Create dataset from metadata skeleton"""
        raise NotImplementedError

    @abstractmethod
    def clone(self, new_data: np.ndarray) -> "Dataset":
        """Create a new dataset instance with different data"""
        raise NotImplementedError

    @abstractmethod
    def get_computing_data(self) -> np.ndarray:
        """Get data in format suitable for ML models"""
        raise NotImplementedError

    @abstractmethod
    def get_computing_shape(self) -> tuple[int, ...]:
        """Get shape of computing data"""
        raise NotImplementedError

    def get_shape_for_model(self) -> str:
        """Get shape string for model configuration"""
        return str(self.get_computing_shape()[1:])

    @abstractmethod
    def to_json(self) -> list[dict]:
        """Convert dataset to JSON format"""
        raise NotImplementedError

    @abstractmethod
    def to_skeleton(self) -> list[dict]:
        """Convert dataset to metadata skeleton"""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, processor: Processor) -> "Dataset":
        """Apply preprocessing to dataset"""
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, processor: Processor) -> "Dataset":
        """Apply postprocessing to dataset"""
        raise NotImplementedError
```

## Creating a Custom Dataset

### Example: Graph Dataset

Let's create a custom dataset for graph data structures:

```python
import numpy as np
from typing import Dict, List, Tuple
from sdg_core_lib.dataset.datasets import Dataset
from sdg_core_lib.dataset.columns import Column, Numeric, Categorical
from sdg_core_lib.preprocess.base_processor import Processor
from sdg_core_lib.dataset.validation_schema import FeatureData, DataSkeleton, SkeletonOut

class GraphDataset(Dataset):
    """
    Custom dataset for graph data with nodes and edges
    """
    
    def __init__(self, nodes: List[Column], edges: List[Column], 
                 adjacency_matrix: np.ndarray):
        super().__init__()
        self.nodes = nodes
        self.edges = edges
        self.adjacency_matrix = adjacency_matrix
        
    @classmethod
    def from_json(cls, json_data: list[dict]) -> "GraphDataset":
        """Create graph dataset from JSON configuration"""
        nodes = []
        edges = []
        adjacency_data = None
        
        for col_data in json_data:
            FeatureData.model_validate(col_data)
            col_type = col_data.get("column_type", "")
            col_name = col_data.get("column_name", "")
            col_values = np.array(col_data.get("column_data", []))
            
            if col_type == "node_feature":
                nodes.append(cls._create_column(col_data))
            elif col_type == "edge_feature":
                edges.append(cls._create_column(col_data))
            elif col_type == "adjacency_matrix":
                adjacency_data = col_values
                
        if adjacency_data is None:
            raise ValueError("Adjacency matrix is required for graph dataset")
            
        return cls(nodes, edges, adjacency_data)
    
    @classmethod
    def from_skeleton(cls, skeleton: list[dict]) -> "GraphDataset":
        """Create graph dataset from metadata skeleton"""
        # Implementation similar to from_json but with empty data
        pass
    
    def clone(self, new_data: np.ndarray) -> "GraphDataset":
        """Create new graph dataset with different data"""
        # Handle new data and recreate the dataset
        pass
    
    def get_computing_data(self) -> np.ndarray:
        """Get data in format suitable for ML models"""
        # Combine node features, edge features, and adjacency matrix
        node_data = np.hstack([node.get_data() for node in self.nodes])
        edge_data = np.hstack([edge.get_data() for edge in self.edges])
        
        # Return concatenated data or structured format
        return np.concatenate([node_data, edge_data, self.adjacency_matrix.flatten()])
    
    def get_computing_shape(self) -> tuple[int, ...]:
        """Get shape of computing data"""
        total_features = sum(node.get_internal_shape()[1] for node in self.nodes)
        total_features += sum(edge.get_internal_shape()[1] for edge in self.edges)
        total_features += self.adjacency_matrix.size
        
        return (len(self.nodes[0].get_data()), total_features)
    
    def to_json(self) -> list[dict]:
        """Convert dataset to JSON format"""
        json_data = []
        
        # Add node features
        for node in self.nodes:
            json_data.append({
                "column_data": node.get_data().tolist(),
                "column_name": node.name,
                "column_type": "node_feature",
                "column_datatype": node.value_type
            })
            
        # Add edge features
        for edge in self.edges:
            json_data.append({
                "column_data": edge.get_data().tolist(),
                "column_name": edge.name,
                "column_type": "edge_feature",
                "column_datatype": edge.value_type
            })
            
        # Add adjacency matrix
        json_data.append({
            "column_data": self.adjacency_matrix.tolist(),
            "column_name": "adjacency_matrix",
            "column_type": "adjacency_matrix",
            "column_datatype": "float64"
        })
        
        return json_data
    
    def to_skeleton(self) -> list[dict]:
        """Convert dataset to metadata skeleton"""
        skeleton = []
        
        # Add node feature metadata
        for i, node in enumerate(self.nodes):
            skeleton.append({
                "feature_name": node.name,
                "feature_position": i,
                "feature_type": "node_feature",
                "type": node.value_type,
                "is_categorical": isinstance(node, Categorical),
                "feature_size": str(node.get_internal_shape()[1])
            })
            
        # Add edge feature metadata
        for i, edge in enumerate(self.edges):
            skeleton.append({
                "feature_name": edge.name,
                "feature_position": len(self.nodes) + i,
                "feature_type": "edge_feature",
                "type": edge.value_type,
                "is_categorical": isinstance(edge, Categorical),
                "feature_size": str(edge.get_internal_shape()[1])
            })
            
        return skeleton
    
    def preprocess(self, processor: Processor) -> "GraphDataset":
        """Apply preprocessing to dataset"""
        # Apply preprocessing to nodes and edges
        processed_nodes = processor.process(self.nodes)
        processed_edges = processor.process(self.edges)
        
        return GraphDataset(processed_nodes, processed_edges, self.adjacency_matrix)
    
    def postprocess(self, processor: Processor) -> "GraphDataset":
        """Apply postprocessing to dataset"""
        # Apply postprocessing to nodes and edges
        postprocessed_nodes = processor.inverse_process(self.nodes)
        postprocessed_edges = processor.inverse_process(self.edges)
        
        return GraphDataset(postprocessed_nodes, postprocessed_edges, self.adjacency_matrix)
    
    # Example helper method to create columns
    @staticmethod
    def _create_column(col_data: dict) -> Column:
        """Helper method to create appropriate column type"""
        col_type = col_data.get("column_type", "")
        col_name = col_data.get("column_name", "")
        col_values = np.array(col_data.get("column_data", []))
        col_datatype = col_data.get("column_datatype", "")
        
        if col_datatype in ["int", "float"]:
            return Numeric(col_name, col_datatype, 0, col_values, col_type)
        else:
            return Column(col_name, col_datatype, 0, col_values, col_type)
        
    # [Insert here your Dataset-Specific helper methods]
```

Any additional helper method is welcome to be added to the class.

## Key Implementation Considerations

### 1. Data Validation

Always validate input data (JSON or Skeletons) using the existing schema validators:

```python
from sdg_core_lib.dataset.validation_schema import FeatureData, DataSkeleton

# In from_json method
FeatureData.model_validate(col_data)

# In from_skeleton method  
DataSkeleton.model_validate(col_data)
```

### 2. Column Management

Use the existing column types or create custom ones:

```python
from sdg_core_lib.dataset.columns import Column, Numeric, Categorical

# Use existing types
numeric_col = Numeric("age", "int64", 0, age_data, "numeric")
categorical_col = Categorical("category", "str", 1, cat_data, "categorical")

# Or create custom column types
class GraphNodeColumn(Column):
    def __init__(self, name: str, value_type: str, position: int, 
                 values: np.ndarray, node_id: str):
        super().__init__(name, value_type, position, values, "node_feature")
        self.node_id = node_id # This may come from an internal Dataset Method
```

### 3. Data Shape Handling

Ensure proper shape handling for different data formats:

```python
def get_computing_shape(self) -> tuple[int, ...]:
    """Handle multi-dimensional data appropriately"""
    if self.is_time_series:
        return (batch_size, sequence_length, feature_dim)
    elif self.is_image:
        return (batch_size, height, width, channels)
    else:
        return (batch_size, feature_dim)
```

## Using Custom Datasets

Custom datasets are used directly where needed. Import your custom dataset where needed:

```python
# In a job
from sdg_core_lib.dataset.datasets import GraphDataset

# Use your custom dataset directly
def create_dataset(dataset_config):
    if dataset_config["dataset_type"] == "graph":
        return GraphDataset.from_json(dataset_config["data"])
    # ... other dataset types
```

### ⚠️ Integrate Custom Datasets

⚠️ Available models generally don't support custom datasets with custom data types. The given example may be an exception, since is build on a fictional Graph Dataset.

⚠️ Processors for such a Dataset may be different from available ones. You may need to [implement your own processor](custom-processors.md).

## Testing Custom Datasets

Create comprehensive tests for your custom dataset:

```python
import pytest
import numpy as np
from your_module import GraphDataset

class TestGraphDataset:
    def test_from_json(self):
        """Test JSON creation"""
        json_data = [
            {
                "column_name": "node_features",
                "column_type": "node_feature", 
                "column_data": [[1, 2], [3, 4]],
                "column_datatype": "float64"
            },
            {
                "column_name": "adjacency_matrix",
                "column_type": "adjacency_matrix",
                "column_data": [[0, 1], [1, 0]],
                "column_datatype": "float64"
            }
        ]
        
        dataset = GraphDataset.from_json(json_data)
        assert len(dataset.nodes) == 1
        assert dataset.adjacency_matrix.shape == (2, 2)
    
    def test_get_computing_data(self):
        """Test data extraction for ML models"""
        # Test implementation
        pass
    
    def test_preprocessing(self):
        """Test preprocessing integration"""
        # Test implementation
        pass
```

## Best Practices

1. **Follow Existing Patterns**: Study the Table and TimeSeries implementations
2. **Proper Type Hints**: Use comprehensive type annotations
3. **Error Handling**: Include meaningful error messages and validation
4. **Documentation**: Add docstrings explaining the dataset's purpose and usage
5. **Performance**: Consider memory efficiency for large datasets
6. **Compatibility**: Ensure compatibility with existing processors and models

## Use Cases

Custom datasets are useful for:

- **Graph Data**: Social networks, molecular structures
- **Image Data**: Computer vision datasets with special preprocessing needs
- **Text Data**: NLP datasets with tokenization requirements
- **Audio Data**: Time series with frequency domain features
- **Multimodal Data**: Combined data from different sources

## Integration with Models

Ensure your custom dataset works with existing models or create model-specific adapters:

```python
def get_shape_for_model(self) -> str:
    """Return shape string compatible with model configuration"""
    if self.model_type == "graph_neural_network":
        return f"({self.num_nodes}, {self.node_features})"
    else:
        return super().get_shape_for_model()
```

This guide provides the foundation for creating custom datasets. Adapt the examples to your specific data requirements and use cases.
