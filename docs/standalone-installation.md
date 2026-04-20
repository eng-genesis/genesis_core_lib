# Standalone Installation Guide

## Overview

This comprehensive guide covers all aspects of installing GENESIS Core Lib in various environments, from basic setups to advanced configurations. Whether you're installing on a local machine, in a container, or in a production environment, this guide has you covered.

## System Requirements

### Minimum Requirements
- **Python**: 3.12 or higher
- **Memory**: 4GB RAM minimum (8GB+ recommended for large datasets)
- **Storage**: 2GB free space for installation and models
- **CPU**: 64-bit processor (multi-core recommended)

### Recommended Requirements
- **Python**: 3.12+ with virtual environment
- **Memory**: 16GB+ RAM for production workloads
- **Storage**: 10GB+ free space
- **GPU**: CUDA-compatible GPU for deep learning models (optional but recommended)
- **CPU**: 4+ cores for parallel processing

### Supported Operating Systems
- **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 9+
- **macOS**: 10.15+ (Catalina and later)
- **Windows**: Windows 10+ (with WSL2 recommended)

## Installation Methods

### Method 1: Standard pip Installation (Recommended)

#### Basic Installation
```bash
# Install from PyPI
pip install sdg-core-lib
```

#### Installation with Specific Version
```bash
# Install specific version
pip install sdg-core-lib==0.1.9

# Install latest version
pip install --upgrade sdg-core-lib
```

#### Installation with Extra Dependencies
```bash
# Install with development dependencies
pip install "sdg-core-lib[dev]"

# Install with test dependencies
pip install "sdg-core-lib[test]"
```

### Method 2: UV Package Manager (Faster)

#### Prerequisites
```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Installation with UV
```bash
# Add to your project
uv add sdg-core-lib

# Add with development dependencies
uv add --dev sdg-core-lib

# Install globally
uv pip install sdg-core-lib
```

### Method 3: Development Installation

#### Clone and Install
```bash
# Clone the repository
git clone https://github.com/emiliocimino/generator_core_lib.git
cd generator_core_lib

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

#### Development Installation with UV
```bash
# Clone the repository
git clone https://github.com/emiliocimino/generator_core_lib.git
cd generator_core_lib

# Install with UV
uv sync --dev
```


## Environment Setup

### Virtual Environment Setup

#### Using venv
```bash
# Create virtual environment
python -m venv genesis-env

# Activate (Linux/macOS)
source genesis-env/bin/activate

# Activate (Windows)
genesis-env\Scripts\activate

# Install the library
pip install sdg-core-lib

# Deactivate when done
deactivate
```

#### Using conda
```bash
# Create conda environment
conda create -n genesis-env python=3.12

# Activate environment
conda activate genesis-env

# Install the library
pip install sdg-core-lib

# Deactivate when done
conda deactivate
```

### Environment Variables

#### Common Environment Variables
```bash
# Training hyperparameters (optional)
export EPOCHS=100
export LEARNING_RATE=0.001
export BATCH_SIZE=32

# Keras backend (automatically set to tensorflow)
export KERAS_BACKEND=tensorflow

# GPU settings (if using CUDA)
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

## Dependency Management

### Core Dependencies

GENESIS Core Lib automatically installs these core dependencies:

#### Machine Learning & Data Science
- **numpy**: 2.0.2+ - Numerical computing
- **pandas**: 2.2.3+ - Data manipulation
- **scikit-learn**: 1.5.2+ - Machine learning utilities
- **scipy**: 1.16.2+ - Scientific computing

#### Deep Learning
- **tensorflow**: 2.18.0+ - Deep learning framework
- **keras**: 3.6.0+ - High-level neural networks API
- **keras-tuner**: 1.4.8+ - Hyperparameter tuning

#### Statistical Analysis
- **statsmodels**: 0.14.5+ - Statistical models and tests
- **seaborn**: 0.13.2+ - Statistical data visualization
- **tslearn**: 0.7.0+ - Time series machine learning

#### Utilities
- **loguru**: 0.7.3+ - Logging library
- **skops**: 0.13.0+ - Model persistence

#### Type Stubs
- **scipy-stubs**: 1.16.2+ - Type annotations for scipy
- **pandas-stubs**: 2.2.3+ - Type annotations for pandas

### Optional Dependencies

#### GPU Support
```bash
# Install TensorFlow with GPU support
pip install tensorflow==2.18.0 --extra-index-url https://download.pytorch.org/whl/cu118

# Verify GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Platform-Specific Installation

### Linux Installation

#### Ubuntu/Debian
```bash
# Update package manager
sudo apt update

# Install Python and build tools
sudo apt install python3.12 python3.12-venv python3.12-dev build-essential

# Install system dependencies for deep learning
sudo apt install libhdf5-dev libhdf5-serial-dev libhdf5-103 libqt5x11extras5 libqt5test5

# Create virtual environment and install
python3.12 -m venv genesis-env
source genesis-env/bin/activate
pip install sdg-core-lib
```

#### CentOS/RHEL/Fedora
```bash
# For modern systems (Fedora/CentOS Stream/RHEL 9+)
sudo dnf install python3.12 python3.12-devel gcc gcc-c++ make hdf5-devel openssl-devel

# For older systems (CentOS 7/RHEL 8)
sudo yum install python3.12 python3.12-devel gcc gcc-c++ make hdf5-devel openssl-devel

# Create virtual environment and install
python3.12 -m venv genesis-env
source genesis-env/bin/activate
pip install sdg-core-lib
```

### macOS Installation

#### Using Homebrew
```bash
# Install Python
brew install python@3.12

# Create virtual environment
python3.12 -m venv genesis-env
source genesis-env/bin/activate

# Install the library
pip install sdg-core-lib
```

#### Using Conda
```bash
# Install Miniconda (if not already installed)
# Visit https://docs.conda.io/en/latest/miniconda.html for latest download

# Create environment and install
conda create -n genesis-env python=3.12
conda activate genesis-env
pip install sdg-core-lib
```

### Windows Installation

#### Using WSL2 (Recommended)
```bash
# Install WSL2
wsl --install

# Install Ubuntu from Microsoft Store
# Then in WSL terminal:

# Update and install Python
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev build-essential

# Install system dependencies
sudo apt install libhdf5-dev libhdf5-serial-dev libssl-dev libffi-dev

# Create virtual environment and install
python3.12 -m venv genesis-env
source genesis-env/bin/activate
pip install sdg-core-lib
```

#### Native Windows Installation
```powershell
# Install Python from python.org
# Ensure "Add to PATH" is checked during installation

# Open PowerShell as Administrator

# Create virtual environment
python -m venv genesis-env

# Activate environment
genesis-env\Scripts\Activate.ps1

# Install the library
pip install sdg-core-lib
```

## GPU Installation

### NVIDIA GPU Setup

#### Install CUDA Toolkit
```bash
# Download and install CUDA Toolkit 11.8 or 12.x
# From: https://developer.nvidia.com/cuda-downloads

# Verify installation
nvidia-smi
nvcc --version
```

#### Install cuDNN
```bash
# Download cuDNN compatible with your CUDA version
# From: https://developer.nvidia.com/cudnn

# Extract and copy to CUDA directory
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

#### Install TensorFlow GPU
```bash
# Install TensorFlow with GPU support
pip install tensorflow==2.18.0

# Verify GPU support
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

### AMD GPU Setup

#### Install ROCm (Experimental)
```bash
# Install ROCm for AMD GPU support
# Follow instructions from: https://rocm.docs.amd.com/

# Install TensorFlow with ROCm support
pip install tensorflow-rocm
```

## Verification

### Basic Verification
```python
# Test basic installation
from sdg_core_lib import Job
print("✅ GENESIS Core Lib installed successfully!")

# Test data generation
dataset_config = {
    "dataset_type": "table",
    "data": [{"x": 1, "y": 2}, {"x": 2, "y": 4}]
}

model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.GANs.TabularGAN",
    "model_name": "test_model"
}

job = Job(n_rows=10, model_info=model_config, dataset=dataset_config)
print("✅ Job creation successful!")
```

### Full Verification
```python
# Complete functionality test
import numpy as np
from sdg_core_lib import Job

# Test data in correct Dataset column-based format
feature1_data = np.random.normal(0, 1, 100).tolist()
feature2_data = np.random.exponential(1, 100).tolist()
category_data = np.random.choice(['A', 'B', 'C'], 100).tolist()

# Configure job with column-based Dataset format
dataset_config = {
    "dataset_type": "table",
    "data": [
        {
            "column_name": "feature1",
            "column_type": "continuous",
            "column_datatype": "float64",
            "column_data": feature1_data
        },
        {
            "column_name": "feature2", 
            "column_type": "continuous",
            "column_datatype": "float64",
            "column_data": feature2_data
        },
        {
            "column_name": "category",
            "column_type": "categorical",
            "column_datatype": "int64",
            "column_data": category_data
        }
    ]
}

model_config = {
    "algorithm_name": "sdg_core_lib.data_generator.models.GANs.implementation.CTGAN.CTGAN",
    "model_name": "verification_model"
}

# Test training
job = Job(
    n_rows=50,
    model_info=model_config,
    dataset=dataset_config,
    save_filepath="./test_models"
)

results, metrics, model, schema = job.train()

print("✅ Full verification successful!")
print(f"Generated {len(results)} rows")
print(f"Quality metrics: {metrics}")
```

### GPU Verification
```python
# Test GPU support
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU Available:", len(tf.config.list_physical_devices('GPU')) > 0)

if tf.config.list_physical_devices('GPU'):
    print("GPU Device:", tf.config.list_physical_devices('GPU')[0])
    print("✅ GPU support verified!")
else:
    print("⚠️ GPU not available, using CPU")
```

## Troubleshooting

### Common Installation Issues

#### Issue 1: Python Version Incompatibility
```bash
# Check Python version
python --version

# If version < 3.12, upgrade Python
# Ubuntu/Debian:
sudo apt install python3.12 python3.12-venv

# macOS:
brew install python@3.12

# Windows:
# Download from python.org
```

#### Issue 2: Memory Error During Installation
```bash
# Increase pip cache size
pip config set global.cache-dir /tmp/pip-cache

# Install with no cache
pip install sdg-core-lib --no-cache-dir

# Use UV for faster, more memory-efficient installation
uv pip install sdg-core-lib
```

#### Issue 3: TensorFlow Installation Problems
```bash
# Install specific TensorFlow version
pip install tensorflow==2.18.0

# For GPU issues, try CPU version first
pip install tensorflow-cpu==2.18.0

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

#### Issue 4: Permission Denied
```bash
# Use user installation
pip install --user sdg-core-lib

# Or use virtual environment
python -m venv genesis-env
source genesis-env/bin/activate
pip install sdg-core-lib
```

#### Issue 5: Dependency Conflicts
```bash
# Create clean environment
python -m venv clean-env
source clean-env/bin/activate

# Install with specific versions
pip install "numpy==2.0.2" "pandas==2.2.3"
pip install sdg-core-lib

# Or use UV for better dependency resolution
uv add sdg-core-lib
```

### Performance Issues

#### Issue 1: Slow Training
```python
# Enable GPU if available
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Adjust batch size
os.environ['BATCH_SIZE'] = '64'

# Use mixed precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
```

#### Issue 2: Memory Errors
```python
# Reduce batch size
os.environ['BATCH_SIZE'] = '16'

# Enable memory growth
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

## Configuration Files

### requirements.txt
```txt
# Core dependencies
numpy==2.0.2
pandas==2.2.3
scikit-learn==1.5.2
tensorflow==2.18.0
keras==3.6.0
loguru==0.7.3

# GENESIS Core Lib
sdg-core-lib==0.1.9
```

### pyproject.toml
```toml
[project]
name = "my-genesis-project"
version = "0.1.0"
dependencies = [
    "sdg-core-lib>=0.1.9",
    "pandas>=2.2.3",
    "numpy>=2.0.2"
]

[tool.uv]
dev-dependencies = [
    "pytest>=7.0.0", 
]
```


## Best Practices

### 1. Use Virtual Environments
Always isolate GENESIS Core Lib in a virtual environment to avoid dependency conflicts.

### 2. Version Pinning
Pin specific versions in production environments for reproducibility.

### 3. GPU Optimization
Enable GPU support for large datasets and complex models.

### 4. Monitoring
Set up logging and monitoring to track performance and errors.

### 5. Regular Updates
Keep dependencies updated for security and performance improvements.

## Next Steps

After successful installation:

1. **Run Quick Start**: Follow the [Quick Start Guide](quick-start.md)
2. **Read Documentation**: Explore [User Documentation](user-documentation.md)
3. **View Examples**: Check the examples directory
4. **Join Community**: Participate in discussions and contribute

## Support

If you encounter installation issues:

- 📖 [Documentation](../README.md)
- 🐛 [Report Issues](https://github.com/emiliocimino/generator_core_lib/issues)
- 💬 [Community Forum](https://github.com/emiliocimino/generator_core_lib/discussions)
- 📧 [Email Support](mailto:emilio.cimino@outlook.it)

Happy installing! 🚀
