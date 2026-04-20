# GENESIS Core Lib

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-AGPLv3-blue.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/sdg-core-lib.svg)](https://pypi.org/project/sdg-core-lib/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/sdg-core-lib.svg)](https://pypi.org/project/sdg-core-lib/)
[![Test and Build Lib](https://github.com/eng-genesis/genesis_core_lib/actions/workflows/lib-stable.yml/badge.svg)](https://github.com/eng-genesis/genesis_core_lib/actions/workflows/lib-stable.yml)
[![codecov](https://codecov.io/gh/eng-genesis/genesis_core_lib/graph/badge.svg?token=RMOVSN2X0H)](https://codecov.io/gh/eng-genesis/genesis_core_lib)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking](https://img.shields.io/badge/type%20checking-mypy-blue.svg)](http://mypy-lang.org/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://github.com/emiliocimino/generator_core_lib/docs)



## What is GENESIS Core Lib?
GENESIS Core Lib is an advanced synthetic data generation library for Python 3.12+ that provides state-of-the-art machine learning models including VAEs (TabularVAE, TimeSeriesVAE) and CTGAN for generating high-quality tabular and time series data. The library features adaptive on-the-fly training with automatic model adaptation to data characteristics, model persistence for reusing trained models, behavior control through custom mathematical functions, and integrated quality evaluation metrics. Designed with privacy preservation in mind and optimized for both CPU and GPU processing, it offers a comprehensive Job API for data augmentation, privacy preservation, and ML model testing across various data types and use cases.

## Why use GENESIS core Lib?
GENESIS Core Lib is the ideal solution for synthetic data generation because it specializes in industrial sensor data and time series applications, offering state-of-the-art models like TimeSeriesVAE that preserve temporal patterns and statistical properties essential for real-world scenarios. It provides immediate access to high-quality synthetic data without the costs and delays of physical sensor deployments, enabling rapid prototyping, algorithm development, and comprehensive testing while maintaining data privacy through synthetic data sharing. The library ensures data fidelity with advanced evaluation metrics including Dynamic Time Warping and correlation preservation, making it perfect for manufacturing IoT, environmental monitoring, energy utilities, and medical device applications where realistic temporal data is critical.

GENESIS Core Lib is a powerful, extensible library for generating high-quality synthetic data using state-of-the-art machine learning models. Perfect for data augmentation, privacy preservation, and ML model testing.

## ✨ Key Features

- **Generative AI Architectures**: Advanced VAEs (TabularVAE, TimeSeriesVAE) and CTGAN for Tabular and Time series data
- **Adaptive Training**: On-the-fly model training with automatic adaptation to your data characteristics
- **Model Persistence**: Save and reuse trained generative models for consistent data generation
- **Behavior Control**: Manipulate generation patterns with custom mathematical functions
- **Integrated Evaluation**: Built-in quality assessment metrics for comprehensive data evaluation
- **High Performance**: Optimized for both CPU and GPU processing

## 🛠️ Quick Start

### Quick Install
```bash
pip install sdg-core-lib
```

### 🚀 Try it

```python
from sdg_core_lib import Job

# Text-based JSON configuration (no file needed)
config = {
    "n_rows": 1000,
    "model": {
        "algorithm_name": "sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE.TabularVAE",
        "model_name": "customer_synthetic_model"
    },
    "dataset": {
        "dataset_type": "table",
        "data": [
            {
                "column_data": [13.71, 13.4, 13.27, 13.17, 14.13, 13.88, 13.24, 13.73],
                "column_name": "alcohol",
                "column_type": "continuous",
                "column_datatype": "float64"
            },
            {
                "column_data": [5.65, 3.91, 4.28, 2.59, 4.1, 3.9, 3.8, 4.2],
                "column_name": "malic_acid",
                "column_type": "continuous",
                "column_datatype": "float64"
            },
            {
                "column_data": [1.28, 1.05, 1.02, 1.03, 1.71, 1.23, 1.07, 1.5],
                "column_name": "ash",
                "column_type": "continuous",
                "column_datatype": "float64"
            }
        ]
    },
    "save_filepath": "./models"
}

# Create and run a synthetic data generation job
job = Job(
    n_rows=config["n_rows"],
    model_info=config["model"],
    dataset=config["dataset"],
    save_filepath=config.get("save_filepath", "./models")
)

# Generate synthetic data
results, metrics, model, schema = job.train()
print(f"Generated {len(results)} synthetic rows")
print(f"Quality metrics: {metrics}")
```

📖 **See [Quick Start Guide](docs/quick-start.md) for detailed examples**


## 📚 Documentation

### 📖 [User Documentation](docs/user-documentation.md)
Complete guide for users including:
- Core concepts and the Job API
- Data Types and [Datasets](docs/user-api-reference/dataset-API-reference.md)
- Fantastic [Models](docs/user-api-reference/model-API-reference.md) and how to use them
- How to handle raw data with [Processors](docs/user-api-reference/processor-API-reference.md) 
- How to control generation of synthetic data with [Functions](docs/user-api-reference/functions-API-reference.md)
- Evaluate your work with [Evaluators](docs/user-api-reference/evaluation-API-reference.md)

### 🔧 [Developer Documentation](docs/developer-documentation.md)
Technical documentation for developers:
- Architecture overview and design patterns
- Extension points and customization
- Development setup and testing
- Code organization and standards

### ⚡ [Quick Start Guide](docs/quick-start.md)
Get started immediately with:
- Installation instructions
- Basic examples and tutorials
- Common use cases
- Troubleshooting tips

### 📋 [Standalone Installation](docs/standalone-installation.md)
Info about installation:
- Requirements
- Installation instructions
- Environment Variables
- GPU Support

## Roadmap
A detailed Roadmap can be found [here](docs/roadmap.md).


## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.


## 📄 License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with TensorFlow and Keras for deep learning models
- Statistical evaluation using scipy and numpy
- Inspired by state-of-the-art synthetic data generation research

## 📞 Support

- 📖 [Documentation](https://github.com/emiliocimino/generator_core_lib/docs)
- 🐛 [Issues](https://github.com/emiliocimino/generator_core_lib/issues)
- 💬 [Discussions](https://github.com/emiliocimino/generator_core_lib/discussions)

---

**GENESIS Core Lib** - *Generating Tomorrow's Data, Today* 🚀
