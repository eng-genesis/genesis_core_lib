from sdg_core_lib.dataset.datasets import Table
from sdg_core_lib.data_generator.models.VAEs.implementation.TabularVAE import TabularVAE
from sdg_core_lib.preprocess.table_processor import TableProcessor
from sdg_core_lib.preprocess.strategies.vae_strategy import TabularVAEPreprocessingStrategy
from sdg_core_lib.evaluate.tables import TabularComparisonEvaluator


# 1. Create Dataset from class (not config)
data_payload = [
    {
        "column_name": "age",
        "column_type": "continuous",
        "column_datatype": "float64",
        "column_data": [25.0, 30.0, 35.0, 40.0, 45.0, 28.0, 32.0, 38.0]
    },
    {
        "column_name": "income",
        "column_type": "continuous",
        "column_datatype": "float64",
        "column_data": [50000.0, 60000.0, 75000.0, 80000.0, 90000.0, 55000.0, 65000.0, 70000.0]
    },
    {
        "column_name": "category",
        "column_type": "categorical",
        "column_datatype": "str",
        "column_data": ["A", "B", "A", "C", "B", "A", "C", "B"]
    }
]

# Create dataset instance
dataset = Table.from_json(data_payload)
print(f"Dataset created with {len(dataset.columns)} columns")


# 2. Preprocess Data
processor = TableProcessor(dir_path="user-api-reference/models/customer_synthetic_model")

# Set strategy for preprocessing
strategy = TabularVAEPreprocessingStrategy()
processor.set_strategy(strategy)

print("Processor created with VAE strategy")

# Preprocess data first
preprocessed_data = dataset.preprocess(processor)

# 3. Extract metadata for model (From Pre-processed Data)
metadata = preprocessed_data.to_skeleton()
input_shape = preprocessed_data.get_shape_for_model()


# 4. Create and Train Model

# Create the model
model = TabularVAE(
    metadata=metadata,
    model_name="customer_synthetic_model",
    input_shape=input_shape,
    load_path=None,  # New model
    latent_dim=2,
    learning_rate=1e-3,
    batch_size=16,
    epochs=100
)
print(f"Model created: {model.model_name}")

# Train the model
model.train(data=preprocessed_data.get_computing_data())
print("Model training completed")

# Save the trained model
model.save("./models/customer_synthetic_model")
print("Model saved to ./models/customer_synthetic_model")

# 5. Generate new synthetic data (inference)
n_rows_to_generate = 100
synthetic_raw = model.infer(n_rows_to_generate)

# Clone the preprocessed structure with synthetic data
synthetic_data = preprocessed_data.clone(synthetic_raw)

# 6. Postprocess the synthetic data
synthetic_data = synthetic_data.postprocess(processor)
print(f"Generated {n_rows_to_generate} synthetic rows")

# 7. Evaluate the generated data Here
evaluator = TabularComparisonEvaluator(
    real_data=dataset,
    synthetic_data=synthetic_data
)

evaluation_report = evaluator.compute()
print("Evaluation completed:")
print(f"  - Statistical similarity metrics: {evaluation_report.get('statistical', {})}")
print(f"  - Adherence metrics: {evaluation_report.get('adherence', {})}")
print(f"  - Novelty metrics: {evaluation_report.get('novelty', {})}")

# 8. Get results in JSON format
results_json = synthetic_data.to_json()
print(f"Final synthetic data ready: {len(results_json)} features generated")

# Example: Access first few rows of generated data
for i, feature in enumerate(results_json[:3]):
    print(f"Feature {i+1}: {feature['column_name']} (type: {feature['column_type']})")
    print(f"  Sample values: {feature['column_data'][:5]}...")