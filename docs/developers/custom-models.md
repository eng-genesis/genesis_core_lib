# Custom Models

## Overview

This guide explains how to create custom machine learning models for the GENESIS Core Lib. Custom models allow you to implement specialized synthetic data generation algorithms beyond the built-in VAEs and GANs.

## Base Model Classes

All custom models should inherit from the base classes in `src/sdg_core_lib/data_generator/models/`:

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, List
from sdg_core_lib.data_generator.models.TrainingInfo import TrainingInfo

class UnspecializedModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(
        self,
        metadata: List[Dict[str, Any]],
        model_name: str,
        input_shape: str = None,
        load_path: str = None,
    ):
        self._metadata = metadata
        self.model_name = model_name
        self.input_shape = self._parse_stringed_input_shape(input_shape)
        self._load_path = load_path
        self._model = None
        self.training_info = None

    @abstractmethod
    def _build(self, input_shape: tuple[int, ...]):
        """Build the model architecture"""
        raise NotImplementedError

    @abstractmethod
    def _load(self, model_filepath: str):
        """Load trained model weights"""
        raise NotImplementedError

    @abstractmethod
    def train(self, data: np.ndarray):
        """Train the model"""
        raise NotImplementedError

    @abstractmethod
    def fine_tune(self, data: np.ndarray, **kwargs):
        """Fine-tune the model"""
        raise NotImplementedError

    @abstractmethod
    def infer(self, n_rows: int, **kwargs):
        """Run inference to generate synthetic data"""
        raise NotImplementedError

    @abstractmethod
    def save(self, folder_path: str):
        """Save the model"""
        raise NotImplementedError

    @abstractmethod
    def set_hyperparameters(self, **kwargs):
        """Set model hyperparameters"""
        raise NotImplementedError

    @classmethod
    def self_describe(cls):
        """Get model description"""
        return {
            "model_name": cls.__name__,
            "description": "Custom model implementation",
            "parameters": [],
            "capabilities": []
        }

    def _instantiate(self):
        """Instantiate the model"""
        if self._load_path is not None:
            self._load(self._load_path)
            return
        if not self._model and self.input_shape:
            self._model = self._build(self.input_shape)

    @staticmethod
    def _parse_stringed_input_shape(stringed_shape: str) -> tuple[int, ...]:
        """Parse string input shape to tuple"""
        brackets = ["(", ")", "[", "]", "{", "}"]
        for b in brackets:
            stringed_shape = stringed_shape.replace(b, "")
        return tuple([int(n) for n in stringed_shape.split(",") if n != ""])
```

## Creating Custom Models

### Example 1: Autoencoder Model

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, Optional
from sdg_core_lib.data_generator.models.UnspecializedModel import UnspecializedModel
from sdg_core_lib.data_generator.models.TrainingInfo import TrainingInfo

class AutoencoderModel(UnspecializedModel):
    """Custom autoencoder model for synthetic data generation"""
    
    def __init__(
        self,
        metadata: List[Dict[str, Any]],
        model_name: str,
        input_shape: str = None,
        load_path: str = None,
        encoding_dim: int = 32,
        hidden_layers: List[int] = None,
        activation: str = "relu",
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        super().__init__(metadata, model_name, input_shape, load_path)
        
        # Model hyperparameters
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers or [128, 64]
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Model components
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        
        # Training history
        self.training_history = None

    def _build(self, input_shape: tuple[int, ...]):
        """Build the autoencoder architecture"""
        # Build encoder
        encoder_inputs = keras.Input(shape=(input_shape[-1],))
        x = encoder_inputs
        
        for units in self.hidden_layers:
            x = keras.layers.Dense(units, activation=self.activation)(x)
        
        # Bottleneck layer
        encoded = keras.layers.Dense(self.encoding_dim, activation=self.activation)(x)
        self.encoder = keras.Model(encoder_inputs, encoded, name="encoder")
        
        # Build decoder
        decoder_inputs = keras.Input(shape=(self.encoding_dim,))
        x = decoder_inputs
        
        # Reverse hidden layers
        for units in reversed(self.hidden_layers):
            x = keras.layers.Dense(units, activation=self.activation)(x)
        
        # Output layer
        decoded = keras.layers.Dense(input_shape[-1], activation="linear")(x)
        self.decoder = keras.Model(decoder_inputs, decoded, name="decoder")
        
        # Build full autoencoder
        autoencoder_inputs = keras.Input(shape=(input_shape[-1],))
        encoded_repr = self.encoder(autoencoder_inputs)
        decoded_output = self.decoder(encoded_repr)
        
        self.autoencoder = keras.Model(autoencoder_inputs, decoded_output, name="autoencoder")
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.autoencoder.compile(optimizer=optimizer, loss="mse")
        
        return self.autoencoder

    def _load(self, model_filepath: str):
        """Load trained model"""
        self.autoencoder = keras.models.load_model(model_filepath)
        self.encoder = self.autoencoder.get_layer("encoder")
        self.decoder = self.autoencoder.get_layer("decoder")

    def train(self, data: np.ndarray):
        """Train the autoencoder"""
        if self.autoencoder is None:
            self._instantiate()
        
        # Prepare training data
        if len(data.shape) > 2:
            # Flatten data for autoencoder
            data = data.reshape(data.shape[0], -1)
        
        # Train the model
        self.training_history = self.autoencoder.fit(
            data, data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Create training info
        self.training_info = TrainingInfo(
            epochs=self.epochs,
            final_loss=self.training_history.history["loss"][-1],
            final_val_loss=self.training_history.history["val_loss"][-1],
            training_time=0  # Would be calculated in actual implementation
        )

    def fine_tune(self, data: np.ndarray, **kwargs):
        """Fine-tune the model"""
        if self.autoencoder is None:
            self.train(data)
            return
        
        # Override hyperparameters for fine-tuning
        fine_tune_epochs = kwargs.get("epochs", 50)
        fine_tune_learning_rate = kwargs.get("learning_rate", self.learning_rate * 0.1)
        
        # Update learning rate
        keras.backend.set_value(
            self.autoencoder.optimizer.learning_rate, 
            fine_tune_learning_rate
        )
        
        # Prepare data
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        
        # Fine-tune
        self.autoencoder.fit(
            data, data,
            epochs=fine_tune_epochs,
            batch_size=self.batch_size,
            verbose=1
        )

    def infer(self, n_rows: int, **kwargs) -> np.ndarray:
        """Generate synthetic data using the trained autoencoder"""
        if self.autoencoder is None:
            raise ValueError("Model must be trained before inference")
        
        # Generate random latent vectors
        latent_dim = self.encoding_dim
        latent_vectors = np.random.normal(0, 1, (n_rows, latent_dim))
        
        # Decode to generate synthetic data
        synthetic_data = self.decoder.predict(latent_vectors)
        
        return synthetic_data

    def save(self, folder_path: str):
        """Save the model"""
        if self.autoencoder is None:
            raise ValueError("No model to save")
        
        self.autoencoder.save(f"{folder_path}/{self.model_name}.h5")
        
        # Save metadata
        import json
        metadata = {
            "model_name": self.model_name,
            "encoding_dim": self.encoding_dim,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "input_shape": self.input_shape
        }
        
        with open(f"{folder_path}/{self.model_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def set_hyperparameters(self, **kwargs):
        """Set model hyperparameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown hyperparameter: {key}")

    @classmethod
    def self_describe(cls):
        """Get model description"""
        return {
            "model_name": "AutoencoderModel",
            "description": "Autoencoder-based synthetic data generation model",
            "parameters": [
                "encoding_dim",
                "hidden_layers", 
                "activation",
                "learning_rate",
                "epochs",
                "batch_size"
            ],
            "capabilities": [
                "tabular_data",
                "feature_learning",
                "dimensionality_reduction"
            ]
        }
```

### Example 2: Normalizing Flow Model

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple
from sdg_core_lib.data_generator.models.UnspecializedModel import UnspecializedModel

class CouplingLayer(nn.Module):
    """Coupling layer for normalizing flows"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Split dimension
        self.split_dim = input_dim // 2
        
        # Neural network for transformation
        self.transform_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (input_dim - self.split_dim) * 2)  # scale and translate
        )
    
    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through coupling layer"""
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        
        if not reverse:
            # Forward transformation
            h = self.transform_net(x1)
            scale, translate = torch.chunk(h, 2, dim=1)
            scale = torch.tanh(scale)  # Constrain scale
            
            y2 = (x2 - translate) * torch.exp(-scale)
            y1 = x1
            
            log_det = -torch.sum(scale, dim=1)
            
            y = torch.cat([y1, y2], dim=1)
            return y, log_det
        else:
            # Reverse transformation
            h = self.transform_net(x1)
            scale, translate = torch.chunk(h, 2, dim=1)
            scale = torch.tanh(scale)
            
            y2 = x2 * torch.exp(scale) + translate
            y1 = x1
            
            log_det = torch.sum(scale, dim=1)
            
            y = torch.cat([y1, y2], dim=1)
            return y, log_det

class NormalizingFlowModel(UnspecializedModel):
    """Normalizing Flow model for synthetic data generation"""
    
    def __init__(
        self,
        metadata: List[Dict[str, Any]],
        model_name: str,
        input_shape: str = None,
        load_path: str = None,
        num_flows: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
    ):
        super().__init__(metadata, model_name, input_shape, load_path)
        
        # Model hyperparameters
        self.num_flows = num_flows
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Model components
        self.flows = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training history
        self.training_losses = []

    def _build(self, input_shape: tuple[int, ...]):
        """Build the normalizing flow model"""
        input_dim = input_shape[-1]
        
        # Create coupling layers
        self.flows = nn.ModuleList()
        for i in range(self.num_flows):
            # Alternate mask patterns
            layer = CouplingLayer(input_dim, self.hidden_dim)
            self.flows.append(layer)
        
        # Move to device
        self.flows.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.flows.parameters(), lr=self.learning_rate)
        
        return self.flows

    def _load(self, model_filepath: str):
        """Load trained model"""
        checkpoint = torch.load(model_filepath, map_location=self.device)
        self.flows.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_losses = checkpoint.get("training_losses", [])

    def train(self, data: np.ndarray):
        """Train the normalizing flow"""
        if self.flows is None:
            self._instantiate()
        
        # Prepare data
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(data))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        self.flows.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch, in dataloader:
                batch = batch[0].to(self.device)
                
                # Forward pass through flows
                z = batch
                log_det_sum = 0
                
                for flow in self.flows:
                    z, log_det = flow(z)
                    log_det_sum += log_det
                
                # Calculate loss (negative log likelihood)
                # Standard normal prior
                log_prob_z = -0.5 * torch.sum(z**2, dim=1) - 0.5 * z.shape[1] * np.log(2 * np.pi)
                log_prob_x = log_prob_z + log_det_sum
                
                loss = -torch.mean(log_prob_x)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.training_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    def fine_tune(self, data: np.ndarray, **kwargs):
        """Fine-tune the model"""
        fine_tune_epochs = kwargs.get("epochs", 50)
        fine_tune_learning_rate = kwargs.get("learning_rate", self.learning_rate * 0.1)
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = fine_tune_learning_rate
        
        # Train for additional epochs
        original_epochs = self.epochs
        self.epochs = fine_tune_epochs
        self.train(data)
        self.epochs = original_epochs

    def infer(self, n_rows: int, **kwargs) -> np.ndarray:
        """Generate synthetic data using the trained flow model"""
        if self.flows is None:
            raise ValueError("Model must be trained before inference")
        
        self.flows.eval()
        
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(n_rows, self.flows[0].input_dim).to(self.device)
            
            # Reverse pass through flows
            x = z
            for flow in reversed(self.flows):
                x, _ = flow(x, reverse=True)
            
            synthetic_data = x.cpu().numpy()
        
        return synthetic_data

    def save(self, folder_path: str):
        """Save the model"""
        if self.flows is None:
            raise ValueError("No model to save")
        
        checkpoint = {
            "model_state_dict": self.flows.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_losses": self.training_losses,
            "model_config": {
                "num_flows": self.num_flows,
                "hidden_dim": self.hidden_dim,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "input_shape": self.input_shape
            }
        }
        
        torch.save(checkpoint, f"{folder_path}/{self.model_name}.pth")

    def set_hyperparameters(self, **kwargs):
        """Set model hyperparameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown hyperparameter: {key}")

    @classmethod
    def self_describe(cls):
        """Get model description"""
        return {
            "model_name": "NormalizingFlowModel",
            "description": "Normalizing Flow model for exact likelihood estimation and generation",
            "parameters": [
                "num_flows",
                "hidden_dim",
                "learning_rate",
                "epochs",
                "batch_size"
            ],
            "capabilities": [
                "tabular_data",
                "exact_likelihood",
                "invertible_transformations"
            ]
        }
```

### Example 3: Diffusion Model

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
from sdg_core_lib.data_generator.models.UnspecializedModel import UnspecializedModel

class DiffusionNet(nn.Module):
    """Neural network for diffusion models"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        
        layers = []
        in_dim = input_dim + 1  # +1 for time embedding
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else input_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
            in_dim = out_dim
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass with time embedding"""
        # Add time dimension
        t_emb = t.unsqueeze(1)
        x_with_time = torch.cat([x, t_emb], dim=1)
        
        return self.network(x_with_time)

class DiffusionModel(UnspecializedModel):
    """Denoising Diffusion Probabilistic Model"""
    
    def __init__(
        self,
        metadata: List[Dict[str, Any]],
        model_name: str,
        input_shape: str = None,
        load_path: str = None,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        super().__init__(metadata, model_name, input_shape, load_path)
        
        # Diffusion parameters
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Model hyperparameters
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Model components
        self.denoising_net = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Precompute diffusion schedule
        self._setup_diffusion_schedule()
        
        # Training history
        self.training_losses = []

    def _setup_diffusion_schedule(self):
        """Setup beta schedule and precompute values"""
        # Linear beta schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Precompute for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Move to device
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(self.device)

    def _build(self, input_shape: tuple[int, ...]):
        """Build the diffusion model"""
        input_dim = input_shape[-1]
        
        self.denoising_net = DiffusionNet(input_dim, self.hidden_dim)
        self.denoising_net.to(self.device)
        
        self.optimizer = optim.Adam(self.denoising_net.parameters(), lr=self.learning_rate)
        
        return self.denoising_net

    def _load(self, model_filepath: str):
        """Load trained model"""
        checkpoint = torch.load(model_filepath, map_location=self.device)
        self.denoising_net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_losses = checkpoint.get("training_losses", [])

    def _q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _p_losses(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute loss for denoising network"""
        noise = torch.randn_like(x_start)
        x_noisy = self._q_sample(x_start, t, noise)
        
        predicted_noise = self.denoising_net(x_noisy, t)
        
        return nn.MSELoss()(predicted_noise, noise)

    def train(self, data: np.ndarray):
        """Train the diffusion model"""
        if self.denoising_net is None:
            self._instantiate()
        
        # Prepare data
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(data))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        self.denoising_net.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch, in dataloader:
                batch = batch[0].to(self.device)
                
                # Sample random timesteps
                t = torch.randint(0, self.num_timesteps, (batch.shape[0],)).to(self.device)
                
                # Compute loss
                loss = self._p_losses(batch, t)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.training_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    def fine_tune(self, data: np.ndarray, **kwargs):
        """Fine-tune the model"""
        fine_tune_epochs = kwargs.get("epochs", 50)
        fine_tune_learning_rate = kwargs.get("learning_rate", self.learning_rate * 0.1)
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = fine_tune_learning_rate
        
        # Train for additional epochs
        original_epochs = self.epochs
        self.epochs = fine_tune_epochs
        self.train(data)
        self.epochs = original_epochs

    def _p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample from p(x_{t-1} | x_t)"""
        betas_t = self.betas[t].unsqueeze(1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].unsqueeze(1)
        
        # Predict noise
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.denoising_net(x, t) / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = betas_t
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def infer(self, n_rows: int, **kwargs) -> np.ndarray:
        """Generate synthetic data using the trained diffusion model"""
        if self.denoising_net is None:
            raise ValueError("Model must be trained before inference")
        
        self.denoising_net.eval()
        
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(n_rows, self.denoising_net.network[0].in_features - 1).to(self.device)
            
            # Reverse diffusion process
            for t in reversed(range(self.num_timesteps)):
                t_tensor = torch.full((n_rows,), t, dtype=torch.long).to(self.device)
                x = self._p_sample(x, t_tensor)
            
            synthetic_data = x.cpu().numpy()
        
        return synthetic_data

    def save(self, folder_path: str):
        """Save the model"""
        if self.denoising_net is None:
            raise ValueError("No model to save")
        
        checkpoint = {
            "model_state_dict": self.denoising_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_losses": self.training_losses,
            "model_config": {
                "num_timesteps": self.num_timesteps,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "hidden_dim": self.hidden_dim,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "input_shape": self.input_shape
            }
        }
        
        torch.save(checkpoint, f"{folder_path}/{self.model_name}.pth")

    def set_hyperparameters(self, **kwargs):
        """Set model hyperparameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown hyperparameter: {key}")

    @classmethod
    def self_describe(cls):
        """Get model description"""
        return {
            "model_name": "DiffusionModel",
            "description": "Denoising Diffusion Probabilistic Model for high-quality synthetic data generation",
            "parameters": [
                "num_timesteps",
                "beta_start",
                "beta_end",
                "hidden_dim",
                "learning_rate",
                "epochs",
                "batch_size"
            ],
            "capabilities": [
                "tabular_data",
                "high_quality_generation",
                "iterative_refinement"
            ]
        }
```

## Model Registration and Factory

Register your custom models for easy instantiation:

```python
from typing import Dict, Type
from sdg_core_lib.data_generator.models.UnspecializedModel import UnspecializedModel

class ModelRegistry:
    """Registry for custom models"""
    
    _models: Dict[str, Type[UnspecializedModel]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[UnspecializedModel]):
        """Register a model class"""
        cls._models[name] = model_class
    
    @classmethod
    def get_model(cls, name: str) -> Type[UnspecializedModel]:
        """Get a model class by name"""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}")
        return cls._models[name]
    
    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all registered models with descriptions"""
        return {
            name: model_class.self_describe()
            for name, model_class in cls._models.items()
        }

# Register custom models
ModelRegistry.register("autoencoder", AutoencoderModel)
ModelRegistry.register("normalizing_flow", NormalizingFlowModel)
ModelRegistry.register("diffusion", DiffusionModel)

class ModelFactory:
    """Factory for creating model instances"""
    
    @staticmethod
    def create_model(
        model_type: str,
        metadata: List[Dict[str, Any]],
        model_name: str,
        **kwargs
    ) -> UnspecializedModel:
        """Create a model instance"""
        model_class = ModelRegistry.get_model(model_type)
        return model_class(metadata, model_name, **kwargs)
```

## Integration with Existing Systems

### Configuration-Based Model Creation

```python
def create_model_from_config(config: Dict[str, Any]) -> UnspecializedModel:
    """Create model from configuration dictionary"""
    model_type = config["model_type"]
    metadata = config.get("metadata", [])
    model_name = config.get("model_name", f"{model_type}_model")
    
    # Extract model-specific parameters
    model_params = {k: v for k, v in config.items() 
                    if k not in ["model_type", "metadata", "model_name"]}
    
    return ModelFactory.create_model(
        model_type=model_type,
        metadata=metadata,
        model_name=model_name,
        **model_params
    )

# Example configuration
model_config = {
    "model_type": "autoencoder",
    "model_name": "my_autoencoder",
    "metadata": [],
    "encoding_dim": 64,
    "hidden_layers": [256, 128],
    "learning_rate": 0.001,
    "epochs": 200
}

model = create_model_from_config(model_config)
```

## Testing Custom Models

```python
import pytest
import numpy as np
from your_module import AutoencoderModel, NormalizingFlowModel, DiffusionModel

class TestCustomModels:
    def test_autoencoder_model(self):
        """Test autoencoder model functionality"""
        # Create dummy metadata
        metadata = [{"feature_name": "test", "feature_type": "continuous"}]
        
        model = AutoencoderModel(
            metadata=metadata,
            model_name="test_autoencoder",
            input_shape="(10,)",
            encoding_dim=4,
            epochs=2  # Short training for test
        )
        
        # Test model building
        model._instantiate()
        assert model.autoencoder is not None
        
        # Test training
        data = np.random.randn(100, 10)
        model.train(data)
        assert model.training_info is not None
        
        # Test inference
        synthetic_data = model.infer(10)
        assert synthetic_data.shape == (10, 10)

    def test_normalizing_flow_model(self):
        """Test normalizing flow model functionality"""
        metadata = [{"feature_name": "test", "feature_type": "continuous"}]
        
        model = NormalizingFlowModel(
            metadata=metadata,
            model_name="test_flow",
            input_shape="(5,)",
            num_flows=2,
            epochs=2  # Short training for test
        )
        
        # Test training
        data = np.random.randn(50, 5)
        model.train(data)
        assert len(model.training_losses) > 0
        
        # Test inference
        synthetic_data = model.infer(10)
        assert synthetic_data.shape == (10, 5)

    def test_diffusion_model(self):
        """Test diffusion model functionality"""
        metadata = [{"feature_name": "test", "feature_type": "continuous"}]
        
        model = DiffusionModel(
            metadata=metadata,
            model_name="test_diffusion",
            input_shape="(3,)",
            num_timesteps=10,  # Short for test
            epochs=2
        )
        
        # Test training
        data = np.random.randn(30, 3)
        model.train(data)
        assert len(model.training_losses) > 0
        
        # Test inference
        synthetic_data = model.infer(5)
        assert synthetic_data.shape == (5, 3)
```

## Best Practices

1. **Model Architecture**: Design architectures appropriate for your data type
2. **Hyperparameter Tuning**: Provide sensible defaults and easy parameter adjustment
3. **Training Monitoring**: Include training history and loss tracking
4. **Model Persistence**: Implement proper save/load functionality
5. **Error Handling**: Handle edge cases and invalid inputs gracefully
6. **Documentation**: Provide clear docstrings and parameter descriptions
7. **Testing**: Create comprehensive tests for all model functionality
8. **Performance**: Optimize for training and inference speed
9. **Memory Management**: Handle large datasets efficiently
10. **Reproducibility**: Ensure consistent results across runs

## Advanced Features

### 1. Ensemble Models

```python
class EnsembleModel(UnspecializedModel):
    """Ensemble of multiple models"""
    
    def __init__(self, models: List[UnspecializedModel], ensemble_method: str = "average"):
        super().__init__([], "ensemble")
        self.models = models
        self.ensemble_method = ensemble_method
    
    def infer(self, n_rows: int, **kwargs) -> np.ndarray:
        """Generate data using ensemble"""
        predictions = []
        
        for model in self.models:
            pred = model.infer(n_rows, **kwargs)
            predictions.append(pred)
        
        if self.ensemble_method == "average":
            return np.mean(predictions, axis=0)
        elif self.ensemble_method == "median":
            return np.median(predictions, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
```

### 2. Adaptive Models

```python
class AdaptiveModel(UnspecializedModel):
    """Model that adapts based on data characteristics"""
    
    def __init__(self, model_candidates: Dict[str, UnspecializedModel]):
        super().__init__([], "adaptive")
        self.model_candidates = model_candidates
        self.selected_model = None
    
    def train(self, data: np.ndarray):
        """Select and train best model based on data characteristics"""
        # Analyze data
        data_characteristics = self._analyze_data(data)
        
        # Select best model
        best_model_name = self._select_best_model(data_characteristics)
        self.selected_model = self.model_candidates[best_model_name]
        
        # Train selected model
        self.selected_model.train(data)
    
    def _analyze_data(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze data characteristics"""
        return {
            "dimensionality": data.shape[1],
            "sample_size": data.shape[0],
            "correlation": np.corrcoef(data.T).mean(),
            "entropy": self._calculate_entropy(data)
        }
```

This guide provides the foundation for creating custom models. Adapt the examples to your specific synthetic data generation requirements and use cases.
