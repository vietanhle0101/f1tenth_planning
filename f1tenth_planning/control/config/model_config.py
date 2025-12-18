from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Neural model/value-function hyperparameters (kept simple; extend as needed).
    """

    hidden_dim: int = 256
    hidden_layers: int = 4
    learning_rate: float = 5e-4
    model_type: str = "nf"  # options: nf, bnn, mlp
    max_epoch: int = 200
    ensemble_size: int = 1
