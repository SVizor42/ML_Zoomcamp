from dataclasses import dataclass, field
from typing import Optional


@dataclass()
class TrainingParams:
    inner_size: Optional[int]
    drop_rate: Optional[float]
    base_model: str = field(default="Xception")
    use_inner_layer: bool = field(default=False)
    use_dropout: bool = field(default=False)
    optimizer: str = field(default='SGD')
    learning_rate: float = field(default=0.001)
    batch_size: int = field(default=32)
    epochs: int = field(default=10)
