from dataclasses import dataclass, field


@dataclass()
class ImageParams:
    width: int = field(default=128)
    height: int = field(default=128)
    n_channels: int = field(default=3)
