import torch
from torch import nn

from pathlib import Path


class SimpleModel(nn.Module):
    def __init__(self, num_classes: int = 6, dropout_rate: float = 0.5):
        super(SimpleModel, self).__init__()

        self.input_size = (1, 512, 512)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=3, padding=3),  # 32x171x171
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=7, stride=3, padding=3),  # 64x57x57
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=7, stride=3, padding=3),  # 128x19x19
            nn.LeakyReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(46208, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes),
        )

        # Initialize the network
        self.InitNetwork()

    def InitNetwork(self) -> None:
        for layer in (self.features, self.classifier):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("leaky_relu"))
                nn.init.zeros_(layer.bias, 0)

    def forward(self, x) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def SaveModel(model: nn.Module, path: Path) -> None:
    print(f"Saving model to {path}")
    path.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict()}, path / "model.pth")
    return
