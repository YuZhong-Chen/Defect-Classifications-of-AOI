import torch
from torch import nn

from pathlib import Path


class SimpleModel(nn.Module):
    def __init__(self, num_classes: int = 6):
        super(SimpleModel, self).__init__()

        # Input size: 1x512x512
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=3, padding=3),  # 32x171x171
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, stride=3, padding=3),  # 64x57x57
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=7, stride=3, padding=3),  # 128x19x19
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(46208, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def SaveModel(model: nn.Module, path: Path) -> None:
    print(f"Saving model to {path}")
    path.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict()}, path / "model.pth")
    return
