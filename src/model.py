import torch
from torch import nn

from pathlib import Path


class SimpleModel(nn.Module):
    def __init__(self, num_classes: int = 6, dropout_rate: float = 0.5):
        super(SimpleModel, self).__init__()

        self.input_size = (1, 256, 256)
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

        # Initialize the network
        self.InitNetwork()

    def InitNetwork(self) -> None:
        for layer in (self.features, self.classifier):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(layer.bias, 0)

    def forward(self, x) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def SaveModel(model: nn.Module, path: Path) -> None:
    print(f"Saving model to {path}")
    path.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict()}, path / "model.pth")
    return


def LoadModel(model: nn.Module, path: Path) -> nn.Module:
    print(f"Loading model from {path}")
    model.load_state_dict(torch.load(path / "model.pth")["model"])
    return model
