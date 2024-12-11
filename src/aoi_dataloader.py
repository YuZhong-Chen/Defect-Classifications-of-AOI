import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, random_split


class AOIDataset(Dataset):
    def __init__(self, is_train=True):
        # Path
        self.root_dir = Path(__file__).parent.parent
        self.data_dir = self.root_dir / "data"
        self.test_images_dir = self.data_dir / "test_images"
        self.test_csv = self.data_dir / "test.csv"
        self.train_images_dir = self.data_dir / "train_images"
        self.train_csv = self.data_dir / "train.csv"

        # Variables
        self.img_labels = pd.read_csv(self.train_csv) if is_train else pd.read_csv(self.test_csv)
        self.img_dir = self.train_images_dir if is_train else self.test_images_dir

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx) -> tuple:
        img_path = self.img_dir / self.img_labels.iloc[idx, 0]
        image = read_image(str(img_path))
        label = self.img_labels.iloc[idx, 1]
        return image, label


def DisplayDatasetImage(dataset: Dataset, idx: int = 0) -> None:
    image = dataset[idx][0].numpy().transpose(1, 2, 0)
    label = dataset[idx][1]
    print(f"Index: {idx}, Label: {label}")
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    return


def SplitDataset(dataset: Dataset, train_size: float = 0.8) -> tuple[Dataset, Dataset]:
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


def GetDataLoader(dataset: Dataset, batch_size: int = 4, train_size: float = 0.8) -> tuple[DataLoader, DataLoader]:
    train_dataset, test_dataset = SplitDataset(dataset, train_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
