import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torchsummary import summary

from model import SimpleModel
from aoi_dataloader import AOIDataset, DisplayDatasetImage, GetDataLoader


# Configurations
BATCH_SIZE = 32
TRAIN_SIZE = 0.8
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    dataset = AOIDataset(is_train=True)
    train_dataloader, test_dataloader = GetDataLoader(dataset, batch_size=BATCH_SIZE, train_size=TRAIN_SIZE)

    model = SimpleModel()
    model.to(DEVICE)

    summary(model, input_size=(1, 512, 512), device=DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_correct = 0
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(tqdm(train_dataloader)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()

        accuracy = total_correct / len(train_dataloader.dataset)
        loss = total_loss / len(train_dataloader.dataset)
        print(f"Epoch: {epoch+1}/{NUM_EPOCHS}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
