import os
import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torchsummary import summary

from model import SimpleModel, SaveModel
from aoi_dataloader import AOIDataset, GetDataLoader
from logger import Logger

# Project Configurations
PROJECT = "aoi-classification"
PROJECT_NAME = PROJECT + "-basic-" + datetime.datetime.now().strftime("%m-%d-%H-%M")
ENABLE_LOGGER = False
ENABLE_WANDB = False

# Training Configurations
config = {}
config["BATCH_SIZE"] = 128
config["TRAIN_SIZE"] = 0.8
config["NUM_EPOCHS"] = 50
config["LEARNING_RATE"] = 0.0002
config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

# Init the project directory
checkpoint_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)
project_dir = checkpoint_dir / PROJECT_NAME
project_dir.mkdir(exist_ok=True)
print(f"Project: {PROJECT_NAME}")

# Init the logger
logger = Logger(PROJECT, PROJECT_NAME, config, project_dir, enable=ENABLE_LOGGER, use_wandb=ENABLE_WANDB)


def main():
    # Load the dataset and dataloader
    dataset = AOIDataset(is_train=True, device=config["DEVICE"])
    train_dataloader, test_dataloader = GetDataLoader(dataset, batch_size=config["BATCH_SIZE"], train_size=config["TRAIN_SIZE"])

    # Create the model
    model = SimpleModel().to(config["DEVICE"])
    summary(model, input_size=model.input_size, device=config["DEVICE"])

    # Specify the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    # Train the model
    for epoch in range(config["NUM_EPOCHS"]):
        total_correct = 0
        total_loss = 0.0

        model.train()
        for batch_idx, (images, labels) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()

        # Logging the results
        accuracy = total_correct / len(train_dataloader.dataset)
        loss = total_loss / len(train_dataloader.dataset)
        logger.Log(epoch + 1, loss, accuracy)
        print(f"Epoch: {epoch + 1}/{config['NUM_EPOCHS']}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save the model
    SaveModel(model, project_dir / "weights")


if __name__ == "__main__":
    main()
