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
config["TRAINING_SET_RATIO"] = 0.8
config["NUM_EPOCHS"] = 500
config["LEARNING_RATE"] = 0.00001
config["TEST_PERIOD"] = 5
config["OPTIMIZER"] = "Adam"  # "Adam" or "SGD"
config["LOSS_FUNC"] = "CrossEntropy"  # "MSE" or "CrossEntropy"
config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

# Enable the cudnn benchmark
if config["DEVICE"] == "cuda":
    torch.backends.cudnn.benchmark = True

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
    # NOTE: If the GPU memory is not enough, set the device to "cpu" and move the data to the GPU when training.
    dataset = AOIDataset(is_train=True, device=config["DEVICE"], load_data=True, use_transform=True)
    train_dataloader, test_dataloader = GetDataLoader(dataset, batch_size=config["BATCH_SIZE"], train_size=config["TRAINING_SET_RATIO"])

    # Create the model
    model = SimpleModel().to(config["DEVICE"])
    summary(model, input_size=model.input_size, device=config["DEVICE"])

    # Specify the loss function
    if config["LOSS_FUNC"] == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    elif config["LOSS_FUNC"] == "MSE":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Loss function {config['loss']} not supported")

    # Specify the optimizer
    if config["OPTIMIZER"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    elif config["OPTIMIZER"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["LEARNING_RATE"], momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    # Start training
    for epoch in range(config["NUM_EPOCHS"]):
        # Train the model
        model.train()
        train_correct = 0
        train_loss = 0.0
        for batch_idx, (images, labels) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # L2 regularization
            l2_lambda = 0.0001
            l2_norm = sum(param.pow(2.0).sum() for param in model.parameters())
            loss += l2_lambda * l2_norm

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()

        # Calculate the training accuracy
        train_accuracy = train_correct / len(train_dataloader.dataset)
        train_loss = train_loss / len(train_dataloader.dataset)
        print(f"Epoch: {epoch + 1}/{config['NUM_EPOCHS']}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        # Test the model
        test_accuracy = None
        test_loss = None
        if (epoch + 1) % config["TEST_PERIOD"] == 0:
            print("Testing the model...")
            model.eval()
            test_correct = 0
            test_loss = 0.0
            with torch.no_grad():
                for images, labels in test_dataloader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    test_correct += (predicted == labels).sum().item()

            # Calculate the test accuracy
            test_accuracy = test_correct / len(test_dataloader.dataset)
            test_loss = test_loss / len(test_dataloader.dataset)
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Logging the results
        logger.Log(epoch + 1, train_loss, train_accuracy, test_loss, test_accuracy)

    # Save the model
    SaveModel(model, project_dir / "weights")


if __name__ == "__main__":
    main()
