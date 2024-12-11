import os
import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model import SimpleModel, LoadModel
from aoi_dataloader import AOIDataset

# Project Configurations
PROJECT_NAME = "aoi-classification-basic-12-12-00-52"

# Training Configurations
config = {}
config["BATCH_SIZE"] = 128
config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

# Init the project directory
root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
checkpoint_dir = root_dir / "checkpoints"
dataset_dir = root_dir / "data"
project_dir = checkpoint_dir / PROJECT_NAME
result_dir = project_dir / "results"
result_dir.mkdir(exist_ok=True)
print(f"Project: {PROJECT_NAME}")


def main():
    # Load the evaluation dataset
    eval_dataset = AOIDataset(is_train=False, device=config["DEVICE"])
    eval_dataloader = DataLoader(eval_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)
    predicted_labels = eval_dataset.img_labels.copy()

    # Create the model
    model = SimpleModel().to(config["DEVICE"])
    model = LoadModel(model, project_dir / "weights")

    # Start evaluating the model
    print("Evaluating the model...")
    model.eval()
    current_index = 0
    with torch.no_grad():
        for images, _ in tqdm(eval_dataloader):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().int().numpy()
            predicted_labels.iloc[current_index : current_index + len(predicted), 1] = predicted
            current_index += len(predicted)
    predicted_labels["Label"] = predicted_labels["Label"].astype(int)
    print("Evaluation completed")

    # Save the predicted labels
    predicted_labels.to_csv(result_dir / "predicted_labels.csv", index=False)


if __name__ == "__main__":
    main()
