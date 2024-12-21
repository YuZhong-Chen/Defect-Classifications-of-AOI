import cv2
import PIL.Image as Image
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torchvision.io import read_image

image_name = "train_00002_1.png"
# image_name = "train_00003_5.png"
# image_name = "train_00007_0.png"
# image_name = "train_00028_4.png"
# image_name = "train_00088_3.png"
# image_name = "train_00125_2.png"

image_path = Path(__file__).parent.parent / "figure" / "data_demo" / image_name

# Load the image
image = Image.open(image_path)

# Apply the transformations
image_horizontal_flip = transforms.RandomHorizontalFlip(p=1)(image)
image_vertical_flip = transforms.RandomVerticalFlip(p=1)(image_horizontal_flip)
image_rotate = transforms.RandomRotation(10)(image_vertical_flip)
image_tensor = transforms.ToTensor()(image_rotate)
image_normalize = transforms.Normalize((0.5), (0.1))(image_tensor)

# Display the image
plt.figure(figsize=(15, 4))

plt.subplot(1, 5, 1)
plt.axis("off")
plt.title("Original image")
original_image = np.array(image)
plt.imshow(original_image, cmap="gray", vmin=0, vmax=255)

plt.subplot(1, 5, 2)
plt.axis("off")
plt.title("Horizontal flip")
horizontal_flip_image = np.array(image_horizontal_flip)
plt.imshow(horizontal_flip_image, cmap="gray", vmin=0, vmax=255)

plt.subplot(1, 5, 3)
plt.axis("off")
plt.title("Vertical flip")
vertical_flip_image = np.array(image_vertical_flip)
plt.imshow(vertical_flip_image, cmap="gray", vmin=0, vmax=255)

plt.subplot(1, 5, 4)
plt.axis("off")
plt.title("Rotate")
rotate_image = np.array(image_rotate)
plt.imshow(rotate_image, cmap="gray", vmin=0, vmax=255)

plt.subplot(1, 5, 5)
plt.axis("off")
plt.title("Normalize")
normalize_image = image_normalize.permute(1, 2, 0).numpy()
plt.imshow(normalize_image)

plt.show()
