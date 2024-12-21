import pandas as pd
from pathlib import Path

# Load the dataset
dataset_path = Path(__file__).parent.parent / "data" / "train.csv"
df = pd.read_csv(dataset_path)
label_counts = df["Label"].value_counts()

# Print the dataset information
print(f"Total data: {len(df)}")
print(f"Total labels: {len(label_counts)}")
print(label_counts)