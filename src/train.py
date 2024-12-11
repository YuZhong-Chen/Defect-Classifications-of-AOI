import numpy as np
from aoi_dataloader import AOIDataset, DisplayDatasetImage, GetDataLoader
from tqdm import tqdm


def main():
    dataset = AOIDataset(is_train=True)

    train_dataloader, test_dataloader = GetDataLoader(dataset, batch_size=4, train_size=0.8)

    for batch_idx, (images, labels) in tqdm(enumerate(train_dataloader)):
        pass


if __name__ == "__main__":
    main()
