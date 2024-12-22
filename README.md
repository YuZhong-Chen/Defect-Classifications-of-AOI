# Defect-Classifications-of-AOI

> Please modify the configurations in `src/train.py` and `src/eval.py` to fit your needs.

## Download the dataset

Please download the dataset from the following link: [AOI Dataset](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4?fbcli%20d=IwAR3NR9UiSB0_u4GxvOfc_xs6b7Bw0dLJfCMnJWpFb5xazd6vzB%20q5bV7ofDs), and put the dataset in the `data` folder.

Note that the dataset is compressed in the `zip` format, and you need to decompress it before using it.

Structure of the `data` folder:
```
data
├── .gitkeep
├── test_images
|   ├── test_00000.png
|   ├── test_00001.png
|   └── ...
├── test.csv
├── train_images
|   ├── train_00000.png
|   ├── train_00001.png
|   └── ...
└── train.csv
```

## Run the Docker container

We have prepared a Docker container for you to run the code. Please follow the instructions below to run the Docker container.

```bash
# Change the directory to the docker folder
cd docker

# Build the Docker image
docker compose build

# Run the Docker container
docker compose up -d
```

## Train the model

After running the Docker container, you can train the model by running the following command:

```bash
# Inside the Docker container
python src/train.py
```

The results will be saved in the `checkpoints` folder.

## Evaluate the model

After training the model, you can evaluate the model by running the following command:

```bash
# Inside the Docker container
python src/eval.py
```