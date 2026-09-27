import os
import tarfile
import urllib.request

import pandas as pd
import torch
from torch.utils.data import Dataset

DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
MAX_LENGTH = 128


def download_imdb(data_dir="."):
    os.makedirs(data_dir, exist_ok=True)
    archive = os.path.join(data_dir, "aclImdb_v1.tar.gz")
    root = os.path.join(data_dir, "aclImdb")
    if not os.path.exists(archive):
        print(f"Downloading {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, archive)
    if not os.path.isdir(root):
        print(f"Extracting {archive} ...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(data_dir, filter="data")
    return root


def load_dataset(folder):
    reviews = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as file:
                reviews.append(file.read())
    return reviews


def load_split(root, split):
    pos = load_dataset(os.path.join(root, split, "pos"))
    neg = load_dataset(os.path.join(root, split, "neg"))
    return pd.DataFrame({
        "review": pos + neg,
        "label": [1] * len(pos) + [0] * len(neg),
    })


def train_val_split(df, train_frac=0.9, seed=42):
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_size = int(train_frac * len(shuffled))
    return shuffled.iloc[:train_size], shuffled.iloc[train_size:]


class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=MAX_LENGTH):
        self.max_length = max_length
        encoding = tokenizer(
            data["review"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encoding["input_ids"]
        self.labels = torch.tensor(data["label"].values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]
