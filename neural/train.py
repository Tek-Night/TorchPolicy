import csv
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from neural.model import TorchPolicyMLP
from data.label import FEATURES, METRICS, POLICIES

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT/"data"/"soft_targets.csv"
MODEL_PATH = ROOT/"neural"/"torchpolicy.pt"

SEED = 42
BATCH_SIZE= 256
EPOCHS = 60
PATIENCE = 10
LR = 0.001

#input and output columns
INPUT_COLUMNS = FEATURES + [f"weight_{metric}" for metric in METRICS]
TARGET_COLUMNS = [f"target_{policy}" for policy in POLICIES]
SCORE_COLUMNS = [f"score_{policy}" for policy in POLICIES]

def load_data():

    with DATA_PATH.open("r", newline = "") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("soft_targets.csv is empty!")

    # extracting 17 input features

    X = np.array(
        [
            [float(row[col]) for col in INPUT_COLUMNS]
            for row in rows
        ],
        dtype = np.float32
    )

    # extracting 6 soft target probabilities
    
    targets = np.array(
        [
            [float(row[col]) for col in TARGET_COLUMNS] for row in rows
        ],
        dtype = np.float32
    )

    # extracting actual scheduler scores
    scores = np.array(
        [
            [float(row[col]) for col in SCORE_COLUMNS] for row in rows
        ],
        dtype = np.float32
    )

    # need workload IDs for splitting dataset later

    workload_ids = np.array(
        [int(row["workload_id"]) for row in rows],
        dtype = np.int64
    )

    if not np.allclose(targets.sum(axis = 1), 1.0, atol = 1e-5):
        raise ValueError("Soft target probabilities do not add up to 1 !!!")

    return X, targets, scores, workload_ids


# loading data 

def split_data(X, targets, scores, workload_ids):

    unique_ids = np.unique(workload_ids)

    # shuffling 
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_ids)

    # splitting boundaries
    train_end = int(0.7*len(unique_ids))
    val_end = train_end + int(0.15*len(unique_ids))

    train_ids = unique_ids[:train_end]
    val_ids = unique_ids[train_end:val_end]
    test_ids = unique_ids[val_end:]

    # ensuring all 21 preference combinations of a certain workload stay together
    def select_data(ids):

        mask = np.isin(workload_ids, ids)

        return (  
            X[mask],
            targets[mask],
            scores[mask]
        )

    train_data = select_data(train_ids)
    val_data = select_data(val_ids)
    test_data = select_data(test_ids)


    return train_data, val_data, test_data

