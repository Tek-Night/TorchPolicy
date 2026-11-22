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

# device agonostic code
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

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


def normalize_data(train_data, val_data, test_data):

    # training inputs
    X_train = train_data[0]

    # calculating mean and standard deviation (only using training data)
  
    mean = X_train.mean(axis = 0)
    std = X_train.std(axis = 0)

    # avoid / by 0 
    std[std == 0] = 1.0

    # normaliziing 
    X_train = (train_data[0] - mean) / std
    X_val = (val_data[0] - mean) / std
    X_test = (test_data[0] - mean)/ std

    # preserving targets & actual score
    train_data = (X_train, train_data[1], train_data[2])
    val_data = (X_val, val_data[1], val_data[2])
    test_data = (X_test, test_data[1], test_data[2])

    return train_data, val_data, test_data, mean, std


# dataloaders finally!!!

def create_dataloaders(train_data, val_data, test_data):

    def make_loader(data, shuffle):

        # np arrays -> tensors
        X_tensor = torch.tensor(data[0], dtype = torch.float32) 

        targets_tensor = torch.tensor(data[1], dtype = torch.float32)

        dataset = TensorDataset(X_tensor, targets_tensor)

        loader = DataLoader(dataset,
                            batch_size = BATCH_SIZE,
                            shuffle = shuffle)

        return loader

    train_loader = make_loader(train_data, shuffle = True)
    val_loader = make_loader(val_data, shuffle= False)
    test_loader = make_loader(test_data, shuffle = False)

    return train_loader, val_loader, test_loader



def soft_target_loss(logits, targets):

    log_probs = F.log_softmax(logits, dim = 1)

    loss = -(targets * log_probs).sum(dim = 1).mean()

    return loss



def train_single_epoch(model, loader, optimizer, device):

    model.train()

    total_loss = 0.0

    for X_batch, targets_batch in loader:

        X_batch = X_batch.to(device)
        targets_batch = targets_batch.to(device)

        optimizer.zero_grad(set_to_none = True)

        logits = model(X_batch)

        loss = soft_target_loss(logits, targets_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()* X_batch.size(0)

    return total_loss/len(loader.dataset)



def eval_loss(model, loader, device):

    total_loss = 0.0

    model.eval() # eval mode 
    with torch.no_grad():

        for X_batch, targets_batch in loader:

            X_batch = X_batch.to(device)
            targets_batch = targets_batch.to(device)

            logits = model(X_batch)

            loss = soft_target_loss(logits, targets_batch)

            total_loss += loss.item()*X_batch.size(0)

    return total_loss/len(loader.dataset)



# ------------------------------------------------------------------------------------------
#                                    MAIN TRAINING LOOP
# ------------------------------------------------------------------------------------------

def train_model(train_loader, val_loader, mean, std, device):

    model = TorchPolicyMLP(
        input_dim= len(INPUT_COLUMNS),
        output_dim= len(POLICIES)
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = LR
    )

    best_val_loss = float("inf")
    epochs_without_improv = 0

    MODEL_PATH.parent.mkdir(parents = True, exist_ok= True)

    print(f"Using device: {device}")

    for epoch in range(EPOCHS):

        train_loss = train_single_epoch(model, train_loader, optimizer, device)

        val_loss = eval_loss(model, val_loader, device= device)

        print(f"Epoch[{epoch+1}/{EPOCHS}]   |   Train loss: {train_loss:.4f}    |   Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss - 1e-6:

            best_val_loss = val_loss
            epochs_without_improv = 0

            # save new best model
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_mean": mean.tolist(),
                "input_std": std.tolist(),
                "input_columns": INPUT_COLUMNS,
                "policies": POLICIES,
                "best_val_loss": best_val_loss
            }, MODEL_PATH)
        else:
            epochs_without_improv+=1

        # early stop to not waste time and compute
        if epochs_without_improv >= PATIENCE:
            print("EARLY STOPPING INITIATED.")
            break

    checkpoint = torch.load(
        MODEL_PATH,
        map_location = device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved. [ {MODEL_PATH} ]")

    return model



def evaluate_decisions(model, test_data, device):

    X_test, _, scores = test_data

    model.eval()

    predictions = []

    with torch.no_grad():

        for start in range(0, len(X_test), BATCH_SIZE):

            X_batch = torch.tensor(
                X_test[start:start + BATCH_SIZE],
                dtype=torch.float32
            ).to(device)

            logits = model(X_batch)

            predicted = logits.argmax(dim=1)

            predictions.extend(
                predicted.cpu().numpy()
            )

    predictions = np.array(predictions)

    # Actual score of the selected policies
    rows = np.arange(len(scores))

    selected_scores = scores[rows, predictions]

    # Best possible score for each example
    optimal_scores = scores.min(axis=1)

    # Calculate regret
    regrets = selected_scores - optimal_scores

    # Count selections that are optimal, including ties
    optimal_choices = np.isclose(
        selected_scores,
        optimal_scores,
        atol=1e-6,
        rtol=1e-5
    )

    optimal_rate = optimal_choices.mean() * 100
    mean_regret = regrets.mean()

    print("\n========== TORCHPOLICY TEST RESULTS ==========")

    print(f"Optimal choice rate: {optimal_rate:.2f}%")
    print(f"Mean regret: {mean_regret:.6f}")

    print("\nPolicy selection distribution:")

    counts = np.bincount(
        predictions,
        minlength=len(POLICIES)
    )

    for policy, count in zip(POLICIES, counts):

        print(
            f"{policy:15s}: {count:4d} "
            f"({count / len(predictions) * 100:.2f}%)"
        )

    # Baseline: Always choose SRTF
    srtf_index = POLICIES.index("srtf")

    srtf_scores = scores[:, srtf_index]

    srtf_regrets = srtf_scores - optimal_scores

    srtf_optimal_rate = np.isclose(
        srtf_scores,
        optimal_scores,
        atol=1e-6,
        rtol=1e-5
    ).mean() * 100

    print("\n========== SRTF BASELINE ==========")

    print(f"Optimal choice rate: {srtf_optimal_rate:.2f}%")
    print(f"Mean regret: {srtf_regrets.mean():.6f}")

    print("==============================================")



# run it
if __name__ == "__main__":

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load dataset
    X, targets, scores, workload_ids = load_data()

    # Split by workload id
    train_data, val_data, test_data = split_data(X, targets, scores, workload_ids)

    # Normalize input
    train_data, val_data, test_data, mean, std = normalize_data(train_data, val_data, test_data)

    # create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data)

    model = train_model(train_loader = train_loader,
                        val_loader= val_loader,
                        mean = mean,
                        std = std,
                        device= device)

    # Evaluate the best checkpoint on unseen test workloads

    test_loss = eval_loss(
        model,
        test_loader,
        device
    )

    print(f"\nTest loss: {test_loss:.5f}")

    evaluate_decisions(
        model,
        test_data,
        device
    )