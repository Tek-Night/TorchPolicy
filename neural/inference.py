import torch
import json
from pathlib import Path
import numpy as np
from neural.model import TorchPolicyMLP
from data.label import FEATURES, METRICS, POLICIES, PROFILES
from data.generate import extract_features

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT/ "neural" / "torchpolicy.pt"
WORKLOADS_PATH = ROOT / "data" / "workloads.jsonl"

#device agnostic code
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location = torch.device(device))

    expected_columns = FEATURES+ [f"weight_{metric}" for metric in METRICS]

    # ensuring checkpoint matches the current proj
    if checkpoint["input_columns"] != expected_columns:
        raise ValueError("Model input columns do not match! :(")
    if checkpoint["policies"] != POLICIES:
        raise ValueError("Model policy ordering does not match! :(")

    mean = np.array(checkpoint["input_mean"], dtype = np.float32)
    std = np.array(checkpoint["input_std"], dtype = np.float32)

    model = TorchPolicyMLP(input_dim=len(expected_columns),
                           output_dim=len(POLICIES)).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval() # evaluation mode

    print(f"Loaded model on {device}")

    return model, mean, std, device


def validate_preferences(preferences):

    if isinstance(preferences, str):
        if preferences not in PROFILES:
            raise ValueError(f"Unkown preference profile: {preferences}!!!")

        preferences = PROFILES[preferences]

    if set(preferences.keys()) != set(METRICS):
        raise ValueError("Invalid preference metrics!")

    weights = {metric: float(preferences[metric]) for metric in METRICS}

    values = np.array(list(weights.values()))

    if not np.all(np.isfinite(values)):
        raise ValueError("Preference weights need to be finite.")
    if np.any(values < 0):
        raise ValueError("Preference weights cannot be -ve.")
    if not np.isclose(sum(values), 1.0, atol = 1e-5):
        raise ValueError("Preference weights must sum to 1")

    return weights


# forward pass
def predict_policy(model, processes, preferences, mean, std, device = device):
    if not processes:
        raise ValueError("Workload is empty!!!")

    weights = validate_preferences(preferences)

    #extracting 12 workload features
    features = extract_features(processes)

    #17 dim input
    input_values = ([features[feature] for feature in FEATURES] + [weights[metric] for metric in METRICS])

    x = np.array(input_values, dtype = np.float32).reshape(1, -1)

    # applying training-set norm
    X = (x-mean)/std

    X_tensor = torch.from_numpy(X).to(device)

    #  forward pass
    model.eval() # evaluation mode

    with torch.no_grad():

        Z = model(X_tensor)

        Z_probs = torch.softmax(Z, dim = 1)

        Z_probs = Z_probs.cpu().numpy()[0]

        # Policy w/ highest probility

        best_index = int(np.argmax(Z_probs))

        policy = POLICIES[best_index]

        probability_dict = {policy: float(Z_probs[i]) for i, policy in enumerate(POLICIES)}

        return {"recommended_policy": policy,
                "probabilities": probability_dict,
                "features": features,
                "weights": weights}


def main():

    model, mean, std, device = load_model()

    with WORKLOADS_PATH.open("r") as file:
        workload = json.loads(file.readline())

    processes = workload["processes"]

    print(" WELCOME TO TORCH POLICY!\n")

    print(f"Workload id: {workload['workload_id']}\nNumber of processes: {len(processes)}")

    print("PROCESSES:")
    for process in processes:
        print(process)

    for profile in PROFILES:
        result = predict_policy(model, processes=processes, preferences= profile, mean = mean, std = std, device= device)

        print(f"\nPreference Profile: {profile}")
        print(f"Recommended policy: {result['recommended_policy']}")

        for policy, probability in result["probabilities"].items():

            print(f"{policy:15}: {probability:.3f}")

if __name__ == "__main__":
    main()