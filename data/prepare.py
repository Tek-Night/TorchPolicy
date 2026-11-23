
import csv
import math
import random
from pathlib import Path
from data.label import (POLICIES, PROFILES, FEATURES, METRICS, calculate_scores)
DATA_DIR = Path(__file__).resolve().parent

TEMPERATURE = 0.1
RANDOM_PREFERENCES = 16
SEED = 42


def generate_preferences(rng):
    """
    Generates random weights that sum to 1!
    """
    values = [rng.gammavariate(0.7, 1.0) for _ in METRICS]

    total = sum(values)

    return {metric: value / total for metric, value in zip(METRICS, values)}


def create_soft_targets(scores, temperature=0.1):
    """
    Schedular costs -> probability distribution
    Lower cost = higher target probability
    """
    if temperature <= 0:
        raise ValueError("temperature must be +ve !!!")

    minimum = min(scores.values())

    exponentials = {policy: math.exp(-(scores[policy] - minimum) / temperature) for policy in POLICIES}

    total = sum(exponentials.values())

    return {policy: exponentials[policy] / total for policy in POLICIES}


def create_training_row(workload, profile, weights):

    # calculating real costs for the scheduling algorithms
    scores = calculate_scores(workload, weights)

    # costs -> soft labels
    targets = create_soft_targets(scores, temperature=TEMPERATURE)

    row = {
        "workload_id": workload["workload_id"],
        "profile": profile
    }
    
    # workload characteristics[INPUT]
    for feature in FEATURES:
        row[feature] = workload[feature]

    # user preference weights[INPUT]
    for metric in METRICS:
        row[f"weight_{metric}"] = weights[metric]

    # actual costs for eval
    for policy in POLICIES:
        row[f"score_{policy}"] = scores[policy]

    # soft training targets
    for policy in POLICIES:
        row[f"target_{policy}"] = targets[policy]


    return row


def prepare():
    rng = random.Random(SEED)

    input_path = DATA_DIR / "benchmarks.csv"
    output_path = DATA_DIR / "soft_targets.csv"

    with input_path.open("r", newline="") as file:
        benchmarks = list(csv.DictReader(file))

    if not benchmarks:
        raise ValueError("benchmarks.csv is empty")

    rows = []

    for workload in benchmarks:

        for profile, weights in PROFILES.items():
            rows.append(
                create_training_row(
                    workload,
                    profile,
                    weights
                )
            )

        for i in range(RANDOM_PREFERENCES):
            weights = generate_preferences(rng)

            rows.append(
                create_training_row(
                    workload,
                    f"random_{i}",
                    weights
                )
            )

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())

        writer.writeheader()
        writer.writerows(rows)

    print(f"Original workloads: {len(benchmarks)}")
    print(f"Fixed profiles per workload: {len(PROFILES)}")
    print(f"Random preferences per workload: {RANDOM_PREFERENCES}")
    print(f"Total training examples: {len(rows)}")
    print(f"Saved: {output_path}")

    # verification of the sum == 1
    sample = rows[0]

    target_sum = sum(sample[f"target_{policy}"] for policy in POLICIES)

    print(f"Sample target sum: {target_sum:.6f}")


if __name__ == "__main__":
    prepare()
