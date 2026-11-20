
import csv
import math
import random
from pathlib import Path

from data.label import (
    POLICIES,
    PROFILES,
    FEATURES,
    METRICS,
    calculate_scores
)

DATA_DIR = Path(__file__).resolve().parent

TEMPERATURE = 0.1
RANDOM_PREFERENCES = 16
SEED = 42


def generate_preferences(rng):
    """
    Generate random weights that sum to 1.

    Uses a Dirichlet-style distribution so that
    different metrics can have different importance.
    """

    values = [
        rng.gammavariate(0.7, 1.0)
        for _ in METRICS
    ]

    total = sum(values)

    return {
        metric: value / total
        for metric, value in zip(METRICS, values)
    }


def create_soft_targets(scores, temperature=0.1):
    """
    Convert scheduler costs into a probability
    distribution.

    Lower costs produce higher target probabilities.
    """

    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    minimum = min(scores.values())

    exponentials = {
        policy: math.exp(
            -(scores[policy] - minimum) / temperature
        )
        for policy in POLICIES
    }

    total = sum(exponentials.values())

    return {
        policy: exponentials[policy] / total
        for policy in POLICIES
    }


def create_training_row(workload, profile, weights):
    # Calculate the real costs for all policies
    scores = calculate_scores(workload, weights)

    # Convert costs into soft labels
    targets = create_soft_targets(
        scores,
        temperature=TEMPERATURE
    )

    row = {
        "workload_id": workload["workload_id"],
        "profile": profile
    }

    # Input: workload characteristics
    for feature in FEATURES:
        row[feature] = workload[feature]

    # Input: user preference weights
    for metric in METRICS:
        row[f"weight_{metric}"] = weights[metric]

    # Actual costs, used for evaluation
    for policy in POLICIES:
        row[f"score_{policy}"] = scores[policy]

    # Soft training targets
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

        # Existing five performance profiles
        for profile, weights in PROFILES.items():
            rows.append(
                create_training_row(
                    workload,
                    profile,
                    weights
                )
            )

        # Additional random preference combinations
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
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Original workloads: {len(benchmarks)}")
    print(f"Fixed profiles per workload: {len(PROFILES)}")
    print(f"Random preferences per workload: {RANDOM_PREFERENCES}")
    print(f"Total training examples: {len(rows)}")
    print(f"Saved: {output_path}")

    # Verify that each target distribution sums to 1
    sample = rows[0]

    target_sum = sum(
        sample[f"target_{policy}"]
        for policy in POLICIES
    )

    print(f"Sample target sum: {target_sum:.6f}")


if __name__ == "__main__":
    prepare()
