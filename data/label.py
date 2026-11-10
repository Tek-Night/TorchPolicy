import csv
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

POLICIES = [
    "fcfs", "sjf", "srtf", "rr",
    "priority_np", "priority_p"
]

FEATURES = [
    "num_processes",
    "mean_burst",
    "std_burst",
    "min_burst",
    "max_burst",
    "total_burst",
    "mean_arrival",
    "std_arrival",
    "arrival_span",
    "mean_priority",
    "std_priority",
    "high_priority_fraction"
]

# Each profile's weights sum to 1
PROFILES = {
    "waiting": {
        "avg_waiting": 0.80,
        "avg_response": 0.05,
        "max_response": 0.05,
        "weighted_waiting": 0.05,
        "context_switches": 0.05
    },
    "interactive": {
        "avg_waiting": 0.10,
        "avg_response": 0.40,
        "max_response": 0.40,
        "weighted_waiting": 0.05,
        "context_switches": 0.05
    },
    "priority": {
        "avg_waiting": 0.10,
        "avg_response": 0.05,
        "max_response": 0.05,
        "weighted_waiting": 0.75,
        "context_switches": 0.05
    },
    "low_overhead": {
        "avg_waiting": 0.15,
        "avg_response": 0.05,
        "max_response": 0.05,
        "weighted_waiting": 0.05,
        "context_switches": 0.70
    },
    "balanced": {
        "avg_waiting": 0.30,
        "avg_response": 0.20,
        "max_response": 0.20,
        "weighted_waiting": 0.20,
        "context_switches": 0.10
    }
}

METRICS = list(PROFILES["balanced"].keys())


def calculate_scores(row, weights):
    scores = {policy: 0.0 for policy in POLICIES}

    for metric in METRICS:
        values = {
            policy: float(row[f"{policy}_{metric}"])
            for policy in POLICIES
        }

        minimum = min(values.values())
        maximum = max(values.values())

        # If every scheduler performs identically,
        # this metric cannot distinguish them.
        if maximum == minimum:
            continue

        for policy in POLICIES:
            normalized = (
                (values[policy] - minimum)
                / (maximum - minimum)
            )

            scores[policy] += weights[metric] * normalized

    return scores


def create_labels():
    input_path = DATA_DIR / "benchmarks.csv"
    output_path = DATA_DIR / "training.csv"

    with input_path.open("r", newline="") as f:
        benchmarks = list(csv.DictReader(f))

    training_rows = []
    skipped_ties = 0

    for row in benchmarks:
        for profile, weights in PROFILES.items():

            scores = calculate_scores(row, weights)
            best_score = min(scores.values())

            # Avoid arbitrary labels when policies tie
            winners = [
                policy for policy, score in scores.items()
                if abs(score - best_score) < 1e-9
            ]

            if len(winners) != 1:
                skipped_ties += 1
                continue

            training_row = {
                "workload_id": row["workload_id"],
                "profile": profile,
                **{
                    feature: row[feature]
                    for feature in FEATURES
                },
                **{
                    f"weight_{metric}": weights[metric]
                    for metric in METRICS
                },
                "best_policy": winners[0]
            }

            training_rows.append(training_row)

    if not training_rows:
        raise ValueError(
            "No uniquely labeled workloads were generated"
        )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=training_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(training_rows)

    distribution = Counter(
        row["best_policy"] for row in training_rows
    )

    print(f"Training examples: {len(training_rows)}")
    print(f"Skipped ambiguous ties: {skipped_ties}")
    print(f"Saved: {output_path}")

    print("\nClass distribution:")
    for policy in POLICIES:
        count = distribution[policy]
        percentage = 100 * count / len(training_rows)
        print(f"{policy:<15} {count:>5} ({percentage:.1f}%)")


if __name__ == "__main__":
    create_labels()
