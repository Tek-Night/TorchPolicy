
import csv
from collections import Counter
from pathlib import Path

from data.label import (
    POLICIES,
    PROFILES,
    calculate_scores
)

DATA_DIR = Path(__file__).resolve().parent


def analyze():
    path = DATA_DIR / "benchmarks.csv"

    with path.open("r", newline="") as file:
        benchmarks = list(csv.DictReader(file))

    print(f"Total workloads: {len(benchmarks)}")

    for profile, weights in PROFILES.items():

        unique_winners = Counter()
        tied_winners = Counter()

        unique_count = 0
        tie_count = 0

        for row in benchmarks:

            scores = calculate_scores(row, weights)
            best_score = min(scores.values())

            winners = [
                policy for policy, score in scores.items()
                if abs(score - best_score) < 1e-9
            ]

            if len(winners) == 1:
                unique_count += 1
                unique_winners[winners[0]] += 1

            else:
                tie_count += 1

                for policy in winners:
                    tied_winners[policy] += 1

        print(f"\n{'=' * 45}")
        print(f"PROFILE: {profile.upper()}")
        print(f"{'=' * 45}")

        print(f"Unique winners: {unique_count}")
        print(f"Tied workloads: {tie_count}")

        print("\nPolicy          Unique wins    In ties")
        print("-" * 43)

        for policy in POLICIES:
            print(
                f"{policy:<15}"
                f"{unique_winners[policy]:<15}"
                f"{tied_winners[policy]}"
            )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    analyze()
