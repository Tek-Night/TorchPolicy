import csv
import json
import statistics
import random
from pathlib import Path
from compare import compare_schedulers

DATA_DIR = Path(__file__).resolve().parent

POLICY_NAMES = {
    "FCFS": "fcfs",
    "SJF": "sjf",
    "SRTF": "srtf",
    "Round Robin": "rr",
    "Priority NP": "priority_np",
    "Priority P": "priority_p"
}

def generate_workload(rng):
    n = rng.randint(4, 18)

    # generating diff arrival patterns
    arrival_pattern = rng.choice([
        "simultaneous", "spread","bursty","sparse"
    ])

    if arrival_pattern == "simultaneous":
        arrivals=[0]*n
    elif arrival_pattern == "spread":
        arrivals = [rng.randint(0,3*n) for _ in range(n)]
    elif arrival_pattern == "bursty":
        arrivals = [
            rng.choice([0, n, 2*n]) + rng.randint(0,2)
            for _ in range(n)
        ]
    else:
        arrivals = rng.sample(range(6*n),n)

    # generate different burst patterns
    burst_pattern = rng.choice([
        "short","long","mixed","varied"
    ])

    if burst_pattern == "short":
        bursts = [rng.randint(1,5) for _ in range(n)]
    elif burst_pattern == "long":
        bursts = [rng.randint(8, 25) for _ in range(n)]
    elif burst_pattern == "mixed":
        bursts = [
            rng.randint(1,4) if rng.random() < 0.7 else rng.randint(15,30) for _ in range(n)
        ]
    else:
        bursts = [rng.randint(1, 20) for _ in range(n)]

    priorities = [rng.randint(1, 5) for _ in range(n)]

    
    processes = [
        {
            "pid": f"P{i + 1}",
            "arrival": arrivals[i],
            "burst": bursts[i],
            "priority": priorities[i]
        }
        for i in range(n)
    ]

    rng.shuffle(processes)

    return processes

def extract_features(processes):
    bursts = [p["burst"] for p in processes]
    arrivals = [p["arrival"] for p in processes]
    priorities = [p["priority"] for p in processes]

    return{
        "num_processes": len(processes),
        "mean_burst": statistics.mean(bursts),
        "std_burst": statistics.pstdev(bursts),
        "min_burst": min(bursts),
        "max_burst": max(bursts),
        "total_burst": sum(bursts),
        "mean_arrival": statistics.mean(arrivals),
        "std_arrival": statistics.pstdev(arrivals),
        "arrival_span": max(arrivals) - min(arrivals),
        "mean_priority": statistics.mean(priorities),
        "std_priority": statistics.pstdev(priorities),
        "high_priority_fraction": priorities.count(1) / len(processes)
    }

def count_context_switches(gantt):
    switches = 0

    for previous, current in zip(gantt, gantt[1:]):
        if(
            previous["pid"] != "IDLE" and current["pid"] != "IDLE" and previous["pid"] != current["pid"]
        ):
            switches = switches + 1

    return switches


def generate(count = 1000,
             seed = 42): # 42 is the answer to the universe

    rng = random.Random(seed)
    workloads = []
    rows=[]

    for workload_id in range(count):
        processes = generate_workload(rng)

        # features describing workload
        
        row = {
            "workload_id": workload_id, **extract_features(processes)
        }

        # running all scheduling algorithms
        comparison = compare_schedulers(processes, quantum=2)

        for name, metrics in comparison.items():
            prefix = POLICY_NAMES[name]

            results = metrics["results"]
            by_pid = {r["pid"]: r for r in results}

            # higher priority process = more weight
            total_weight = sum(1 / p["priority"] for p in processes)

            weighted_waiting = sum(
                (1 / p["priority"]) * by_pid[p["pid"]]["waiting"] for p in processes
            ) / total_weight

            row[f"{prefix}_avg_waiting"] = metrics["avg_waiting"]
            row[f"{prefix}_avg_turnaround"] = metrics["avg_turnaround"]
            row[f"{prefix}_avg_response"] = metrics["avg_response"]

            row[f"{prefix}_max_waiting"] = max(
                r["waiting"] for r in results
            )

            row[f"{prefix}_max_response"] = max(
                r["response"] for r in results
            )

            row[f"{prefix}_weighted_waiting"] = weighted_waiting

            row[f"{prefix}_context_switches"] = (
                count_context_switches(metrics["gantt"])
            )

        rows.append(row)

        workloads.append({
            "workload_id":workload_id,
            "processes": processes
            })

    # saving benchmark metrics
    csv_path = DATA_DIR / "benchmarks.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    # saving original processes list
    json_path = DATA_DIR / "workloads.jsonl"

    with json_path.open("w") as file:
        for workload in workloads:
            file.write(json.dumps(workload) + "\n")

    print(f"Generated {count} workloads")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")

if __name__ == "__main__":
    generate()