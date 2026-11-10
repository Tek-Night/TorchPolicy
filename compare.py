
from schedulers.fcfs import fcfs
from schedulers.sjf import sjf
from schedulers.srtf import srtf
from schedulers.round_robin import round_robin
from schedulers.priority import (
    priority_non_preemptive,
    priority_preemptive
)


def compare_schedulers(processes, quantum=2):

    schedulers = {
        "FCFS": fcfs,
        "SJF": sjf,
        "SRTF": srtf,
        "Round Robin": lambda p: round_robin(p, quantum),
        "Priority NP": priority_non_preemptive,
        "Priority P": priority_preemptive
    }

    comparison = {}

    for name, scheduler in schedulers.items():

        # Give each scheduler its own copy of the workload
        workload = [p.copy() for p in processes]

        results, gantt = scheduler(workload)

        avg_waiting = sum(
            r["waiting"] for r in results
        ) / len(results)

        avg_turnaround = sum(
            r["turnaround"] for r in results
        ) / len(results)

        avg_response = sum(
            r["response"] for r in results
        ) / len(results)

        comparison[name] = {
            "avg_waiting": avg_waiting,
            "avg_turnaround": avg_turnaround,
            "avg_response": avg_response,
            "gantt": gantt,
            "results": results
        }

    return comparison


if __name__ == "__main__":

    processes = [
        {"pid": "P1", "arrival": 0, "burst": 5, "priority": 3},
        {"pid": "P2", "arrival": 1, "burst": 3, "priority": 1},
        {"pid": "P3", "arrival": 2, "burst": 1, "priority": 2}
    ]

    comparison = compare_schedulers(processes)

    print(
        f"{'Algorithm':<18}"
        f"{'Waiting':<12}"
        f"{'Turnaround':<14}"
        f"{'Response':<12}"
    )

    print("-" * 56)

    for name, metrics in comparison.items():
        print(
            f"{name:<18}"
            f"{metrics['avg_waiting']:<12.2f}"
            f"{metrics['avg_turnaround']:<14.2f}"
            f"{metrics['avg_response']:<12.2f}"
        )

    best = min(
        comparison,
        key=lambda name: comparison[name]["avg_waiting"]
    )

    print(f"\nBest scheduler (waiting time): {best}")
