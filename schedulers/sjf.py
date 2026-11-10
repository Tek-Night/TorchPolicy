def sjf(processes):
    """
    Shortest job first(non premp)
    Args:
        process: has pid, arrival, burst
    Returns:
        results, gantt
    """

    remaining = list(enumerate(processes))

    current_time = 0
    results = []
    gantt = []

    while remaining:

        #arrived processes
        available = [
            (i,p) for i,p in remaining if p["arrival"] <= current_time
        ]

        #if no process available, CPU idle
        if not available:
            next_arrival = min(p["arrival"] for _, p in remaining)

            gantt.append({
                "pid": "IDLE",
                "start": current_time,
                "end": next_arrival
            })

            current_time = next_arrival
            continue

        #selecting shortest available job and handle ties
        index, selected = min(available,
                              key = lambda item: (
                                    item[1]["burst"],
                                    item[1]["arrival"],
                                    item[0]
                              ))

        start = current_time
        completion = start + selected["burst"]
        turnaround = completion - selected["arrival"]
        waiting = turnaround - selected["burst"]
        response = start - selected["arrival"]

        results.append({
            "pid": selected["pid"],
            "completion": completion,
            "turnaround": turnaround,
            "waiting": waiting,
            "response": response
        })

        gantt.append({
            "pid": selected["pid"],
            "start": start,
            "end": completion
        })

        current_time=completion

        remaining=[
            (i, p) for i,p in remaining if i != index 
        ]

    return results, gantt
