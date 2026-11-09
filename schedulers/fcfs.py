def fcfs(processes):

    """
    First come first serve

    Args:
    processes: contains pid, arrival time, burst time

    Returns:
    returns, gantt
    """
    processes = sorted(processes, 
                       key= lambda p: p["arrival"])

    current_time = 0
    results = []
    gantt = []

    for p in processes:

        #CPU idle until next arrival occurs

        if current_time < p["arrival"]:
            gantt.append({
                "pid": "IDLE",
                "start": current_time,
                "end": p["arrival"]
            })

            current_time = p["arrival"]

        start = current_time

        completion = start+p["burst"]

        turnaround = completion - p["arrival"]
        waiting = turnaround - p["burst"]
        response = start - p["arrival"]

        results.append({

            "pid": p["pid"],
            "completion": completion,
            "turnaround": turnaround,
            "waiting": waiting,
            "response": response
        })

        gantt.append({

            "pid": p["pid"],
            "start": start,
            "end": completion
        })

        current_time = completion

    return results, gantt