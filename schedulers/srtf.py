def srtf(processes):
    """
    Shortest remaining time first (preemp)
    Args:
        process (has pid, arrival, burst)
    Returns:
        results, gantt
    """

    n = len(processes)

    if n==0:
        return [],[]

    for p in processes: # error handling
        if p["arrival"] < 0 or p["burst"]<= 0:
            raise ValueError("Arrival must be greater than or equal to 0 and burst must be greater than 0")

    remaining = [p["burst"] for p in processes]
    first_start= [None]*n
    current_time = 0
    completed = 0
    results = []
    gantt = []

    def add_segment(pid, start, end):

        if gantt and gantt[-1]["pid"] == pid:
            if gantt[-1]["end"] == start:
                gantt[-1]["end"] = end
                return

        gantt.append({
            "pid": pid,
            "start": start,
            "end": end
        })

    while completed < n:
        # arrived processes that are not finished yet
        available = [
            i for i in range(n) if processes[i]["arrival"] <= current_time and remaining[i] > 0
        ]

        # if no process available, CPU idle
        if not available:
            next_arrival = min(
                processes[i]["arrival"] for i in range(n) if remaining[i] > 0
            )

            add_segment("IDLE", current_time, next_arrival)
            current_time = next_arrival
            continue

        # choosing p with shortest remaining time and handling ties
        selected = min(available,
                       key = lambda i: (remaining[i],
                                        processes[i]["arrival"],
                                        i
                                        )
                        )
        p = processes[selected]

        if first_start[selected] is None:
            first_start[selected] = current_time

        # executing process one time unit at a time
        add_segment(
            p["pid"],
            current_time,
            current_time+1
        )

        remaining[selected] = remaining[selected] - 1
        current_time = current_time + 1

        # check if process finished
        if remaining[selected] == 0:
            completed = completed + 1

            completion = current_time
            turnaround = completion - p["arrival"]
            waiting = turnaround - p["burst"]
            response = first_start[selected] - p["arrival"]

            results.append({
                "pid": p["pid"],
                "completion": completion,
                "turnaround": turnaround,
                "waiting": waiting,
                "response": response
            })

    results.sort(key=lambda r: r["pid"]) # didn't like it coming in reverse order so had it sort

    return results, gantt

