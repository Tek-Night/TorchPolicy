from collections import deque

def round_robin(processes, quantum):
    """
    Round robin CPU scheduling (preemp)

    Args:
        processes: pid, arrival and burst
        quantum: max CPU time per turn
    Returns:
        results, gantt
    """

    n = len(processes)

    if n == 0:
        return [], []

    if not isinstance(quantum, int) or quantum <= 0:
        raise ValueError("Quantum must be positive integer")

    for p in processes: # error handling
        if p["arrival"] < 0 or p["burst"] <= 0:
            raise ValueError(
                "Arrival must be greater than or equal to 0 and burst must be greater than 0"
            )

    # sort by arrival 
    order = sorted(
        range(n),
        key = lambda i: (processes[i]["arrival"], i)
    )

    remaining= [p["burst"] for p in processes]
    first_start= [None]*n
    current_time = 0
    completed = 0
    next_index = 0
    ready_queue = deque()
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

    def add_arrivals():
        nonlocal next_index

        while next_index < n:
            i = order[next_index]

            if processes[i]["arrival"]> current_time:
                break

            ready_queue.append(i)
            next_index = next_index + 1

    while completed< n:

        # adding arrived processes
        add_arrivals()

        # if ready queue empty, CPU idle
        if not ready_queue:

            next_arrival = processes[order[next_index]]["arrival"]

            add_segment("IDLE", current_time, next_arrival)

            current_time = next_arrival
            continue

        # pick first process from RQ
        selected = ready_queue.popleft()
        p = processes[selected]

        if first_start[selected] is None:
            first_start[selected] = current_time

        # executing for quantum time or remaining burst time
        execution_time = min(quantum, remaining[selected])

        add_segment(
            p["pid"],
            current_time,
            current_time + execution_time
        )

        current_time = current_time + execution_time
        remaining[selected] = remaining[selected] - execution_time

        # add processes arriving during this execution time
        add_arrivals()

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
        else:
            # unfinished process returns to back of queue
            ready_queue.append(selected)

        # output in original input order
        
        position= {p["pid"]: i for i, p in enumerate(processes)}

        results.sort(key = lambda r: position[r["pid"]])

    return results, gantt
