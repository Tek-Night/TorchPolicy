def priority_schedule(processes, preemptive = False):
    """
    Priority CPU scheduling
    Lower priority number = higher priority

    Args:
        processes: list,arrival, burst, priority

    Returns:
        results, gantt
    """

    n = len(processes)

    if n == 0:
        return [],[]

    for p in processes:
        if p["arrival"] < 0 or p["burst"] <= 0:
            raise ValueError(
                "Arrival must be greater than or equal to 0 and burst must be greater than 0"
            )

    remaining = [p["burst"] for p in processes]

    first_start = [None]*n

    current_time = 0
    completed=0
    results = [None]*n
    gantt=[]

    def add_segment(pid,start,end):

        if gantt and gantt[-1]["pid"] == pid:
            if gantt[-1]["end"] == start:
                gantt[-1]["end"] = end
                return

        gantt.append({
            "pid": pid,
            "start": start,
            "end": end
        })

    while completed< n:

        # finding arrived but unfinished processes
        available = [i for i in range(n) if processes[i]["arrival"] <= current_time and remaining[i] > 0]

        # CPU idle if no process available
        if not available:
            next_arrival = min(processes[i]["arrival"] for i in range(n) if remaining[i] > 0)

            add_segment("IDLE", current_time, next_arrival)
            current_time = next_arrival
            continue

        # selecting higher priority process and breaking ties
        selected = min(available,
                       key = lambda i: (
                           processes[i]["priority"],
                           processes[i]["arrival"],
                           i
                       ))
        p = processes[selected]

        if first_start[selected] is None:
            first_start[selected] = current_time

        # preemptive: runs for 1 time unit
        # non-preemptive: runs till completion
        if preemptive:
            execution_time = 1
        else:
            execution_time = remaining[selected]

        add_segment(
            p["pid"],
            current_time,
            current_time+execution_time
        )

        current_time = current_time + execution_time
        remaining[selected] = remaining[selected] - execution_time

        # checking if process finished
        if remaining[selected] == 0:
            completed = completed + 1

            completion = current_time
            turnaround = completion - p["arrival"]
            waiting = turnaround - p["burst"]
            response = first_start[selected] - p["arrival"]

            results[selected] = {
                "pid": p["pid"],
                "completion": completion,
                "turnaround": turnaround,
                "waiting": waiting,
                "response": response
            }

    return results, gantt

# for easy calling 
def priority_non_preemptive(processes):
    return priority_schedule(processes, preemptive= False)

def priority_preemptive(processes):
    return priority_schedule(processes, preemptive= True)

