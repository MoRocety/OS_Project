import heapq

# Define the Process class
class Process:
    def __init__(self, pid, arrival_time, burst_time):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time

    # This method is used by the heapq module to compare Process objects based on their burst time
    def __lt__(self, other):
        return self.burst_time < other.burst_time

    # This method returns a string representation of the Process object
    def __str__(self):
        return f"Process ID: {self.pid}, Arrival Time: {self.arrival_time}, Burst Time: {self.burst_time}"

# Define the SRTF scheduling algorithm
def srtf(processes):
    print("\n\t Starting simulation.\n")

    # Sort the processes by arrival time
    yet_to_arrive_queue = sorted(processes, key=lambda p: p.arrival_time)
    ready_queue = []
    current_time = 0
    result = []
    old_process = None
    current_process = None
    process_completed = False

    # Main loop
    while yet_to_arrive_queue or ready_queue:
        # Move processes from 'yet to arrive' to 'ready' queue if they have arrived
        while yet_to_arrive_queue and yet_to_arrive_queue[0].arrival_time <= current_time:
            process = yet_to_arrive_queue.pop(0)
            heapq.heappush(ready_queue, process)

        # If the ready queue is not empty, execute the process at the front
        if ready_queue:
            current_process = heapq.heappop(ready_queue)
            current_time += 1
            current_process.burst_time -= 1

            # If the process is not finished, add it back to the ready queue
            if current_process.burst_time > 0:
                heapq.heappush(ready_queue, current_process)
            else:
                result.append(current_process)
                process_completed = True
                old_process = current_process
                current_process = None

        else:
            current_time += 1

        # Print the status at each time unit
        print(f"Time {current_time} => ", end="")
        if current_process:
            print(f"Currently executing: {current_process}")
            process_completed = False
        elif process_completed:
            print(f"Process completed: {old_process}")
            process_completed = False
            old_process = None
        else:
            print("CPU is currently idle.")

    print("\n\t All processes terminated, exiting simulation.")
    return result

# Example usage:
processes = [Process(1, 0, 5), Process(2, 7, 3), Process(3, 9, 2)]
completed_processes = srtf(processes)
