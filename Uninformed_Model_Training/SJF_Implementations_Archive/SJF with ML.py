import heapq
import joblib

# Load the model
model = joblib.load('decision_tree_regression_model.joblib')

class Process:
    process_id = 1
    def __init__(self, arrival_time, used_memory, user_id, golden_runtime):
        self.process_id = Process.process_id
        Process.process_id += 1

        self.arrival_time = arrival_time
        self.used_memory = used_memory
        self.user_id = user_id

        # Dividing by 100 to make the simulation easier on the eye
        self.golden_runtime = golden_runtime//100

        self.features = [self.arrival_time, self.used_memory] + self.user_id
        self.burst_time = model.predict([self.features])[0]//100

    def __lt__(self, other):
        return self.burst_time < other.burst_time

    def __str__(self):
        return f"Process ID: {self.process_id}, Arrival Time: {self.arrival_time}, Predicted Burst Time: {self.burst_time}, Actual: {self.golden_runtime}"

# Define the SRTF scheduling algorithm
def srtf(processes):
    print("\n\t Starting simulation.\n")

    # Sort the processes by arrival time
    yet_to_arrive_queue = sorted(processes, key=lambda p: p.arrival_time)
    ready_queue = []
    current_time = min([p.arrival_time for p in yet_to_arrive_queue])
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
            current_process.golden_runtime -= 1

            # If the process is not finished, add it back to the ready queue
            if current_process.golden_runtime > 0:
                heapq.heappush(ready_queue, current_process)
            else:
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
    return

raw_processes = [
                    [545,127100,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2112],
                    [494,633768,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,23842],
                    [396,61592,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1084]
                ]

processes = []

for rp in raw_processes:
    processes.append(Process(rp[0], rp[1], rp[2:-1], rp[-1]))

srtf(processes)



