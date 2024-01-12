import heapq
import joblib

# Load the models
uninformed_model = joblib.load('../Uninformed_Model_Training/models/decision_tree_regression_model.joblib')
informed_model = joblib.load('../Informed_Model_Training/models/decision_tree_regression_model_informed.joblib')


class Process:
    process_id = 1

    def __init__(self, arrival_time, used_memory, orig_id, golden_runtime):
        self.process_id = Process.process_id
        Process.process_id += 1

        self.arrival_time = arrival_time
        self.used_memory = used_memory
        self.orig_id = orig_id
        self.golden_runtime = golden_runtime

        self.predicted_burst_time = None
        self.waiting_time = None
        self.completion_time = None
        self.turnaround_time = None

    def __lt__(self, other):
        return self.predicted_burst_time < other.predicted_burst_time

    def __str__(self):
        return (f"Process ID: {self.process_id}, Arrival Time: {self.arrival_time}, "
                f"Predicted Burst Time: {self.predicted_burst_time}, Actual Burst Time: {self.golden_runtime}")
    

class UninformedProcess(Process):
    def __init__(self, arrival_time, used_memory, orig_id, golden_runtime):
        super().__init__(arrival_time, used_memory, orig_id, golden_runtime)
        self.features = [self.arrival_time, self.used_memory] + list(self.orig_id)
        self.predicted_burst_time = uninformed_model.predict([self.features])[0]

class InformedProcess(Process):
    def __init__(self, arrival_time, used_memory, orig_id, golden_runtime, average_cpu_time):
        super().__init__(arrival_time, used_memory, orig_id, golden_runtime)
        self.average_cpu_time = average_cpu_time
        self.features = [self.arrival_time, self.used_memory, self.average_cpu_time] + list(self.orig_id)
        self.predicted_burst_time = informed_model.predict([self.features])[0]


def srtf(processes):
    print("\n\tStarting simulation.\n")

    yet_to_arrive_queue = sorted(processes, key=lambda p: p.arrival_time)
    ready_queue = []
    current_time = min(p.arrival_time for p in yet_to_arrive_queue)
    cpu_idle_time = 0
    old_process = None
    current_process = None

    while (yet_to_arrive_queue or ready_queue) or (current_process is not None):
        # Adding all process that need are ready for the ready queue
        while yet_to_arrive_queue and yet_to_arrive_queue[0].arrival_time <= current_time:
            process = yet_to_arrive_queue.pop(0)
            heapq.heappush(ready_queue, process)
            
            # Since new challenger process, we add it if no current process or we'll check if it needs to be preempted

            # No preemption, CPU was IDLE
            if not current_process:
                current_process = heapq.heappop(ready_queue)
                print(f"At Time {current_time} => Process {process.process_id} added to idle CPU.")
            
            # Preemption, Process added back to ready queue
            elif process.golden_runtime < current_process.golden_runtime:
                old_process = current_process
                current_process = heapq.heappop(ready_queue)
                heapq.heappush(ready_queue, old_process)
                print(f"At Time {current_time} => Context switched, Process {current_process.process_id} replaced old Process {old_process.process_id}.")
                
            # No preemption, Process just added to ready queue
            else:
                print(f"At Time {current_time} => Process {process.process_id} added to ready queue.")

        # If CPU is IDLE
        if not current_process and ready_queue:
            current_process = heapq.heappop(ready_queue)
        
        # If CPU is occpuied
        if current_process:
            # Since time is passing, we decrement the golden and predicted runtime of the current process
            current_process.golden_runtime -= 1
            current_process.predicted_burst_time -= 1

            # If a process has run its course, we make it leave the CPU
            if current_process.golden_runtime == 0:
                print(f"At Time {current_time} => Process {current_process.process_id} has finished executing.")
                current_process.completion_time = current_time
                current_process = None
        
        # Yet to arrive queue is not empty but CPU is empty
        if current_process is None:
            cpu_idle_time += 1

        # Increment at the end of each time unit
        current_time += 1

    for process in processes:
        process.turnaround_time = process.completion_time - process.arrival_time
        process.waiting_time = process.turnaround_time - process.golden_runtime

    print("\n\t All processes terminated, exiting simulation.\n")
    return

raw_processes_uninformed = [
                    [475,76564,0.0,1.0,0.0,1861],
                    [340,631020,0.0,1.0,0.0,34298],
                    [620,613316,0.0,1.0,0.0,23544]
                ]

raw_processes_informed = [
                    [911,25660,8,0.0,1.0,0.0,169],
                    [308,596840,38874,0.0,1.0,0.0,1000755],
                    [515,72944,8,0.0,1.0,0.0,1929]
                ]

processes = []

for rp in raw_processes_uninformed:
    processes.append(UninformedProcess(rp[0], rp[1], rp[2:-1], rp[-1]))

for rp in raw_processes_informed:
    processes.append(InformedProcess(rp[0], rp[1], rp[3:-1], rp[-1], rp[2]))

srtf(processes)
