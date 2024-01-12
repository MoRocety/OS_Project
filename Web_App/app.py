
import heapq
import joblib
from flask import Flask
app = Flask(__name__)


# Load the models
uninformed_model = joblib.load('../Uninformed_Model_Training/models/decision_tree_regression_model.joblib')
informed_model = joblib.load('../Informed_Model_Training/models/decision_tree_regression_model_informed.joblib')

class Process:
    process_id_count = 1

    def __init__(self, arrival_time, used_memory, orig_id, golden_runtime):
        self.process_id = Process.process_id_count
        Process.process_id_count += 1

        self.arrival_time = arrival_time
        self.used_memory = used_memory
        self.orig_id = orig_id
        self.golden_runtime = golden_runtime

        self.predicted_burst_time = None
        self.start_time = None
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

    while yet_to_arrive_queue or ready_queue:
        while yet_to_arrive_queue and yet_to_arrive_queue[0].arrival_time <= current_time:
            process = yet_to_arrive_queue.pop(0)
            heapq.heappush(ready_queue, process)
            print(f"At Time: {current_time} => Process added to ready queue: {process}")

        if ready_queue:
            current_process = heapq.heappop(ready_queue)
            if current_process.start_time is None:
                current_process.start_time = current_time
                print(f"At Time: {current_time} => Process assigned to CPU: {current_process}")

            # Only print when a process is assigned to CPU
            current_time += 1
            current_process.predicted_burst_time -= 1
            current_process.golden_runtime -= 1

            if current_process.golden_runtime > 0:
                heapq.heappush(ready_queue, current_process)
            else:
                print(f"At Time: {current_time} => Process completed: {current_process}")
                current_process.completion_time = current_time
                current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
                # Waiting time is turnaround time minus the actual burst time (golden_runtime)
                
        else:
            # Only print CPU idle when transitioning from non-idle to idle
            if cpu_idle_time == 0:
                print(f"At Time: {current_time} => CPU Idle")
            current_time += 1
            cpu_idle_time += 1

    print("\n\tAll processes terminated, exiting simulation.\n")
    
    print("\tTurnaround, Waiting, and Completion Times:")
    for process in processes:
        print(f"Process ID: {process.process_id}, Turnaround Time: {process.turnaround_time}, "
              f"Completion Time: {process.completion_time}")

    print(f"\n\tTotal CPU Idle Time: {cpu_idle_time}")

raw_processes_uninformed = [
                    [475,76564,0.0,1.0,0.0,1861],
                    [340,631020,0.0,1.0,0.0,34298],
                    [620,613316,0.0,1.0,0.0,23544]
                ]

raw_processes_informed = [
                    [911,25660,8,0.0,1.0,0.0,169],
                    [308,596840,38874,0.0,1.0,0.0,39545],
                    [515,72944,8,0.0,1.0,0.0,1929]
                ]

processes = []

for rp in raw_processes_uninformed:
    processes.append(UninformedProcess(rp[0], rp[1], rp[2:-1], rp[-1]))

for rp in raw_processes_informed:
    processes.append(InformedProcess(rp[0], rp[1], rp[3:-1], rp[2], rp[-1]))

srtf(processes)


