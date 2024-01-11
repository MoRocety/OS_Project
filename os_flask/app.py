from flask import Flask, render_template, request
import heapq
import joblib
import random
from flask import jsonify


app = Flask(__name__)

# Load the model
model = joblib.load('decision_tree_regression_model.joblib')

class Process:
    process_id_counter = 1  # Class variable to keep track of the process IDs

    def __init__(self, arrival_time, used_memory, golden_runtime, user_id):
        self.process_id = Process.process_id_counter
        Process.process_id_counter += 1

        # Rest of your __init__ method remains the same
        self.arrival_time = arrival_time
        self.used_memory = used_memory
        self.user_id = user_id

        self.features = [self.arrival_time, self.used_memory] + self.user_id
        self.burst_time = model.predict([self.features])[0]
        self.golden_runtime = golden_runtime

    # Rest of your class remains the same
    def __lt__(self, other):
        return self.burst_time < other.burst_time

    def __repr__(self):
        return f"Process ID: {self.process_id}, Arrival Time: {self.arrival_time}, Burst Time: {self.burst_time}, Actual: {self.golden_runtime}"

def srtf(processes):
    MAX_TIME_UNITS = 1000  # Adjust this value as needed
    simulation_result = []
    current_time = 0

    yet_to_arrive_queue = sorted(processes, key=lambda p: p.arrival_time)
    ready_queue = []
    old_process = None
    current_process = None

    while (yet_to_arrive_queue or ready_queue) and current_time < MAX_TIME_UNITS:
        print(f"\nTime {current_time}:")

        while yet_to_arrive_queue and yet_to_arrive_queue[0].arrival_time <= current_time:
            process = yet_to_arrive_queue.pop(0)
            heapq.heappush(ready_queue, process)
            print(f"  Process {process.process_id} added to ready queue. Queue: {ready_queue}")

        if ready_queue:
            current_process = heapq.heappop(ready_queue)
            current_time += 1
            current_process.golden_runtime -= 1

            if current_process.golden_runtime > 0:
                heapq.heappush(ready_queue, current_process)
                simulation_result.append(f"  Currently executing - {str(current_process)}")
            else:
                old_process = current_process
                current_process = None
                simulation_result.append(f"  Process completed - {str(old_process)}")

        else:
            current_time += 1
            simulation_result.append("  CPU is currently idle.")

    if current_time >= MAX_TIME_UNITS:
        simulation_result.append(f"\nSimulation terminated early after {MAX_TIME_UNITS} time units.")

    print("Final Queue:", ready_queue)
    return simulation_result

@app.route('/')
def index():
    print("Rendering index.html")
    return render_template('index.html')


@app.route('/initialize', methods=['POST'])
def initialize():
    num_processes_to_initialize = int(request.json['numProcesses'])

    # Hardcoded input processes
    raw_processes = [
        [100, 961104, 476, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [50, 937320, 473, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [20, 7740, 59, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]

    # Randomly select processes to simulate
    selected_processes = random.sample(raw_processes, min(num_processes_to_initialize, len(raw_processes)))

    # Create Process objects
    processes = [Process(rp[0], rp[1], rp[2], rp[3:]) for rp in selected_processes]

    # Convert processes to a list of dictionaries for JSON response
    processes_data = [{'process_id': p.process_id, 'arrival_time': p.arrival_time, 'burst_time': p.burst_time,
                       'golden_runtime': p.golden_runtime} for p in processes]

    return jsonify({'processes': processes_data})


if __name__ == '__main__':
    app.run(debug=True)