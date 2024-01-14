import heapq
import joblib
from flask import Flask, render_template, jsonify, request
import csv
from random import sample
from math import ceil

# Global variables
processes = []
app = Flask(__name__)

# Load the models
uninformed_model = joblib.load('../Uninformed_Model_Training/models/decision_tree_regression_model.joblib')
informed_model = joblib.load('../Informed_Model_Training/models/decision_tree_regression_model_informed.joblib')


# Abstract Process class
class Process:
    process_id = 1

    def __init__(self, arrival_time, used_memory, orig_id, golden_burst_time):
        self.process_id = Process.process_id
        Process.process_id += 1

        self.arrival_time = ceil(pow(arrival_time, 1/3))                      
        self.used_memory = used_memory
        self.orig_id = orig_id

        self.actual_burst_time = golden_burst_time
        self.golden_burst_time = golden_burst_time

        self.predicted_burst_time = None
        self.golden_predicted_burst_time = None

        self.waiting_time = None
        self.completion_time = None
        self.turnaround_time = None

    def __lt__(self, other):
        return self.predicted_burst_time < other.predicted_burst_time
    
# Uninformed and Informed Process classes
class UninformedProcess(Process):
    def __init__(self, arrival_time, used_memory, orig_id, golden_burst_time):
        super().__init__(arrival_time, used_memory, orig_id, golden_burst_time)
        self.features = [pow(self.arrival_time, 3), self.used_memory] + list(self.orig_id)
        self.predicted_burst_time = uninformed_model.predict([self.features])[0]
        self.golden_predicted_burst_time = uninformed_model.predict([self.features])[0]

class InformedProcess(Process):
    def __init__(self, arrival_time, used_memory, orig_id, golden_burst_time, average_cpu_time):
        super().__init__(arrival_time, used_memory, orig_id, golden_burst_time)
        self.average_cpu_time = average_cpu_time
        self.features = [pow(self.arrival_time, 3), self.used_memory, self.average_cpu_time] + list(self.orig_id)
        self.predicted_burst_time = informed_model.predict([self.features])[0]
        self.golden_predicted_burst_time = informed_model.predict([self.features])[0]


# Default Page
@app.route('/')
def index():
    # Clear processes list on page load
    global processes
    Process.process_id = 1  # Reset the process id 
    processes = []  # Clear the processes list
    return render_template('index.html')


@app.route('/add_processes', methods=['POST'])
def add_processes():
    data = request.json  # Assuming the user sends a JSON object in the request

    # Check if the required data is present in the request
    if 'num_processes' not in data or 'process_type' not in data:
        return jsonify({'error': 'Invalid request format'}), 400

    num_processes = int(data['num_processes'])
    process_type = data['process_type']

    # Read processes from CSV based on the process type
    if process_type == 'informed':
        file_path = '../Informed_Model_Training/test_data_fossil_encoded.csv'
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header
            raw_processes = [list(map(float, row)) for row in reader]

        # Randomly sample unique processes
        selected_processes = sample(raw_processes, min(num_processes, len(raw_processes)))

        for rp in selected_processes:
            processes.append(InformedProcess(rp[0], rp[1], rp[3:-1], rp[2], rp[-1]))

    elif process_type == 'uninformed':
        file_path = '../Uninformed_Model_Training/test_data_fossil_encoded.csv'
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header
            raw_processes = [list(map(float, row)) for row in reader]

        # Randomly sample unique processes
        selected_processes = sample(raw_processes, min(num_processes, len(raw_processes)))

        for rp in selected_processes:
            processes.append(UninformedProcess(rp[0], rp[1], rp[2:-1], rp[-1]))

    else:
        return jsonify({'error': 'Invalid process type'}), 400

    # Return the current state of the process list as JSON
    process_list_json = [{'process_id': p.process_id,
                          'arrival_time': p.arrival_time,
                          'used_memory': p.used_memory,
                          'orig_id': p.orig_id,
                          'golden_burst_time': p.golden_burst_time,
                          'golden_predicted_burst_time': p.golden_predicted_burst_time,
                          'waiting_time': p.waiting_time,
                          'completion_time': p.completion_time,
                          'turnaround_time': p.turnaround_time}
                         for p in processes]

    return jsonify(process_list_json)


# Endpoint to clear processes
@app.route('/clear_processes', methods=['GET'])
def clear_processes():
    global processes
    Process.process_id = 1  # Reset the process id 
    processes = []  # Clear the processes list
    return jsonify({'message': 'Processes cleared successfully'}), 200


# Separate function for simulation logic
def run_simulation():
    # Simulation Output
    for process in processes:
        print(process.arrival_time, process.golden_burst_time)

    simulation_output = [] 

    print("\n\tStarting simulation.\n")
    
    # Initializing variables
    for x in processes:
        x.actual_burst_time = x.golden_burst_time
        x.predicted_burst_time = x.golden_predicted_burst_time

    yet_to_arrive_queue = sorted(processes, key=lambda p: p.arrival_time)
    ready_queue = []
    current_time = min(p.arrival_time for p in yet_to_arrive_queue)
    cpu_idle_time = 0
    old_process = None
    current_process = None

    simulation_output.append((0, f"Starting simulation."))

    # Main Loop
    while (yet_to_arrive_queue or ready_queue) or (current_process is not None):
        # Adding all process that need are ready for the ready queue
        # print(current_time, [(x.process_id, x.actual_burst_time) for x in yet_to_arrive_queue], [(x.process_id, x.actual_burst_time) for x in ready_queue], current_process.actual_burst_time if current_process else None)
        while yet_to_arrive_queue and yet_to_arrive_queue[0].arrival_time <= current_time:
            process = yet_to_arrive_queue.pop(0)
            heapq.heappush(ready_queue, process)

            # Since new challenger process, we add it if no current process or we'll check if it needs to be preempted

            # No preemption, CPU was idle
            if not current_process:
                current_process = heapq.heappop(ready_queue)
                simulation_output.append((current_time, f" Process {process.process_id} added to idle CPU."))
            
            # Preemption, Process added back to ready queue
            elif process.actual_burst_time < current_process.actual_burst_time:
                old_process = current_process
                current_process = heapq.heappop(ready_queue)
                heapq.heappush(ready_queue, old_process)
                simulation_output.append((current_time, f"Context switched, Process {current_process.process_id} replaced Process {old_process.process_id}."))
                
            # No preemption, Process added to ready queue
            else:
                simulation_output.append((current_time, f"Process {process.process_id} added to ready queue."))

        # If CPU is idle
        if not current_process and ready_queue:
            current_process = heapq.heappop(ready_queue)
        
        # If CPU is occupied
        if current_process:
            # Since time is passing, we decrement the golden and predicted runtime of the current process
            current_process.actual_burst_time -= 1
            current_process.predicted_burst_time -= 1

            # If a process has run its course, we make it leave the CPU
            if current_process.actual_burst_time == 0:
                simulation_output.append((current_time, f"Process {current_process.process_id} has finished executing."))
                current_process.completion_time = current_time + 1
                current_process = None
        
        # Yet to arrive queue is not empty but CPU is empty
        if current_process is None:
            cpu_idle_time += 1

        # Increment at the end of each time unit
        current_time += 1

    # Computing the waiting and turnaround time for each process
    for process in processes:
        process.turnaround_time = process.completion_time - process.arrival_time
        process.waiting_time = process.turnaround_time - process.golden_burst_time

    simulation_output.append((current_time-1, f"All Processes terminated."))

    print("\n\t All Processes terminated, exiting simulation.\n")
    return simulation_output


# Flask endpoint for running the simulation
@app.route('/simulate', methods=['GET'])
def flask_run_simulation():
    simulation_result = run_simulation()

    # Collect process information after the simulation
    process_info = [{'process_id': p.process_id,
                     'arrival_time': p.arrival_time,
                     'used_memory': p.used_memory,
                     'orig_id': p.orig_id,
                     'golden_burst_time': p.golden_burst_time,
                     'golden_predicted_burst_time': p.golden_predicted_burst_time,
                     'waiting_time': p.waiting_time,
                     'completion_time': p.completion_time,
                     'turnaround_time': p.turnaround_time}
                    for p in processes]

    # Return both simulation results and process information
    return jsonify({'simulation_result': simulation_result, 'process_info': process_info})



if __name__ == '__main__':
    app.run(debug=True)
