import multiprocessing
import traceback
from neural_network_4_output_version import train
from simulation_parameters import SimulationParameters
import os
import sys
import traceback
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import time
from support_layer import kill_process_PID
# Custom stream to capture print output.
class QueueStream:
    def __init__(self, queue):
        self.queue = queue

    def write(self, msg):
        if msg.strip():
            self.queue.put(msg)

    def flush(self):
        pass

# The worker function wraps train() to catch errors.
def worker(sim_params, error_queue):
    try:
        process_id = os.getpid()  # Get the current process ID
        result = train(sim_params, process_id)
        return result
    except Exception:
        # print("Error")
        # # # If an error occurs, capture the traceback and push it into the error queue.
        kill_process_PID(process_id)
        error_queue.put((sim_params, traceback.format_exc()))
        return None

if __name__ == "__main__":
    print(multiprocessing.get_start_method())

    # Prepare a list of simulation parameters.
    sim_params_list = []
    #gamma_value = [1e-8, 5e-9]
    gamma_value = [3e-8, 5e-9]
    gamma_value_list = [[3e-6, 5e-7], [3e-7, 5e-8], [3e-8, 5e-9]]
    #gamma_value = [2e-5, 3e-6]
    batch_size = 5
    beta = 5e-5
    scale_factor_list = [0.4, 0.5] 
    #scale_factor_list = [0.5]
    bias = 0
    #bias_list = [0]
    #bias_list = [0.2]

    
    for gamma_value in gamma_value_list:
        for scale_factor in scale_factor_list:
            sim_params_list.append(SimulationParameters(scale_factor, bias, batch_size, beta, gamma_value))
    
    # Use a Manager to create an error queue that can be shared with worker processes.
    manager = multiprocessing.Manager()
    error_queue = manager.Queue()
    # Use a Pool to run the worker function in parallel.
    # sim_params = sim_params_list[0]
    # result = worker(sim_params, error_queue)
    
    
    #results = train(sim_params_list[0])
    
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # pool.starmap allows us to pass a tuple of arguments (sim_params, error_queue) to each worker.
        results = pool.starmap(worker, [(sim, error_queue) for sim in sim_params_list])
    
    # After processing, collect any errors from the error queue.
    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())
    
    # Print the errors, if any.
    if errors:
        print("Errors encountered in some simulations:")
        for sim, err in errors:
            print(f"\nError in simulation with beta={sim.beta}, scale_factor={sim.scale_factor}:")
            print(err)
    else:
        print("No errors encountered.")
    
    # Optionally, print the simulation results.
    print("\nResults from simulations:")
    for result in results:
        print(result)

