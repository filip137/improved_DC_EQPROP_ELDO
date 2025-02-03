from layer_class import *
from initializer import *
from datasets import * 
import numpy as np
import time
from support_layer import *
from eldo_support_functions import *
from loss_functions import * 
from sklearn.model_selection import train_test_split
from datetime import datetime
import signal
import subprocess
import torch
from torch.utils.data import DataLoader, TensorDataset
import json
from ac_plots import *
import matplotlib.pyplot as plt




def parse_aex_file_fsst(filename, start_index, end_index, simulation_type = "AC"):
    # Dictionary to store extracted data
    parsed_data = {}

    # Open the file and process line by line
    current_index = 0
    with open(filename, 'r') as file:
        
        for line in file:
            current_index += 1
            if current_index < start_index:
                continue
            stripped_line = line.strip()
            # Skip empty lines
            if line.strip() == "":
                continue

            # Identify lines with the format *V(...)
            if simulation_type == "DC":
                if stripped_line.startswith("*V("):
                    # Extract node name and value using regex
                    parts = stripped_line.split()
                    node_name = parts[0][3:-1].strip("'\"")
                    node_value = float(parts[2])
                    parsed_data[node_name] = node_value
                if current_index >= end_index:
                    break

            if simulation_type == "AC":
                if stripped_line.startswith("*VR("):
                    # Extract node name and value using regex
                    parts = stripped_line.split()
                    node_name = parts[0][4:-1].strip("'\"")
                    node_value = float(parts[2])
                    parsed_data[node_name] = node_value
                if current_index >= end_index:
                    break
                
            elif simulation_type == "FSST":
                if stripped_line.startswith("*YVAL("):
                    parts = stripped_line.split()
                    # parts[0] is something like "*YVAL(V(V_OUT_0_1),10MEG)"

                    # Use a local string variable instead of `signal`:
                    signal_str = parts[0]
                    start_idx = signal_str.find('V(') + 2  # position after 'V('
                    end_idx = signal_str.find(')', start_idx)
                    node_name = signal_str[start_idx:end_idx].split(',')[0]
                    # For simplicity, treat parts[-1] as the real value
                    node_value = float(parts[-1])
                    parsed_data[node_name] = node_value

                if current_index >= end_index:
                    break




    return parsed_data



"""
TO EXTRACT VOLTAGE
.EXTRACT FSST YVAL(V(INPUT_FIRST_AMP), 2Meg) YVAL(V(OUTPUT_FIRST_AMP), 2Meg) YVAL(V(INPUT_SECOND_AMP), 2Meg)
.EXTRACT FSST YVAL(V(OUTPUT_SECOND_AMP), 1Meg) YVAL(V(INPUT_THIRD_AMP), 1Meg) YVAL(V(OUTPUT_THIRD_AMP), 2Meg)
.EXTRACT FSST YVAL(V(OUTPUT_NETWORK_1), 1Meg) YVAL(V(OUTPUT_NETWORK_2), 1Meg) 
"""


def create_input_array(vac2, n_of_iterations):
    input_dict = {'VAC1' : 0, 'VAC2' : 0.1}
    volt_array1 = np.linspace(-0.6,0.6, n_of_iterations)
    volt_array2 = vac2 * np.ones(n_of_iterations)
    stacked_array = np.column_stack((volt_array1, volt_array2))
    return stacked_array


def main():
    
    sample_file ="/home/filip/simulations/sample_files/eldo_samples/python_generated_netlists/3moons_netlist.cir"
    output_dir = "/home/filip/simulations/sample_files/eldo_samples/first_harmonic_tests"
    filename = '/home/filip/simulations/sample_files/eldo_samples/first_harmonic_tests/3moons_netlist.aex'
    start_index = 4
    n_of_node_voltages = 8 
    eldo_process = start_eldo_simulation(sample_file, output_dir, m_thread = True, noascii =  True, debug=False)
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, eldo_process))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, eldo_process))
    stored_voltages = []
    input_dict = {'VAC1' : 0, 'VAC2' : -0.3}
    vac2_v = np.linspace(-0.5,0.5, 11)
    for vac2 in vac2_v: 
        stored_voltages = []
        input_voltages = create_input_array(vac2, 100)
            
            
            
        input_keys_list = list(input_dict.keys())
            
        for voltage in input_voltages:
            for j, key in enumerate(input_keys_list):
                input_dict[key] = voltage[j]  # Directly assign the value from X to the corresponding key
            set_input_voltages(eldo_process, input_dict, debug = True)
            run_eldo_simulation(eldo_process, debug = True)
            end_index = start_index + n_of_node_voltages
            lines_of_interest = wait_for_eldos_completion(eldo_process, debug = False)
            voltage_dict_free = parse_aex_file_fsst(filename, start_index, end_index, simulation_type = 'FSST')
            stored_voltages.append(list(voltage_dict_free.values()))
            start_index += n_of_node_voltages + 3
                
        labels = [
            "IN_1",  # Abbreviated from INPUT_FIRST_AMP
            "OUT_1", # Abbreviated from OUTPUT_FIRST_AMP
            "IN_2",  # Abbreviated from INPUT_SECOND_AMP
            "OUT_2", # Abbreviated from OUTPUT_SECOND_AMP
            "IN_3",  # Abbreviated from INPUT_THIRD_AMP
            "OUT_3", # Abbreviated from OUTPUT_THIRD_AMP
            "NET_1", # Abbreviated from OUTPUT_NETWORK_1
            "NET_2"  # Abbreviated from OUTPUT_NETWORK_2
        ]

        
        
                        
        labels_cur = [
            "IN_1",  # Abbreviated from INPUT_FIRST_AMP
            "OUT_1", # Abbreviated from OUTPUT_FIRST_AMP
            "IN_2",  # Abbreviated from INPUT_SECOND_AMP
            "OUT_2", # Abbreviated from OUTPUT_SECOND_AMP
            "IN_3",  # Abbreviated from INPUT_THIRD_AMP
            "OUT_3" # Abbreviated from OUTPUT_THIRD_AMP
        ]

        
        
        
        
        
        
        
        # Plot each column on the same figure
        plt.figure(figsize=(8, 5))
        stored_voltages_arr = np.array(stored_voltages)
        labels_amp = ["A1", "A2", "A3"]
        # amp1 = stored_voltages_arr[:,0]/stored_voltages_arr[:,1]
        # amp2 = stored_voltages_arr[:,2]/stored_voltages_arr[:,3]
        # amp3 = stored_voltages_arr[:,4]/stored_voltages_arr[:,5]
        amp1 = stored_voltages_arr[:,1]/stored_voltages_arr[:,0]
        amp2 = stored_voltages_arr[:,3]/stored_voltages_arr[:,2]
        amp3 = stored_voltages_arr[:,5]/stored_voltages_arr[:,4]
        amp = np.column_stack((amp1, amp2, amp3))
        
        colors = ['blue', 'blue', 'orange', 'orange', 'green', 'green', 'black', 'black']  # Customize colors
        colors_2 = ['blue', 'orange', 'green']  # Customize colors

        line_widths = [1, 4, 1, 4, 1, 4, 4, 4]  # Thinline for INPUTs, thicker for OUTPUTs
        
        for i in range(amp.shape[1]):
            plt.plot(input_voltages[:,0], amp[:, i], marker='o', label=labels_amp[i], color=colors_2[i])
        
        
        plt.ylim(0, 8)
        plt.xlabel("VAC1")  # or "Iteration" / "Sweep #"
        plt.ylabel("Voltage amplification")
        plt.title(f"First Harmonic currents for VAC2 = {vac2}")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
       
        
        # for i in range(stored_voltages_arr.shape[1]):
        #     plt.plot(input_voltages[:,0], stored_voltages_arr[:, i], marker='o', label=labels[i])
        
        
        # for i in range(stored_voltages_arr.shape[1]):
        #     plt.plot(
        #         input_voltages[:, 0],          # x-axis data
        #         stored_voltages_arr[:, i],     # y-axis data
        #         label=labels_cur[i],               # legend label
        #         color=colors[i],               # line color
        #         linewidth=line_widths[i]      # line thickness# optional marker
        #         )
        
        
        
        
        for i in range(stored_voltages_arr.shape[1]):
            plt.plot(
                input_voltages[:, 0],          # x-axis data
                stored_voltages_arr[:, i],     # y-axis data
                label=labels[i],               # legend label
                color=colors[i],               # line color
                linewidth=line_widths[i]      # line thickness# optional marker
                )

        #plt.ylim(0, 8)
        plt.xlabel("VAC1")  # or "Iteration" / "Sweep #"
        plt.ylabel("Voltage amplification")
        plt.title(f"First Harmonic voltages for VAC2 = {vac2}")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
       
    return amp        
        
if __name__ == "__main__":

    amp = main()        
    np.save("amp_data.npy", amp)
        
        
        