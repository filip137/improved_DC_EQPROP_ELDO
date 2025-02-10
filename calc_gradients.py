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
    volt_array1 = np.linspace(-0.2,0.2, n_of_iterations)
    volt_array2 = vac2 * np.ones(n_of_iterations)
    stacked_array = np.column_stack((volt_array1, volt_array2))
    return stacked_array


def main():
    
    sample_file ="/home/filip/simulations/gradient_calculatinons/gradient_netlist.cir"
    output_dir = "/home/filip/simulations/gradient_calculatinons"
    filename = '/home/filip/simulations/gradient_calculatinons/gradient_netlist.aex'
    start_index = 4
    n_of_node_voltages = 10 
    eldo_process = start_eldo_simulation(sample_file, output_dir, m_thread = True, noascii =  True, debug=True)
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, eldo_process))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, eldo_process))
    stored_voltages = []
    input_dict = {'VAC1' : 0, 'VAC2' : -0.3}
    
    resistor_value_dict = {
    "R_0_1_1": 200.91,
    "R_0_1_2": 300.96,
    "R_0_1_3": 531.5,
    "R_0_2_1": 421.31,
    "R_0_2_2": 863.2,
    "R_0_2_3": 241.86,
    "R_1_1_1": 643.82,
    "R_1_1_2": 734.03,
    "R_1_2_1": 124.67,
    "R_1_2_2": 643.53,
    "R_1_3_1": 341.3,
    "R_1_3_2": 673.78
}

    vac2_v = [0]
    for vac2 in vac2_v: 
        stored_voltages = []
        input_voltages = create_input_array(vac2, 100)
            
            
            
        input_keys_list = list(input_dict.keys())
            
        for voltage in input_voltages:
            for j, key in enumerate(input_keys_list):
                input_dict[key] = voltage[j]  # Directly assign the value from X to the corresponding key
            set_resistances(eldo_process, resistor_value_dict, debug = True)
            set_input_voltages(eldo_process, input_dict, debug = True)
            run_eldo_simulation(eldo_process, debug = True)
            end_index = start_index + n_of_node_voltages
            lines_of_interest = wait_for_eldos_completion(eldo_process, debug = False)
            voltage_dict_free = parse_aex_file_fsst(filename, start_index, end_index, simulation_type = 'FSST')
            stored_voltages.append(list(voltage_dict_free.values()))
            start_index += n_of_node_voltages + 3
            output1f = voltage_dict_free["V_OUT_1_1"]
            output2f = voltage_dict_free["V_OUT_1_2"]
            
            
            
            
            delta_arr = np.linspace(0.0001, 0.1, 1000)
            loss_grads = []
            og_res = resistor_value_dict["R_0_1_2"]
            for delta in delta_arr:
                cond = 1/resistor_value_dict["R_0_1_2"]
                nudge = delta * cond
                new_cond = cond + nudge
                new_res = 1/new_cond
                resistor_value_dict["R_0_1_2"] = new_res
                set_resistances(eldo_process, resistor_value_dict, debug = True)
                
                
                run_eldo_simulation(eldo_process, debug = True)
                end_index = start_index + n_of_node_voltages
                lines_of_interest = wait_for_eldos_completion(eldo_process, debug = False)
                voltage_dict_weight_nudged = parse_aex_file_fsst(filename, start_index, end_index, simulation_type = 'FSST')
                stored_voltages.append(list(voltage_dict_weight_nudged.values()))
                output1n = voltage_dict_weight_nudged["V_OUT_1_1"]
                output2n = voltage_dict_weight_nudged["V_OUT_1_2"]
                
                
                
                weight_grad1 = (output1n - output1f)/nudge
                weight_grad2 = (output2n - output2f)/nudge
                loss_grad = [output1f * weight_grad1, output2f * weight_grad2]
                loss_grads.append(loss_grad)
                
                
                start_index += n_of_node_voltages + 3
                end_index = start_index + n_of_node_voltages
                resistor_value_dict["R_0_1_2"] = og_res
            loss_grads_arr = np.array(loss_grads)
            grad1_list = []
            beta_arr = np.linspace(1e-8, 1e-5, 1000)
            voltage_nudge_list = []
            for beta in beta_arr:
                #Start the nudging phase
                #resistor_value_dict["R_0_1_2"] -= nudge
                set_resistances(eldo_process, resistor_value_dict, debug = True)
                nudge_current1 = -beta *  output1f
                nudge_current2 = -beta *  output2f
                
                inudge_dict = {"INUDGE_1" : nudge_current1, "INUDGE_2" : 0}
                set_currents_nudge_mode(eldo_process, inudge_dict, debug = True)
                
                run_eldo_simulation(eldo_process, debug = True)
                end_index = start_index + n_of_node_voltages
                lines_of_interest = wait_for_eldos_completion(eldo_process, debug = False)
                voltage_dict_vol_nudged = parse_aex_file_fsst(filename, start_index, end_index, simulation_type = 'FSST')
                #stored_voltages.append(list(voltage_dict_weight_nudged.values()))
                
                voltage_output_f = voltage_dict_free["V_OUT_1_1"]
                voltage_output_n = voltage_dict_vol_nudged["V_OUT_1_1"]            
                
                voltage1f = voltage_dict_free["V_IN_0_1"]
                voltage2f = voltage_dict_free["V_OUT_0_2"]
                diff_F = voltage1f - voltage2f
                
                voltage1n = voltage_dict_vol_nudged["V_IN_0_1"]
                voltage2n = voltage_dict_vol_nudged["V_OUT_0_2"]
                diff_N = voltage1n - voltage2n
                
                
                voltage_nudge = voltage_output_n/voltage_output_f
                voltage_nudge_list.append(voltage_nudge)
                
                grad1 = 1/beta * (diff_N ** 2 - diff_F ** 2)
                grad1_list.append(grad1)
                start_index += n_of_node_voltages + 3
                end_index = start_index + n_of_node_voltages
                        # Plot Loss Gradient vs Delta
            plt.figure(figsize=(8, 5))
            plt.plot(delta_arr, loss_grads_arr[:, 0], marker='o', label="Loss Grad (V_OUT_1_1)")
            plt.plot(delta_arr, loss_grads_arr[:, 1], marker='o', label="Loss Grad (V_OUT_1_2)")
            plt.xlabel("Delta")
            plt.ylabel("Loss Gradient")
            plt.title("Loss Gradient vs Delta")
            plt.legend(loc="best")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

                        # Plot 2: grad1_list vs Beta
            # -------------------------------------------
            plt.figure(figsize=(8, 5))
            plt.plot(beta_arr, grad1_list, marker='o', label="grad1")
            plt.xlabel("Beta")
            plt.ylabel("grad1 Value")
            plt.ylim((-10,10))
            plt.title("grad1 vs Beta")
            plt.legend(loc="best")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            
            plt.figure(figsize=(8, 5))
            plt.plot(beta_arr, voltage_nudge_list, label="nudge percentage")
            plt.title("nudgepercentage vs Beta")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            
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
        
        