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





def parse_aex_file_from_end_current_version(filename, n_of_variables, simulation_type, voltage_dict, current_dict1, seek_position = None):

    
    # Part 1: Read the entire file into a list of lines.
    start_read = time.perf_counter()
    with open(filename, 'r') as file:
        file.seek(seek_position)
        lines = file.readlines()
    end_read = time.perf_counter()

    
    # Part 2: Collect the last n_of_node_voltages non-blank lines by iterating backwards.
    start_collect = time.perf_counter()
    selected_lines = []
    count = 0
    for line in reversed(lines):
        if line.strip() == "":
            break
        selected_lines.append(line)
        count += 1
        if count >= n_of_variables:
            break
    
    # Part 3: Process the selected lines.
    start_process = time.perf_counter()
    updated_keys = set()
    for line in reversed(selected_lines):
        stripped_line = line.strip()
        
        if simulation_type == "DC":
            if stripped_line.startswith("*V("):
                parts = stripped_line.split()
                # Extract node name and value.
                node_name = parts[0][3:-1].strip("'\"")
                node_value = float(parts[2])
                if node_name in voltage_dict_free:
                    voltage_dict_free[node_name] = node_value
                    updated_keys.add(node_name)
        elif simulation_type == "AC":
            if stripped_line.startswith("*VR("):
                parts = stripped_line.split()
                node_name = parts[0][4:-1].strip("'\"")
                node_value = float(parts[2])
                if node_name in voltage_dict_free:
                    voltage_dict_free[node_name] = node_value
                    updated_keys.add(node_name)
        elif simulation_type == "FSST":
            if stripped_line.startswith("*YVAL("):
                parts = stripped_line.split()
                signal_str = parts[0]
                # Check if the signal is an ISUB measurement or a V measurement.
                if "ISUB(" in signal_str and current_dict is not None:
                    start_idx = signal_str.find("ISUB(") + len("ISUB(")
                    end_idx = signal_str.find(")", start_idx)
                    node_name_current = signal_str[start_idx:end_idx]
                    node_value_curr = float(parts[-1])
                    current_dict[node_name_current] = node_value_curr

                elif "V(" in signal_str:
                    start_idx = signal_str.find("V(") + 2  # position after 'V('
                    end_idx = signal_str.find(")", start_idx)
                    node_name_vol = signal_str[start_idx:end_idx].split(',')[0]
                    node_value_vol = float(parts[-1])
                    voltage_dict_free[node_value_vol] = node_value_vol
                    updated_keys.add(node_value_vol)
                else:
                    continue  # if the line does not match expected formats, skip it

                # # Extract the node value; using the last token (this takes the value after the comma).
                # if node_name_vol in voltage_dict_free:
                #     voltage_dict_free[node_name] = node_value_vol
                #     updated_keys.add(node_name)
                # if node_name_current in current_dict:
                #     current_dict[node_name_current] = node_value_curr
                
    
    # Return the timings list.
    return voltage_dict_free, current_dict



def create_input_array(vac1, vac2, n_of_iterations):
    volt_array1 = np.linspace(-vac1, vac1, n_of_iterations)
    volt_array2 = vac2 * np.ones(n_of_iterations)
    stacked_array = np.column_stack((volt_array1, volt_array2))
    return stacked_array


def main():
    
    simulation_type = "FSST"
    if simulation_type == "DC":
        sample_file ="/home/filip/simulations/gradient_calculatinons/DC_gradient/aex_files_20250220_125922.cir"
        output_dir = "/home/filip/simulations/gradient_calculatinons/DC_gradient"
        aex_result_file = "/home/filip/simulations/gradient_calculatinons/DC_gradient/aex_files_20250220_125922.aex"    
    elif simulation_type == "FSST":
        sample_file ="/home/filip/simulations/gradient_calculatinons/AC_gradient/gradient_calc_ac_current_extract.cir"
        output_dir = "/home/filip/simulations/gradient_calculatinons/AC_gradient"
        aex_result_file = "/home/filip/simulations/gradient_calculatinons/AC_gradient/gradient_calc_ac_current_extract.aex"
    
    start_index = 4
    
    
    eldo_process = start_eldo_simulation(sample_file, output_dir, m_thread = True, noascii =  True, debug=True)
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, eldo_process))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, eldo_process))
    stored_voltages = []
    
    
    
    
    voltage_dict_free = {
    "V_IN_0_1": None,
    "V_IN_0_2": None,
    "V_IN_0_3": None,
    "V_OUT_0_1": None,
    "V_OUT_0_2": None,
    "V_OUT_0_3": None,
    "V_OUT_0_4": None,
    "V_IN_1_1": None,
    "V_IN_1_2": None,
    "V_IN_1_3": None,
    "V_IN_1_4": None,
    "V_OUT_1_1": None,
    "V_OUT_1_2": None
}
    voltage_dict_weight_nudged  = {
    "V_IN_0_1": None,
    "V_IN_0_2": None,
    "V_IN_0_3": None,
    "V_OUT_0_1": None,
    "V_OUT_0_2": None,
    "V_OUT_0_3": None,
    "V_OUT_0_4": None,
    "V_IN_1_1": None,
    "V_IN_1_2": None,
    "V_IN_1_3": None,
    "V_IN_1_4": None,
    "V_OUT_1_1": None,
    "V_OUT_1_2": None
}
    
    voltage_dict_vol_nudged  = {
    "V_IN_0_1": None,
    "V_IN_0_2": None,
    "V_IN_0_3": None,
    "V_OUT_0_1": None,
    "V_OUT_0_2": None,
    "V_OUT_0_3": None,
    "V_OUT_0_4": None,
    "V_IN_1_1": None,
    "V_IN_1_2": None,
    "V_IN_1_3": None,
    "V_IN_1_4": None,
    "V_OUT_1_1": None,
    "V_OUT_1_2": None
}
    
    
    resistor_value_dict = {
    "R_0_1_1": 55552.22,
    "R_0_1_2": 81545.46,
    "R_0_1_3": 205455.22,
    "R_0_1_4": 13559.51,
    "R_0_2_1": 34519.84,
    "R_0_2_2": 245155.15,
    "R_0_2_3": 15545.95,
    "R_0_2_4": 25535.53,
    "R_0_3_1": 254545.94,
    "R_0_3_2": 55525.95,
    "R_0_3_3": 7135.6,
    "R_0_3_4": 43553.98,
    "R_1_1_1": 44555.68,
    "R_1_1_2": 10555.75,
    "R_1_2_1": 21455.23,
    "R_1_2_2": 34555.42,
    "R_1_3_1": 05445.42,
    "R_1_3_2": 095155.44,
    "R_1_4_1": 55545.04,
    "R_1_4_2": 27545.24
}

    current_dict = {
    "XI011.AMP_INPUT": None,
    "XI011.AMP_OUTPUT": None,
    "XI022.AMP_INPUT": None,
    "XI022.AMP_OUTPUT": None,
    "XI033.AMP_INPUT": None,
    "XI033.AMP_OUTPUT": None
} 
    
    n_of_amp_currents = 6
    n_of_node_voltages = len(list(voltage_dict_free.values()))

    n_of_variables = n_of_amp_currents + n_of_node_voltages
    
    if simulation_type == "DC":
        input_dict = {"VDC1" : 0, "VDC2" : 0}
    elif simulation_type == "FSST":
        input_dict = {"VAC1" : 0, "VAC2" : 0}    
    vac1 = -1
    vac2_v = [0.2]
    
    

    
    offset = 0
    
    for vac2 in vac2_v: 
        stored_voltages = []
        stored_currents = []
        input_voltages = create_input_array(vac1, vac2, 10)
            
            
            
        input_keys_list = list(input_dict.keys())
            
        for voltage in input_voltages:
            for j, key in enumerate(input_keys_list):
                input_dict[key] = voltage[j]  # Directly assign the value from X to the corresponding key
            set_resistances(eldo_process, resistor_value_dict, debug = True)
            set_input_voltages(eldo_process, input_dict, debug = True)
            run_eldo_simulation(eldo_process, debug = True)
            wait_for_eldos_completion(eldo_process, debug = True)
            voltage_dict_free, current_dict = parse_aex_file_from_end_current_version(aex_result_file, n_of_variables, simulation_type, voltage_dict_free, current_dict, offset)            
            stored_voltages.append(list(voltage_dict_free.values()))
            stored_currents.append(list(current_dict.values()))
            #start_index += n_of_node_voltages + 3
            output1f = voltage_dict_free["V_OUT_1_1"]
            output2f = voltage_dict_free["V_OUT_1_2"]
            
            
            
            
            # delta_arr = np.linspace(0.001, 0.15, 200)
            # loss_grads = []
            # og_res = resistor_value_dict["R_0_1_1"]
            # res_nudge_list = []
            # for delta in delta_arr:
            #     cond = 1/resistor_value_dict["R_0_1_1"]
            #     nudge = delta * cond
            #     new_cond = cond + nudge
            #     new_res = 1/new_cond
            #     resistor_value_dict["R_0_1_1"] = new_res
            #     set_resistances(eldo_process, resistor_value_dict, debug = True)
                
                
            #     run_eldo_simulation(eldo_process, debug = False)
            #     lines_of_interest = wait_for_eldos_completion(eldo_process, debug = False)
            #     end_index = start_index + n_of_node_voltages
            #     parse_aex_file_from_end_current_version(filename, n_of_variables, simulation_type, voltage_dict_vol_nudged, current_dict, offset)                
            #     stored_voltages.append(list(voltage_dict_weight_nudged.values()))
            #     output1n = voltage_dict_weight_nudged["V_OUT_1_1"]
            #     output2n = voltage_dict_weight_nudged["V_OUT_1_2"]
            #     res_nudge_list.append([output1n/output1f])
                
            #     weight_grad1 = 1/2 * (output1n ** 2 - output1f ** 2)/nudge
            #     weight_grad2 = 1/2 * (output1n ** 2 - output1f ** 2)/nudge
            #     loss_grad = [weight_grad1, weight_grad2]
            #     loss_grads.append(loss_grad)
                
                
            #     start_index += n_of_node_voltages + 3
            #     end_index = start_index + n_of_node_voltages
            #     resistor_value_dict["R_0_1_1"] = og_res
                
            # loss_grads_arr = np.array(loss_grads)
            # grad1_list = []
            # beta_arr = np.linspace(1e-12, 1e-7, 200)
            # voltage_nudge_list = []
            # #truncate__aex_file(filename, 10)
            
            # file_size = os.path.getsize(aex_result_file)
            # offset = file_size

            
            
            # for beta in beta_arr:
            #     #Start the nudging phase
            #     #resistor_value_dict["R_0_1_2"] -= nudge
            #     start_index = 4
            #     set_resistances(eldo_process, resistor_value_dict, debug = False)
            #     nudge_current1 = 4 * beta *  output1f
            #     nudge_current2 = beta *  output2f
                
            #     inudge_dict = {"INUDGE_1" : nudge_current1, "INUDGE_2" : 0}
            #     set_currents_nudge_mode(eldo_process, inudge_dict, debug = True)
                
            #     run_eldo_simulation(eldo_process, debug = False)
            #     end_index = start_index + n_of_node_voltages
            #     lines_of_interest = wait_for_eldos_completion(eldo_process, debug = False)
            #     parse_aex_file_from_end_current_version(filename, n_of_variables, simulation_type, voltage_dict_vol_nudged, current_dict, offset)
            #     #stored_voltages.append(list(voltage_dict_weight_nudged.values()))
                
            #     voltage_output_f = voltage_dict_free["V_OUT_1_1"]
            #     voltage_output_n = voltage_dict_vol_nudged["V_OUT_1_1"]            
                
            #     voltage1f = voltage_dict_free["V_IN_0_1"]
            #     voltage2f = voltage_dict_free["V_OUT_0_1"]
            #     diff_F = voltage1f - voltage2f
                
            #     voltage1n = voltage_dict_vol_nudged["V_IN_0_1"]
            #     voltage2n = voltage_dict_vol_nudged["V_OUT_0_1"]
            #     diff_N = voltage1n - voltage2n
                
                
            #     voltage_nudge = voltage_output_n/voltage_output_f
            #     voltage_nudge_list.append(voltage_nudge)
                
            #     grad1 = (1/(2*beta)) * (diff_N ** 2 - diff_F ** 2)
            #     grad1_list.append(grad1)
                
            #     inudge_dict = {"INUDGE_1" : 0, "INUDGE_2" : 0}
            #     set_currents_nudge_mode(eldo_process, inudge_dict, debug = True)
            #             # Plot Loss Gradient vs Delta
                        
                        
            # fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                        
            #             # Plot 1: Loss Gradient vs Delta
            # axes[0].plot(delta_arr, loss_grads_arr[:, 0], marker='o', label="conductance perturbation estimation")
            # axes[0].set_xlabel("Delta")
            # axes[0].set_ylabel("Loss Gradient")
            # axes[0].set_title(f"Loss Gradient vs Delta for {voltage} and {vac2}  - weight in the first layer")
            # axes[0].legend(loc="best")
            # axes[0].grid(True)
            
            # # Plot 2: Gradient Estimation vs Beta
            # axes[1].plot(beta_arr, grad1_list, marker='o', label="nudge_estimation")
            # axes[1].set_xlabel("Beta")
            # axes[1].set_ylabel("Gradient Estimation")
            # axes[1].set_title("Gradient Estimation vs Beta  - weight in the first layer")
            # axes[1].legend(loc="best")
            # axes[1].grid(True)
            
            # plt.tight_layout()
            # plt.show()            
            
            # labels = [
            #     "IN_1",  # Abbreviated from INPUT_FIRST_AMP
            #     "OUT_1", # Abbreviated from OUTPUT_FIRST_AMP
            #     "IN_2",  # Abbreviated from INPUT_SECOND_AMP
            #     "OUT_2", # Abbreviated from OUTPUT_SECOND_AMP
            #     "IN_3",  # Abbreviated from INPUT_THIRD_AMP
            #     "OUT_3", # Abbreviated from OUTPUT_THIRD_AMP
            #     "NET_1", # Abbreviated from OUTPUT_NETWORK_1
            #     "NET_2"  # Abbreviated from OUTPUT_NETWORK_2
            # ]

            
            
                            
            # labels_cur = [
            #     "IN_1",  # Abbreviated from INPUT_FIRST_AMP
            #     "OUT_1", # Abbreviated from OUTPUT_FIRST_AMP
            #     "IN_2",  # Abbreviated from INPUT_SECOND_AMP
            #     "OUT_2", # Abbreviated from OUTPUT_SECOND_AMP
            #     "IN_3",  # Abbreviated from INPUT_THIRD_AMP
            #     "OUT_3" # Abbreviated from OUTPUT_THIRD_AMP
            # ]

            
            
            
            
            
            
            
            # Plot each column on the same figure
        plt.figure(figsize=(8, 5))
        stored_voltages_arr = np.array(stored_voltages)
        labels_amp = ["A1", "A2", "A3"]
            # amp1 = stored_voltages_arr[:,0]/stored_voltages_arr[:,1]
            # amp2 = stored_voltages_arr[:,2]/stored_voltages_arr[:,3]
            # amp3 = stored_voltages_arr[:,4]/stored_voltages_arr[:,5]
        amp1 = stored_voltages_arr[:,7]/stored_voltages_arr[:,3]
        amp2 = stored_voltages_arr[:,8]/stored_voltages_arr[:,4]
        amp3 = stored_voltages_arr[:,9]/stored_voltages_arr[:,5]
        amp = np.column_stack((amp1, amp2, amp3))
            
        colors = ['blue', 'blue', 'orange', 'orange', 'green', 'green', 'black', 'black']  # Customize colors
        colors_2 = ['blue', 'orange', 'green']  # Customize colors
        line_widths = [1, 4, 1, 4, 1, 4, 4, 4]  # Thinline for INPUTs, thicker for OUTPUTs
            
        plt.figure(figsize=(8, 6))
        
        for i in range(amp.shape[1]):
            # For the first three curves, override the label with A1, A2, A3
            if i < 3:
                label = f"A{i+1}"
            else:
                label = labels_amp[i]
            plt.plot(input_voltages[:, 0], amp[:, i], marker='o', label=label, color=colors_2[i])
        
        plt.xlabel("VAC1")  # or "Iteration" / "Sweep #"
        plt.ylabel("Voltage amplification")
        plt.title(f"First Harmonic Amplification for VAC2 = {vac2}")
        plt.ylim((0,10))
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
                    
 
            # Plot each column on the same figure
        plt.figure(figsize=(8, 5))
        stored_currents_arr = np.array(stored_currents_arr)
        labels_bamp = ["B1", "B2", "B3"]
        bamp1 = stored_currents_arr[:,0]/stored_currents_arr[:,1]
        bamp2 = stored_currents_arr[:,2]/stored_currents_arr[:,3]
        bamp3 = stored_currents_arr[:,4]/stored_currents_arr[:,5]

        bamp = np.column_stack((bamp1, bamp2, bamp3))
            
        colors = ['blue', 'blue', 'orange', 'orange', 'green', 'green', 'black', 'black']  # Customize colors
        colors_2 = ['blue', 'orange', 'green']  # Customize colors
        line_widths = [1, 4, 1, 4, 1, 4, 4, 4]  # Thinline for INPUTs, thicker for OUTPUTs
            
        plt.figure(figsize=(8, 6))
        
        for i in range(amp.shape[1]):
            # For the first three curves, override the label with A1, A2, A3
            if i < 3:
                label = f"B{i+1}"
            else:
                label = labels_bamp[i]
            plt.plot(input_voltages[:, 0], bamp[:, i], marker='o', label=label, color=colors_2[i])
        
        plt.xlabel("VAC1")  # or "Iteration" / "Sweep #"
        plt.ylabel("Current amplification")
        plt.title(f"First Harmonic Current Amplification for VAC2 = {vac2}")
        plt.ylim((0,10))
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
                                
 
    
 
            
            # #truncate__aex_file(filename, 10)
            # plt.figure(figsize=(8, 5))
            # plt.plot(delta_arr, loss_grads_arr[:, 0], marker='o', label="conductance perturbation estimation")
            # #plt.plot(delta_arr, loss_grads_arr[:, 1], marker='o', label="Loss Grad (V_OUT_1_2)")
            # plt.xlabel("Delta")
            # plt.ylabel("Loss Gradient")
            # plt.title(f"Loss Gradient vs Delta for voltages {vac1} and {vac2}")
            # plt.legend(loc="best")
            # plt.grid(True)
            # plt.tight_layout()
            # plt.show()


            #             # Plot 2: grad1_list vs Beta
            # # -------------------------------------------
            # plt.figure(figsize=(8, 5))
            # plt.plot(beta_arr, grad1_list, marker='o', label="nudge_estimation")
            # #plt.ylim(-200, 200)
            # plt.xlabel("Beta")
            # plt.ylabel("The gradient estimation")
            # plt.legend(loc="best")
            # plt.grid(True)
            # plt.tight_layout()
            # plt.show()
            




            # plt.figure(figsize=(8, 5))
            # plt.plot(delta_arr, res_nudge_list, label="nudge percentage - resistance")
            # plt.title("The output voltage change due to delta conductance change")
            # plt.grid(True)
            # plt.tight_layout()
            # plt.show()
            




            
            # plt.figure(figsize=(8, 5))
            # plt.plot(beta_arr, voltage_nudge_list, label="nudge percentage - current")
            # plt.title("The output voltage change vs Beta")
            # plt.grid(True)
            # plt.tight_layout()
            # plt.show()
            
            

            

        
        
        # #plt.ylim(0, 8)
        # plt.xlabel("VAC1")  # or "Iteration" / "Sweep #"
        # plt.ylabel("Voltage amplification")
        # plt.title(f"First Harmonic currents for VAC2 = {vac2}")
        # plt.legend(loc="upper right")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
       
        
        # # for i in range(stored_voltages_arr.shape[1]):
        # #     plt.plot(input_voltages[:,0], stored_voltages_arr[:, i], marker='o', label=labels[i])
        
        
        # # for i in range(stored_voltages_arr.shape[1]):
        # #     plt.plot(
        # #         input_voltages[:, 0],          # x-axis data
        # #         stored_voltages_arr[:, i],     # y-axis data
        # #         label=labels_cur[i],               # legend label
        # #         color=colors[i],               # line color
        # #         linewidth=line_widths[i]      # line thickness# optional marker
        # #         )
        
        
        
        
        # for i in range(stored_voltages_arr.shape[1]):
        #     plt.plot(
        #         input_voltages[:, 0],          # x-axis data
        #         stored_voltages_arr[:, i],     # y-axis data
        #         label=labels[i],               # legend label
        #         color=colors[i],               # line color
        #         linewidth=line_widths[i]      # line thickness# optional marker
        #         )

        # #plt.ylim(0, 8)
        # plt.xlabel("VAC1")  # or "Iteration" / "Sweep #"
        # plt.ylabel("Voltage amplification")
        # plt.title(f"First Harmonic voltages for VAC2 = {vac2}")
        # plt.legend(loc="upper right")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
       
    return        
        
if __name__ == "__main__":

    amp = main()        
    np.save("amp_data.npy", amp)
        
        