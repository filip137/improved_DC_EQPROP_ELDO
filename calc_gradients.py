from layer_class import *
from initializer import *
from datasets import * 
import numpy as np
import time
import math
from support_layer import *
from eldo_support_functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from loss_functions import * 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from datetime import datetime
import signal
import subprocess
import torch
from torch.utils.data import DataLoader, TensorDataset
import json
from ac_plots import *
from simulation_parameters import SimulationParameters
from layers_initialization import initialize_network_layers
from netlist_generation.netlist_generation import netlist_builder
from transconductance_calculations import calculate_transconductance
from neural_network_4_output_version import MyNetwork
import multiprocessing
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting (in some versions)
from matplotlib import cm  # For colormap support
#import plotly.graph_objects as go



"""
.EXTRACT FSST YVAL(ISUB(XI011.AMP_INPUT), 1MEG) YVAL(ISUB(XI011.AMP_OUTPUT), 1MEG)
.EXTRACT FSST YVAL(ISUB(XI022.AMP_INPUT), 1MEG) YVAL(ISUB(XI022.AMP_OUTPUT), 1MEG)
.EXTRACT FSST YVAL(ISUB(XI033.AMP_INPUT), 1MEG) YVAL(ISUB(XI033.AMP_OUTPUT), 1MEG)
"""

#what I need for the current calculation

def parse_aex_file_from_end_current_version(filename, n_of_variables, simulation_type, voltage_dict1, current_dict1, seek_position = None):

    voltage_dict = voltage_dict1.copy()
    # Part 1: Read the entire file into a list of lines.
    start_read = time.perf_counter()
    with open(filename, 'r') as file:
        file.seek(seek_position)
        lines = file.readlines()
    end_read = time.perf_counter()

    
    # Part 2: Collect the last n_of_node_voltages non-blank lines by iterating backwards.
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
                if "ISUB(" in signal_str and current_dict1 is not None:
                    start_idx = signal_str.find("ISUB(") + len("ISUB(")
                    end_idx = signal_str.find(")", start_idx)
                    node_name_current = signal_str[start_idx:end_idx]
                    node_value_curr = float(parts[-1])
                    current_dict1[node_name_current] = node_value_curr

                elif "V(" in signal_str:
                    start_idx = signal_str.find("V(") + 2  # position after 'V('
                    end_idx = signal_str.find(")", start_idx)
                    node_name_vol = signal_str[start_idx:end_idx].split(',')[0]
                    node_value_vol = float(parts[-1])
                    voltage_dict[node_name_vol] = node_value_vol
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
    return voltage_dict, current_dict1

def send_quit_command_to_eldo(process, debug):
    """Sends a command to the Eldo subprocess, ensuring it's still open."""
    command = "QUIT"
    if process.poll() is None:  # None means the process is still running
        if debug: print(f"Sending command: {command}")
        try:
            process.stdin.write(command + "\n")
            process.stdin.flush()
            print("Eldo process terminated.")
        except Exception as e:
            print(f"Error sending command: {e}")
    else:
        print("Cannot send command, subprocess has terminated.")



class SimulationParameters:
    def __init__(self, scale_factor, bias, batch_size, beta, gamma_values):
        # Network initialization parameters
        self.simulation_type = "FSST"  # FSST OR DC
        self.network_size = [5, 12, 4]  # [input, hidden, output]
        self.freq = "1MEG"
        self.neuron = "amp_ss"  # amp_ss or perfect_amp
        self.amplifier = "ThreeTerminalBiDirAmp" # NewBiDirAmp or OldBiDirAmp or ThreeTerminalBiDirAmp
        if self.amplifier == "ThreeTerminalBiDirAmp":
            self.non_lin = True
        else:
            self.non_lin = False
        
        self.vdc_bias1 = 2.4  # Ensure this matches the netlist
        self.pmos_nonlin_bias = 2.3
        self.synapse = "fet" # fet or resistor
        #[1e-6, 2e-7]
        self.gamma_values = gamma_values
        self.cs_bias = True
        # Simulation hyper parameters

        self.beta = beta
        self.batch_size = batch_size
        self.nudging_mode = "current"
        self.loss_function = "MSE"
        self.bounds = {"min_conductance" : 1e-7,
                       "max_conductance" : 7e-5}

        # Output files
        self.sample_file = "/home/filip/simulations/aex_files/13_4"
        self.output_dir = "/home/filip/simulations/aex_files/13_4"
        self.trained_models_dir = "/home/filip/simulations/trained_models/fet_with_nonlin_vary_beta"
        # Dataset parameters
        self.dataset = "moons"
        self.n_of_epochs = 40
        self.scale_factor = scale_factor
        self.noise = 0.1
        self.bias = bias
        self.num_samples = 2400

        
        self.initializer = {
            "initializer": {
                "init_type": "random_uniform",
                "params": {
                    "L": 7e-5,
                    "U": 1e-7
                },
                "seed" : 41
                }}

        

        # Layer parameters: a subset of network parameters (for example, layer sizes)
        self.layer_parameters = {
            "simulation_type" : self.simulation_type,
            "network_size": self.network_size,
            "gamma_values": self.gamma_values,
            "freq" : self.freq,
            "neuron" : self.neuron,
            "nudging_mode" : self.nudging_mode,
            "bounds" : self.bounds,
            "initializer" : self.initializer,
            "synapse" : self.synapse,
            "cs_bias" : self.cs_bias,
            "non_lin" : self.non_lin

        }

        #the additional network parameters needed to be defined when generating the netlist
        self.network_parameters_for_netlist= {
            "VDC_BIAS1" : self.vdc_bias1,
            "PMOS_NONLIN_BIAS" : self.pmos_nonlin_bias,
            "FORM" : 0,
            "LOW_NOISE_OPTION" : 0
        }



def main():

    

    gamma_value = [1e-8, 5e-9]
    batch_size = 5
    beta = 1e-5
    scale_factor = 0.5
    bias = 0.1
    
    sim_params = SimulationParameters(scale_factor, bias, batch_size, beta, gamma_value)
    synapse = sim_params.synapse
    process_id = 0
    input_files = create_filenames(sim_params.output_dir, sim_params.sample_file, process_id)
    full_subfolder_path, new_sample_file, aex_file_path = input_files

    simulation_type = sim_params.simulation_type

    layers = initialize_network_layers(sim_params) #here also the weight matrices are initialized
    builder = netlist_builder(layers, sim_params)

    all_nodes = extract_all_nodes_voltages(layers)
    transcon_calc = True
    
    fet_identifiers = ["0_2_1", "1_3_2"]
    builder.build_netlist(new_sample_file, transcon_calc, fet_identifiers, printfile = None)

    
    net = MyNetwork(layers, sim_params, input_files, all_nodes)

        
    ##Prepare the input dataset

    X_t, Y_t = prepare_moons_data(sim_params.num_samples, noise=0.1, random_state=41)


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_t) * sim_params.scale_factor

    #input_function = onehot_pos_neg_inputs_1bias_double_input



    #pids = get_eldo_pids(eldo_identifier = 'eldo_64.exe')
    #signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, eldo_process))
    min1, max1 = X_scaled[:, 0].min() - 0.1, X_scaled[:, 0].max() + 0.1
    min2, max2 = X_scaled[:, 1].min() - 0.1, X_scaled[:, 1].max() + 0.1
    num_points = 2
    
    x1grid = np.linspace(min1, max1, num_points)
    x2grid = np.linspace(min2, max2, num_points)
    
    # Create a meshgrid from the grid points
    xx, yy = np.meshgrid(x1grid, x2grid)
    grid = np.c_[xx.ravel(), yy.ravel()]
    #sim_params = SimulationParameters()
    layers = initialize_network_layers(sim_params) #here also the weight matrices are initialized
    builder = netlist_builder(layers, sim_params)


    #just create 5 inputs out of 2
    #X_in, Y = onehot_pos_neg_inputs_1bias_double_input(grid, Y_t, net.bias)
    
    
    #input_function = pos_inputs_1bias
    X_in, Y = onehot_pos_neg_inputs_1bias_double_input(grid, Y_t, net.bias)
    
    
    stored_voltages = []
    

    #
    input_node_names = []
    output_node_names = []
    
    n_of_neurons = sim_params.network_size[1]
    
    
    

    
    
    for i in range(1, n_of_neurons+1):
        node_input_name = f"XI0{i}{i}.AMP_INPUT"
        node_output_name = f"XI0{i}{i}.AMP_OUTPUT"
        input_node_names.append(node_input_name)
        output_node_names.append(node_output_name)
    
    input_current_dict = dict.fromkeys(input_node_names)
    output_current_dict = dict.fromkeys(output_node_names)
    current_dict = {**input_current_dict, **output_current_dict}
    
    n_of_amp_currents = len(input_node_names) + len(output_node_names)
    n_of_node_voltages = len(all_nodes)
    n_of_variables = n_of_amp_currents + n_of_node_voltages

    
    aex_file_path = net.aex_file_path
    voltage_dict_free = dict.fromkeys(net.all_nodes, None)


    eldo_process = start_eldo_simulation(new_sample_file, full_subfolder_path, m_thread = True, noascii =  True, debug=True)

    offset = 0

    resisitve_layers = [layer for layer in layers if getattr(layer, 'trainable', None)]
    
    input_amp_voltages = []
    output_amp_voltages = []        

    input_amp_currents = []
    output_amp_currents = []
    output_values_list = []
    debug = False

    for node in current_dict.keys():
        command = f".EXTRACT FSST YVAL(ISUB({node}), 1MEG)"
        send_command_to_eldo(eldo_process, command, debug)

    

    input_dict = layers[0].inputs
    input_keys_list = list(input_dict.keys())

    for X in X_in:
        if os.path.exists(aex_file_path):
            file_size = os.path.getsize(aex_file_path)
            offset = file_size
        for i, key in enumerate(input_keys_list):
            input_dict[key] = X[i]  # Directly assign the value from X to the corresponding key
        
        
        set_input_voltages(eldo_process, input_dict, debug)
                #move at the end
        run_eldo_simulation(eldo_process, debug)
        lines_of_interest = wait_for_eldos_completion(eldo_process, debug)

        voltage_dict_free, current_dict = parse_aex_file_from_end_current_version(aex_file_path, n_of_variables, simulation_type, voltage_dict_free, current_dict, offset)
        
        for key, value in current_dict.items():
            try:
                if "AMP_INPUT" in key:
                    input_current_dict[key] = value
                elif "AMP_OUTPUT" in key:
                    output_current_dict[key] = value
            except:
                continue
        
        input_amp_currents.append(list(input_current_dict.values()))
        output_amp_currents.append(list(output_current_dict.values()))        
        
        for layer in resisitve_layers:
            layer.update__free_voltages(voltage_dict_free)
        
            
        input_amp_voltages.append(list(resisitve_layers[0].output_free_voltages.values()))
        output_amp_voltages.append(list(resisitve_layers[1].input_free_voltages.values()))
        
        
        outputs = resisitve_layers[1].output_free_voltages #the outputs are just the outputs of the last layer
        output_values = list(outputs.values())
        output_values_list.append(output_values)
        
            
            
            
            # delta_arr = np.linspace(0.001, 0.15, 10)
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
            #     voltage_dict_weight_nudged, current_dict = parse_aex_file_from_end_current_version(aex_result_file, n_of_variables, simulation_type, voltage_dict_free, current_dict, offset)            
            #     #stored_voltages.append(list(voltage_dict_weight_nudged.values()))
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
            # beta_arr = np.linspace(1e-10, 1e-7, 10)
            # voltage_nudge_list = []
            # #truncate__aex_file(filename, 10)
            
            # file_size = os.path.getsize(aex_result_file)
            # offset = file_size

            
            
            # for beta in beta_arr:
            #     #Start the nudging phase
            #     #resistor_value_dict["R_0_1_2"] -= nudge
            #     start_index = 4
            #     set_resistances(eldo_process, resistor_value_dict, debug = False)
            #     nudge_current1 = -4 *  beta *  output1f
            #     nudge_current2 = + 4* beta *  output2f
                
            #     inudge_dict = {"INUDGE_1" : nudge_current1, "INUDGE_2" : 0}
            #     set_currents_nudge_mode(eldo_process, inudge_dict, debug = True)
                
            #     run_eldo_simulation(eldo_process, debug = False)
            #     end_index = start_index + n_of_node_voltages
            #     lines_of_interest = wait_for_eldos_completion(eldo_process, debug = False)
            #     voltage_dict_vol_nudged, current_dict = parse_aex_file_from_end_current_version(aex_result_file, n_of_variables, simulation_type, voltage_dict_free, current_dict, offset)            
            #     #stored_voltages.append(list(voltage_dict_weight_nudged.values()))
                
            #     voltage_output_f = voltage_dict_free["V_OUT_1_1"]
            #     voltage_output_n = voltage_dict_vol_nudged["V_OUT_1_1"]            
                
            #     voltage1f = voltage_dict_free["V_IN_1_1"]
            #     voltage2f = voltage_dict_free["V_OUT_1_1"]
            #     diff_F = voltage1f - voltage2f
                
            #     voltage1n = voltage_dict_vol_nudged["V_IN_1_1"]
            #     voltage2n = voltage_dict_vol_nudged["V_OUT_1_1"]
            #     diff_N = voltage1n - voltage2n
                
                
            #     voltage_nudge = voltage_output_n/voltage_output_f
            #     voltage_nudge_list.append(voltage_nudge)
                
            #     grad1 = (1/(2*beta)) * (diff_N ** 2 - diff_F ** 2)
            #     grad1_list.append(grad1)
                
            #     inudge_dict = {"INUDGE_1" : 0, "INUDGE_2" : 0}
            #     set_currents_nudge_mode(eldo_process, inudge_dict, debug = True)
                        
                        
            # fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                        
            #             # Plot 1: Loss Gradient vs Delta
            # axes[0].plot(delta_arr, loss_grads_arr[:, 0], marker='o', label="conductance perturbation estimation")
            # axes[0].set_xlabel("Delta")
            # axes[0].set_ylabel("Gradient Estimation")
            # axes[0].set_title(f"Gradient vs Delta for {voltage} and {vac2} - cond perturbation")
            # axes[0].legend(loc="best")
            # axes[0].grid(True)
            
            # # Plot 2: Gradient Estimation vs Beta
            # axes[1].plot(beta_arr, grad1_list, marker='o', label="nudge_estimation")
            # axes[1].set_xlabel("Beta")
            # axes[1].set_ylabel("Gradient Estimation")
            # axes[1].set_title("Gradient Estimation vs Beta - nudge perturbation")
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

            
            
            
            
            
    plot_amplification = False
    if plot_amplification:
        
        def plot_voltage_amplification(input_amp_voltages, output_amp_voltages, xx, yy, plots_per_fig=6):
            """
            Plots the voltage amplification contours.
            
            Parameters:
                input_amp_voltages (array-like): Input voltages from amplifiers.
                output_amp_voltages (array-like): Output voltages from amplifiers.
                xx, yy (ndarray): 2D grid coordinates used for reshaping the amplifier data.
                plots_per_fig (int): Number of subplots per figure (default is 6).
            """
            # Calculate voltage amplification and clip the result between 2 and 6
            input_voltages_arr = np.transpose(np.array(input_amp_voltages))
            output_voltages_arr = np.transpose(np.array(output_amp_voltages))
            amp = output_voltages_arr / input_voltages_arr            
            clipped_amp = np.clip(amp, 2, 6)
            
            # Determine the number of amplifier channels and figures needed.
            n_plots = clipped_amp.shape[0]
            n_figs = math.ceil(n_plots / plots_per_fig)
            
            for fig_idx in range(n_figs):
                # Create a figure with a 3x2 grid.
                fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
                axes = axes.flatten()  # Flatten for easier iteration.
                
                for j in range(plots_per_fig):
                    global_index = fig_idx * plots_per_fig + j  # Current amplifier index.
                    if global_index >= n_plots:
                        axes[j].axis('off')  # Hide unused axes.
                        continue
                    
                    # Reshape the 1D amplifier slice into a 2D array for plotting.
                    amp_col = clipped_amp[global_index, :]
                    amp_2d = amp_col.reshape(xx.shape)
                    
                    # Plot the filled contour with explicit color limits.
                    cf = axes[j].contourf(xx, yy, amp_2d, vmin=0, vmax=10, cmap='viridis')
                    fig.colorbar(cf, ax=axes[j], label="Amplitude")
                    
                    # Set labels, title, and grid.
                    axes[j].set_xlabel("VAC1", fontsize=14)
                    axes[j].set_ylabel("VAC2", fontsize=14)
                    axes[j].set_title(f"Voltage Amplification Amplifier {global_index+1}", fontsize=16)
                    axes[j].tick_params(axis='both', which='major', labelsize=14)
                    axes[j].grid(True)
                
                fig.tight_layout()
                plt.show()
    
    
        def plot_current_amplification(input_amp_currents, output_amp_currents, xx, yy, plots_per_fig=6):
            """
            Plots the current amplification contours.
            
            Parameters:
                input_amp_currents (array-like): Input currents from amplifiers.
                output_amp_currents (array-like): Output currents from amplifiers.
                xx, yy (ndarray): 2D grid coordinates used for reshaping the amplifier data.
                plots_per_fig (int): Number of subplots per figure (default is 6).
            """
            # Calculate current amplification and clip the values between -2 and 0.
            input_currents_arr = np.transpose(np.array(input_amp_currents))
            output_input_currents_arr = np.transpose(np.array(output_amp_currents))
            current_amp = input_currents_arr / output_input_currents_arr
            clipped_current_amp = np.clip(current_amp, -2, 0)
            
            # Determine the number of amplifier channels and figures needed.
            n_plots = clipped_current_amp.shape[0]
            n_figs = math.ceil(n_plots / plots_per_fig)
            
            for fig_index in range(n_figs):
                # Create a figure with a 3x2 grid.
                fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
                axes = axes.flatten()
                
                for j in range(plots_per_fig):
                    global_index = fig_index * plots_per_fig + j
                    if global_index >= n_plots:
                        axes[j].axis('off')
                        continue
                    
                    # Reshape the 1D amplifier slice to match grid dimensions.
                    amp_col = clipped_current_amp[global_index, :]
                    amp_2d = amp_col.reshape(xx.shape)
                    
                    # Plot the filled contour.
                    contour = axes[j].contourf(xx, yy, amp_2d, cmap='viridis')
                    fig.colorbar(contour, ax=axes[j], label="Amplitude")
                    
                    # Set standard axis labels, title, and grid.
                    axes[j].set_xlabel("VAC1", fontsize=14)
                    axes[j].set_ylabel("VAC2", fontsize=14)
                    axes[j].set_title(f"Current Amplifier {global_index+1}", fontsize=16)
                    axes[j].tick_params(axis='both', which='major', labelsize=14)
                    axes[j].grid(True)
                
                fig.tight_layout()
                plt.show()
        
    
    outputs_arr = np.transpose(np.array(output_amp_voltages)) #or output_values_list
    n_plots = outputs_arr.shape[0]
    #Loop over each amplifier (data slice).
    for global_index in range(n_plots):
        # Create a new figure with two subplots side by side.
        fig = plt.figure(figsize=(15, 8))
        
        # ----------------------------
        # Left subplot: 3D Surface Plot
        # ----------------------------
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        
        # Extract and reshape the data for the current amplifier to match the grid.
        input_voltages_col = outputs_arr[global_index, :]
        input_voltages_2d = input_voltages_col.reshape(xx.shape)
        
        # Compute dynamic color limits based on the actual data range.
        zmin, zmax = input_voltages_2d.min(), input_voltages_2d.max()
        
        # Plot the 3D surface using a dynamic range for color mapping.
        surf = ax3d.plot_surface(xx, yy, input_voltages_2d, 
                                 cmap='viridis', 
                                 vmin=zmin, vmax=zmax,
                                 edgecolor='none',
                                 antialiased=True)
        
        # Optionally, add a colorbar to indicate amplitude values.
        fig.colorbar(surf, ax=ax3d, pad=0.1, shrink=0.6, label="Amplitude")
        
        # Set labels and title for the 3D plot.
        ax3d.set_xlabel("VAC1", fontsize=14)
        ax3d.set_ylabel("VAC2", fontsize=14)
        ax3d.set_zlabel("Amplitude", fontsize=14)
        ax3d.set_title(f"Amplifier {global_index+1}: 3D Input Voltages for synapse {synapse}", fontsize=16)
        ax3d.grid(True)
        
        # Adjust viewing angle if desired (optional).
        # ax3d.view_init(elev=30, azim=45)
        
        # -----------------------------------
        # Right subplot: 2D Diagonal Plot
        # -----------------------------------
        ax2d = fig.add_subplot(1, 2, 2)
        
        # For a square grid, the diagonal is given by the elements where the row index equals the column index.
        diag_x = np.diag(xx)               # x-coordinates along the diagonal
        diag_voltage = np.diag(input_voltages_2d)
        
        # Plot the diagonal points.
        ax2d.plot(diag_x, diag_voltage, 'o-', label="Amplitude along diagonal")
        ax2d.set_xlabel("Diagonal Coordinate (x = y)", fontsize=14)
        ax2d.set_ylabel("Amplitude", fontsize=14)
        ax2d.set_title(f"Amplifier {global_index+1}: Diagonal Amplitude for synapse {synapse}", fontsize=16)
        ax2d.grid(True)
        ax2d.legend()
        
        # Adjust layout so the subplots don't overlap.
        fig.tight_layout()
        plt.show()
 
    

                     
    
    
    
    
    try:
        calculate_transconductance(eldo_process, fet_identifiers, aex_file_path, all_nodes, debug)
        send_quit_command_to_eldo(eldo_process, debug)
        return eldo_process
    except:
        send_quit_command_to_eldo(eldo_process, debug)
        return eldo_process
    # for i in range(amp.shape[1]):
    #         # For the first three curves, override the label with A1, A2, A3
    #     if i < 3:
    #         label = f"B{i+1}"
    #     else:
    #         label = labels_bamp[i]
    #     plt.plot(input_voltages[:, 0], bamp[:, i], marker='o', label=label, color=colors_2[i])
        
    #     plt.xlabel("VAC1", fontsize = 14)  # or "Iteration" / "Sweep #"
    #     plt.ylabel("Current amplification", fontsize = 14)
    #     plt.title("First Harmonic Current Amplification for the three Amplifiers", fontsize = 16)
    #     plt.tick_params(axis = 'both', which = 'major', labelsize = 14)

    #     plt.ylim((-10,10))
    #     plt.legend(loc="upper right")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()
                                
 
    
 
            
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
       
        
if __name__ == "__main__":
    try:
        eldo_process = main()        
        np.save("amp_data.npy", amp)
        send_quit_command_to_eldo(eldo_process, debug = False)
    except:
        print("Eldo terminated prematurely")