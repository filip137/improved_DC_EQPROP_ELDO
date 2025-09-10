from initializer import *
import datasets as ds
import numpy as np
import time
import traceback
from netlist_generation_files import (
    BaseLayer,
    InputLayer,
    DenseLayer,
    NonLinearLayer,
    OutputLayer,
    netlist_builder,
    SimulationParameters,
    initialize_network_layers,
)


from support_layer import *
from eldo_support_functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from loss_functions import * 
from sklearn.model_selection import train_test_split
from datetime import datetime
import signal
import subprocess
import torch
from torch.utils.data import DataLoader, TensorDataset
import json
from ac_plots import *
from transconductance_calculations import calculate_transconductance
import multiprocessing
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting (in some versions)
from matplotlib import cm  # For colormap support
import warnings
warnings.filterwarnings(
    "ignore",
    message="Failed to set pipe buffer size"
)
#Initialize a complete neural network and build a netlist


class MyNetwork:
    def __init__(self, layers, sim_params, input_files, all_nodes):
        # Unpack configuration details
        
        self.layers = layers
        simulation_details = sim_params
        self.loss_fn = MSE(0.5)
        self.beta = sim_params.beta

        self.nudging_mode = sim_params.nudging_mode        
        
        
        # Unpack network details.
        self.freq = sim_params.freq
        self.simulation_type = sim_params.simulation_type
        full_subfolder_path, self.new_sample_file, self.result_file = input_files



        #dataset_details = config.get("dataset_details", {})
        #self.bias = dataset_details["bias"]
        
        self.batch_size = sim_params.batch_size
        self.scale_factor = sim_params.scale_factor
        self.bias = sim_params.bias #this is again the FSST?
        self.diode_connected_flash_params = sim_params.diode_connected_flash_params
        
        
        
        self.all_nodes = all_nodes
        
            
            
    def set_inputs_and_run_simulation(input_dict, eldo_process, simulation_type, q, debug):
        
        
        set_input_voltages(eldo_process, input_dict, debug = False)
                #move at the end
        #disable_current_sources(eldo_process, inudge_dict, debug = False)
                #time.sleep(0.01)
        start_simulation = time.time()
        run_simulation_and_wait(eldo_process, simulation_type, q, debug=debug)
                
                                
                
                
                #lines_of_interest = wait_for_eldos_completion(eldo_process, debug)
        wait_for_eldos_completion_old(eldo_process, debug)
        end_simulation = time.time() - start_simulation
        simulation_time_list.append(end_simulation)
        start_vol_extract = time.time()
                #voltage_dict_free = parse_aex_file_no_end(aex_result_file, start_index, simulation_type, voltage_dict_free)
        parse_aex_file_from_end_timed(aex_result_file, n_of_node_voltages, simulation_type, voltage_dict_free, offset)
#                timings_list.append(timings)
        vol_extract_time = time.time() - start_vol_extract
        vol_extract_list.append(vol_extract_time)

                
        for layer in resistive_layers:
            layer.update__free_voltages(voltage_dict_free)
                     
                #print(f"Voltage extraction {volt_extract_end}")
        outputs = layer.output_free_voltages #the outputs are just the outputs of the last layer
        output_values = np.array(list(outputs.values()))
        output_list.append(output_values)

    def free_nudged_train(self, eldo_process, X_train, Y_train, targets, epoch, optimizer, debug):
        
        
        layers = self.layers
        beta = self.beta
        batch_size = self.batch_size
        loss_fn = self.loss_fn
        result_file = self.result_file
        simulation_type = self.simulation_type
        diode_connected_flash_params = self.diode_connected_flash_params
        
        n_of_node_voltages = len(self.all_nodes)
        
        q = self.q
        
        
        input_layer = layers[0]
        output_layer = layers[-1]
        input_dict = input_layer.inputs
        input_keys_list = list(input_dict.keys())
        
        inudge_dict = output_layer.parameters
        inudge_keys_list = list(inudge_dict.keys())
        
        resistive_layers = [layer for layer in layers if getattr(layer, 'trainable', None)]
        
        num_batches = int(np.ceil(len(X_train) / batch_size))
         
        for layer in resistive_layers:
            layer.gamma *= 1000
    
        prediction_list = []
        loss_list = []
        output_list = []
        output_list_nudge = []
        weight_matrices_1 = []
        weight_matrices_2 = []
        ratios_list1 = []
        ratios_list2 = []
        
        start_index = 4
        vol_extract_list = []
        simulation_time_list = []
        voltage_dict_free = dict.fromkeys(self.all_nodes[0], None) #
        voltage_dict_nudge = dict.fromkeys(self.all_nodes[0], None)
        
        timings_list = []
        offset = 0

        debug = True
        

            #print(f"File size before clearing: {file_size} bytes")
        
        #Write the initialized synapses
        if epoch == 1:
            command_nmos = "SET P(W_NONLIN_NMOS)=4u"
            send_command_to_eldo(eldo_process, command_nmos, debug)
            command_pmos = "SET P(W_NONLIN_PMOS)=2u"
            send_command_to_eldo(eldo_process, command_pmos, debug)
            command_non_lin = "SET P(VDD_PMOS_NONLIN)=1.5"
            send_command_to_eldo(eldo_process, command_non_lin, debug)
            initialize_synapses(eldo_process, resistive_layers, diode_connected_flash_params, simulation_type, debug)
        
        for i in range(num_batches):
            # if os.path.exists(aex_result_file):
            #     file_size = os.path.getsize(aex_result_file)
            #     offset = file_size
            if os.path.exists(result_file):
                offset = os.path.getsize(result_file)
            else:
                offset = 0
            batch_start_time = time.time()
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))  # Ensure not to exceed the dataset length

            # Select the batch
            X_batch = X_train[start_idx:end_idx]
            Y_batch = Y_train[start_idx:end_idx]
            
            
            for layer in resistive_layers:
                layer.zero_grad()
            
            
            for X, Y in zip(X_batch, Y_batch):
                sample_start_time = time.time()
                for j, key in enumerate(input_keys_list):
                    input_dict[key] = X[j]  # Directly assign the value from X to the corresponding key
                    
                beta_r = beta * np.random.uniform(-5,5)
                #beta_r = beta
                disable_current_sources(eldo_process, inudge_dict, debug)
                #time.sleep(0.01)
                start_simulation = time.time()
                
                set_input_voltages(eldo_process, input_dict, debug)


                ##### 
                run_simulation_and_wait(eldo_process, simulation_type, q, debug)  
                ###Here I need to set the PMOS CS TO OFF (after the first simulation)
                command_pmos_cs = f"SET P(PMOS_CS_VDD_NEG)=0"
                send_command_to_eldo(eldo_process, command_pmos_cs, debug)
                

                #for plotting there's a function read_update_and_plot

                results = read_update(
                    eldo_process,
                    result_file,
                    voltage_dict_free,
                    offset,
                    simulation_type,
                    transcon_calc=True,
                    debug=debug
                    )

            
                vol_extract_time = time.time() - start_vol_extract
                vol_extract_list.append(vol_extract_time)

                #this stays the same
                for layer in resistive_layers:
                    layer.update__free_voltages(voltage_dict_free)

                                    
                
                #print(f"Voltage extraction {volt_extract_end}")
                outputs = layer.output_free_voltages #the outputs are just the outputs of the last layer
                output_values = np.array(list(outputs.values()))
                output_list.append(output_values)
                
                mode = self.nudging_mode
                #target = Y * self.boundary
                if targets != None:
                    Y_index = np.argmax(Y)
                    target = targets[Y_index]
                else:
                    target = (Y/4) * self.scale_factor
                #predicted value
                #pred_free = loss_fn(outputs, target, beta_r, mode)
                
                #now I can simply zip the currents to the parameters of the last layer and repeat
                
                def calc_losses_set_currents(inudge_dict, outputs, target, beta_r, mode):
                    sample_losses, currents = loss_fn(outputs, target, beta_r, mode)
                    inudge_keys_list = list(inudge_dict.keys())
                    for k, key in enumerate(inudge_keys_list):
                        inj_currents =  currents.flatten() 
                        inudge_dict[key] = inj_currents[k]    
                        
                    set_currents_nudge_mode(eldo_process, inudge_dict, debug)
                    return sample_losses
                
                sample_losses = calc_losses_set_currents(inudge_dict, outputs, target, beta_r, mode)
                run_simulation_and_wait(eldo_process, simulation_type, q, debug)                                 
                #output_layer.parameters = output_layer.update_parameters(flat_currents)

                #wait_for_eldos_completion_drain(eldo_process, q, debug)

                #mode = 'test' 
                voltage_dict_nudge =  read_update(eldo_process, result_file, voltage_dict_free, offset, simulation_type, f0, t0, n_of_node_voltages, debug)
                
                #parse_aex_file_from_end_offset(aex_result_file, n_of_node_voltages, simulation_type, voltage_dict_nudge, offset)
                python_update_time = time.time
                for layer in resistive_layers:
                    layer.update__nudge_voltages(voltage_dict_nudge)
                
                
                outputs_n = layer.output_nudge_voltages
                outputs_n_values = list(outputs_n.values())#the outputs are just the outputs of the last layer
                output_list_nudge.append(outputs_n_values)   
                sample_losses_n, voltages_n = loss_fn(outputs_n, target, beta_r, mode)
                
                
                for layer in resistive_layers:
                    layer.run_update_process(batch_size, beta_r) #this accumulates the gradients
                    
                    

                #print(f"Python update time {python_update_time_end}")
                    
                loss_list.append(sample_losses)
                
                
                                       
            #At the end of the batch update all resistances
            #this also needs to be changed if I am writing with the current pulses

                        
                        
            update_synapses(eldo_process, resistive_layers, diode_connected_flash_params, simulation_type, debug)
                        
            #Here easier to differentiate between TRAN analysis or FSST analysis
            # for j, layer in enumerate(resistive_layers):
            #     if self.write_mode == "w_voltage_source":
            #         layer.update_W(mode = None, clip = None)
            #         layer.update_synapse_dict()
            #         set_resistances(eldo_process, layer.synapse_dict, debug)
            #     elif self.write_mode == "w_current_source":
            #         #so I need another dict, which will accumulate the deltaG updates and transform them into current pulses
            #         layer.update_W(mode = None, clip = None)
            #         layer.update_deltaI_dict()
            #         set_resistances(eldo_process, layer.synapse_dict, debug)
                # ratios = compute_cosine_similarity(layer.deltaG, layer.W_old, layer.W)
                # if j == 0:
                #     ratios_list1.append(ratios)
                # elif j == 1:
                #     ratios_list2.append(ratios)
                #print(f"Norm of the deltaG {np.linalg.norm(layer.deltaG, ord=2)}")
                #print(f"finished batch {i}")
                
            weight_matrices_1.append(layers[1].W)
            weight_matrices_2.append(layers[3].W)
                
            

            
        
        ratio1_mean = np.array(np.mean(ratios_list1))
        ratio2_mean = np.array(np.mean(ratios_list2))
        ratio_list = [ratio1_mean, ratio2_mean]

        #loss_fn(output_list, mode='test')
        predictions = loss_fn(output_list, mode='test')
        epoch_acc = np.mean(loss_fn.verify_result(Y_train, np.array(predictions)))
        print(epoch_acc)
        # Convert one-hot encoded Y_train to class indices

        # Compute accuracy by comparing predictions with true_labels
        mean_time_extract = np.mean(np.array(vol_extract_list))
        #print(f"Average voltage extract time {mean_time_extract}")
        mean_time_simulation = np.mean(np.array(simulation_time_list))
        #print(f"Average simulation time {mean_time_simulation}")
        output_nodes = [0, 1, 2, 3]
        plot_free_and_nudged(output_list, output_list_nudge, output_nodes, beta, epoch)

        

        diff1, diff2 =  output_plot(output_list)
        disable_current_sources(eldo_process, inudge_dict, debug)
        mean_loss = np.mean(np.array(loss_list), axis = 1)
        #print(mean_loss)
        return {
        "accuracy" : epoch_acc,
        "loss_list": mean_loss,
        "weight_matrix_1": weight_matrices_1[-1],
        "weight_matrix_2": weight_matrices_2[-1],
        "epoch_accuracy": epoch_acc,
        "output_list" : [diff1, diff2],
        "ratio_list" : [ratio1_mean, ratio2_mean]}

            
            
    def free_test(self, eldo_process, X_grid, X_bias, Y_in, input_function, epoch, draw_grid, debug):
        
        # def signal_handler(sig, frame):
        #     if True:
        #         print("Interrupt received, sending quit command to subprocess.")
        #         send_command_to_eldo(eldo_process, "QUIT", debug)
        #     else:
        #         print("Interrupt received, exiting without sending command.")
        #     sys.exit(0)  # Exit the program
        
        if draw_grid:
            X_in, Y = onehot_pos_neg_inputs_1bias_double_input(X_grid, Y_in, self.bias, output_scale=1)
            
        input_layer = self.layers[0]
        output_layer = self.layers[-1]
        input_dict = input_layer.inputs
        input_keys_list = list(input_dict.keys())


        simulation_type = self.simulation_type
        n_of_node_voltages = len(self.all_nodes)
        voltage_dict_free = dict.fromkeys(self.all_nodes, None)


        layers = self.layers
        resistive_layers = [layer for layer in layers if getattr(layer, 'trainable', None)]
        output_layer = resistive_layers[-1]
        binary_list = []
        prediction_list = []
        


        result_file = self.result_file

        counter = 0
        
        
        input_amp_voltages = []

        output_amp_voltages = [] 
 
        
        
        if os.path.exists(result_file):
            file_size = os.path.getsize(result_file)
            offset = file_size
        print(f"File size before draw grid: {file_size} bytes")
        for X in X_in:
            for i, key in enumerate(input_keys_list):
                input_dict[key] = X[i]  # Directly assign the value from X to the corresponding key
                
            set_input_voltages(eldo_process, input_dict, debug)
                #move at the end
            run_eldo_simulation(eldo_process, debug)
            lines_of_interest = wait_for_eldos_completion(eldo_process, debug)

            #end_index = start_index + n_of_node_voltages
            parse_aex_file_from_end_offset(result_file, n_of_node_voltages, simulation_type, voltage_dict_free, offset)


            for layer in resistive_layers:
                layer.update__free_voltages(voltage_dict_free)
                
                    
            input_amp_voltages.append(list(resistive_layers[0].output_free_voltages.values()))
            output_amp_voltages.append(list(resistive_layers[1].input_free_voltages.values()))
            output_layer.update__free_voltages(voltage_dict_free)
            
            outputs = output_layer.output_free_voltages #the outputs are just the outputs of the last layer
            output_values = list(outputs.values())
            prediction_list.append(output_values)
            #start_index += n_of_node_voltages + 3


            prediction = self.loss_fn(output_values, mode='test') #need to decide where to initialize this function - remember that loss_fn comes from the already initialized MSE
            binary_prediction = self.loss_fn.binary_prediction(prediction)
            binary_list.extend(binary_prediction)
            
            
        binary_array = np.array(binary_list).reshape(-1,1)
        
        
        #accuracy = np.mean(np.equal(binary_array, Y_in)) * 100
        #print(f"Accuracy: {accuracy:.2f}%")
        
        return binary_array, input_amp_voltages, output_amp_voltages
    
    def calc_desired_outputs(self, eldo_process, X, Y, epoch, debug):
        input_layer = self.layers[0]
        output_layer = self.layers[-1]
        input_dict = input_layer.inputs
        input_keys_list = list(input_dict.keys())


        simulation_type = self.simulation_type
        n_of_node_voltages = len(self.all_nodes)
        voltage_dict_free = dict.fromkeys(self.all_nodes, None)


        
        resistive_layers = [layer for layer in self.layers if getattr(layer, 'type', None) == 'resistive']
        output_layer = resistive_layers[-1]
        binary_list = []
        output_list = []
        
        #clear_aex_file(aex_result_file)

        aex_result_file = self.aex_file_path
        if epoch > 1:
            file_size = os.path.getsize(aex_result_file)
            offset = file_size
        else:
            offset = 0
        
        # Run simulations for the first 200 samples in X
        X_in = X[:200]
        
        for X_sample in X_in:  # Renamed variable to avoid confusion with X (the dataset)
            
            for i, key in enumerate(input_keys_list):
                input_dict[key] = X_sample[i]  # Assign the value from X_sample to the corresponding key
            
            set_input_voltages(eldo_process, input_dict, debug)
            run_eldo_simulation(eldo_process, debug)
            wait_for_eldos_completion(eldo_process, debug)
            parse_aex_file_from_end_timed(aex_result_file, n_of_node_voltages, simulation_type, voltage_dict_free, offset)
            output_layer.update__free_voltages(voltage_dict_free)
            
            outputs = output_layer.output_free_voltages  # The outputs from the last layer
            output_values = list(outputs.values())
            output_list.append(output_values)
        
        # Assuming Y is a 2D array of one-hot encoded labels (each row like [1, 0] or [0, 1])
        # Use .all(axis=1) to check complete row matches and np.where(...)[0] to get indices
        indices0 = np.where((Y[:200] == np.array([1, 0])).all(axis=1))[0]
        indices1 = np.where((Y[:200] == np.array([0, 1])).all(axis=1))[0]
        
        # Convert the list of outputs to a NumPy array
        outputs_arr = np.array(output_list)
        
        # Extract outputs corresponding to the two classes using the computed indices
        X0 = outputs_arr[indices0]
        X1 = outputs_arr[indices1]
        
        # Compute the mean of outputs for each class along axis 0
        X0_mean = np.mean(X0, axis=0)
        X1_mean = np.mean(X1, axis=0)
        
        # Combine the means into a target array
        X0_target = X0_mean.reshape(2,-1)[:,0] - X0_mean.reshape(2,-1)[:,1]
        X1_target = X1_mean.reshape(2,-1)[:,0] - X1_mean.reshape(2,-1)[:,1]
        targets = np.array([X0_target, X1_target])
        
        return targets
    

    def draw_grid(self, eldo_process, X_val, Y_val, input_function, epoch, plot_directory, debug):
        # Start a new figure for this iteration
        plt.figure()
        
        # Define bounds of the domain
        min1, max1 = X_val[:, 0].min() - 0.1, X_val[:, 0].max() + 0.1
        min2, max2 = X_val[:, 1].min() - 0.1, X_val[:, 1].max() + 0.1
        num_points = 10
    
        x1grid = np.linspace(min1, max1, num_points)
        x2grid = np.linspace(min2, max2, num_points)
    
        # Create a meshgrid from the grid points
        xx, yy = np.meshgrid(x1grid, x2grid)
        grid = np.c_[xx.ravel(), yy.ravel()]
    
        Y_indices = np.argmax(Y_val, axis=1)
    
        # Make predictions for the grid
        X_bias = X_val[0, -1]
        draw_grid_flag = True
        y_predictions, input_amp_voltages, output_amp_voltages = self.free_test(eldo_process, grid, X_bias, Y_val, input_function, epoch, draw_grid_flag, debug=False)
    
    
        
    
    
        def draw_vol_amplification(xx, yy, input_amp_voltages, output_amp_voltages, epoch, plot_directory):
            input_voltages_arr = np.transpose(np.array(input_amp_voltages))
            output_voltages_arr = np.transpose(np.array(output_amp_voltages))
            amp = output_voltages_arr / input_voltages_arr            
            clipped_amp = np.clip(amp, 0, 10)
            
            #voltage amplifications
            for i in range(clipped_amp.shape[0]):
                amp_col = clipped_amp[i, :]
                amp_2d = amp_col.reshape(xx.shape)
            
                plt.figure(figsize=(8, 6))
    
                # Set explicit limits using vmin and vmax
                contour = plt.contourf(xx, yy, amp_2d, cmap='viridis')
    
                # Add a colorbar with defined limits
                plt.colorbar(contour, label="Amplitude")
    
                plt.xlabel("VAC1", fontsize=14)
                plt.ylabel("VAC2", fontsize=14)
                plt.title(f"First Harmonic Voltage Amplification Amplifier {i+1}", fontsize=16)
                plt.tick_params(axis='both', which='major', labelsize=14)
                plt.grid(True)
                plt.tight_layout()
    
    
                vol_amp__directory = os.path.join(plot_directory, "voltage_amplification_graphs")    
                # Ensure the directory exists
                if not os.path.exists(vol_amp__directory):
                    os.makedirs(vol_amp__directory)
    
                save_path = os.path.join(vol_amp__directory, f"Voltage_Amplification_{i+1}_epoch_{epoch}.png")
                plt.savefig(save_path, bbox_inches='tight')
    
                plt.close()
        
        
            #PLOT THE INPUTS VS THE OUTPUTS OF THE AMPLIFIERS
        
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
                
                # Plot the 3D surface.
                surf = ax3d.plot_surface(xx, yy, input_voltages_2d, cmap='viridis', vmin=0, vmax=10)
                # Add a colorbar to indicate amplitude values.
                #fig.colorbar(surf, ax=ax3d, label="Amplitude")
                
                # Set labels and title for the 3D plot.
                ax3d.set_xlabel("VAC1", fontsize=14)
                ax3d.set_ylabel("VAC2", fontsize=14)
                ax3d.set_zlabel("Amplitude", fontsize=14)
                ax3d.set_title(f"Amplifier {global_index+1}: 3D Input Voltages for synapse {synapse}", fontsize=16)
                ax3d.grid(True)
                
                # -----------------------------------
                # Right subplot: 2D Diagonal Plot
                # -----------------------------------
                ax2d = fig.add_subplot(1, 2, 2)
                
                # For a square grid, the diagonal is given by the elements where the row index equals the column index.
                diag_x = np.diag(xx)               # x-coordinates along the diagonal (same as y in a square grid)
                diag_voltage = np.diag(input_voltages_2d)
                
                # Plot the diagonal points.
                ax2d.plot(diag_x, diag_voltage, 'o-', label="Amplitude along diagonal")
                ax2d.set_xlabel("Diagonal Coordinate (x = y)", fontsize=14)
                ax2d.set_ylabel("Amplitude", fontsize=14)
                ax2d.set_title(f"Amplifier {global_index+1}: Diagonal Amplitude for synapse {synapse}", fontsize=16)
                ax2d.grid(True)
                ax2d.legend()
                
                
                amp_outputs_graphs_directory = os.path.join(plot_directory, "amp_outputs_graphs")   
                
                if not os.path.exists(amp_outputs_graphs_directory):
                    os.makedirs(amp_outputs_graphs_directory)
                
                
                
                save_path = os.path.join(amp_outputs_graphs_directory, f"Outputs of the amplifier_{global_index+1} at epoch_{epoch}.png")
                plt.savefig(save_path, bbox_inches='tight')
    
                plt.close()
        
        
        
        
    
        # Convert predictions into an array and reshape back into a grid
        def draw_boundary(xx, yy, y_predictions, X_val, Y_val, epoch, plot_directory):
            y_predictions = np.array(y_predictions)
            zz = y_predictions.reshape(xx.shape)
    
            # Plot the grid of x, y, and z values as a surface
            contour = plt.contourf(xx, yy, zz, cmap='Paired')
            cbar = plt.colorbar(contour)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Class 0', 'Class 1'])
        
            # Separate the points by class
            class0 = X_val[Y_indices == 0]
            class1 = X_val[Y_indices == 1]
    
            # Scatter plot for the validation set with legend
            plt.scatter(class0[:, 0], class0[:, 1], c='blue', edgecolor='k', marker='o', s=20, label='Class 0')
            plt.scatter(class1[:, 0], class1[:, 1], c='red', edgecolor='k', marker='o', s=20, label='Class 1')
    
            # Add legend, titles, and labels
            plt.legend()
            plt.title(f"Decision Boundary for test dataset after epoch {epoch}")
            plt.xlabel('VAC1')
            plt.ylabel('VAC2')
    
            # Ensure the plot_directory exists; if not, create it
            if not os.path.exists(plot_directory):
                os.makedirs(plot_directory)
            
            # Save underlying data using epoch in the file name
            data_to_save = {
                'xx': xx,
                'yy': yy,
                'zz': zz,
                'class0': class0,
                'class1': class1,
                'X_val': X_val,
                'Y_val': Y_val,
                'epoch': epoch
            }
            data_path = os.path.join(plot_directory, f"plot_data_{epoch}.npz")
            np.savez(data_path, **data_to_save)
            
            # Save the plot with the epoch included in the file name
            save_path = os.path.join(plot_directory, f"decision_boundary_{epoch}.png")
            plt.savefig(save_path, bbox_inches='tight')
            
            # Close the figure to start fresh in the next iteration
            plt.close()


        draw_vol_amplification(xx, yy, input_amp_voltages, output_amp_voltages,epoch, plot_directory)
        draw_boundary(xx, yy, y_predictions, X_val, Y_val, epoch, plot_directory)
        
        
        
        fet_identifiers = ["0_2_1", "1_3_2"]      
        all_nodes = self.all_nodes
        aex_file = aex_file_path
        calculate_transconductance(eldo_process, fets_identifier, aex_file, all_nodes, debug)

def predict(process, X, vol_sources, debug, resistor_value_dict, node_to_vdc, node_voltages, output_nodes, boundary, bias):
    
    input_layer = layers[0]
    output_layer = layers[-1]
    input_dict = input_layer.parameters
    input_keys_list = list(input_dict.keys())

        
    resistive_layers = [layer for layer in layers if getattr(layer, 'type', None) == 'resistive']
        
    acc_list = []
        
    disable_current_sources(eldo_process, inudge_dict, debug)
    
    pred_outputs = []
    true_outputs = []
    vol_values = []
    X_pos =  X  
    X_neg = -X 
    X_in = np.hstack((X_pos, X_neg, X_bias))

    
    for i in range(0, X_in.shape[0]):        
        X_vec = X_in[i, :]
        

        
        #set up everything for the free phase
        #input_values = create_voltage_source_to_value_dict(node_to_vdc, X_vec)   
        
        input_values = dict(zip(vol_sources, X_vec))
        set_input_voltages(process, input_values, debug)
        
        
        #run simulation and wait for the results
        run_eldo_simulation(process, debug)
        wait_for_eldos_completion(process, debug)
        
        #extract the results
        mode = "get_voltage"
        free_node_voltages = extract_results(process, mode, node_voltages, resistor_value_dict, debug) #contains voltages at nodes at the end of free phase
      
        #calculate the losses and the nudging current
        #losses = loss_function_moon(Y_vec, free_node_voltages, output_nodes) # losses is a dictionary as well
        predicted_output = predicted_value(free_node_voltages, output_nodes, boundary)
        pred_outputs.append(predicted_output)
        vol_values.append(voltage_values(free_node_voltages, output_nodes))

    return pred_outputs
    

def train(sim_params, process_id = None): #probably objective function

    fet_identifiers = ["0_1_1", "1_1_1"]
    transcon_calc = False

    #sim_params = SimulationParameters()
    layers = initialize_network_layers(sim_params) #here also the weight matrices are initialized
    builder = netlist_builder(layers, sim_params)
    
    
    #Here I need to generate a new file name
    ########
    simulation_type = sim_params.simulation_type
    input_files = create_filenames(sim_params.output_dir, sim_params.sample_file, simulation_type, process_id)
    full_subfolder_path, new_sample_file, result_file_path = input_files
    #printfile = os.path.join(full_subfolder_path, "PRINTFILE.TXT")


    all_nodes = extract_all_nodes_voltages(layers)
    builder.build_netlist(new_sample_file, transcon_calc, fet_identifiers, result_file_path)

    
    net = MyNetwork(layers, sim_params, input_files, all_nodes)

         
    simulation = "moons_simulation"
        # Load and center the dataset
    if simulation == "moons_simulation":
        X_t, Y_t = ds.prepare_moons_data(sim_params.num_samples, noise=0.1, random_state=41)
    elif simulation == "digits_simulation":
        X_t, Y_t = ds.prepare_digits_data()
    elif simulation == "iris_simulation":
        X_t, Y_t = ds.prepare_iris_data()
        
        #X, Y = linear_regression(n_of_samples = 1200,a = float(1), b = float(7))
        
        
    #scaler = StandardScaler()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_t) * sim_params.scale_factor

    input_function = ds.onehot_pos_neg_inputs_1bias_double_input
    X, Y = ds.onehot_pos_neg_inputs_1bias_double_input(X_scaled, Y_t, net.bias)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)




    #pids = get_eldo_pids(eldo_identifier = 'eldo_64.exe')
    m_thread = True
    noascii =  True
    debug = False
    eldo_process, q = start_eldo_simulation_drain(new_sample_file, full_subfolder_path, m_thread, noascii, debug)
    setattr(net, 'q', q)
    #wait_for_eldos_completion_initialization(eldo_process, debug)

    #signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, eldo_process))
    #signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, eldo_process))
    
    
    
    
    
        #net.draw_grid(eldo_process, X_train, y_train, epoch, n_of_node_voltages, debug = False)
    accuracy_list = []
    big_loss_list = []
    diff1_list = []
    diff2_list = []
    ratio1_mean_list = []
    ratio2_mean_list = []        
        
    weight_matrices_1 = [net.layers[1].W]
    weight_matrices_2 = [net.layers[3].W]
    initial_weights1 = net.layers[1].W
    initial_weights2 = net.layers[3].W
        
    dynamic_outputs = False
    targets = None
    draw_grid_to_plot = [1, 2, 5, 9, 15, 25, 35]
    

    
    
    
    # Define the base directory
    
    base_dir = sim_params.trained_models_dir#"/home/filip/simulations/trained_models"
            
    # Generate a subfolder name with the current date and time
    time_str = datetime.now().strftime("%H%M%S")
    if process_id:
        full_subfolder_path = f"{simulation}_{time_str}_{process_id}"
    else:
        full_subfolder_path = f"{simulation}_{time_str}"
    # Combine the base directory with the subfolder name
    output_dir_path = os.path.join(base_dir, full_subfolder_path)
    os.makedirs(output_dir_path, exist_ok=True)
    
    
        
    plot_directory = os.path.join(output_dir_path, "plots")       
    os.makedirs(plot_directory, exist_ok=True)
    
    
    
    try: 
        for epoch in range(1, sim_params.n_of_epochs +1):
            #calculate_transconductance(eldo_process, fet_identifiers, new_sample_file, net.aex_file_path, net.all_nodes, debug)
            start_epoch = time.time()
                #NEED TO MANUALLY SET THEM, Inew_sample_file+NITIALLY EVERYTHING IS 0
                #set_input_voltages(eldo_process, bias_dict, debug = True)
                #print(f"Starting epoch {epoch} for boundary {boundary}, batch size {batch_size} and scale factor {scale_factor}")
            if dynamic_outputs:
                targets = net.calc_desired_outputs(eldo_process, X, Y, epoch, debug)

            optimizer = None
            results = net.free_nudged_train(eldo_process, X_train, y_train, targets, epoch, optimizer, debug)
            #if epoch in draw_grid_to_plot:
            #   net.draw_grid(eldo_process, X_train, y_train, input_function, epoch, plot_directory, debug = False)
    
            accuracy_list.append(results["accuracy"])
            big_loss_list.append(results["loss_list"])
            weight_matrix1 = results["weight_matrix_1"]
            weight_matrix2 = results["weight_matrix_2"]
                
            diff1, diff2 = results["output_list"]
            diff1_list.extend(diff1)
            diff2_list.extend(diff2)
                #plot_moving_averages(diff1_list, diff2_list, window_size=40)
                
                
                
            ratio1_mean, ratio2_mean = results["ratio_list"]
            ratio1_mean_list.append(ratio1_mean)
            ratio2_mean_list.append(ratio2_mean)
                
                
            end_epoch = time.time()
            total_time = end_epoch - start_epoch
            #print(f"Time for epoch {total_time}")
                #rel_change1 = compute_relative_changes(weight_matrices, epsilon=1e-10)
                
                
                #plot_weight_histogram(weight_matrix1, bins=50, title='Weight Matrix 1 Histogram')
                #plot_weight_histogram(weight_matrix2, bins=50, title='Weight Matrix 2 Histogram')
            weight_matrices_1.append(results["weight_matrix_1"])
            weight_matrices_2.append(results["weight_matrix_2"])
            
                
            abs_change1 = compute_absolute_changes(weight_matrices_1)
            abs_change2 = compute_absolute_changes(weight_matrices_2)
            
            

            
            
            #plot_average_relative_change(rel_change1, title="Average Relative Weight Change per Epoch")
            #plot_average_relative_change(rel_change2, title="Average Relative Weight Change per Epoch")

            #plot_average_loss(big_loss_list)
            
            
            #plot_weight_evolution(weight_matrices_1, title='Weight Evolution Over Epochs')
            #plot_weight_evolution(weight_matrices_2, title='Weight Evolution Over Epochs')
            
            #net.draw_grid(eldo_process, X_train, y_train, input_function, epoch, debug = False)
            #accuracy = net.free_test(eldo_process, layers, X_test, y_test, epoch, loss_fn, metrics = None, debug = False)

                    # Save the configuration to a JSON file
    
        
            
            # --- Save Weight Matrices ---
            # Assume results is a dictionary containing your trained weight matrices:
            # results["weight_matrix_1"] and results["weight_matrix_2"]
        weights_file_path = os.path.join(output_dir_path, "weights.npz")
        np.savez(weights_file_path, 
                 weight_matrix_1=results["weight_matrix_1"], 
                 weight_matrix_2=results["weight_matrix_2"])
       #print(f"Weights saved to {weights_file_path}")
            
    
    
        config_file_path = os.path.join(output_dir_path, "config") 
        save_sim_parameters(sim_params, config_file_path)
    
    
    
        plot_accuracy(accuracy_list, plot_directory)
        plot_cosine_similarity(ratio1_mean_list, plot_directory, title = "Cosine sim for weight matrix 1")
        plot_cosine_similarity(ratio2_mean_list, plot_directory, title  = "Cosine sim for weight matrix 2")
        window_size = 40
        plot_moving_averages(diff1_list, diff2_list, window_size, plot_directory)
        plot_average_loss(big_loss_list, plot_directory)
        plot_average_relative_change(abs_change1, title="Average Absolute Weight Change per Epoch weightmatrix 1")
        plot_average_relative_change(abs_change2, title="Average Absolute Weight Change per Epoch weightmatrix 2")
    
        plot_weight_matrix_evolution_lines(weight_matrices_1, plot_directory, title='Weight Matrix 1 Evolution Over Epochs')
        plot_weight_matrix_evolution_lines(weight_matrices_2, plot_directory, title='Weight Matrix 2 Evolution Over Epochs')
        #delete_file_with_chi_extension(new_sample_file)
        
        
        
                # Save the key metrics data for future use
        metrics_data = {
            'accuracy_list': accuracy_list,                # accuracy per epoch
            'losses': big_loss_list,                        # loss values (big_loss_list)
            'weight_matrix_evolution_1': weight_matrices_1,   # evolution of weight matrix 1
            'weight_matrix_evolution_2': weight_matrices_2    # evolution of weight matrix 2
        }
        
        data_path = os.path.join(plot_directory, 'metrics_data.npz')
        np.savez_compressed(data_path, **metrics_data)
        
        
        
        send_quit_command(eldo_process)
        remove_directory(full_subfolder_path)
        return accuracy_list
        
    except Exception as e:
    # 1) Print the exception type, message, and full traceback
        print(f"Caught error: {type(e).__name__}: {e}")
        traceback.print_exc()       
        config_file_path = os.path.join(output_dir_path, "config") 
        save_sim_parameters(sim_params, config_file_path)
                # Save the key metrics data for future use
        metrics_data = {
            'accuracy_list': accuracy_list,                # accuracy per epoch
            'losses': big_loss_list,                        # loss values (big_loss_list)
            'weight_matrix_evolution_1': weight_matrices_1,   # evolution of weight matrix 1
            'weight_matrix_evolution_2': weight_matrices_2    # evolution of weight matrix 2
        }
        send_quit_command(eldo_process)
        remove_directory(full_subfolder_path)
        
    
if __name__ == "__main__":
    gamma_value =  [1e-8, 5e-9]

    gamma_values = [1e-8, 5e-9]
    batch_size = 8
    beta = 5e-5
    #scale_factor_list = [0.4, 0.5] 
    #bias_list = [0.2, 0.3, 0.4, 0.5, 0.6]
    #good scale factor is 0.4
    scale_factor_list = [0.3] 
    scale_factor = 0.3
    bias_list = [0.2]    
    bias = 0.3
    sim_params = SimulationParameters(scale_factor, bias, batch_size, beta, gamma_value, mode = "TESTING")
    train(sim_params, process_id = None)