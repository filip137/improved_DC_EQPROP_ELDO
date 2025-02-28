from layer_class import *
from initializer import *
from datasets import * 
import numpy as np
import time
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
from netlist_data import NETLIST_DEFINITIONS, PARAMS
#Initialize a complete neural network and build a netlist


class MyNetwork:
    def __init__(self, config, files, scale_factor):
        # Unpack configuration details
        
        simulation_details = config.get("simulation_details", {})
        self.boundary = simulation_details.get("boundary", 0.05)
        self.loss_fn = MSE(self.boundary)
        self.gradient_clip = simulation_details.get("gradient_clip", {5e-7})
        self.beta = simulation_details["beta"]

        
        
        
        # Unpack network details.
        network_details = config.get("network_details", {})
        self.nonlin_parameters_ss = network_details.get("nonlin_parameters_ss", {})
        self.nonlin_parameters_perfect = network_details.get("nonlin_parameters_perfect", {})
        self.ac_biases = network_details.get("AC_biases", {})
        self.frequency = network_details.get("frequency", None)
        self.neuron = network_details.get("neuron", {})
        self.freq = network_details.get("frequency", {})
        self.simulation_type = network_details.get("simulation_type", {"DC"})
        network_config = network_details.get("network_files", {})
        full_subfolder_path, self.new_sample_file, self.aex_file_path = files



        dataset_details = config.get("dataset_details", {})
        self.bias = dataset_details["bias"]
        self.batch_size = dataset_details["batch_size"]
        self.scale_factor = scale_factor
        
        
        self.layers = self.initialize_network_layers(simulation_details, network_details)
        self.all_nodes = extract_all_nodes_voltages(self.layers)
        
    #just builds the netlist
    def build_netlist(self):
        
        freq = self.freq
        neuron = self.neuron
        file_name = self.new_sample_file
        all_nodes = self.all_nodes
        parameter_lines = self.extract_parameters()
        #parameter_lines.append(PARAMS["amp_ss1"])
            
            
        #parameter_lines.extend(amp_parameter_lines)
        network_description = self.extract_connections()
        
        # Get current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate header with current date and time
        header = f"***\n" \
                 f"*** Generated for: eldoD\n" \
                 f"*** Generated on: {current_datetime}\n" \
                 f"*** Design library name: tests\n" \
                 f"*** Design cell name: 4moons\n" \
                 f"*** Design view name: schematic\n" \
                 f".GLOBAL\n"
                 
                 
        
        #not really sure how/if I can avoid doing this
        if neuron == "amp_ss":
            mid_sect = NETLIST_DEFINITIONS["amp_ss"]
        
        elif neuron == "perfect_amp":
            mid_sect = NETLIST_DEFINITIONS["perfect_amp"]
        
        counter = 0 
        
        if self.simulation_type == "FSST":
            vm_list = []
            for node in all_nodes:
                if counter == 0:
                    vm_list.append(".EXTRACT FSST")
                vm_node = f"YVAL(V({node}), {freq})"
                vm_list.append(vm_node)
                counter += 1
                if counter == 5:
                    # Append the current line and reset for a new one
                    vm_list.append("\n")
                    counter = 0
           # Join the vm_list into a string, removing unnecessary spaces and ensuring formatting
            vm_string = " ".join(filter(None, vm_list)).replace(" \n.", "\n.")
            # vi_string = " ".join(filter(None, vi_list)).replace(" \n.", "\n.")
            
            
            #f".AC LIST {freq}\n"
            simulation_details = (
                
                f".SST FUND1={freq} NHARM1=3\n"
                f"{vm_string}\n" ##here also append vi string if it's needed
                ".OPTION AEX\n"
                ".OPTION NOASCII\n"
                ".END\n"
            )


        if self.simulation_type == "DC":
            vdc_list = []
            for node in all_nodes:
                if counter == 0:
                    vdc_list.append(".EXTRACT DC")
                vdc_node = f"V({node})"
                vdc_list.append(vdc_node)
                counter += 1
                if counter == 5:
                    # Append the current line and reset for a new one
                    vdc_list.append("\n")
                    counter = 0
           # Join the vm_list into a string, removing unnecessary spaces and ensuring formatting
            vdc_string = " ".join(filter(None, vdc_list)).replace(" \n.", "\n.")
            # vi_string = " ".join(filter(None, vi_list)).replace(" \n.", "\n.")
            
            
            simulation_details = (
                
                f".DC\n"
                f"{vdc_string}\n" ##here also append vi string if it's needed
                ".OPTION AEX\n"
                ".OPTION NOASCII\n"
                ".END\n"
            )
                             
        with open(file_name, 'w') as file:
            file.write(header)
            for line in parameter_lines:
                file.write(line + '\n')  # Ensure line is a string and add a newline
            file.write(mid_sect)
            for section in network_description:
                for line in section:
                    file.write(line)  # Write each line from the sections
            file.write(simulation_details)
        print(f"Network description and parameters have been saved to {file_name}.")
        
    
    def extract_parameters(self):
        # ampv = self.ampv
        # ampc = self.ampc
        form = ["FORM=0"]
        all_parameters = []
        
        # Add parameters from each layer with `.PARAM` prefix
        for layer in self.layers:
            if layer.parameters:
                for key, value in layer.parameters.items():
                    line = f".PARAM {key}={value}"
                    all_parameters.append(line)
            else:
                pass
        # Add other parameters with `.PARAM` prefix
        # all_parameters.extend([f".PARAM {item}" for item in ampv])
        # all_parameters.extend([f".PARAM {item}" for item in ampc])
        all_parameters.extend([f".PARAM {item}\n" for item in form])
        
        return all_parameters
    
    def extract_connections(self):
        extracted_connections = []
        for layer in self.layers:
            lines = layer.connections  # Assuming 'connections' is already a list
            extracted_connections.extend(lines)  # Use 'extend' instead of 'append' to add all elements of 'lines'
        return extracted_connections

    
  

    def free_nudged_train(self, eldo_process, X_train, Y_train, targets, epoch, optimizer = None, debug = False):
        
        
        layers = self.layers
        beta = self.beta
        batch_size = self.batch_size
        loss_fn = self.loss_fn
        aex_result_file = self.aex_file_path
        simulation_type = self.simulation_type

        
        n_of_node_voltages = len(self.all_nodes)
        
        
        
        input_layer = layers[0]
        output_layer = layers[-1]
        input_dict = input_layer.inputs
        input_keys_list = list(input_dict.keys())
        
        inudge_dict = output_layer.parameters
        inudge_keys_list = list(inudge_dict.keys())
        
        resistive_layers = [layer for layer in layers if getattr(layer, 'type', None) == 'resistive']
        
        num_batches = int(np.ceil(len(X_train) / batch_size))
         
        for layer in resistive_layers:
            layer.gamma *= 1
    
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
        voltage_dict_free = dict.fromkeys(self.all_nodes, None)
        voltage_dict_nudge = dict.fromkeys(self.all_nodes, None)
        
        timings_list = []
        offset= 0
        if epoch > 1:
            file_size = os.path.getsize(aex_result_file)
            offset = file_size
            #print(f"File size before clearing: {file_size} bytes")
        
        
        for i in range(num_batches):
            #if i > 0:
                #start_index += 3
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
                    
                beta_r = beta
                set_input_voltages(eldo_process, input_dict, debug = False)
                #move at the end
                disable_current_sources(eldo_process, inudge_dict, debug = False)
                #time.sleep(0.01)
                start_simulation = time.time()
                run_eldo_simulation(eldo_process, debug)
                
                
                end_index = start_index + n_of_node_voltages
                
                
                
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
                
                mode = "train-voltage"
                #target = Y * self.boundary
                if targets != None:
                    Y_index = np.argmax(Y)
                    target = targets[Y_index]
                else:
                    target = (Y/4) * self.scale_factor
                sample_losses, target_voltages = loss_fn(outputs, target, beta_r, mode = "train-voltage")#need to decide where to initialize this function - remember that loss_fn comes from the already initialized MSE
                #predicted value
                #pred_free = loss_fn(outputs, target, beta_r, mode)
                
                #now I can simply zip the currents to the parameters of the last layer and repeat
                target_voltages = target_voltages.flatten()

                
                if mode == "train-current":
                    for k, key in enumerate(inudge_keys_list):
                        inudge_dict[key] = flat_currents[k]    
                        
                if mode == "train-voltage":
                    voltage_index = 0
                    for key in inudge_keys_list:
                        if key.startswith("VNUDGE"):
                            inudge_dict[key] = target_voltages[voltage_index]
                            voltage_index += 1
                        elif key.startswith("RNUDGE"):
                            inudge_dict[key] = 0.01
                                        
                #output_layer.parameters = output_layer.update_parameters(flat_currents)
                
                set_currents_nudge_mode(eldo_process, inudge_dict, debug = False)
                run_eldo_simulation(eldo_process, debug)
                #lines_of_interest = wait_for_eldos_completion(eldo_process, debug)
                wait_for_eldos_completion_old(eldo_process, debug)

                mode = 'test' 

                start_index += n_of_node_voltages + 3
                end_index = start_index + n_of_node_voltages
                
                parse_aex_file_from_end_timed(aex_result_file, n_of_node_voltages, simulation_type, voltage_dict_nudge, offset)
                 



                start_index += n_of_node_voltages + 3

                python_update_time = time.time
                for layer in resistive_layers:
                    layer.update__nudge_voltages(voltage_dict_nudge)
                
                
                outputs_n = layer.output_nudge_voltages
                outputs_n_values = list(outputs_n.values())#the outputs are just the outputs of the last layer
                output_list_nudge.append(outputs_n_values)   
                sample_losses_n, voltages_n = loss_fn(outputs_n, target, beta_r, mode = "train-voltage")
                
                
                for layer in resistive_layers:
                    layer.run_update_process(batch_size, beta_custom = beta_r)
                    
                    

                #print(f"Python update time {python_update_time_end}")
                    
                loss_list.append(sample_losses)
                
                
              
                
              
            #At the end of the batch update all resistances
            for i, layer in enumerate(resistive_layers):
                layer.update_W(mode = "clip_updates", clip = self.gradient_clip)
                layer.update_res_dict()
                set_resistances(eldo_process, layer.resistor_dict, debug = False)
                #time.sleep(0.01)
                ratios = compute_cosine_similarity(layer.deltaG, layer.W_old, layer.W)
                if i == 0:
                    ratios_list1.append(ratios)
                elif i == 1:
                    ratios_list2.append(ratios)
                #print(f"Norm of the deltaG {np.linalg.norm(layer.deltaG, ord=2)}")
                #print(f"finished batch {i}")
                
            weight_matrices_1.append(layers[1].W)
            weight_matrices_2.append(layers[3].W)
                
            

            
            #truncate__aex_file(aex_result_file, 10)
            #clear_aex_file(aex_result_file)

            #print(f"Batch duration {batch_duration}")
        
        
        
    #    num_iterations = len(timings_list)
#        num_parts = len(timings_list[0])
        # average_timings = [0.0] * num_parts
        # for timings in timings_list:
        #     for i, t in enumerate(timings):
        #         average_timings[i] += t
        # average_timings = [t / num_iterations for t in average_timings]

        # print("Average timings (in seconds):", average_timings)
        
        
        #plot_cosine_similarity(ratios_list1)
        ratio1_mean = np.array(np.mean(ratios_list1))
        ratio2_mean = np.array(np.mean(ratios_list2))
        ratio_list = [ratio1_mean, ratio2_mean]
        #plot_cosine_similarity(ratios_list2)
        #reset_chi_file(eldo_process, debug = False)
        #reset_extract_file(eldo_process, debug = True)
        #truncate__aex_file(aex_result_file, 10)
        #clear_aex_file(aex_result_file)
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
        #gamma = None
        #plot_free_and_nudged(output_list, output_list_nudge, output_nodes, beta, gamma = None, epoch = None)
        #plot_weight_matrix_evolution_lines(weight_matrices_1, interval=10, title = "First weight matrix")    
        #plot_weight_matrix_evolution_lines(weight_matrices_2, interval = 10,title = "Second weight matrix")  
        #plot_weight_matrix_evolution_separate(weight_matrices_1, interval=10, title = "First weight matrix")    
        #plot_weight_matrix_evolution_separate(weight_matrices_2, interval=10, title = "Second weight matrix")    

        #clear_aex_file(aex_result_file)
            #print("Successfully deleted aex file")
        
        #output_plot(output_list)
        #plot_loss(loss_list, epoch)
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
            X_in, Y = input_function(X_grid, Y_in, scale_factor = self.scale_factor, bias = self.bias, output_scale=1)
            
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
        prediction_list = []
        
        #clear_aex_file(aex_result_file)

        if epoch == 0:
            start_index = 4
        else:
            start_index = 3 
        start_index = 3
        aex_result_file = self.aex_file_path
        #clear_aex_file(aex_result_file)
        #truncate__aex_file(aex_result_file, 10)
        counter = 0
        file_size = os.path.getsize(aex_result_file)
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
            parse_aex_file_from_end_timed(aex_result_file, n_of_node_voltages, simulation_type, voltage_dict_free, offset)
            output_layer.update__free_voltages(voltage_dict_free)
            
            outputs = output_layer.output_free_voltages #the outputs are just the outputs of the last layer
            output_values = list(outputs.values())
            prediction_list.append(output_values)
            #start_index += n_of_node_voltages + 3


            prediction = self.loss_fn(output_values, mode='test') #need to decide where to initialize this function - remember that loss_fn comes from the already initialized MSE
            binary_prediction = self.loss_fn.binary_prediction(prediction)
            binary_list.extend(binary_prediction)
            
            # if counter > 10:
            #     truncate__aex_file(aex_result_file, 10)
            #     counter = 0
            #     start_index = 3
            # counter += 1
            
        binary_array = np.array(binary_list).reshape(-1,1)
        #accuracy = np.mean(np.equal(binary_array, Y_in)) * 100
        #print(f"Accuracy: {accuracy:.2f}%")
        
        return binary_array
    
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
    

    def draw_grid(self, eldo_process, X_val, Y_val, input_function, epoch, debug):
        
        # Define bounds of the domain
        min1, max1 = X_val[:, 0].min() - 0.1, X_val[:, 0].max() + 0.1
        min2, max2 = X_val[:, 1].min() - 0.1, X_val[:, 1].max() + 0.1
        
        num_points = 50
    
        x1grid = np.linspace(min1, max1, num_points)
        x2grid = np.linspace(min2, max2, num_points)
    
    # Create a meshgrid from the grid points
        xx, yy = np.meshgrid(x1grid, x2grid)
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        
        Y_indices = np.argmax(Y_val, axis=1) 
    # Make predictions for the grid
        X_bias = X_val[0,-1]
        draw_grid = True
        y_predictions = self.free_test(eldo_process, grid, X_bias, Y_val, input_function, epoch, draw_grid, debug = False)
    
    # Convert predictions into an array and reshape back into a grid
        y_predictions = np.array(y_predictions)
        zz = y_predictions.reshape(xx.shape)
    
    # Plot the grid of x, y, and z values as a surface
        contour = plt.contourf(xx, yy, zz, cmap='Paired')
        cbar = plt.colorbar(contour)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Class 0', 'Class 1'])
        Y_val = Y_indices.ravel()
    # Separate the points by class
        class0 = X_val[Y_indices == 0]
        class1 = X_val[Y_indices == 1]
    
    # Scatter plot for the validation set with legend
        plt.scatter(class0[:, 0], class0[:, 1], c='blue', edgecolor='k', marker='o', s=20, label='Class 0')
        plt.scatter(class1[:, 0], class1[:, 1], c='red', edgecolor='k', marker='o', s=20, label='Class 1')
    
    # Add legend
        plt.legend()
    
    # Add titles and labels
        plt.title(f"Decision Boundary for test dataset after epoch {epoch} and boundary {self.boundary}")
        plt.xlabel('VDC1')
        plt.ylabel('VDC2')
    
    # Show plot
        plt.show()


    def initialize_network_layers(self, simulation_details, network_details):
        
        
        init_config = simulation_details["initializer"]
        fc_layers = simulation_details["layers"]["fully_connected"]
        lower_cond_bound = simulation_details["layers"]["lower_cond_bound"]
        upper_cond_bound = simulation_details["layers"]["upper_cond_bound"]
        
        
        lr_layer1 = simulation_details["learning_rate_factors"]["lr_layer1"]
        lr_layer2 = simulation_details["learning_rate_factors"]["lr_layer2"]
        
        gamma_layer1 = simulation_details["gamma_values"]["layer1"]
        gamma_layer2 = simulation_details["gamma_values"]["layer2"]
        
        beta = simulation_details["beta"]
        
        
        
        
        
        nonlin_parameters = network_details["nonlin_parameters_perfect"]
        vdc_bias = network_details["AC_biases"]["source_dc_bias"]
        freq = network_details["frequency"]
        
        
        
        synapse = "resistive"
        iparams = init_config["init_type"]
        wparams = init_config["params"]
        weight_initializer = Initializer(iparams, wparams)
        print(f"Doing simulations for {wparams}")
        layers = []
        
        simulation_type = self.simulation_type
        if simulation_type == "DC":
            nonlin_parameters = network_details["nonlin_parameters_perfect"]
        elif simulation_type == "FSST":
            nonlin_parameters = network_details["nonlin_parameters_ss"]

        
        
        input_layer = InputLayer(fc_layers[0], vdc_bias, freq, simulation_type, which_layer=0)
        layers.append(input_layer)
        
        # First Dense Layer
        n_of_inputs = fc_layers[0]
        n_of_outputs = fc_layers[1]
        which_layer = 0
        layer1 = DenseLayer(
            n_of_inputs, n_of_outputs, which_layer, synapse, 
            lr_layer1, gamma_layer1, beta, initializer=weight_initializer,
            lower_cond_bound=lower_cond_bound, upper_cond_bound=upper_cond_bound
        )
        layer1.initialize_res()
        layer1.initialize_W()
        layers.append(layer1)
        
        # Non-linear Layer
        layer2 = NonLinearLayer(n_of_outputs, which_layer, nonlin_parameters, neuron_type = self.neuron)
        layers.append(layer2)
        
        # Second Dense Layer
        n_of_inputs = fc_layers[1]
        n_of_outputs = fc_layers[2]
        which_layer = 1
        layer3 = DenseLayer(
            n_of_inputs, n_of_outputs, which_layer, synapse, 
            lr_layer2, gamma_layer2, beta, initializer=weight_initializer,
            lower_cond_bound=lower_cond_bound, upper_cond_bound=upper_cond_bound
        )
        layer3.initialize_res()
        layer3.initialize_W()
        layers.append(layer3)
        
        # Output Layer
        n_of_inputs = fc_layers[2]
        layer4 = OutputLayer(n_of_inputs, freq, simulation_type, which_layer)
        layers.append(layer4)
        
        return layers

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
    







def main(): #probably objective function



    # Load the configuration file
    with open("config_DC.json", "r") as file:
        config = json.load(file)
    
    # #  
    simulation_details = config["simulation_details"]
    network_details = config["network_details"]
    dataset_details = config["dataset_details"]
    print(simulation_details)
    print(network_details)

    ########################
    network_config = network_details["network_files"]
    input_files = create_filenames(network_config)
    full_subfolder_path, new_sample_file, aex_file_path = input_files
    
    #Extract dataset
    dataset_config = dataset_details
    n_of_epochs = dataset_config["n_of_epochs"]
    scale_factor = dataset_config["scale_factor"]
    noise = dataset_config["noise"]
    bias = dataset_config["bias"]
    num_samples = dataset_config["num_samples"]
    batch_size = dataset_config["batch_size"]
    

    
    # Initialize weight initializer
    # init_config['params'] = {key: float(value) for key, value in init_config['params'].items()}
    # weight_initializer = Initializer(init_type=init_config["init_type"], params=init_config["params"])
    
    
    #Initialize the network
    
    # Initialize the network layers
    
    scale_factor_list = [4]
    #batch_size = 32
    for scale_factor in scale_factor_list:
        #layers = initialize_network_layers(simulation_details, network_details)

        
        #Initialize the network
        net = MyNetwork(config, input_files, scale_factor)
        #Build the netlist
        net.build_netlist()
         
        simulation = "iris_simulation"
        # Load and center the dataset
        if simulation == "moons_simulation":
            X_t, Y_t = prepare_moons_data(num_samples, noise=0.1, random_state=41)
        elif simulation == "digits_simulation":
            X_t, Y_t = prepare_digits_data()
        elif simulation == "iris_simulation":
            X_t, Y_t = prepare_iris_data()
        
        #X, Y = linear_regression(n_of_samples = 1200,a = float(1), b = float(7))
        
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_t) * scale_factor
        #X_scaled = X_scaled

        # Process the scaled wine data with onehot_pos_neg_inputs.
        # Note: scale_factor, bias, and output_scale should be defined previously.
        input_function = onehot_pos_neg_inputs_1bias
        X, Y = onehot_pos_neg_inputs_1bias_double_input(X_scaled, Y_t, bias = 1, output_scale=1)
        

        
        # Scale the features using StandardScaler
        #scaler = StandardScaler()

        
        # Optionally, you can visualize the data here (if needed)
        #plot_moons_data(X[:, :2], Y)  # Adjust function name if necessary
        
        # Split the processed data into training and test sets.
        #from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
        
        #vac+bias is the bias of the ac voltage source, vbias is the additional bias that is currently not used
        #bias_dict = {"VAC_BIAS" : v_ac_bias}
        
        #This does not work and it really should work
        pids = get_eldo_pids(eldo_identifier = 'eldo_64.exe')
        eldo_process = start_eldo_simulation(new_sample_file, full_subfolder_path, m_thread = True, noascii =  True, debug=True )
        signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, eldo_process))
        signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, eldo_process))
        
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
        for epoch in range(1, n_of_epochs +1):
            start_epoch = time.time()
            #NEED TO MANUALLY SET THEM, INITIALLY EVERYTHING IS 0
            #set_input_voltages(eldo_process, bias_dict, debug = True)
            #print(f"Starting epoch {epoch} for boundary {boundary}, batch size {batch_size} and scale factor {scale_factor}")
            if dynamic_outputs:
                targets = net.calc_desired_outputs(eldo_process, X, Y, epoch= 0, debug = False)
            results = net.free_nudged_train(eldo_process, X_train, y_train, targets, epoch, optimizer = None, debug = False)
            # if epoch in draw_grid_to_plot:
            #     net.draw_grid(eldo_process, X_train, y_train, input_function, epoch, debug = False)

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
            print(f"Time for epoch {total_time}")
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
        # Define the base directory
        base_dir = "/home/filip/simulations/trained_models"
        
        # Generate a subfolder name with the current date and time
        date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_subfolder_path = f"{simulation}_{date_time_str}"
        
        # Combine the base directory with the subfolder name
        output_dir_path = os.path.join(base_dir, full_subfolder_path)
        os.makedirs(output_dir_path, exist_ok=True)
                # Save the configuration to a JSON file
        config_file_path = os.path.join(output_dir_path, "config.json")
        with open(config_file_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {config_file_path}")
        
        # --- Save Weight Matrices ---
        # Assume results is a dictionary containing your trained weight matrices:
        # results["weight_matrix_1"] and results["weight_matrix_2"]
        weights_file_path = os.path.join(output_dir_path, "weights.npz")
        np.savez(weights_file_path, 
                 weight_matrix_1=results["weight_matrix_1"], 
                 weight_matrix_2=results["weight_matrix_2"])
        print(f"Weights saved to {weights_file_path}")
        
        
        
        
        
        

        plot_directory = os.path.join(output_dir_path, "plots")       
        os.makedirs(plot_directory, exist_ok=True)

        plot_accuracy(accuracy_list, plot_directory)
        plot_cosine_similarity(ratio1_mean_list, plot_directory, title = "Cosine sim for weight matrix 1")
        plot_cosine_similarity(ratio2_mean_list, plot_directory, title  = "Cosine sim for weight matrix 2")
        window_size = 40
        plot_moving_averages(diff1_list, diff2_list, window_size, plot_directory)
        plot_average_loss(big_loss_list, plot_directory)
        plot_average_relative_change(abs_change1, title="Average Absolute Weight Change per Epoch weightmatrix 1")
        plot_average_relative_change(abs_change2, title="Average Absolute Weight Change per Epoch weightmatrix 2")

        plot_weight_matrix_evolution_lines(weight_matrices_1, plot_directory, title='Weight Evolution Over Epochs')
        plot_weight_matrix_evolution_lines(weight_matrices_2, plot_directory, title='Weight Evolution Over Epochs')
        delete_file_with_chi_extension(new_sample_file)


        
    
if __name__ == "__main__":
    main()