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
    def __init__(self, layers, loss_fn, boundary, aex_file_path, new_sample_file):
        self.layers = layers
        self.loss_function = loss_fn
        self.loss_fn = loss_fn
        self.boundary = boundary
        self.aex_file_path = aex_file_path
        self.new_sample_file = new_sample_file
        self.simulation_type = "FSST"
    #just builds the netlist
    def build_netlist(self, file_name, all_nodes, freq = "1Meg"):
        

        
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
                 f"*** Design cell name: 2moons\n" \
                 f"*** Design view name: schematic\n" \
                 f".GLOBAL\n"
                 
                 
        neuron = 'amp_ss'
        #not really sure how/if I can avoid doing this
        if neuron == "amp_ss":
            mid_sect = NETLIST_DEFINITIONS["amp_ss"]
        
        elif neuron == "perfect_amp":
            mid_sect = NETLIST_DEFINITIONS["perfect_amp"]
        
        vm_list = []
        counter = 0 
        
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

        # vi_list = []
        # counter = 0 
        
        # for node in all_nodes:
        #     if counter == 0:
        #         vi_list.append(".EXTRACT")
        #     vi_node = f"vi({node})"
        #     vi_list.append(vi_node)
        #     counter += 1
        #     if counter == 5:
        #         # Append the current line and reset for a new one
        #         vi_list.append("\n")
        #         counter = 0



        
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
        
        
        
        
    # currently this is not used
    # def build_network(self, diode_dict, synapse):
        
    #     layers = []
    #     input_layer = InputLayer(self.fc_layers[0], self.fc_layers[0], which_layer = 0)
    #     layers.append(input_layer)
    #     synapse = "resistor"
    #     diode_dict = {"VDIODE1" : None, "VDIODE2" : None} #this are strictly parameters for the hidden layer
    #     for i in range(1, len(self.fc_layers)):
    #         n_of_inputs = self.fc_layers[i-1]
    #         n_of_outputs = self.fc_layers[i]
    #         which_layer = i-1
    #         layer1 = DenseLayer(n_of_inputs, n_of_outputs, which_layer, synapse)
    #         layers.append(layer1)
    #         if i == len(self.fc_layers)-1:
    #             break
    #         else:
    #             layer2 = NonLinearLayer(n_of_outputs, n_of_outputs, which_layer, diode_dict)
    #             layers.append(layer2)
    #         #layers.append(layer1)
            
    #     layers.append(OutputLayer(self.fc_layers[-1], self.fc_layers[-1], which_layer = 1))
        
    #     return layers
    
    
    
    
    
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

    
  

    def free_nudged_train(self, eldo_process, layers, X_train, Y_train, beta, epochs, batch_size, loss_fn, n_of_node_voltages, aex_file_path, optimizer = None, debug = False):
        
        # def signal_handler(sig, frame):
        #     if True:
        #         print("Interrupt received, sending quit command to subprocess.")
        #         send_command_to_eldo(eldo_process, "QUIT", debug)
        #     else:
        #         print("Interrupt received, exiting without sending command.")
        #     sys.exit(0)  # Exit the program
        
     
        input_layer = layers[0]
        output_layer = layers[-1]
        input_dict = input_layer.inputs
        input_keys_list = list(input_dict.keys())
        
        inudge_dict = output_layer.parameters
        inudge_keys_list = list(inudge_dict.keys())
        
        resistive_layers = [layer for layer in layers if getattr(layer, 'type', None) == 'resistive']
        
        num_batches = int(np.ceil(len(X_train) / batch_size))
         
    
        prediction_list = []
        loss_list = []
        output_list = []
        output_list_nudge = []
        weight_matrices_1 = []
        weight_matrices_2 = []
        
        
        start_index = 4
        n_of_node_voltages = n_of_node_voltages
        aex_result_file = self.aex_file_path
        simulation_type = 'FSST'
        
        for i in range(num_batches):
            if i > 0:
                start_index = 4
            batch_start_time = time.time()
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))  # Ensure not to exceed the dataset length

            # Select the batch
            X_batch = X_train[start_idx:end_idx]
            Y_batch = Y_train[start_idx:end_idx]
            
            for X, Y in zip(X_batch, Y_batch):
                sample_start_time = time.time()
                for j, key in enumerate(input_keys_list):
                    input_dict[key] = X[j]  # Directly assign the value from X to the corresponding key
                    
                beta_r = beta * np.random.randint(0, 100) 
                set_input_voltages(eldo_process, input_dict, debug)
                #move at the end
                disable_current_sources(eldo_process, inudge_dict, debug)
                simulation_start_time = time.time()
                run_eldo_simulation(eldo_process, debug)
                
                
                end_index = start_index + n_of_node_voltages
                
                
                
                lines_of_interest = wait_for_eldos_completion(eldo_process, debug)
                voltage_dict_free = parse_aex_file_no_end(aex_result_file, start_index, simulation_type)
                simulation_end_time = time.time() - simulation_start_time
                #print(f"Simulation duration {simulation_end_time}")
                volt_extract = time.time()
                
                for layer in resistive_layers:
                    layer.update__free_voltages(voltage_dict_free)
                     
                volt_extract_end = time.time() - volt_extract
                #print(f"Voltage extraction {volt_extract_end}")
                outputs = layer.output_free_voltages #the outputs are just the outputs of the last layer
                output_values = np.array(list(outputs.values()))
                output_list.append(output_values)
                
                mode = "train"
                target = (2 * Y - 1) * self.boundary
                sample_losses, currents = loss_fn(outputs, target, beta_r, mode)#need to decide where to initialize this function - remember that loss_fn comes from the already initialized MSE
                #predicted value
                mode = 'test' 
                pred_free = loss_fn(outputs, target, beta_r, mode)
                
                #now I can simply zip the currents to the parameters of the last layer and repeat
                flat_currents = currents.flatten()

                

                for k, key in enumerate(inudge_keys_list):
                    inudge_dict[key] = flat_currents[k]    
                    
                    
                    
                #output_layer.parameters = output_layer.update_parameters(flat_currents)
                
                set_currents_nudge_mode(eldo_process, inudge_dict, debug)
                run_eldo_simulation(eldo_process, debug)
                lines_of_interest = wait_for_eldos_completion(eldo_process, debug)
                            
                
                start_index += n_of_node_voltages + 3
                end_index = start_index + n_of_node_voltages
                
                voltage_dict_nudge = parse_aex_file_no_end(aex_result_file, start_index, simulation_type)
                 



                start_index += n_of_node_voltages + 3

                python_update_time = time.time
                for layer in resistive_layers:
                    layer.update__nudge_voltages(voltage_dict_nudge)
                
                
                outputs_n = layer.output_nudge_voltages
                outputs_n_values = list(outputs_n.values())#the outputs are just the outputs of the last layer
                output_list_nudge.append(outputs_n_values)   
                
                
                python_update_time = time.time()
                
                for layer in resistive_layers:
                    layer.run_update_process(mode = "discrete")
                weight_matrices_1.append(layers[1].W)
                weight_matrices_2.append(layers[3].W)
                python_update_time_end = time.time() - python_update_time
                #print(f"Python update time {python_update_time_end}")
                    
                loss_list.append(sample_losses)
                
                sample_duration = time.time() - sample_start_time
                
              
            res_start_time = time.time()   
            #At the end of the batch update all resistances
            for layer in resistive_layers:
                layer.update_res_dict()
                set_resistances(eldo_process, layer.resistor_dict, debug)
            
                
            

            #res_duration = time.time() - res_start_time                
            #batch_duration = time.time() - batch_start_time
            
            #prediction = loss_fn(output_list[batch_size*i:(batch_size+batch_size*i)], mode='test')
            #prediction_list.extend(prediction)
            truncate__aex_file(aex_result_file, 10)
            #clear_aex_file(aex_result_file)

            #print(f"Batch duration {batch_duration}")
        
        
        predictions = loss_fn(output_list, mode='test')
        epoch_acc = np.mean(loss_fn.verify_result(Y_train, np.array(predictions)))
        print(f"Accuracy after epoch {epoch_acc}")
        output_nodes = [0,1,2,3, 4, 5]
        plot_free_and_nudged(output_list, output_list_nudge, output_nodes, beta = None, gamma = None)
        plot_weight_matrix_evolution_lines(weight_matrices_1, interval=10, title = "First weight matrix")    
        plot_weight_matrix_evolution_lines(weight_matrices_2, interval = 10,title = "Second weight matrix")  
        plot_weight_matrix_evolution_separate(weight_matrices_1, interval=10, title = "First weight matrix")    
        plot_weight_matrix_evolution_separate(weight_matrices_2, interval=10, title = "Second weight matrix")    

        #clear_aex_file(aex_result_file)
            #print("Successfully deleted aex file")
        
        output_plot(output_list)
        plot_loss(loss_list)
        disable_current_sources(eldo_process, inudge_dict, debug)

        return loss_list
            
            
    def free_test(self, eldo_process, X_grid, X_bias, Y_in, epoch, n_of_node_voltages, draw_grid, debug):
        
        # def signal_handler(sig, frame):
        #     if True:
        #         print("Interrupt received, sending quit command to subprocess.")
        #         send_command_to_eldo(eldo_process, "QUIT", debug)
        #     else:
        #         print("Interrupt received, exiting without sending command.")
        #     sys.exit(0)  # Exit the program
        
        if draw_grid:
            X_pos =  X_grid
            X_neg = -X_grid 
            X_bias_arr_pos = X_bias*np.ones((X_pos.shape[0], 1))
            X_bias_arr_neg = - X_bias*np.ones((X_pos.shape[0], 1))
            X_in = np.hstack((X_pos, X_neg, X_bias_arr_pos, X_bias_arr_neg))
            
        input_layer = self.layers[0]
        output_layer = self.layers[-1]
        input_dict = input_layer.inputs
        input_keys_list = list(input_dict.keys())

        
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
        for X in X_in:
            for i, key in enumerate(input_keys_list):
                input_dict[key] = X[i]  # Directly assign the value from X to the corresponding key
                
            set_input_voltages(eldo_process, input_dict, debug)
                #move at the end
            run_eldo_simulation(eldo_process, debug)
            lines_of_interest = wait_for_eldos_completion(eldo_process, debug)

            end_index = start_index + n_of_node_voltages
            voltage_dict_free = parse_aex_file_no_end(aex_result_file, start_index, self.simulation_type)
            output_layer.update__free_voltages(voltage_dict_free)
            
            outputs = output_layer.output_free_voltages #the outputs are just the outputs of the last layer
            output_values = list(outputs.values())
            prediction_list.append(output_values)
            start_index += n_of_node_voltages + 3


            prediction = self.loss_fn(output_values, mode='test') #need to decide where to initialize this function - remember that loss_fn comes from the already initialized MSE
            binary_prediction = self.loss_fn.binary_prediction(prediction)
            binary_list.extend(binary_prediction)
            if counter > 10:
                truncate__aex_file(aex_result_file, 10)
                counter = 0
                start_index = 3
            counter += 1
            
        binary_array = np.array(binary_list).reshape(-1,1)
        #accuracy = np.mean(np.equal(binary_array, Y_in)) * 100
        #print(f"Accuracy: {accuracy:.2f}%")
        
        return binary_array
    


    def draw_grid(self, eldo_process, X_val, Y_val, epoch, n_of_node_voltages, debug):
        
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
        y_predictions = self.free_test(eldo_process, grid, X_bias, Y_val, epoch, n_of_node_voltages, draw_grid, debug = False)
    
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
    



def initialize_network_layers(simulation_details, network_details):
    """
    Initializes the network layers and returns a list of layers.

    Parameters:
    - fc_layers: List[int], the sizes of fully connected layers.
    - weight_initializer: Initializer, the weight initializer object.
    - diode_dict: Dict, configuration for the diode layer.
    - synapse: str, the type of synapse to use. Default is "resistor".
    - lr_layer1: float, learning rate for the first layer.
    - gamma_layer1: float, gamma value for the first layer.
    - lr_layer2: float, learning rate for the second layer.
    - gamma_layer2: float, gamma value for the second layer.
    - beta: float, beta value for all layers.
    - lower_cond_bound: float, lower bound for conductance.
    - upper_cond_bound: float, upper bound for conductance.

    Returns:
    - layers: List, initialized layers of the network.
    """
    
    
    init_config = simulation_details["initializer"]
    fc_layers = simulation_details["layers"]["fully_connected"]
    lower_cond_bound = simulation_details["layers"]["lower_cond_bound"]
    upper_cond_bound = simulation_details["layers"]["upper_cond_bound"]
    
    
    lr_layer1 = simulation_details["learning_rate_factors"]["lr_layer1"]
    lr_layer2 = simulation_details["learning_rate_factors"]["lr_layer2"]
    
    gamma_layer1 = simulation_details["gamma_values"]["layer1"]
    gamma_layer2 = simulation_details["gamma_values"]["layer2"]
    
    beta = simulation_details["beta"]
    
    
    
    
    
    
    nonlin_parameters = network_details["nonlin_parameters"]
    vdc_bias = network_details["AC_biases"]["source_dc_bias"]
    freq = network_details["frequency"]
    
    
    
    synapse = "resistive"
    iparams = init_config["init_type"]
    wparams = init_config["params"]
    weight_initializer = Initializer(iparams, wparams)
    print(f"Doing simulations for {wparams}")
    layers = []
    
    
    
    input_layer = InputLayer(fc_layers[0], vdc_bias, freq, which_layer=0)
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
    layer2 = NonLinearLayer(n_of_outputs, which_layer, nonlin_parameters, neuron_type = 'AMPLIFICATION_SS')
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
    layer4 = OutputLayer(n_of_inputs, freq, which_layer)
    layers.append(layer4)
    
    return layers



def main(): #probably objective function



    # Load the configuration file
    with open("config_amp_imp.json", "r") as file:
        config = json.load(file)
    
    # # However I still need to feed both of these to the 
    simulation_details = config["simulation_details"]
    network_details = config["network_details"]
    
    
    
    
    
    
    # init_config = config["initializer"]
    # fc_layers = config["layers"]["fully_connected"]
    # lower_cond_bound = config["layers"]["lower_cond_bound"]
    # upper_cond_bound = config["layers"]["upper_cond_bound"]
    
    # nonlin_parameters = config["nonlin_parameters"]
    # v_ac_bias = config["AC_biases"]["source_dc_bias"]
    
    
    # lr_layer1 = config["learning_rate_factors"]["lr_layer1"]
    # lr_layer2 = config["learning_rate_factors"]["lr_layer2"]
    
    # gamma_layer1 = config["gamma_values"]["layer1"]
    # gamma_layer2 = config["gamma_values"]["layer2"]
    
    # beta = config["beta"]
    
    # loss_config = config["loss"]
    # boundary = loss_config["boundary"]
    
    # loss_fn = MSE(boundary)  # Assuming MSE is defined elsewhere
    
    # network_config = config["network"]
    # ########################
    # # 1. Create a new subfolder in aex_files
    # ########################
    # # Use a prefix like "my_experiment", then generate a timestamped folder name
    # subfolder_name = generate_filename("my_experiment", extension=None)
    # full_subfolder_path = os.path.join(network_config["output_dir"], subfolder_name)
    
    # # Create the directory
    # os.makedirs(full_subfolder_path, exist_ok=True)
    
    # print("New subfolder:", full_subfolder_path)
    
    # ########################
    # # 2. Extract a base name from sample_file
    # ########################
    # # e.g. if sample_file is "/home/filip/.../my_network_netlist"
    # #      then base_sample_name = "my_network_netlist"
    # base_sample_file = os.path.basename(network_config["sample_file"])
    # base_sample_name, _ = os.path.splitext(base_sample_file)
    
    # ########################
    # # 3. Generate .cir and .aex filenames using the base name
    # ########################
    # # (They will each have their own timestamp or can share one if desired)
    # cir_filename = generate_filename(base_sample_name, extension=".cir")
    # aex_filename = generate_filename(base_sample_name, extension=".aex")
    
    ########################
    # 4. Join those filenames with the subfolder path
    ########################
    network_config = network_details["network_files"]
    full_subfolder_path, new_sample_file, aex_file_path = create_filenames(network_config)
    

    #Extract dataset
    dataset_config = simulation_details["dataset"]
    n_of_epochs = dataset_config["n_of_epochs"]
    scale_factor = dataset_config["scale_factor"]
    noise = dataset_config["noise"]
    bias = dataset_config["bias"]
    num_samples = dataset_config["num_samples"]
    batch_size = dataset_config["batch_size"]
    
    beta = simulation_details["beta"]
    
    # Initialize weight initializer
    # init_config['params'] = {key: float(value) for key, value in init_config['params'].items()}
    # weight_initializer = Initializer(init_type=init_config["init_type"], params=init_config["params"])
    
    
    #Initialize the network
    
    # Initialize the network layers
    layers = initialize_network_layers(simulation_details, network_details)
    
    # Decide the loss function
    boundary = simulation_details["loss"]["boundary"]

    bias = 0
    scale_factor = 1
    bsize_arr = [1]
    boundary = 0.1
    for batch_size in bsize_arr:
        loss_fn = MSE(boundary)
        
        
        #Initialize the network
        net = MyNetwork(layers, loss_fn, boundary, aex_file_path, new_sample_file)
        #Build the netlist
        all_nodes = extract_all_nodes_voltages(layers)
        n_of_node_voltages = len(all_nodes)
        net.build_netlist(new_sample_file, all_nodes)
         
         
        num_samples = 800  # (if needed elsewhere; not used directly here)
        
        # Load and center the wine dataset
        X_t, Y_t = prepare_simple_dataset(num_samples)
        
        # Scale the features using StandardScaler
        #scaler = StandardScaler()
        #scaler = MinMaxScaler(feature_range=(-0.7, 0.7))
        #X_t_scaled = scaler.fit_transform(X_t)
        
        # Process the scaled wine data with onehot_pos_neg_inputs.
        # Note: scale_factor, bias, and output_scale should be defined previously.
        #X, Y = onehot_pos_neg_inputs(X_t_scaled, Y_t, scale_factor, bias, output_scale=1)
        
        # Optionally, you can visualize the data here (if needed)
        # plot_moons_data(X[:, :2], Y)  # Adjust function name if necessary
        
        # Split the processed data into training and test sets.
        #from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_t, Y_t, test_size=0.2, random_state=4)
        
        #vac+bias is the bias of the ac voltage source, vbias is the additional bias that is currently not used
        #bias_dict = {"VAC_BIAS" : v_ac_bias}
        
        #This does not work and it really should work
        pids = get_eldo_pids(eldo_identifier = 'eldo_64.exe')
        eldo_process = start_eldo_simulation(new_sample_file, full_subfolder_path, m_thread = True, noascii =  True, debug=True )
        signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, eldo_process))
        signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, eldo_process))
        
        n_of_epochs = 50
        epoch = 0
        #net.draw_grid(eldo_process, X_train, y_train, epoch, n_of_node_voltages, debug = False)
        for epoch in range(1, n_of_epochs +1):
            #NEED TO MANUALLY SET THEM, INITIALLY EVERYTHING IS 0
            #set_input_voltages(eldo_process, bias_dict, debug = True)
            print(f"Starting epoch {epoch} for boundary {boundary}, batch size {batch_size} and scale factor {scale_factor}")
            losses = net.free_nudged_train(eldo_process, layers, X_train, y_train, beta, epoch, batch_size, loss_fn, n_of_node_voltages, aex_file_path, optimizer = None, debug = False)
    
            #net.draw_grid(eldo_process, X_train, y_train, epoch, n_of_node_voltages, debug = False)
            #accuracy = net.free_test(eldo_process, layers, X_test, y_test, epoch, loss_fn, metrics = None, debug = False)
            
        delete_file_with_chi_extension(new_sample_file)


        
    
if __name__ == "__main__":
    main()