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
#Initialize a complete neural network and build a netlist
class MyNetwork:
    def __init__(self, fc_layers, layers, ampv, ampc, diode_list, loss_fn, boundary):
        self.fc_layers = fc_layers
        self.layers = layers
        self.loss_function = loss_fn
        self.ampv = ampv
        self.ampc = ampc
        self.diode_list = diode_list
        self.loss_fn = loss_fn
        self.boundary = boundary
        
        
    #just builds the netlist
    def build_netlist(self, file_name):
        
        
        parameter_lines = self.extract_parameters()
        network_description = self.extract_connections()
        
        # Get current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate header with current date and time
        header = f"***\n" \
                 f"*** Generated for: eldoD\n" \
                 f"*** Generated on: {current_datetime}\n" \
                 f"*** Design library name: tests\n" \
                 f"*** Design cell name: idk\n" \
                 f"*** Design view name: schematic\n" \
                 f".GLOBAL\n"
        
        #not really sure how/if I can avoid doing this
        mid_sect = ".LIB /cao/DK/ST/HCMOS9A_10.9/Addon_NVM_H9A@2018.4.1/tools/eldo/model_oxram/OxRRAM.lib OxRRAM_TT\n" \
                   ".LIB /home/filip/CMOS130/corners.eldo\n" \
                   ".LIB /home/filip/Documents/MyDiode.lib\n\n" \
                   "*** Library name: tests\n" \
                   "*** Cell name: neuron\n" \
                   "*** View name: schematic\n" \
                   ".SUBCKT NEURON VIN VOUT\n" \
                   "    D0 VIN NET6 diode1\n" \
                   "    D1 NET7 VIN diode1\n" \
                   "    V2 NET6 0 DC VDIODE1\n" \
                   "    V3 NET7 0 DC VDIODE2\n" \
                   "    F0 0 VIN EVCVS1 {AMPC}\n" \
                   "    EVCVS1 VOUT 0 VIN 0 AMP\n" \
                   ".ENDS\n" \
                   "*** End of subcircuit definition.\n\n" \
                   "*** Library name: tests\n" \
                   "*** Cell name: kendal_non_linear_moons_easy\n" \
                   "*** View name: schematic\n"

        # simulation_details = ".OP\n" \
        #                      ".PROBE V" \
        #                      ".OPTION NOASCII" \
        #                      ".OPTION PROBEOP2\n" \
        #                      ".DC\n" \
        #                      ".END\n"


    
        simulation_details = (
            ".DC\n"
            ".EXTRACT DC V(*)\n"
            ".OPTION AEX\n" # Removed the trailing backslash here
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
    def build_network(self, diode_dict, synapse):
        
        layers = []
        input_layer = InputLayer(self.fc_layers[0], self.fc_layers[0], which_layer = 0)
        layers.append(input_layer)
        synapse = "resistor"
        diode_dict = {"VDIODE1" : None, "VDIODE2" : None} #this are strictly parameters for the hidden layer
        for i in range(1, len(self.fc_layers)):
            n_of_inputs = self.fc_layers[i-1]
            n_of_outputs = self.fc_layers[i]
            which_layer = i-1
            layer1 = DenseLayer(n_of_inputs, n_of_outputs, which_layer, synapse)
            layers.append(layer1)
            if i == len(self.fc_layers)-1:
                break
            else:
                layer2 = NonLinearLayer(n_of_outputs, n_of_outputs, which_layer, diode_dict)
                layers.append(layer2)
            #layers.append(layer1)
            
        layers.append(OutputLayer(self.fc_layers[-1], self.fc_layers[-1], which_layer = 1))
        
        return layers
    
    
    def extract_parameters(self):
        ampv = self.ampv
        ampc = self.ampc
        form = ["FORM=0"]
        all_parameters = []
        
        # Add parameters from each layer with `.PARAM` prefix
        for layer in self.layers:
            for key, value in layer.parameters.items():
                line = f".PARAM {key}={value}"
                all_parameters.append(line)
        
        # Add other parameters with `.PARAM` prefix
        all_parameters.extend([f".PARAM {item}" for item in ampv])
        all_parameters.extend([f".PARAM {item}" for item in ampc])
        all_parameters.extend([f".PARAM {item}\n" for item in form])
    
        return all_parameters
    
    def extract_connections(self):
        extracted_connections = []
        for layer in self.layers:
            lines = layer.connections  # Assuming 'connections' is already a list
            extracted_connections.extend(lines)  # Use 'extend' instead of 'append' to add all elements of 'lines'
        return extracted_connections

    
    def initialize_network(self):
        for layer in layers:
            layer.ini
  

    def free_nudged_train(self, eldo_process, layers, X_train, Y_train, beta, epochs, batch_size, loss_fn, n_of_node_voltages, optimizer = None, metrics = None, debug = False):
        
        # def signal_handler(sig, frame):
        #     if True:
        #         print("Interrupt received, sending quit command to subprocess.")
        #         send_command_to_eldo(eldo_process, "QUIT", debug)
        #     else:
        #         print("Interrupt received, exiting without sending command.")
        #     sys.exit(0)  # Exit the program
        
     
        input_layer = layers[0]
        output_layer = layers[-1]
        input_dict = input_layer.parameters
        input_keys_list = list(input_dict.keys())
        
        inudge_dict = output_layer.parameters
        inudge_keys_list = list(inudge_dict.keys())
        
        resistive_layers = [layer for layer in layers if getattr(layer, 'type', None) == 'resistive']
        
        num_batches = int(np.ceil(len(X_train) / batch_size))
    
    
        loss_list = []
        
        start_index = 4
        n_of_node_voltages = n_of_node_voltages
        
        
        for i in range(num_batches):
            batch_start_time = time.time()
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))  # Ensure not to exceed the dataset length

            # Select the batch
            X_batch = X_train[start_idx:end_idx]
            Y_batch = Y_train[start_idx:end_idx]
            
            for X, Y in zip(X_batch, Y_batch):
                sample_start_time = time.time()
                for i, key in enumerate(input_keys_list):
                    input_dict[key] = X[i]  # Directly assign the value from X to the corresponding key
                    
                    
                set_input_voltages(eldo_process, input_dict, debug = True)
                #move at the end
                disable_current_sources(eldo_process, inudge_dict, debug)
                simulation_start_time = time.time()
                run_eldo_simulation(eldo_process, debug)
                
                
                end_index = start_index + n_of_node_voltages
                
                
                
                lines_of_interest = wait_for_eldos_completion(eldo_process, debug = False)
                voltage_dict_free = parse_aex_file("/home/filip/simulations/simulations/my_network_netlist2.aex", start_index, end_index)
                simulation_end_time = time.time() - simulation_start_time
                print(f"Simulation duration {simulation_end_time}")
                volt_extract = time.time()
                
                for layer in resistive_layers:
                    layer.update__free_voltages(voltage_dict_free)
                     
                volt_extract_end = time.time() - volt_extract
                print(f"Voltage extraction {volt_extract_end}")
                outputs = layer.output_free_voltages #the outputs are just the outputs of the last layer
                sample_losses, currents = loss_fn(outputs, target = self.boundary, beta = beta, mode='train') #need to decide where to initialize this function - remember that loss_fn comes from the already initialized MSE
                #now I can simply zip the currents to the parameters of the last layer and repeat
                flat_currents = currents.flatten()
                
                for i, key in enumerate(inudge_keys_list):
                    input_dict[key] = flat_currents[i]    
                    
                    
                    
                #output_layer.parameters = output_layer.update_parameters(flat_currents)
                
                set_currents_nudge_mode(eldo_process, inudge_dict, debug)
                run_eldo_simulation(eldo_process, debug)
                lines_of_interest = wait_for_eldos_completion(eldo_process, debug)
                            
                
                start_index += n_of_node_voltages + 3
                end_index = start_index + n_of_node_voltages
                
                voltage_dict_nudge = parse_aex_file("/home/filip/simulations/simulations/my_network_netlist2.aex", start_index, end_index)


                start_index += n_of_node_voltages + 3

                python_update_time = time.time
                for layer in resistive_layers:
                    layer.update__nudge_voltages(voltage_dict_nudge)
                
                python_update_time = time.time()
                
                for layer in resistive_layers:
                    layer.run_update_process()
                
                python_update_time_end = time.time() - python_update_time
                print(f"Python update time {python_update_time_end}")
                    
                loss_list.append(sample_losses)
                
                sample_duration = time.time() - sample_start_time
                print(f"Sample update time {sample_duration}")
                
                
            res_start_time = time.time()   
            #At the end of the batch update all resistances
            for layer in resistive_layers:
                layer.update_res_dict()
                set_resistances(eldo_process, layer.resistor_dict, debug)
                
            res_duration = time.time() - res_start_time                
            batch_duration = time.time() - batch_start_time
            print(f"Batch duration {batch_duration}")
            
        return loss_list
            
            
    def free_test(self, eldo_process, layers, X_test, Y_test, epoch, loss_fn, metrics = None, debug = False):
        
        # def signal_handler(sig, frame):
        #     if True:
        #         print("Interrupt received, sending quit command to subprocess.")
        #         send_command_to_eldo(eldo_process, "QUIT", debug)
        #     else:
        #         print("Interrupt received, exiting without sending command.")
        #     sys.exit(0)  # Exit the program
        
     
        input_layer = layers[0]
        output_layer = layers[-1]
        input_dict = input_layer.parameters
        input_keys_list = list(input_dict.keys())

        
        resistive_layers = [layer for layer in layers if getattr(layer, 'type', None) == 'resistive']
        
        acc_list = []
        
        disable_current_sources(eldo_process, inudge_dict, debug)
        
        for X, Y in zip(X_test, Y_test):
            sample_start_time = time.time()
            for i, key in enumerate(input_keys_list):
                input_dict[key] = X[i]  # Directly assign the value from X to the corresponding key
            set_input_voltages(eldo_process, input_dict, debug)
                #move at the end
            run_eldo_simulation(eldo_process, debug)
            lines_of_interest = wait_for_eldos_completion(eldo_process, debug)
            simulation_end_time = time.time() - simulation_start_time
            print(f"Simulation duration {simulation_end_time}")
            volt_extract = time.time()
            for layer in resistive_layers:
                layer.input_free_voltages = extract_voltage_from_list(lines_of_interest, layer.input_free_voltages)
                layer.output_free_voltages = extract_voltage_from_list(lines_of_interest, layer.output_free_voltages)
                     
            volt_extract_end = time.time() - volt_extract
            print(f"Voltage extraction {volt_extract_end}")
            outputs = layer.output_free_voltages #the outputs are just the outputs of the last layer
            prediction = loss_fn(outputs, target = self.boundary, beta = None, mode='test') #need to decide where to initialize this function - remember that loss_fn comes from the already initialized MSE
            c = loss_fn.verify_result(Y_test, prediction)
            acc_list.append(c)
            
        accuracy = np.mean(acc_list) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        
        return accuracy
    

def initialize_network_layers(fc_layers, weight_initializer, diode_dict, synapse="resistor", 
                              lr_layer1=1, gamma_layer1=0.03, lr_layer2=3, gamma_layer2=0.01, 
                              beta=0.01, lower_cond_bound=1e-6, upper_cond_bound=10):
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
    layers = []
    
    # Input layer
    input_layer = InputLayer(fc_layers[0], which_layer=0)
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
    layer2 = NonLinearLayer(n_of_outputs, which_layer, diode_dict)
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
    layer4 = OutputLayer(n_of_inputs, which_layer)
    layers.append(layer4)
    
    return layers



def main(): #probably objective function



    # Load the configuration file
    with open("config.json", "r") as file:
        config = json.load(file)
    
    # Extract configurations
    init_config = config["initializer"]
    fc_layers = config["layers"]["fully_connected"]
    fc_layers = [5, 128, 10]  # Overriding the value
    lower_cond_bound = config["layers"]["lower_cond_bound"]
    upper_cond_bound = config["layers"]["upper_cond_bound"]
    
    diode_dict = config["diodes"]
    ampv = [config["amplification"]["ampv"]]
    ampc = [config["amplification"]["ampc"]]
    diode_dict_list = [f"{key} = {value}" for key, value in diode_dict.items()]
    
    lr_layer1 = config["learning_rate_factors"]["lr_layer1"]
    lr_layer2 = config["learning_rate_factors"]["lr_layer2"]
    
    gamma_layer1 = config["gamma_values"]["layer1"]
    gamma_layer2 = config["gamma_values"]["layer2"]
    
    beta = config["beta"]
    
    loss_config = config["loss"]
    boundary = loss_config["boundary"]
    loss_fn = MSE(boundary)  # Assuming MSE is defined elsewhere
    
    network_config = config["network"]
    sample_file = network_config["sample_file"]
    output_dir = network_config["output_dir"]
    
    dataset_config = config["dataset"]
    n_of_epochs = dataset_config["n_of_epochs"]
    scale_factor = dataset_config["scale_factor"]
    noise = dataset_config["noise"]
    bias = dataset_config["bias"]
    num_samples = dataset_config["num_samples"]
    batch_size = dataset_config["batch_size"]
    
    # Initialize weight initializer
    init_config['params'] = {key: float(value) for key, value in init_config['params'].items()}
    weight_initializer = Initializer(init_type=init_config["init_type"], params=init_config["params"])
    
    

    n_of_node_voltages = fc_layers[0] + 2*fc_layers[1] + fc_layers[2]
    #Initialize the network
    
    # Initialize the network layers
    layers = initialize_network_layers(
        fc_layers=fc_layers, 
        weight_initializer=weight_initializer, 
        diode_dict=diode_dict, 
        synapse="resistor", 
        lr_layer1=lr_layer1, 
        gamma_layer1=gamma_layer1, 
        lr_layer2=lr_layer2, 
        gamma_layer2=gamma_layer2, 
        beta=beta, 
        lower_cond_bound=lower_cond_bound, 
        upper_cond_bound=upper_cond_bound
    )
    
    # Decide the loss function
    loss_fn = MSE(boundary)
    
    
    #Initialize the network
    net = MyNetwork(fc_layers, layers, ampv, ampc, diode_dict_list, loss_fn, boundary)
    #Build the netlist
    sample_file = "/home/filip/simulations/sample_files/eldo_samples/python_generated_netlists/my_network_netlist2.cir"
    output_dir = "/home/filip/simulations/simulations"
    net.build_netlist(sample_file)
     
     
    #Start a simulation
    X_t, Y_t = prepare_moons_data(num_samples, noise=0.10, random_state=4)
    X, Y = generate_1_bias_pos_neg_inputs(X_t, Y_t, scale_factor, bias, output_scale = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
    
    
    
    
    #This does not work and it really should work
    pids = get_eldo_pids(eldo_identifier = 'eldo_64.exe')
    eldo_process = start_eldo_simulation(sample_file, output_dir, m_thread = True, noascii =  True, debug=False)
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, eldo_process))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, eldo_process))
    
    for epoch in range(1, n_of_epochs +1):
        losses = net.free_nudged_train(eldo_process, layers, X_train, y_train, beta, epoch, batch_size, loss_fn, n_of_node_voltages, optimizer = None, metrics = None)
        accuracy = net.free_test(eldo_process, layers, X_test, y_test, epoch, loss_fn, metrics = None, debug = False)
        
     
     
     
     
     
     
     
     


    
if __name__ == "__main__":
    main()