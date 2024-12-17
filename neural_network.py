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
        self.amp_parameters_file = "/home/filip/simulations/improved_simulation_functions/amp_parameters.inc"
        
    #just builds the netlist
    def build_netlist(self, file_name, all_nodes, freq = "100Meg"):
        

        
        parameter_lines = self.extract_parameters()
        with open(self.amp_parameters_file, "r") as file:
            amp_parameter_lines = file.readlines()
            
            
        parameter_lines.extend(amp_parameter_lines)
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
                 
                 
        
        mid_sect = ".LIB /cao/DK/ST/HCMOS9A_10.9/Addon_NVM_H9A@2018.4.1/tools/eldo/model_oxram/OxRRAM.lib OxRRAM_TT\n" \
                   ".LIB /home/filip/CMOS130/corners.eldo\n" \
                   "*** Library name: amplifiers\n" \
                   "*** Cell name: vcvs_ss\n" \
                   "*** View name: schematic\n" \
                   ".SUBCKT VCVS_SS INPUT_VCVS OUTPUT_VCVS\n" \
                   "    R5 VDD OUTPUT_CS_2 RD\n" \
                   "    R3 VDD OUTPUT_CS_1 RD\n" \
                   "    R21 VDD NET1 RBIAS1\n" \
                   "    R19 NET7 0 RS\n" \
                   "    R8 OUTPUT_VCVS 0 COMMON_DRAIN_RES\n" \
                   "    R4 NET8 0 RS\n" \
                   "    R1 NET1 0 RBIAS2\n" \
                   "XM13 VDD OUTPUT_CS_2 OUTPUT_VCVS 0 EN5V0_BS3JU w=CD1_W l=CD1_L nfing=1\n" \
                   "+ncrsd=1 number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0\n" \
                   "+dmu_mdev=0 soa=1 lpe=0\n" \
                   "XM12 OUTPUT_CS_2 OUTPUT_CS_1 NET7 0 EN5V0_BS3JU w=2e-06 l=CS2_L nfing=1\n" \
                   "+ncrsd=1 number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0\n" \
                   "+dmu_mdev=0 soa=1 lpe=0\n" \
                   "XM11 OUTPUT_CS_1 NET1 NET8 0 EN5V0_BS3JU w=CS1_W l=2e-06 nfing=1 ncrsd=1\n" \
                   "+number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0 dmu_mdev=0\n" \
                   "+soa=1 lpe=0\n" \
                   "    C4 INPUT_VCVS NET1 CAP\n" \
                   "    V0 VDD 0 DC VDD\n" \
                   ".ENDS\n" \
                   "*** End of subcircuit definition.\n\n" \
                   "*** Library name: amplifiers\n" \
                   "*** Cell name: cccs_ss\n" \
                   "*** View name: schematic\n" \
                   ".SUBCKT CCCS_SS I_OUT V_IN V_OUT\n" \
                   "XM1 NET11 NET14 NET09 0 EN5V0_BS3JU w=WIDTH_NMOS_DIFF_A\n" \
                   "+l=LENGTH_NMOS_DIFF_A nfing=1 ncrsd=1 number=1 srcefirst=1 ngcon=1\n" \
                   "+mismatch=1 po2act=-1 dvt_mdev=0 dmu_mdev=0 soa=1 lpe=0\n" \
                   "XM0 NET9 NET7 NET11 0 EN5V0_BS3JU w=WIDTH_NMOS_DIFF_A l=LENGTH_NMOS_DIFF_A\n" \
                   "+nfing=1 ncrsd=1 number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1\n" \
                   "+dvt_mdev=0 dmu_mdev=0 soa=1 lpe=0\n" \
                   "XM9 I_OUT NET28 NET27 0 EN5V0_BS3JU w=1.2e-06 l=LENGH_CASC_2 nfing=1\n" \
                   "+ncrsd=1 number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0\n" \
                   "+dmu_mdev=0 soa=1 lpe=0\n" \
                   "XM8 NET27 NET16 0 0 EN5V0_BS3JU w=0.8e-6 l=LENGH_CASC_1 nfing=1 ncrsd=1\n" \
                   "+number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0 dmu_mdev=0\n" \
                   "+soa=1 lpe=0\n" \
                   "XM10 NET16 NET09 0 0 EN5V0_BS3JU w=8e-07 l=0.5e-6 nfing=1 ncrsd=1 number=1\n" \
                   "+srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0 dmu_mdev=0 soa=1\n" \
                   "+lpe=0\n" \
                   "    R0 V_IN V_OUT R_SHUNT\n" \
                   "    R9 VDD NET7 RBIAS1\n" \
                   "    R11 VDD NET14 RBIAS1\n" \
                   "    R12 NET14 0 RBIAS2\n" \
                   "    R10 NET7 0 RBIAS2\n" \
                   "    R23 VDD NET16 R_D_DIFF_AMP\n" \
                   "    I5 VDD I_OUT DC IBIAS_CASCODE\n" \
                   "    I2 NET11 0 DC IBIAS_DIFF_A\n" \
                   "    V0 VDD 0 DC VDD\n" \
                   "    V3 NET28 0 DC V_CASCODE\n" \
                   "    C0 V_IN NET7 CAP\n" \
                   "    C1 NET14 V_OUT CAP\n" \
                   "XM2 VDD NET9 NET9 VDD EP5V0_BS3JU w=1.6e-06 l=0.5e-6 nfing=1 ncrsd=1\n" \
                   "+number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0 dmu_mdev=0\n" \
                   "+soa=1 lpe=0\n" \
                   "XM3 NET09 NET9 VDD VDD EP5V0_BS3JU w=1.6e-06 l=0.5e-6 nfing=1 ncrsd=1\n" \
                   "+number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0 dmu_mdev=0\n" \
                   "+soa=1 lpe=0\n" \
                   ".ENDS\n" \
                   "*** End of subcircuit definition.\n\n" \
                   "*** Library name: amplifiers\n" \
                   "*** Cell name: amplification_ss\n" \
                   "*** View name: schematic\n" \
                   ".SUBCKT AMPLIFICATION_SS AMP_INPUT AMP_OUTPUT\n" \
                   "    XI0 AMP_INPUT NET6 VCVS_SS\n" \
                   "    XI1 AMP_INPUT NET6 AMP_OUTPUT CCCS_SS\n" \
                   ".ENDS\n" \
                   "*** End of subcircuit definition.\n"

        
        
        vm_list = []
        counter = 0 
        
        for node in all_nodes:
            if counter == 0:
                vm_list.append(".EXTRACT")
            vm_node = f"vdb({node})"
            vm_list.append(vm_node)
            counter += 1
            if counter == 5:
                # Append the current line and reset for a new one
                vm_list.append("\n")
                counter = 0

        vi_list = []
        counter = 0 
        
        for node in all_nodes:
            if counter == 0:
                vi_list.append(".EXTRACT")
            vi_node = f"vp({node})"
            vi_list.append(vi_node)
            counter += 1
            if counter == 5:
                # Append the current line and reset for a new one
                vi_list.append("\n")
                counter = 0



        
        # Join the vm_list into a string, removing unnecessary spaces and ensuring formatting
        vm_string = " ".join(filter(None, vm_list)).replace(" \n.", "\n.")
        vi_string = " ".join(filter(None, vi_list)).replace(" \n.", "\n.")
        
        
        
        simulation_details = (
            f".AC LIST {freq}\n"
            f"{vm_string}\n"
            f"{vi_string}\n"
            ".OPTION AEX\n"
            ".OPTION NOASCII\n"
            ".DC\n"
            ".PRINTFILE DC V(*) file=\"/home/filip/simulations/sample_files/eldo_samples/python_generated_netlists/output.txt\"\n"
            ".PRINTFILE AC VI(*) file=\"/home/filip/simulations/sample_files/eldo_samples/python_generated_netlists/output_ac.txt\"\n"
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
        input_dict = input_layer.inputs
        input_keys_list = list(input_dict.keys())
        
        inudge_dict = output_layer.parameters
        inudge_keys_list = list(inudge_dict.keys())
        
        resistive_layers = [layer for layer in layers if getattr(layer, 'type', None) == 'resistive']
        
        num_batches = int(np.ceil(len(X_train) / batch_size))
         
    
    
        loss_list = []
        output_list = []
        
        start_index = 4
        n_of_node_voltages = n_of_node_voltages
        aex_result_file = "/home/filip/simulations/sample_files/eldo_samples/output_files/my_network_netlist2.aex"
        
        for i in range(num_batches):
            if i > 0:
                start_index = 3
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
                voltage_dict_free = parse_aex_file(aex_result_file, start_index, end_index)
                simulation_end_time = time.time() - simulation_start_time
                print(f"Simulation duration {simulation_end_time}")
                volt_extract = time.time()
                
                for layer in resistive_layers:
                    layer.update__free_voltages(voltage_dict_free)
                     
                volt_extract_end = time.time() - volt_extract
                print(f"Voltage extraction {volt_extract_end}")
                outputs = layer.output_free_voltages #the outputs are just the outputs of the last layer
                output_values = list(outputs.values())
                output_list.append(output_values)
                sample_losses, currents = loss_fn(outputs, target = self.boundary, beta = beta, mode='train') #need to decide where to initialize this function - remember that loss_fn comes from the already initialized MSE
                #now I can simply zip the currents to the parameters of the last layer and repeat
                flat_currents = currents.flatten()
                
                for i, key in enumerate(inudge_keys_list):
                    inudge_dict[key] = flat_currents[i]    
                    
                    
                    
                #output_layer.parameters = output_layer.update_parameters(flat_currents)
                
                set_currents_nudge_mode(eldo_process, inudge_dict, debug = True)
                run_eldo_simulation(eldo_process, debug)
                lines_of_interest = wait_for_eldos_completion(eldo_process, debug)
                            
                
                start_index += n_of_node_voltages + 3
                end_index = start_index + n_of_node_voltages
                
                voltage_dict_nudge = parse_aex_file(aex_result_file, start_index, end_index)


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
            clear_aex_file(aex_result_file)
            print("Successfully deleted aex file")
        
        output_plot(output_list)

        
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
    freq = "10Meg"
    vdc_bias = 1.3
    
    
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
    layer4 = OutputLayer(n_of_inputs, freq, which_layer)
    layers.append(layer4)
    
    return layers



def main(): #probably objective function



    # Load the configuration file
    with open("config.json", "r") as file:
        config = json.load(file)
    
    # Extract configurations
    init_config = config["initializer"]
    fc_layers = config["layers"]["fully_connected"]
    lower_cond_bound = config["layers"]["lower_cond_bound"]
    upper_cond_bound = config["layers"]["upper_cond_bound"]
    
    diode_dict = config["diodes"]
    ampv = [config["amplification"]["ampv"]]
    ampc = [config["amplification"]["ampc"]]
    diode_dict_list = [f"{key} = {value}" for key, value in diode_dict.items()]
    v_ac_bias = config["AC_biases"]["source_dc_bias"]
    
    
    
    
    
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
    output_dir = "/home/filip/simulations/sample_files/eldo_samples/output_files"
    all_nodes = extract_all_nodes_voltages(layers)
    n_of_node_voltages = len(all_nodes)
    net.build_netlist(sample_file, all_nodes)
     
     
    #Start a simulation
    X_t, Y_t = prepare_moons_data(num_samples, noise = 0.10, random_state=4)
    X, Y = generate_pos_neg_inputs(X_t, Y_t, scale_factor = 0.4, output_scale = 0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
    
    #vac+bias is the bias of the ac voltage source, vbias is the additional bias that is currently not used
    vbias = 0
    bias_dict = {"VAC_BIAS" : v_ac_bias, "VBIAS" : vbias}
    
    #This does not work and it really should work
    pids = get_eldo_pids(eldo_identifier = 'eldo_64.exe')
    eldo_process = start_eldo_simulation(sample_file, output_dir, m_thread = True, noascii =  True, debug=False)
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, eldo_process))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, eldo_process))
    
    for epoch in range(1, n_of_epochs +1):
        #Set biases
        set_input_voltages(eldo_process, bias_dict, debug = True)
        losses = net.free_nudged_train(eldo_process, layers, X_train, y_train, beta, epoch, batch_size, loss_fn, n_of_node_voltages, optimizer = None, metrics = None)
        accuracy = net.free_test(eldo_process, layers, X_test, y_test, epoch, loss_fn, metrics = None, debug = False)
        
     
     
     
     
     
     
     
     


    
if __name__ == "__main__":
    main()