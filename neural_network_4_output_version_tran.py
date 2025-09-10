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
    initialize_network_layers,
)
from simulation_parameters_folder import SimulationParametersTran
from save_and_load_functions import (
save_all, save_sim_parameters, 
best_epoch_from_dir, load_sim_parameters, 
load_simulation_parameters_new)
from support_layer import *
from eldo_support_functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from loss_functions import * 
from sklearn.model_selection import train_test_split
from save_and_load_functions import (
save_all, save_sim_parameters, 
best_epoch_from_dir, load_sim_parameters, 
load_simulation_parameters_new)

from datetime import datetime
import logging
import os
import logging.handlers
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
        self.load_weights = sim_params.load_weights
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
        
            
    def _setup_tran(
       self,
       eldo_process,
       result_file,
       offset,
       voltage_dict,
       sim_type,
       q,
       f0,
       t0,
       debug
   ):
       # Perform TRAN-specific DC and bias setup
       results = read_update(
           eldo_process,
           result_file,
           voltage_dict,
           offset,
           sim_type,
           transcon_calc=True,
           debug=debug,
           f0=f0,
           t0=t0
       )
       # Extract DC operating point and set initial conditions
       dc_ds = results["dc_ds_end"]
       dc_gate = results["dc_gate_end"]
       voltage_dict = results["ac_voltages"]
       set_the_ic_voltages(eldo_process, dc_gate, dc_ds, debug)

       return voltage_dict

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

        prediction_list = []
        loss_list = []
        output_list = []
        output_list_nudge = []
        weight_matrices_1 = []
        weight_matrices_2 = []
        ratios_list1 = []
        ratios_list2 = []
        
        vol_extract_list = []
        voltage_dict_free = dict.fromkeys(self.all_nodes[0], None) #
        voltage_dict_nudge = dict.fromkeys(self.all_nodes[0], None)
        
        timings_list = []
        offset = 0


            #print(f"File size before clearing: {file_size} bytes")
        f0, t0 = 1e6, 20e-6
        #Write the initialized synapses
        if epoch == 1:
            #write the weights
            if self.load_weights:
                h5_path =  "/home/filip/simulations/testing_plots/fet_FSST_0802/FSST_114247_262613/plots/metrics_data.h5"
                best_idx, best_accuracy, wm_evo1_at_best, wm_evo2_at_best = best_epoch_from_dir(h5_path)
                W1 = wm_evo1_at_best
                W2 = wm_evo2_at_best
                
                resistive_layers[0].W = W1
                resistive_layers[1].W = W2
                
                update_synapses(eldo_process, resistive_layers, diode_connected_flash_params, simulation_type, debug)
                run_simulation_and_wait(eldo_process, simulation_type, q, debug)
                voltage_dict_free = self._setup_tran(eldo_process, result_file, offset, voltage_dict_free, simulation_type, q, f0, t0, debug)
            else:
                initialize_synapses(eldo_process, resistive_layers, diode_connected_flash_params, simulation_type, debug)
                update_synapses(eldo_process, resistive_layers, diode_connected_flash_params, simulation_type, debug)
                run_simulation_and_wait(eldo_process, simulation_type, q, debug)
                voltage_dict_free = self._setup_tran(eldo_process, result_file, offset, voltage_dict_free, simulation_type, q, f0, t0, debug)
            #command_pmos_cs = f"SET P(PMOS_CS_VDD_NEG)=0"


        
        for i in range(num_batches):

            batch_start_time = time.time()
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))  # Ensure not to exceed the dataset length

            # Select the batch
            X_batch = X_train[start_idx:end_idx]
            Y_batch = Y_train[start_idx:end_idx]
            
            if i > 1:
                with open(result_file, "r+") as f:
                    f.truncate(0)
            
            
            for layer in resistive_layers:
                layer.zero_grad()
            
            
            for X, Y in zip(X_batch, Y_batch):
                sample_start_time = time.time()
                for j, key in enumerate(input_keys_list):
                    input_dict[key] = - X[j]  # no inverting the inputs for the TRAN
                    
                beta_r = beta * np.random.uniform(-5,5)
                disable_current_sources(eldo_process, inudge_dict, debug)
                
                set_input_voltages(eldo_process, input_dict, debug)


                ##### 
                try:
                    offset = os.path.getsize(result_file)
                except FileNotFoundError:
                    print("need to run the first simulation")
                    offset = 0

                run_simulation_and_wait(eldo_process, simulation_type, q, debug)  

                #for plotting there's a function read_update_and_plot
                
                voltage_dict_free = self._setup_tran(eldo_process, result_file, offset, voltage_dict_free, simulation_type, q, f0, t0, debug)



                
                
        #this stays the same
                for layer in resistive_layers:
                    layer.update__free_voltages(voltage_dict_free)


                #print(f"Voltage extraction {volt_extract_end}")
                outputs = layer.output_free_voltages #the outputs are just the outputs of the last layer
                output_values = np.array(list(outputs.values()))
                output_list.append(output_values)
                
                mode = self.nudging_mode
                if targets != None:
                    Y_index = np.argmax(Y)
                    target = targets[Y_index]
                else:
                    target = (Y/4) * self.scale_factor

                
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
                offset = os.path.getsize(result_file)
                run_simulation_and_wait(eldo_process, simulation_type, q, debug)
                voltage_dict_nudge = self._setup_tran(eldo_process, result_file, offset, voltage_dict_nudge, simulation_type, q, f0, t0, debug)
                
                for layer in resistive_layers:
                    layer.update__nudge_voltages(voltage_dict_nudge)
                
                
                outputs_n = layer.output_nudge_voltages
                outputs_n_values = list(outputs_n.values())#the outputs are just the outputs of the last layer
                output_list_nudge.append(outputs_n_values)   
                sample_losses_n, voltages_n = loss_fn(outputs_n, target, beta_r, mode)
                
                
                for layer in resistive_layers:
                    layer.run_update_process(batch_size, beta_r) #this accumulates the gradients
                    
                    
                loss_list.append(sample_losses)
                

            update_synapses(eldo_process, resistive_layers, diode_connected_flash_params, simulation_type, debug)
            batch_end_time = batch_start_time - time.time()        
                
            weight_matrices_1.append(layers[1].W)
            weight_matrices_2.append(layers[3].W)
                
            

            
        
        ratio1_mean = np.array(np.mean(ratios_list1))
        ratio2_mean = np.array(np.mean(ratios_list2))
        ratio_list = [ratio1_mean, ratio2_mean]

        #loss_fn(output_list, mode='test')
        predictions = loss_fn(output_list, mode='test')
        epoch_acc = np.mean(loss_fn.verify_result(Y_train, np.array(predictions)))
        #print(epoch_acc)
        # Convert one-hot encoded Y_train to class indices

        # Compute accuracy by comparing predictions with true_labels
        #print(f"Average voltage extract time {mean_time_extract}")
        #print(f"Average simulation time {mean_time_simulation}")
        output_nodes = [0, 1, 2, 3]
        #plot_free_and_nudged(output_list, output_list_nudge, output_nodes, beta, epoch)

        

        diff1, diff2 =  output_plot(output_list)
        #disable_current_sources(eldo_process, inudge_dict, debug)
        mean_loss = np.mean(np.array(loss_list), axis = 1)
        #print(mean_loss)
        return {
        "accuracy" : epoch_acc,
        "loss_list": mean_loss,
        "weight_matrix_1": weight_matrices_1[-1],
        "weight_matrix_2": weight_matrices_2[-1],
        "epoch_accuracy": epoch_acc,
        "output_list" : [diff1, diff2]}

            

def train(sim_params, process_id = None, logger = None): #probably objective function

    #Set-up logging
    mode = "train"
    
    
    if logger is None:
        logger = logging.getLogger(f"worker-{os.getpid()}")
        logger.info(f"Started train for {sim_params.gamma_values}, pid={process_id}")

    
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


    # Setup output directories
    base_dir = sim_params.trained_models_dir
    ts = datetime.now().strftime("%H%M%S")
    folder = (
        f"{simulation_type}_{ts}_{process_id}"
        if process_id else
        f"{simulation_type}_{ts}"
    )
    
    out_dir = os.path.join(base_dir, folder)
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    data_path = os.path.join(plot_dir, 'metrics_data.h5')
    config_file_path = os.path.join(out_dir, "config") 

    
    

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
    

    
    
    
    
    try: 
        for epoch in range(1, sim_params.n_of_epochs +1):
            logger.info(f"??  In train(): received logger {logger!r}")
            assert logger is not None, "train() must be passed a logger"
            #calculate_transconductance(eldo_process, fet_identifiers, new_sample_file, net.aex_file_path, net.all_nodes, debug)
            start_epoch = time.time()
            if dynamic_outputs:
                targets = net.calc_desired_outputs(eldo_process, X, Y, epoch, debug)

            optimizer = None
            #if epoch in draw_grid_to_plot:
            #   net.draw_grid(eldo_process, X_train, y_train, input_function, epoch, plot_directory, debug = False)
            results = net.free_nudged_train(eldo_process, X_train, y_train, targets, epoch, optimizer, debug)
            accuracy = results["accuracy"]
            loss = results["loss_list"]
            try:
                logger.info(f"Epoch {epoch}: accuracy={accuracy:.4f}, loss={loss[-1]}")  # full durable record
            except:
                pass
            
                
            accuracy_list.append(accuracy)
            big_loss_list.append(loss)
            weight_matrix1 = results["weight_matrix_1"]
            weight_matrix2 = results["weight_matrix_2"]
                
            diff1, diff2 = results["output_list"]
            diff1_list.extend(diff1)
            diff2_list.extend(diff2)
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
    
    
        save_all(
            data_path,
            results,
            weight_matrices_1,
            weight_matrices_2,
            accuracy_list,
            big_loss_list
        )
    
    
    
        config_file_path = os.path.join(out_dir, "config") 
        save_sim_parameters(sim_params, config_file_path)
    
    
    
        plot_accuracy(accuracy_list, plot_dir)

        window_size = 40
        plot_moving_averages(diff1_list, diff2_list, window_size, plot_dir)
        plot_average_loss(big_loss_list, plot_dir)
        #plot_average_relative_change(abs_change1, title="Average Absolute Weight Change per Epoch weightmatrix 1")
        #plot_average_relative_change(abs_change2, title="Average Absolute Weight Change per Epoch weightmatrix 2")
    
        plot_weight_matrix_evolution_lines(weight_matrices_1, plot_dir, title='Weight Matrix 1 Evolution Over Epochs')
        plot_weight_matrix_evolution_lines(weight_matrices_2, plot_dir, title='Weight Matrix 2 Evolution Over Epochs')
        #delete_file_with_chi_extension(new_sample_file)
        
        
        
        
        
        
        send_quit_command(eldo_process)
        remove_directory(full_subfolder_path)
        return accuracy_list
        
    except (Exception, KeyboardInterrupt) as e:
    # 1) Print the exception type, message, and full traceback
        logger.exception(f"Worker {process_id} failed during {mode} for {sim_params}")     
        save_all(
            data_path,
            results,
            weight_matrices_1,
            weight_matrices_2,
            accuracy_list,
            big_loss_list
        )
        send_quit_command(eldo_process)
        remove_directory(full_subfolder_path)
        raise
        
    
if __name__ == "__main__":
    ###need to change the biasing and the simulation type
    gamma_values =  [3e-8, 25e-9]
    batch_size = 2
    beta = 5e-5
    # #scale_factor_list = [0.4, 0.5] 
    # #bias_list = [0.2, 0.3, 0.4, 0.5, 0.6]
    # #good scale factor is 0.4
    # scale_factor_list = [0.4] 
    scale_factor = 0.4
    # bias_list = [0.2]    
    bias =  0.3
    sim_params = SimulationParametersTran(scale_factor, bias, batch_size, beta, gamma_values)
    train(sim_params, process_id = 5)