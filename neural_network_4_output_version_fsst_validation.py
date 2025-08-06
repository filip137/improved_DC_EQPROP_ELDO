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

from simulation_parameters_folder import SimulationParametersFSST

from support_layer import *
from eldo_support_functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from loss_functions import * 
from sklearn.model_selection import train_test_split

from save_and_load_functions import (
save_all, save_sim_parameters, 
best_epoch_from_dir, load_sim_parameters)

from datetime import datetime
import logging
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
        
            

    def free_test(self, eldo_process, X_grid, Y_in, input_function, mode, h5_path, debug):
        
        
        if mode == "draw_grid":
            X_in, Y = onehot_pos_neg_inputs_1bias_double_input(X_grid, Y_in, self.bias, output_scale=1)
        
        elif mode == "validate":
            X_in, Y = X_grid, Y_in
        
        ###Here I first need to write weights
        
        debug = True
        
        input_layer = self.layers[0]
        output_layer = self.layers[-1]
        input_dict = input_layer.inputs
        input_keys_list = list(input_dict.keys())


        simulation_type = self.simulation_type
        diode_connected_flash_params = self.diode_connected_flash_params
        q = self.q
        
        n_of_node_voltages = len(self.all_nodes)
        voltage_dict_free = dict.fromkeys(self.all_nodes[0], None) 


        layers = self.layers
        resistive_layers = [layer for layer in layers if getattr(layer, 'trainable', None)]
        
        
        #write the weights
        best_idx, best_accuracy, wm_evo1_at_best, wm_evo2_at_best = best_epoch_from_dir(h5_path)
        W1 = wm_evo1_at_best
        W2 = wm_evo2_at_best
        
        resistive_layers[0].W = W1
        resistive_layers[1].W = W2
        
        update_synapses(eldo_process, resistive_layers, diode_connected_flash_params, simulation_type, debug)
        
        output_layer = resistive_layers[-1]
        binary_list = []
        prediction_list = []
        


        result_file = self.result_file

        counter = 0
        
        
        input_amp_voltages = []

        output_amp_voltages = [] 
 
        
 
        
        
        for X in X_in:
            for i, key in enumerate(input_keys_list):
                input_dict[key] = - X[i]  # Directly assign the value from X to the corresponding key
                
            set_input_voltages(eldo_process, input_dict, debug)


                ##### 
            try:
                offset = os.path.getsize(result_file)
            except FileNotFoundError:
                print("need to run the first simulation")
                offset = 0

            run_simulation_and_wait(eldo_process, simulation_type, q, debug)  

                #for plotting there's a function read_update_and_plot
                
            results = read_update(
                    eldo_process,
                    result_file,
                    voltage_dict_free,
                    offset,
                    simulation_type,
                    transcon_calc=False,
                    debug=debug
                    )
                
            voltage_dict_free = results["voltages"]



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
            
        binary_array = np.asarray(binary_list)
        binary_array = np.squeeze(binary_array)
        Y_flat = np.argmax(Y, axis=1).astype(int) 
        accuracy = np.mean(np.equal(binary_array, Y_flat)) * 100
        
        return accuracy
    
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
        draw_grid_flag = True
        y_predictions, input_amp_voltages, output_amp_voltages = self.free_test(eldo_process, grid, Y_val, input_function, epoch, draw_grid_flag, debug=False)
    
    
        
    
    
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

    

def validate(sim_params, h5metrics_data_path): #probably objective function

    process_id = None
    transcon_calc = None
    fet_identifiers = None

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
    # os.makedirs(out_dir, exist_ok=True)
    # plot_dir = os.path.join(out_dir, "plots")
    # os.makedirs(plot_dir, exist_ok=True)
    
    # data_path = os.path.join(plot_dir, 'metrics_data.h5')
    # config_file_path = os.path.join(out_dir, "config") 

    
    

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
        
        
    #Maybe I can simply save the input dataset?
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_t) * sim_params.scale_factor

    input_function = ds.onehot_pos_neg_inputs_1bias_double_input
    X, Y = ds.onehot_pos_neg_inputs_1bias_double_input(X_scaled, Y_t, net.bias)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)




    #pids = get_eldo_pids(eldo_identifier = 'eldo_64.exe')
    m_thread = True
    noascii =  True
    debug = False
    eldo_process, q = start_eldo_simulation_drain(new_sample_file, full_subfolder_path, m_thread, noascii, debug)
    setattr(net, 'q', q)

    try: 
            #calculate_transconductance(eldo_process, fet_identifiers, new_sample_file, net.aex_file_path, net.all_nodes, debug)
        start_epoch = time.time()


        optimizer = None
            #if epoch in draw_grid_to_plot:
            #   net.draw_grid(eldo_process, X_train, y_train, input_function, epoch, plot_directory, debug = False)
        mode = "validate"
        accuracy = net.free_test(eldo_process, X_test, y_test, input_function, mode, h5metrics_data_path, debug)


        return accuracy_list
        
    except Exception as e:
        send_quit_command(eldo_process)
        remove_directory(full_subfolder_path)
        raise
        
    
if __name__ == "__main__":
    
    h5metrics_data_path =  "/home/filip/simulations/testing_plots/fet_FSST_0802/FSST_114247_262613/plots/metrics_data.h5"
    scale_factor =  0.3
    bias = 0.3
    batch_size, beta, gamma_values = 1, 5e-5, [3e-9, 3e-9]
    sim_params = SimulationParametersFSST(scale_factor, bias, batch_size, beta, gamma_values)
    validate(sim_params, h5metrics_data_path)