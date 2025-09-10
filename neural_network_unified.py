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
from simulation_parameters_folder import SimulationParametersTran, SimulationParametersFSST
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


class MyNetwork:
    def __init__(self, layers, sim_params, input_files, all_nodes):
        # Core config
        self.layers = layers
        self.load_weights = sim_params.load_weights
        self.loss_fn = MSE(0.5)
        self.beta = sim_params.beta
        self.nudging_mode = sim_params.nudging_mode

        # Sim specifics
        self.freq = sim_params.freq
        self.simulation_type = sim_params.simulation_type  # "FSST" | "TRAN" | "DC"
        full_subfolder_path, self.new_sample_file, self.result_file = input_files

        # Network / device params
        self.batch_size = sim_params.batch_size
        self.scale_factor = sim_params.scale_factor
        self.bias = sim_params.bias
        self.diode_connected_flash_params = sim_params.diode_connected_flash_params
        self.all_nodes = all_nodes
        self.output_scale = sim_params.output_scale
        self.weights_h5_path = getattr(sim_params, "weights_h5_path", None)
        self.h5_file = getattr(sim_params, "h5_file", None)

        # Optional TRAN knobs
        self.f0 = getattr(sim_params, "f0", 1e6)
        self.t0 = getattr(sim_params, "t0", 20e-6)

    # ---------------- Helpers (explicit mode handling with elif) ----------------
    def _apply_inputs(self, input_dict, X_sample):
        """Apply inputs with correct sign per mode."""
        mode = self.simulation_type.upper()
        if mode == "FSST":
            for j, key in enumerate(input_dict.keys()):
                input_dict[key] = -X_sample[j]
        elif mode == "TRAN" or mode == "DC":
            for j, key in enumerate(input_dict.keys()):
                input_dict[key] = X_sample[j]
        else:
            raise ValueError(f"Unknown simulation_type: {self.simulation_type}")

    def _read_update_any(self, eldo_process, offset, voltage_dict, debug):
        """
        Unified read_update for FREE/NUDGE in FSST, TRAN, and DC.
        Returns a normalized dict:
          {
            "voltages": <dict>     # FSST/DC: out["voltages"], TRAN: out["ac_voltages"]
            "dc_gate_end": <dict or None>,
            "dc_ds_end":   <dict or None>,
          }
        """
        mode = self.simulation_type.upper()
        if mode == "TRAN":
            out = read_update(
                eldo_process,
                self.result_file,
                voltage_dict,
                offset,
                self.simulation_type,
                transcon_calc=True,
                debug=debug,
                f0=self.f0,
                t0=self.t0,
            )
            return {
                "voltages": out["ac_voltages"],
                "dc_gate_end": out.get("dc_gate_end"),
                "dc_ds_end": out.get("dc_ds_end"),
            }
        elif mode == "FSST" or mode == "DC":
            out = read_update(
                eldo_process,
                self.result_file,
                voltage_dict,
                offset,
                self.simulation_type,
                transcon_calc=False,
                debug=debug,
            )
            return {
                "voltages": out["voltages"],
                "dc_gate_end": None,
                "dc_ds_end": None,
            }
        else:
            raise ValueError(f"Unknown simulation_type: {self.simulation_type}")

    def _set_ic_from(self, eldo_process, update_out, debug):
        """Call set_the_ic_voltages for TRAN using returned DC info."""
        mode = self.simulation_type.upper()
        if mode == "TRAN":
            dc_g = update_out.get("dc_gate_end")
            dc_s = update_out.get("dc_ds_end")
            if dc_g is not None and dc_s is not None:
                set_the_ic_voltages(eldo_process, dc_g, dc_s, debug)
        elif mode == "FSST" or mode == "DC":
            return
        else:
            raise ValueError(f"Unknown simulation_type: {self.simulation_type}")

    # -------------------- Training (FREE/NUDGE per sample) --------------------
    def free_nudged_train(self, eldo_process, X_train, Y_train, targets, epoch, optimizer, debug):
        layers = self.layers
        beta = self.beta
        batch_size = self.batch_size
        loss_fn = self.loss_fn
        simtype = self.simulation_type
        diode_params = self.diode_connected_flash_params
        q = self.q

        input_layer = layers[0]
        output_layer = layers[-1]
        input_dict = input_layer.inputs
        inudge_dict = output_layer.parameters
        out_keys = list(inudge_dict.keys())
        resistive_layers = [ly for ly in layers if getattr(ly, "trainable", None)]

        num_batches = int(np.ceil(len(X_train) / batch_size))

        loss_list = []
        output_list = []
        output_list_nudge = []
        weight_matrices_1, weight_matrices_2 = [], []
        voltage_dict_free = dict.fromkeys(self.all_nodes[0], None)
        voltage_dict_nudge = dict.fromkeys(self.all_nodes[0], None)

        # ---------- Epoch 1: program / load weights once, then write ----------
        if epoch == 1:
            if self.load_weights:
                mode = simtype.upper()
                # Prefer a path passed in from multiprocessing
                h5_path = self.h5_file

                # if not os.path.isdir(h5_path):
                #     raise FileNotFoundError(f"weights_h5_path not found: {h5_path}")
        
                best_idx, best_accuracy, wm_evo1_at_best, wm_evo2_at_best = best_epoch_from_dir(h5_path)
                resistive_layers[0].W = wm_evo1_at_best
                resistive_layers[1].W = wm_evo2_at_best
            else:
                initialize_synapses(eldo_process, resistive_layers, diode_params, simtype, debug)

            update_synapses(eldo_process, resistive_layers, diode_params, simtype, debug)

        # ----------------------- Mini-batch loop -----------------------
        for i in range(num_batches):
            try:
                offset = os.path.getsize(self.result_file)
            except FileNotFoundError:
                offset = 0

            X_batch = X_train[i * batch_size: min((i + 1) * batch_size, len(X_train))]
            Y_batch = Y_train[i * batch_size: min((i + 1) * batch_size, len(Y_train))]

            for ly in resistive_layers:
                ly.zero_grad()

            for Xs, Ys in zip(X_batch, Y_batch):
                # Inputs (mode-dependent)
                self._apply_inputs(input_dict, Xs)
                disable_current_sources(eldo_process, inudge_dict, debug)

                # Beta sampling (mode-dependent)
                modeU = simtype.upper()
                if modeU == "TRAN":
                    beta_r = beta * np.random.uniform(-5, 5)
                elif modeU == "FSST" or modeU == "DC":
                    beta_r = beta * np.random.uniform(-5, 5)
                else:
                    raise ValueError(f"Unknown simulation_type: {simtype}")

                # ---------------- FREE ----------------
                set_input_voltages(eldo_process, input_dict, debug)
                try:
                    offset = os.path.getsize(self.result_file)
                except FileNotFoundError:
                    offset = 0
                run_simulation_and_wait(eldo_process, simtype, q, debug)

                free_out = self._read_update_any(eldo_process, offset, voltage_dict_free, debug)
                voltage_dict_free.update(free_out["voltages"])
                # STRICT TRAN behavior: reset IC after every FREE
                self._set_ic_from(eldo_process, free_out, debug)

                for ly in resistive_layers:
                    ly.update__free_voltages(voltage_dict_free)

                outputs = resistive_layers[-1].output_free_voltages
                output_values = np.array(list(outputs.values()))
                output_list.append(output_values)

                # Target for this sample
                mode_loss = self.nudging_mode
                if self.output_scale == None:
                    self.output_scale = 4

                if targets is not None:
                    Y_index = np.argmax(Ys)
                    target = targets[Y_index]
                else:
                    target = (Ys / self.output_scale) * self.scale_factor

                # Compute loss and set nudging currents on HW outputs
                sample_losses, currents = loss_fn(outputs, target, beta_r, mode_loss)
                flat = currents.flatten()
                for k, key in enumerate(out_keys):
                    val = flat[k]
                    if modeU == "DC":
                        val *= -1  # your DC sign flip
                    inudge_dict[key] = val
                set_currents_nudge_mode(eldo_process, inudge_dict, debug)
                loss_list.append(sample_losses)

                # ---------------- NUDGE ----------------
                offset = os.path.getsize(self.result_file)
                run_simulation_and_wait(eldo_process, simtype, q, debug)

                nudge_out = self._read_update_any(eldo_process, offset, voltage_dict_nudge, debug)
                voltage_dict_nudge.update(nudge_out["voltages"])
                # STRICT TRAN behavior: reset IC after every NUDGE
                self._set_ic_from(eldo_process, nudge_out, debug)

                for ly in resistive_layers:
                    ly.update__nudge_voltages(voltage_dict_nudge)

                outputs_n = resistive_layers[-1].output_nudge_voltages
                outputs_n_values = list(outputs_n.values())
                output_list_nudge.append(outputs_n_values)
                _sample_losses_n, _voltages_n = loss_fn(outputs_n, target, beta_r, mode_loss)

                # Accumulate gradients per EP rule inside layer
                for ly in resistive_layers:
                    ly.run_update_process(batch_size, beta_r)

            # Write back updated device states once per batch
            update_synapses(eldo_process, resistive_layers, diode_params, simtype, debug)
            weight_matrices_1.append(layers[1].W)
            weight_matrices_2.append(layers[3].W)

        # ----------------------- Epoch summary -----------------------
        predictions = loss_fn(output_list, mode="test")
        epoch_acc = np.mean(loss_fn.verify_result(Y_train, np.array(predictions)))
        diff1, diff2 = output_plot(output_list)
        mean_loss = np.mean(np.array(loss_list), axis=1) if len(loss_list) else np.array([])

        return {
            "accuracy": epoch_acc,
            "loss_list": mean_loss,
            "weight_matrix_1": weight_matrices_1[-1] if weight_matrices_1 else layers[1].W,
            "weight_matrix_2": weight_matrices_2[-1] if weight_matrices_2 else layers[3].W,
            "epoch_accuracy": epoch_acc,
            "output_list": [diff1, diff2],
        }
    def validate(self, eldo_process, X_test, Y_test, optimizer, debug):
        layers = self.layers
        beta = self.beta
        loss_fn = self.loss_fn
        simtype = self.simulation_type
        diode_params = self.diode_connected_flash_params
        q = self.q

        input_layer = layers[0]
        output_layer = layers[-1]
        input_dict = input_layer.inputs
        inudge_dict = output_layer.parameters
        out_keys = list(inudge_dict.keys())
        resistive_layers = [ly for ly in layers if getattr(ly, "trainable", None)]


        loss_list = []
        output_list = []
        voltage_dict_free = dict.fromkeys(self.all_nodes[0], None)

        # ---------- Epoch 1: program / load weights once, then write ----------
        if epoch == 1:
            if self.load_weights:
                mode = simtype.upper()
                # Prefer a path passed in from multiprocessing
                h5_path = self.h5_file

                # if not os.path.isdir(h5_path):
                #     raise FileNotFoundError(f"weights_h5_path not found: {h5_path}")
        
                best_idx, best_accuracy, wm_evo1_at_best, wm_evo2_at_best = best_epoch_from_dir(h5_path)
                resistive_layers[0].W = wm_evo1_at_best
                resistive_layers[1].W = wm_evo2_at_best
            else:
                initialize_synapses(eldo_process, resistive_layers, diode_params, simtype, debug)

            update_synapses(eldo_process, resistive_layers, diode_params, simtype, debug)

        # ----------------------- Mini-batch loop -----------------------
        for i in range(num_batches):
            try:
                offset = os.path.getsize(self.result_file)
            except FileNotFoundError:
                offset = 0

            X_batch = X_train[i * batch_size: min((i + 1) * batch_size, len(X_train))]
            Y_batch = Y_train[i * batch_size: min((i + 1) * batch_size, len(Y_train))]

            for ly in resistive_layers:
                ly.zero_grad()

            for Xs, Ys in zip(X_batch, Y_batch):
                # Inputs (mode-dependent)
                self._apply_inputs(input_dict, Xs)
                disable_current_sources(eldo_process, inudge_dict, debug)

                # Beta sampling (mode-dependent)
                modeU = simtype.upper()
                if modeU == "TRAN":
                    beta_r = beta * np.random.uniform(-5, 5)
                elif modeU == "FSST" or modeU == "DC":
                    beta_r = beta * np.random.uniform(0, 5)
                else:
                    raise ValueError(f"Unknown simulation_type: {simtype}")

                # ---------------- FREE ----------------
                set_input_voltages(eldo_process, input_dict, debug)
                try:
                    offset = os.path.getsize(self.result_file)
                except FileNotFoundError:
                    offset = 0
                run_simulation_and_wait(eldo_process, simtype, q, debug)

                free_out = self._read_update_any(eldo_process, offset, voltage_dict_free, debug)
                voltage_dict_free.update(free_out["voltages"])
                # STRICT TRAN behavior: reset IC after every FREE
                self._set_ic_from(eldo_process, free_out, debug)

                for ly in resistive_layers:
                    ly.update__free_voltages(voltage_dict_free)

                outputs = resistive_layers[-1].output_free_voltages
                output_values = np.array(list(outputs.values()))
                output_list.append(output_values)

                # Target for this sample
                mode_loss = self.nudging_mode
                if targets is not None:
                    Y_index = np.argmax(Ys)
                    target = targets[Y_index]
                else:
                    target = (Ys / 4) * self.scale_factor

                # Compute loss and set nudging currents on HW outputs
                sample_losses, currents = loss_fn(outputs, target, beta_r, mode_loss)
                flat = currents.flatten()
                for k, key in enumerate(out_keys):
                    val = flat[k]
                    if modeU == "DC":
                        val *= -1  # your DC sign flip
                    inudge_dict[key] = val
                set_currents_nudge_mode(eldo_process, inudge_dict, debug)
                loss_list.append(sample_losses)

                # ---------------- NUDGE ----------------
                offset = os.path.getsize(self.result_file)
                run_simulation_and_wait(eldo_process, simtype, q, debug)

                nudge_out = self._read_update_any(eldo_process, offset, voltage_dict_nudge, debug)
                voltage_dict_nudge.update(nudge_out["voltages"])
                # STRICT TRAN behavior: reset IC after every NUDGE
                self._set_ic_from(eldo_process, nudge_out, debug)

                for ly in resistive_layers:
                    ly.update__nudge_voltages(voltage_dict_nudge)

                outputs_n = resistive_layers[-1].output_nudge_voltages
                outputs_n_values = list(outputs_n.values())
                output_list_nudge.append(outputs_n_values)
                _sample_losses_n, _voltages_n = loss_fn(outputs_n, target, beta_r, mode_loss)

                # Accumulate gradients per EP rule inside layer
                for ly in resistive_layers:
                    ly.run_update_process(batch_size, beta_r)

            # Write back updated device states once per batch
            update_synapses(eldo_process, resistive_layers, diode_params, simtype, debug)
            weight_matrices_1.append(layers[1].W)
            weight_matrices_2.append(layers[3].W)

        # ----------------------- Epoch summary -----------------------
        predictions = loss_fn(output_list, mode="test")
        epoch_acc = np.mean(loss_fn.verify_result(Y_train, np.array(predictions)))
        diff1, diff2 = output_plot(output_list)
        mean_loss = np.mean(np.array(loss_list), axis=1) if len(loss_list) else np.array([])

        return {
            "accuracy": epoch_acc,
            "loss_list": mean_loss,
            "weight_matrix_1": weight_matrices_1[-1] if weight_matrices_1 else layers[1].W,
            "weight_matrix_2": weight_matrices_2[-1] if weight_matrices_2 else layers[3].W,
            "epoch_accuracy": epoch_acc,
            "output_list": [diff1, diff2],
        }
def train(sim_params, process_id = None, logger = None): #probably objective function

        
    
    if logger is None:
        logger = logging.getLogger(f"worker-{os.getpid()}")  # should already have a QueueHandler
        logger.info(f"Started train for {sim_params.gamma_values}, pid={process_id}")
    else:
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
    
    data_path = os.path.join(plot_dir, 'metrics_data')
    config_file_path = os.path.join(out_dir, "config") 

    
    

    all_nodes = extract_all_nodes_voltages(layers)
    builder.build_netlist(new_sample_file, transcon_calc, fet_identifiers, result_file_path)

    
    net = MyNetwork(layers, sim_params, input_files, all_nodes)
         
    simulation_dataset = sim_params.dataset
        # Load and center the dataset
    if simulation_dataset == "moons_simulation":
        X_t, Y_t = ds.prepare_moons_data(sim_params.num_samples, noise=0.1, random_state=41)
    elif simulation_dataset == "digits_simulation":
        X_t, Y_t = ds.prepare_digits_data()
    elif simulation_dataset == "iris_simulation":
        X_t, Y_t = ds.prepare_iris_data()
    else:
        raise ValueError("Invalid dataset")
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
        plot_average_relative_change(abs_change1, title="Average Absolute Weight Change per Epoch weightmatrix 1")
        plot_average_relative_change(abs_change2, title="Average Absolute Weight Change per Epoch weightmatrix 2")
    
        plot_weight_matrix_evolution_lines(weight_matrices_1, plot_dir, title='Weight Matrix 1 Evolution Over Epochs')
        plot_weight_matrix_evolution_lines(weight_matrices_2, plot_dir, title='Weight Matrix 2 Evolution Over Epochs')
        #delete_file_with_chi_extension(new_sample_file)
        
        
        
        
        
        
        send_quit_command(eldo_process)
        remove_directory(full_subfolder_path)
        return accuracy_list
        
    except Exception as e:
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
    scale_factor = 0.4
    bias =  0.3
    load_weights = False
    h5_file = None
    sim_params = SimulationParametersFSST(scale_factor, bias, batch_size, beta, gamma_values, load_weights, h5_file)
    train(sim_params, process_id = 5)