from initializer import *
import datasets as ds
import numpy as np
import time

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



class NetworkAnalyzer:
    def __init__(
        self,
        layers,
        sim_params,
        input_files,
        all_nodes
    ):
        # Basic network and simulation parameters
        self.layers = layers
        self.simulation_details = sim_params
        self.nudging_mode = sim_params.nudging_mode
        self.freq = sim_params.freq
        self.simulation_type = sim_params.simulation_type

        # File paths
        (
            self.full_subfolder_path,
            self.new_sample_file,
            self.result_file
        ) = input_files

        # Dataset parameters
        self.scale_factor = sim_params.scale_factor
        self.bias = sim_params.bias
        self.diode_connected_flash_params = sim_params.diode_connected_flash_params

        # Node list
        self.all_nodes = all_nodes

    def initialize_synapses(
        self,
        eldo_process,
        resistive_layers,
        simulation_type,
        debug
    ):
        params = self.diode_connected_flash_params
        full_voltage_range = params["full_voltage_range"]
        offset1 = params["offset1layer"]
        offset2 = params["offset2layer"]

        for layer in resistive_layers:
            layer.update_synapse_dict(
                full_voltage_range,
                offset1,
                offset2
            )
            set_synapses(eldo_process, layer.synapse_dict, debug)

    def read_update_and_plot(
        self,
        eldo_process,
        result_file,
        voltage_dict_free,
        offset,
        simulation_type,
        n_of_node_voltages,
        debug
    ):
        if simulation_type == "FSST":
            return parse_aex_file_from_end_offset(
                result_file,
                n_of_node_voltages,
                simulation_type,
                voltage_dict_free,
                offset
            )

        # TRAN simulation processing
        df_last, col_names = open_and_read_txt_file(result_file, offset)
        f0 = 1e6
        t0 = 15e-6

        dc_gate, ac_gate, dc_ds, ac_ds, used_cols = compute_fourier_coefficients2(
            df_last,
            f0,
            t0,
            n_harmonics=1,
            n_periods=4
        )

        plot_specified_columns(
            df_last,
            [
                "V(VG_0_1_2)",
                "V(VG_1_1_1)",
                "V(V_OUT_0_1)",
                "V(V_OUT_1_1)",
                "V(V_IN_1_1)"
            ],
            time_col=None
        )
        plot_specified_columns(
            df_last,
            [
                "V(XI011.XI1.OUTPUT_CS_1)",
                "V(XI011.XI1.OUTPUT_CS_2)",
                "V(XI011.NET05)"
            ],
            time_col=None
        )

        # Update free voltages with AC results
        for node, coeffs in ac_ds.items():
            if node in voltage_dict_free:
                voltage_dict_free[node] = coeffs

        # Prepare DC parameter dictionaries
        dc_gate_params = {f"W_{n[3:]}": round(v, 3) for n, v in dc_gate.items()}
        dc_ds_params = make_param_dict(dc_ds, prefix="V_END_")

        # Send commands to Eldo
        for key, val in dc_gate_params.items():
            send_command_to_eldo(
                eldo_process,
                f"SET P({key})={val}",
                debug
            )
        for key, val in dc_ds_params.items():
            send_command_to_eldo(
                eldo_process,
                f"SET P({key})={val}",
                debug
            )

        return voltage_dict_free






    def draw_grid(self, eldo_process, X_val, Y_val, input_function, epoch, plot_directory, debug):
        # Start a new figure for this iteration
        plt.figure()
        
        # Define bounds of the domain
        min1, max1 = -0.5, 0.5
        min2, max2 = -0.5, 0.5
        num_points = 20
    
        x1grid = np.linspace(min1, max1, num_points)
        x2grid = np.linspace(min2, max2, num_points)
    
        # Create a meshgrid from the grid points
        xx, yy = np.meshgrid(x1grid, x2grid)
        grid = np.c_[xx.ravel(), yy.ravel()]
    
    
        # Make predictions for the grid
        X_bias = 0.3
        draw_grid_flag = True
        y_predictions, input_amp_voltages, output_amp_voltages = self.free_test(eldo_process, grid, X_bias, Y_val, input_function, epoch, draw_grid_flag, debug=False)
        
        
        
    def analyze_and_plot(self, eldo_process, X_analyze, debug):
        layers = self.layers
        result_file = self.result_file
        simulation_type = self.simulation_type
        
        
        n_of_node_voltages = len(self.all_nodes)
        
        q = self.q
        
        
        input_layer = layers[0]
        output_layer = layers[-1]
        input_dict = input_layer.inputs
        input_keys_list = list(input_dict.keys())

        
        resistive_layers = [layer for layer in layers if getattr(layer, 'trainable', None)]
        
         
        for layer in resistive_layers:
            layer.gamma *= 1

        voltage_dict_free = dict.fromkeys(self.all_nodes[0], None) #
        voltage_dict_nudge = dict.fromkeys(self.all_nodes[0], None)
        
        offset = 0

        # Context and training lists
        resistive = [l for l in self.layers if getattr(l, 'trainable', False)]
        
        for layer in resistive:
            layer.gamma *= 1


        # Initial parameter settings
        initial_cmds = [
            "SET P(W_NONLIN_NMOS)=4u",
            "SET P(W_NONLIN_PMOS)=4u",
            "SET P(VDD_PMOS_NONLIN)=2.4"
        ]
        for cmd in initial_cmds:
            send_command_to_eldo(eldo_process, cmd, True)
            
        self.initialize_synapses(
            eldo_process,
            resistive,
            simulation_type,
            True
        )

        # Run input-value simulations
        for inputs in X_analyze:
            if os.path.exists(result_file):
                offset = os.path.getsize(result_file)
            else:
                offset = 0
            for j, key in enumerate(input_keys_list):
                input_dict[key] = X_analyze[j]  # Directly assign the value from X to the corresponding key
                
                
            set_input_voltages(eldo_process, input_dict, True)
            run_simulation_and_wait(
                eldo_process,
                simulation_type,
                q,
                True
            )
            self.read_update_and_plot(
                eldo_process,
                result_file,
                voltage_dict_free,
                offset,
                simulation_type,
                len(all_nodes),
                True
            )

        return net, X_train, X_test, y_train, y_test


def empty_queue(q):
    """
    Remove and discard all items from the given Queue.
    Works with both queue.Queue and multiprocessing.Queue.
    """
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass

def main_setup(sim_params, process_id):
    """
    Core setup for netlist build and directory initialization.
    Returns: layers, sample_file, result_file, out_dir, plot_dir, all_nodes
    """
    fet_ids = ["0_1_1", "1_1_1"]; transcalc = False
    layers = initialize_network_layers(sim_params)
    builder = netlist_builder(layers, sim_params)
    simulation_type = sim_params.simulation_type

    full_subfolder_path, new_sample_file, result_file = create_filenames(
        sim_params.output_dir, sim_params.sample_file, simulation_type, process_id
    )
    all_nodes = extract_all_nodes_voltages(layers)
    net = NetworkAnalyzer(layers, sim_params, (full_subfolder_path, new_sample_file, result_file), all_nodes)

    builder.build_netlist(new_sample_file, transcalc, fet_ids, result_file)

    base_dir = sim_params.trained_models_dir
    ts = datetime.now().strftime("%H%M%S")
    folder = f"{simulation_type}_{ts}_{process_id}" if process_id else f"{simulation_type}_{ts}"
    out_dir = os.path.join(base_dir, folder); os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "plots");    os.makedirs(plot_dir, exist_ok=True)
    
    
    
    
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
    
    
    
    
    
    
    X, Y = ds.onehot_pos_neg_inputs_1bias_double_input(X_small, Y_t, bias)
    m_thread = True; noascii = True; debug = True
    eldo_process, q = start_eldo_simulation_drain(new_sample_file, full_subfolder_path, m_thread, noascii, debug)
    setattr(net, 'q', q)
    run_simulation_and_wait(eldo_process, simulation_type, q, debug)  
    net.analyze_and_plot(eldo_process, X_analyze, debug)

    ###here I run the analyyer
    






def plot_voltage(df, voltage_cols='all', time_col=None, title=None,
                 xlabel='Time', ylabel='Voltage'):
    if time_col is None:
        time_col = next((c for c in df.columns if c.lower() == 'time'), None)
        if time_col is None:
            raise KeyError("No time column found. Expected column named 'time'.")

    # Determine voltage columns
    if voltage_cols is 'all':
        volt_cols = [c for c in df.columns if c != time_col]
    elif isinstance(voltage_cols, str):
        volt_cols = [voltage_cols]
    else:
        volt_cols = list(voltage_cols)

    t = df[time_col].values
    # Split into chunks of max 4
    chunks = [volt_cols[i:i+4] for i in range(0, len(volt_cols), 4)]
    figs_axes = []

    for idx, chunk in enumerate(chunks, 1):
        n = len(chunk)
        cols = 2 if n > 1 else 1
        rows = int(np.ceil(n / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), squeeze=False)
        for j, col in enumerate(chunk):
            r, c = divmod(j, cols)
            ax = axs[r][c]
            if col not in df.columns:
                raise KeyError(f"Voltage column '{col}' not found in DataFrame.")
            ax.plot(t, df[col].values)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(col)
        # Hide unused subplots
        total_subplots = rows * cols
        for j in range(n, total_subplots):
            r, c = divmod(j, cols)
            fig.delaxes(axs[r][c])
        # Figure title
        if title:
            fig_title = title if isinstance(title, str) and len(chunks) == 1 else f"{title} (Fig {idx})" if isinstance(title, str) else title[idx-1]
            fig.suptitle(fig_title)
        figs_axes.append((fig, axs))

    return figs_axes

    










def extract_last_voltages(df_last):
    """
    Extracts the final gate voltages and other voltages from df_last.

    Returns
    -------
    gate_voltage_dict : dict
        Maps each gate column VG_x_y_z to key W_x_y_z with its last VALUE.
    end_voltage_dict : dict
        Maps each non-gate voltage column V_x_y_z to key V_END_x_y_z with its last VALUE.
    """
    # 1) Find all gate-voltage columns (prefix "VG_")
    gate_cols = [c for c in df_last.columns if c.startswith('V(VG_')]
    # 2) Find all other voltage columns (prefix "V_" but not "VG_")
    other_cols = [c for c in df_last.columns if c.startswith('V(V_') and not c.startswith('V(VG_')]

    # 3) Build gate dict
    gate_voltage_dict = {
        f"W_{col.split('_', 1)[1]}": df_last[col].iloc[-1]
        for col in gate_cols
    }
    # 4) Build end-voltage dict
    end_voltage_dict = {
        f"V_END_{col.split('_', 1)[1]}": df_last[col].iloc[-1]
        for col in other_cols
    }

    return gate_voltage_dict, end_voltage_dict


     
def analyze_network():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    new_sample_file = os.path.join(script_dir, "big_network3.cir")
    full_subfolder_path = script_dir
    
    debug = True
    
    eldo_process, q = start_eldo_simulation_drain(new_sample_file, full_subfolder_path, m_thread = True, noascii =  True, debug = False)
    
    
    pulses = np.linspace(-10e-9, 15e-9, 20)
    v_forward  = np.linspace(1, 2.6, 16)
    v_backward = np.flip(v_forward)
    v_pulses   = np.hstack((v_forward, v_backward))
    
    
    init_ic = np.linspace(-1, 1, 10)
    caps = np.linspace(8e-15, 15e-15, 10)
    caps = [10e-15]
    f0 = 1e6
    t0 = 15e-6
    sim_time = 50e-6
    txt_file_path = os.path.join(script_dir, 'TRANDATA.TXT')
    gm_list = []
    gm_list3 = []
    gm_list4 = []

    dc_gate_vol_list = []
    dc_gate2_vol_list = []
    n_of_pulses = 32
    dc_gate_vol_end = None
    dc_gate_vol_end2 = None
    dc_gate_end2 = None
    neg_pulses = np.linspace(0.85, 0.95, 10)
    #neg_pulse = 1.1e-9
    dc_gate_end011 = None
    dc_gate_end111 = None
    dc_gate_end013 = None
    df_last = None
    for neg_pulse in neg_pulses:
        i = 0
        gm_list = []
        gm_list013 = []
        gm_list111 = []

        dc_gate_vol_list = []
        dc_gate_vol_list013 =[]
        dc_gate_vol_list111 =[]
        
        dc_gate_vol_list = []
        dc_gate_vol_list013end =[]
        dc_gate_vol_list111end =[]
        
        
        dc_gate_vol_list_end = []
        #send_command_to_eldo(eldo_process, f"SET P(CAP)={cap}", debug)
        
        for i in range(n_of_pulses):
            if os.path.exists(txt_file_path):
                offset = os.path.getsize(txt_file_path)
            else:
                offset = 0
            if i == 0:
                pulse = 0
                send_command_to_eldo(eldo_process, f"SET P(Iw_0_1_1)={pulse}", debug)                
            # elif i == 1:
            #     pulse = -20e-9
            #     send_command_to_eldo(eldo_process, f"SET P(Iw_0_1_1)={pulse}", debug)
            else:
            #send_command_to_eldo(eldo_process, f"SET P(I_write)={pulse}", debug)
                gate_voltage_dict, end_voltage_dict = extract_last_voltages(df_last)
                for key, value in gate_voltage_dict.items():
                    command = f"SET P({key})={value}"
                    send_command_to_eldo(eldo_process, command, debug)
                for key, value in end_voltage_dict.items():
                    command = f"SET P({key})={value}"
                    send_command_to_eldo(eldo_process, command, debug)
                
                v_pulse = v_pulses[i]
                #send_command_to_eldo(eldo_process, f"SET P(W_0_1_1)={v_pulse}", debug)
                #send_command_to_eldo(eldo_process, f"SET P(W_0_1_3)={v_pulse}", debug)
                #send_command_to_eldo(eldo_process, f"SET P(W_1_1_1)={v_pulse}", debug)
                # dc_gate_value = np.real(dc_gate_end2)
                # weight011 = np.real(dc_gate_end011)
                # weight111 = np.real(dc_gate_end111)
                # weight013 = np.real(dc_gate_end013)
                # command = f"SET P(W_0_1_1)={weight011}"
                # send_command_to_eldo(eldo_process, command, debug)
                # command = f"SET P(W_1_1_1)={weight111}"
                # send_command_to_eldo(eldo_process, command, debug)
                # command = f"SET P(W_0_1_3)={weight013}"
                # send_command_to_eldo(eldo_process, command, debug)
                # choose base amplitude
                base_pulse = 2e-9
        
                # compute block index and sign: block 0 ? +, block 1 ? ?, block 2 ? +, ?
                block = i // 16
                sign  = (-1) ** block      # +1 for even blocks, ?1 for odd blocks
                if sign == -1:
                    base_pulse = neg_pulse * base_pulse
                else:
                    base_pulse = 2e-9
                pulse = base_pulse * sign
                send_command_to_eldo(eldo_process, f"SET P(Iw_0_1_1)={pulse}", debug)
                send_command_to_eldo(eldo_process, f"SET P(Iw_1_1_3)={pulse}", debug)
                send_command_to_eldo(eldo_process, f"SET P(Iw_1_1_1)={pulse}", debug)
                
                
            t0 = 15e-6
            run_trans_and_DC(eldo_process, sim_time, q, debug)
            df_last, col_names = open_and_read_txt_file(txt_file_path, offset)
            results, voltage_cols = compute_fourier_coefficients(
                df_last, f0, t0, n_harmonics=1, n_periods=4
            )
            dc_gate, ac_gate, dc_ds, ac_ds, used_columns = compute_fourier_coefficients2(df_last, f0, t0,
                                             n_harmonics=1,
                                             n_periods=4,
                                             voltage_cols=None)
            dc_gate_mid = results['V(VG_0_1_1)']['DC']
            dc_gate_vol_list.append(dc_gate_mid)
            
            dc_gate_mid013 = results['V(VG_0_1_3)']['DC']
            dc_gate_vol_list013.append(dc_gate_mid013)
            
            
            dc_gate_mid111 = results['V(VG_1_1_1)']['DC']
            dc_gate_vol_list111.append(dc_gate_mid111)
            
            #dc_vol_mid = results['V(V_IN_0_1)']['DC']
            #dc_gate_vol_mid = results['V(VG_0_1_1)']['DC']
            gm = abs(results['ISUB(XM_0_1_1.S)']['AC1']) / abs(results['V(V_IN_0_1)']['AC1'] - results['V(V_OUT_0_1)']['AC1'])
            gm_list.append(gm)
            gm111 = abs(results['ISUB(XM_1_1_1.S)']['AC1']) / abs(results['V(V_IN_1_1)']['AC1'] - results['V(V_OUT_1_1)']['AC1'])
            gm_list111.append(gm111)
            gm013 = abs(results['ISUB(XM_0_1_3.S)']['AC1']) / abs(results['V(V_IN_0_1)']['AC1'] - results['V(V_OUT_0_3)']['AC1'])
            gm_list013.append(gm013)
            # t0 = 25e-6
            # results, voltage_cols = compute_fourier_coefficients(
            #     df_last, f0, t0, n_harmonics=1,n_periods=5
            # )     
            dc_gate_end = results['V(VG_1_1_1)']['DC']
            #dc_gate_end = results['V(VG_0_1_1)']['DC']
            #dc_vol_mid = results['V(V_IN_0_1)']['DC']
            #dc_cur_end = abs(results[voltage_cols[0]][0])
            #dc_gate_vol_end = df_last['V(VG_0_1_1)'].iloc[-1]
            
            

            dc_gate_end011 = df_last['V(VG_0_1_1)'].iloc[-1]
            dc_gate_end111 = df_last['V(VG_1_1_1)'].iloc[-1]
            dc_gate_end013 = df_last['V(VG_0_1_3)'].iloc[-1]


            dc_gate_vol_list013end.append(dc_gate_end013)
            dc_gate_vol_list111end.append(dc_gate_end111)
            




            dc_gate2_vol_list.append(dc_gate_end2)
            #plt.figure()
            # plot_voltage(df_last, voltage_cols= 'ISUB(XM_0_1_1.S)', time_col=None, title=None,
            #                 xlabel='Time', ylabel='Voltage')
            # plot_voltage(df_last, voltage_cols= 'V(VG_0_1_1)', time_col=None, title=None,
            #                 xlabel='Time', ylabel='Voltage')
            # plot_voltage(df_last, voltage_cols= 'V(V_OUT_0_1)', time_col=None, title=None,
            #                 xlabel='Time', ylabel='Voltage')
            
        
        def plot_series(x, y, *,
                        xlabel, ylabel, title,
                        split_half=False,
                        annotate_indices=None,
                        labels=None):
            """
            Generic plot helper.
        
            Parameters
            ----------
            x : 1D array-like
            y : 1D array-like, same length as x
            xlabel, ylabel, title : str
            split_half : bool
                If True, splits both x and y in half and plots the first half as
                "Forward" and second as "Backward" with different colors.
            annotate_indices : iterable of ints, optional
                Points in y (and corresponding x) to annotate with their numeric value.
            labels : tuple(str, str), optional
                Names for the two halves when split_half=True (default ("Forward","Backward")).
            """
            x = np.asarray(x)
            y = np.asarray(y)
            plt.figure()
        
            if split_half:
                half = len(x)//2
                lab_fwd, lab_bwd = labels or ("Forward", "Backward")
                plt.plot(x[:half],   y[:half],   'o-', label=lab_fwd)
                plt.plot(x[half:],   y[half:],   's-', label=lab_bwd)
                plt.legend()
            else:
                plt.plot(x, y, 'o-')
        
            if annotate_indices:
                for idx in annotate_indices:
                    xi, yi = x[idx], y[idx]
                    plt.annotate(f"{yi:.3f}", xy=(xi, yi),
                                 xytext=(0, 8), textcoords='offset points',
                                 ha='center', color='C2')
                    plt.plot(xi, yi, 'o', color='C2')
        
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        
        # === usage ===
        
        scale = 1e-5
        half_len = len(v_pulses)//2
        
        # 1) gm vs. initial gate voltage, forward/backward
        for data, label in ((gm_list,    'gm011'),
                            (gm_list013, 'gm013'),
                            (gm_list111, 'gm111')):
            plot_series(v_pulses, np.array(data)/scale,
                        xlabel='Initial Gate Voltage (V)',
                        ylabel=f'{label} (×10?? S)',
                        title=f'{label} vs. Initial Gate Voltage',
                        split_half=True)
        
        # 2) gm vs. pulse number (simple)
        for data, label in ((gm_list,    'gm011'),
                            (gm_list013, 'gm013'),
                            (gm_list111, 'gm111')):
            pulse_nums = list(range(1, n_of_pulses + 1))
            plot_series(pulse_nums, np.array(data)/scale,
                        xlabel='Pulse Number',
                        ylabel=f'{label} (×10?? S)',
                        title=f'{label} vs. Pulse Number')
        
        # 3) DC gate voltages at mid-pulse
        for data, label in ((dc_gate_vol_list,    '0_1_1'),
                            (dc_gate_vol_list111, '1_1_1'),
                            (dc_gate_vol_list013, '0_1_3')):
            pulse_nums = list(range(1, n_of_pulses + 1))
            plot_series(pulse_nums, data,
                        xlabel='Pulse Number',
                        ylabel=f'DC Gate Voltage Mid {label} (V)',
                        title='Gate Voltage at Mid of Reading Pulse')
        
        # 4) DC gate voltages at end-pulse with annotation at [0] and [15]
        for data, label in ((dc_gate_vol_list111end, '1_1_1'),
                            (dc_gate_vol_list013end, '0_1_3')):
            pulse_nums = list(range(1, n_of_pulses + 1))
            plot_series(pulse_nums, data,
                        xlabel='Pulse Number',
                        ylabel=f'DC Gate Voltage End {label} (V)',
                        title='Gate Voltage at End of Reading Pulse',
                        annotate_indices=(0, 15))
        
                      
            
        # --- 2) Plot pulse number vs DC gate voltage at end ---
        # plt.figure()
        # plt.plot(pulse_nums, dc_gate2_vol_list, marker='o')
        # plt.xlabel('Pulse Number')
        # plt.ylabel('DC Gate Voltage Mid (V)')
        # plt.title(f'Gate Voltage at End of Pulse vs. Pulse Number for negpulse {neg_pulse}')
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
                    
        
        
    # Determine the folder where this script is located    
    # Build the full path to data.txt
    # data_path = os.path.join(script_dir, 'DATA.TXT')
    
    # data = np.loadtxt(data_path, comments='#')
    
    # t = data[:, 0]
    # v = data[:, 1]
    # i = data[:, 2]
    
    # pulse_data = []
    
    # # 2) Loop over pulses 1?9
    # for pulse_num in range(1, 30):
    #     # define the start/end of this pulse window
    #     start_t = 13e-6 + (pulse_num - 1) * 10e-6  # 13µs, 23µs, 33µs, ?
    #     end_t   = 20e-6 + (pulse_num - 1) * 10e-6  # 20µs, 30µs, 40µs, ?
    
    #     # 3) Build the boolean mask for this time interval:
    #     mask = (t >= start_t) & (t <= end_t)
    
    #     # 4) Use the mask to slice out just those points
    #     pulse_t = t[mask]
    #     pulse_v = v[mask]
    #     pulse_i = i[mask]
    
    #     pulse_data.append({
    #         'pulse': pulse_num,
    #         'start_time': start_t,
    #         'end_time': end_t,
    #         't': pulse_t,
    #         'v': pulse_v,
    #         'i': pulse_i
    #     })                      
    
    # f0 = 1e6  # 1 MHz
    # T0 = 1 / f0
    # # Compute first harmonic and transconductance for each pulse
    # results = []
    # for p in pulse_data:
    #     t_p = p['t']
    #     v_p = p['v']
    #     i_p = p['i']
    #     # 1st harmonic coefficients via trapezoidal integration
    #     X1_v = np.trapz(v_p * np.exp(-1j * 2 * np.pi * f0 * t_p), t_p) / T0
    #     X1_i = np.trapz(i_p * np.exp(-1j * 2 * np.pi * f0 * t_p), t_p) / T0
    #     # Transconductance (complex ratio)
    #     g1 = X1_i / X1_v
    
    #     results.append({
    #         'Pulse': p['pulse'],
    #         'Voltage Amp [V]': np.abs(X1_v),
    #         'Voltage Phase [deg]': np.angle(X1_v, deg=True),
    #         'Current Amp [A]': np.abs(X1_i),
    #         'Current Phase [deg]': np.angle(X1_i, deg=True),
    #         'Transcond Amp [S]': np.abs(g1),
    #         'Transcond Phase [deg]': np.angle(g1, deg=True)
    #     })
    
    # # Display results
    # df = pd.DataFrame(results)
    # # Plot all three on the same figure
    # plt.figure()
    # plt.plot(df['Pulse'], df['Voltage Amp [V]'], label='Voltage Amp [V]')
    # plt.xlabel('Pulse Number')
    # plt.ylabel('Voltage Amplitude')
    # plt.title('Voltage FSST')
    # plt.figure()
    # plt.plot(df['Pulse'], df['Current Amp [A]'], label='Current Amp [A]')
    # plt.ylabel('Current Amplitude')
    # plt.xlabel('Pulse Number')
    # plt.title('Current FSST')
    # plt.figure()
    # plt.plot(df['Pulse'], df['Transcond Amp [S]'], label='Transconductance Amp [S]')
    # plt.xlabel('Pulse Number')
    # plt.ylabel('Amplitude')
    # plt.title('Transconductance Amplitudes')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
#tools.display_dataframe_to_user('Pulse-by-Pulse 1st Harmonic & Transconductance', df)


if __name__ == "__main__":
    gamma_value =  [1e-8, 5e-9]

    gamma_values = [1e-8, 5e-9]
    batch_size = 2
    beta = 5e-5
    #scale_factor_list = [0.4, 0.5] 
    #bias_list = [0.2, 0.3, 0.4, 0.5, 0.6]
    #good scale factor is 0.4
    scale_factor_list = [0.1] 
    scale_factor = 0.1
    bias_list = [0.2]    
    bias = 0.3
    sim_params = SimulationParameters(scale_factor, bias, batch_size, beta, gamma_value)
    main_setup(
        sim_params,
        process_id = None
    )