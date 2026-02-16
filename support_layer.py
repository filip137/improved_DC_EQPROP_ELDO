#############
## SUPPORT FUNCTIONS FOR THE LAYER CLASS APPROACH ##

import subprocess, threading, queue, time
import time
import re
import signal
import os
import sys
import datetime
import numpy as np
import json
import shutil
import random
import pandas as pd
import traceback
from io import StringIO
import matplotlib.pyplot as plt

def plot_current(time, current):
    """
    Plot the substrate current (ISUB) versus time.

    Parameters
    ----------
    time : array-like
        Time samples (s).
    current : array-like
        ISUB samples (A).
    """
    plt.figure()
    plt.plot(time, current)
    plt.xlabel("Time (s)")
    plt.ylabel("ISUB (A)")
    plt.title("Drain source Current vs. Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def remove_directory(directory_path):
    try:
        shutil.rmtree(directory_path)  # remove actual variable path
        print(f"Successfully removed directory {directory_path}")
    except FileNotFoundError:
        print(f"Warning: directory not found: {directory_path}")
    except PermissionError:
        print(f"Error: permission denied when removing {directory_path}")
    except OSError as e:
        print(f"Error removing {directory_path}: {e}")





def generate_filename(base_path, extension, identifier=None):
    """
    Generate a filename or folder path with a datetime stamp or a custom identifier.
    
    Args:
        base_path (str): The base path or initial part of the file name.
        extension (str): File extension (use None if you don't want an extension).
        identifier (str): Optional custom identifier to append to the file name.
    
    Returns:
        str: The generated path or filename with a datetime stamp or identifier.
    """
    now = datetime.datetime.now()
    datetime_stamp = now.strftime("%H%M%S")
    
    if extension is None:
        extension = ""  # If extension is None, use an empty string
    
    if identifier:
        path = f"{base_path}_{identifier}{extension}"
    else:
        path = f"{base_path}_{datetime_stamp}{extension}"
    
    return path

def open_and_read_txt_file(filename, seek_position):

    with open(filename, 'r') as file:
        # Determine where to start reading
        if seek_position is None:
            file.seek(seek_position)
        lines = file.readlines()

    # Find all "# TIME" headers in the buffer
    header_idxs = [i for i, line in enumerate(lines) if line.strip().startswith('# TIME')]
    if not header_idxs:
        raise ValueError("No '# TIME' header found in the read segment.")
    last_hdr_i = header_idxs[-2]

    # Parse column names
    header_line = lines[last_hdr_i].lstrip('#').strip()
    col_names = header_line.split()

    # Gather data lines until a blank or '# TEMPERATURE' line is encountered
    data_lines = []
    for line in lines[last_hdr_i + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith('# TEMPERATURE'):
            break
        data_lines.append(line)
    if not data_lines:
        raise ValueError("No data found after the last '# TIME' header in segment.")

    # Load into DataFrame
    df_last = pd.read_csv(
        StringIO(''.join(data_lines)),
        sep=r'\s+',
        header=None,
        names=col_names
    )
    return df_last, col_names


def compute_cosine_similarity(deltaG, W_before, W_after):
    """
    Computes the reduction ratios for the gradient update and the weight matrix.
    
    Parameters:
        deltaG (np.array): Original gradient update.
        deltaG_clipped (np.array): Clipped gradient update.
        W_before (np.array): Weight matrix before the update.
        W_after (np.array): Weight matrix after applying the clipped update and bounds clipping.
    
    Returns:
        dict: A dictionary with keys:
            - 'gradient_reduction_ratio': norm(deltaG_clipped) / norm(deltaG)
            - 'weight_reduction_ratio': norm(W_after) / norm(W_before)
    """
    # Compute norms for the gradient update
    deltag_arr = deltaG.flatten()
    deltaw_arr = (W_before - W_after).flatten()
    
    # Compute the dot product
    dot_product = np.dot(deltag_arr, deltaw_arr)
    
    # Compute the norms of the vectors
    norm_deltag_arr = np.linalg.norm(deltag_arr)
    norm_deltaw_arr = np.linalg.norm(deltaw_arr)
    
    # Avoid division by zero
    epsilon = 1e-10
    cosine_sim = dot_product / ((norm_deltag_arr * norm_deltaw_arr))
    
    return cosine_sim

def create_filenames(output_dir, sample_file, simulation_type, process_id = None):
    """
    Creates a new subfolder in the output directory, generates circuit filenames,
    and returns the full paths for the .cir and .aex files.
    """
    # Step 1: Create a new subfolder
    subfolder_name = generate_filename("my_experiment", extension = None)
    full_subfolder_path = os.path.join(output_dir, subfolder_name)
    if process_id is not None:
        full_subfolder_path = full_subfolder_path + "_" + str(process_id)
        
    os.makedirs(full_subfolder_path, exist_ok=True)
    
    # Step 2: Extract base name from sample_file
    base_sample_file = os.path.basename(sample_file)
    base_sample_name, _ = os.path.splitext(base_sample_file)
    
    # Step 3: Generate filenames
    base_netlist_name = "netlist"
    cir_filename = base_netlist_name + ".cir"
    if simulation_type == "FSST" or simulation_type == "DC":
        result_file = base_netlist_name + ".aex"
    elif simulation_type == "TRAN":
        result_file = generate_filename(base_sample_name, extension=".TXT")
    
    
    
    # Step 4: Construct full paths
    new_sample_file = os.path.join(full_subfolder_path, cir_filename)
    result_file_path = os.path.join(full_subfolder_path, result_file)
    
    return full_subfolder_path, new_sample_file, result_file_path




def extract_all_nodes_voltages(layers):
    drain_source_nodes = []
    gate_nodes_list = []
    for layer in layers:
        if layer.trainable:
            in_node = layer.input_node_list
            out_node = layer.output_node_list
            gate_nodes = layer.gate_node_list
            gate_nodes_list.extend(gate_nodes)
            drain_source_nodes.extend(in_node)
            drain_source_nodes.extend(out_node)
    

    return drain_source_nodes, gate_nodes_list

def extract_free_node_voltages(layers):
    """
    Return drain/source nodes from trainable layers, excluding input nodes.
    """
    drain_source_nodes = []
    gate_nodes_list = []
    for layer in layers:
        if layer.trainable:
            in_node = layer.input_node_list
            out_node = layer.output_node_list
            gate_nodes = layer.gate_node_list
            gate_nodes_list.extend(gate_nodes)
            drain_source_nodes.extend(in_node)
            drain_source_nodes.extend(out_node)

    drain_source_nodes = [
        node for node in drain_source_nodes if not node.startswith("V_IN_0_")
    ]
    return drain_source_nodes, gate_nodes_list

def extract_last_layer_node_voltages(layers):
    """
    Return drain/source nodes from the last trainable layer only.
    """
    for layer in reversed(layers):
        if layer.trainable:
            drain_source_nodes = []
            drain_source_nodes.extend(layer.input_node_list)
            drain_source_nodes.extend(layer.output_node_list)
            return drain_source_nodes
    return []

def parse_aex_file(filename, start_index, end_index, simulation_type = "AC"):
    # Dictionary to store extracted data
    parsed_data = {}

    # Open the file and process line by line
    current_index = 0
    with open(filename, 'r') as file:
        
        for line in file:
            current_index += 1
            if current_index < start_index:
                continue
            stripped_line = line.strip()
            # Skip empty lines
            if line.strip() == "":
                continue

            # Identify lines with the format *V(...)
            if simulation_type == "DC":
                if stripped_line.startswith("*V("):
                    # Extract node name and value using regex
                    parts = stripped_line.split()
                    node_name = parts[0][3:-1].strip("'\"")
                    node_value = float(parts[2])
                    parsed_data[node_name] = node_value
                if current_index >= end_index:
                    break

            if simulation_type == "AC":
                if stripped_line.startswith("*VR("):
                    # Extract node name and value using regex
                    parts = stripped_line.split()
                    node_name = parts[0][4:-1].strip("'\"")
                    node_value = float(parts[2])
                    parsed_data[node_name] = node_value
                if current_index >= end_index:
                    break
                
                
            elif simulation_type == "FSST":
                if stripped_line.startswith("*YVAL("):
                    parts = stripped_line.split()
                    # parts[0] is something like "*YVAL(V(V_OUT_0_1),10MEG)"

                    # Use a local string variable instead of `signal`:
                    signal_str = parts[0]
                    start_idx = signal_str.find('V(') + 2  # position after 'V('
                    end_idx = signal_str.find(')', start_idx)
                    node_name = signal_str[start_idx:end_idx].split(',')[0]
                    # For simplicity, treat parts[-1] as the real value
                    node_value = float(parts[-1])
                    parsed_data[node_name] = node_value

                if current_index >= end_index:
                    break




    return parsed_data


def plot_two_dc_sources(dc1, dc2,
                        label1='Source 1', label2='Source 2',
                        x=None,
                        x_label='Index',
                        y_label='DC value',
                        title=None,
                        unit=1e-5):
    """
    Plot dc1 and dc2 (array-like) on the same axes, scaling values by `unit`
    so that the y-axis is in multiples of that unit (e.g., 1e-5).
    """
    dc1 = np.asarray(dc1, dtype=float)
    dc2 = np.asarray(dc2, dtype=float)
    if x is None:
        x = np.arange(len(dc1))
    else:
        x = np.asarray(x)
    if len(x) != len(dc1) or len(dc1) != len(dc2):
        raise ValueError("dc1, dc2 and x must all have the same length")

    # scale data
    scaled_dc1 = dc1 / unit
    scaled_dc2 = dc2 / unit

    plt.figure()
    plt.plot(x, scaled_dc1, marker='o', label=label1)
    plt.plot(x, scaled_dc2, marker='s', linestyle='--', label=label2)
    plt.xlabel(x_label)
    plt.ylabel(f"{y_label} (×{unit:.0e} A)")
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def delete_file_with_chi_extension(sample_file):
    """
    Given a filename ending with '.cir', this function replaces the extension with '.chi'
    and deletes the resulting file.
    
    Parameters:
        sample_file (str): The original filename with a '.cir' extension.
                           Example: 'new_sample_file.cir'
    """
    # Split the filename into base and extension parts
    base, ext = os.path.splitext(sample_file)
    
    # Check if the original file has the expected '.cir' extension
    if ext.lower() != '.cir':
        print(f"Warning: The input file does not have a '.cir' extension. Proceeding with replacement.")
    
    # Construct the new filename with the '.chi' extension
    chi_file = base + '.chi'
    
    # Check if the file exists before attempting deletion
    if os.path.exists(chi_file):
        try:
            os.remove(chi_file)
            print(f"Deleted file: {chi_file}")
        except Exception as e:
            print(f"Error deleting {chi_file}: {e}")
    else:
        print(f"File {chi_file} does not exist.")



def parse_aex_file_no_end(filename, start_index, simulation_type, voltage_dict_free):
    # Keep track of which keys have been updated
    updated_keys = set()

    current_index = 0
    with open(filename, 'r') as file:
        for line in file:
            current_index += 1

            # Skip lines until we reach the starting index
            if current_index < start_index:
                continue

            stripped_line = line.strip()

            # Break out of the loop if a blank line is encountered
            if stripped_line == "":
                break

            if simulation_type == "DC":
                if stripped_line.startswith("*V("):
                    parts = stripped_line.split()
                    # Extract node name and value
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
                    # Example: "*YVAL(V(V_OUT_0_1),10MEG)"
                    signal_str = parts[0]
                    start_idx = signal_str.find('V(') + 2  # position after 'V('
                    end_idx = signal_str.find(')', start_idx)
                    node_name = signal_str[start_idx:end_idx].split(',')[0]
                    node_value = float(parts[-1])
                    if node_name in voltage_dict_free:
                        voltage_dict_free[node_name] = node_value
                        updated_keys.add(node_name)

    # After processing, ensure that all keys in voltage_dict_free were updated.
    missing_keys = set(voltage_dict_free.keys()) - updated_keys
    if missing_keys:
        raise ValueError(f"Not all keys were updated. Missing keys: {missing_keys}")

    return voltage_dict_free




def parse_aex_file_from_end_timed(filename, n_of_node_voltages, simulation_type, voltage_dict_free, seek_position = None):
    timings = []  # List to store timings for each part.
    start_total = time.perf_counter()
    
    # Part 1: Read the entire file into a list of lines.
    start_read = time.perf_counter()
    with open(filename, 'r') as file:
        file.seek(seek_position)
        lines = file.readlines()
    end_read = time.perf_counter()
    read_time = end_read - start_read
    timings.append(read_time)
    
    # Part 2: Collect the last n_of_node_voltages non-blank lines by iterating backwards.
    start_collect = time.perf_counter()
    selected_lines = []
    count = 0
    for line in reversed(lines):
        if line.strip() == "":
            break
        selected_lines.append(line)
        count += 1
        if count >= n_of_node_voltages:
            break
    end_collect = time.perf_counter()
    collect_time = end_collect - start_collect
    timings.append(collect_time)
    
    # Part 3: Process the selected lines.
    start_process = time.perf_counter()
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
                start_idx = signal_str.find('V(') + 2  # position after 'V('
                end_idx = signal_str.find(')', start_idx)
                node_name = signal_str[start_idx:end_idx].split(',')[0]
                node_value = float(parts[-1])
                if node_name in voltage_dict_free:
                    voltage_dict_free[node_name] = node_value
                    updated_keys.add(node_name)
    end_process = time.perf_counter()
    process_time = end_process - start_process
    timings.append(process_time)
    
    # Part 4: Check for missing keys.
    start_check = time.perf_counter()
    missing_keys = set(voltage_dict_free.keys()) - updated_keys
    if missing_keys:
        raise ValueError(f"Not all keys were updated. Missing keys: {missing_keys}")
    end_check = time.perf_counter()
    check_time = end_check - start_check
    timings.append(check_time)
    
    total_time = time.perf_counter() - start_total
    timings.append(total_time)
    
    # Return the timings list.
    return timings


def parse_aex_file_from_end_offset(
    filename,
    simulation_type,
    voltage_dict_free,
    transcon_calc: bool = False,
    seek_position=None
):
    
    # Part 1: read from seek_position
    with open(filename, 'r') as f:
        f.seek(seek_position or 0)
        lines = f.readlines()
    # total_lines = len(lines)
    # total_chars = sum(len(l) for l in lines)  # total character count
    # total_bytes = sum(len(l.encode('utf-8')) for l in lines)  # total byte size
    
    # print(f"[DEBUG] Read {total_lines} lines, {total_chars} chars, {total_bytes} bytes "
    #   f"from offset {seek_position or 0}")
    # Part 2: grab everything from the end up to the TEMPERATURE line
    selected_lines = []
    for line in reversed(lines):
        if line.lstrip().startswith("TEMPERATURE ="):
            break
        selected_lines.append(line)

    # Prepare optional dict for ISUB currents
    current_dict = {} if transcon_calc else None
    dc_current_dict = {} if transcon_calc else None
    dc_dict = {} if transcon_calc else None

    # Part 3: parse selected_lines
    updated_keys = set()
    for line in reversed(selected_lines):
        stripped = line.strip()
        node = None
        val  = None

        if simulation_type == "DC" and stripped.startswith("*V("):
            parts = stripped.split()
            node  = parts[0][3:-1]
            val   = float(parts[2])

        elif simulation_type == "AC" and stripped.startswith("*VR("):
            parts = stripped.split()
            node  = parts[0][4:-1]
            val   = float(parts[2])

        elif simulation_type == "FSST" and stripped.startswith("*YVAL("):
            parts = stripped.split()
            sig   = parts[0]
            si    = sig.find("V(") + 2
            ei    = sig.find(")", si)
            node  = sig[si:ei].split(",")[0]
            val   = float(parts[-1])

        # update voltage_dict_free if matched
        if node and node in voltage_dict_free:
            voltage_dict_free[node] = val
            updated_keys.add(node)

        if transcon_calc and stripped.startswith("*YVAL(ISUB("):
            # split off the right?hand side of the ?=? 
            lhs, rhs = stripped.split("=", 1)
        
            # extract the node name between ISUB( ... )
            start = lhs.find("ISUB(") + len("ISUB(")
            end   = lhs.find(")", start)
            cur_node = lhs[start:end]
        
            # split the RHS at commas, strip whitespace, ignore empty strings
            vals = [v.strip() for v in rhs.split(",") if v.strip()]
        
            # vals[0] is the left number, vals[1] is the right number
            cur_val = float(vals[1])
        
            current_dict[cur_node] = cur_val
            
        if transcon_calc and stripped.startswith("*ISUB("):
            # split off the right?hand side of the ?=? 
            lhs, rhs = stripped.split("=", 1)
        
            # extract the node name between ISUB( ... )
            start = lhs.find("ISUB(") + len("ISUB(")
            end   = lhs.find(")", start)
            cur_node = lhs[start:end]
        
            # split the RHS at commas, strip whitespace, ignore empty strings
            vals = [v.strip() for v in rhs.split(",") if v.strip()]
        
            # vals[0] is the left number, vals[1] is the right number
            dc_cur_val = float(vals[0])
        
            dc_current_dict[cur_node] = dc_cur_val           
            
        if transcon_calc and stripped.startswith("*V("):
            # split off the right?hand side of the ?=? 
            lhs, rhs = stripped.split("=", 1)
        
            # extract the node name between ISUB( ... )
            start = lhs.find("V(") + len("V(")
            end   = lhs.find(")", start)
            vol_node = lhs[start:end]
        
            # split the RHS at commas, strip whitespace, ignore empty strings
            vals = [v.strip() for v in rhs.split(",") if v.strip()]
        
            # vals[0] is the left number, vals[1] is the right number
            dc_val = float(vals[0])
        
            dc_dict[vol_node] = dc_val
            
    # Part 4: sanity check for voltages only
    missing = set(voltage_dict_free) - updated_keys
    if missing:
        raise ValueError(f"Missing updates for: {missing!r}")

    # Return one or two dicts depending on transcon_calc
    result_dict = {
    "voltages": voltage_dict_free,
    "currents": current_dict,
    "dc_voltages": dc_dict,
    "dc_currents": dc_current_dict}
    return result_dict







def make_param_dict(orig_dict, prefix="V_END_"):
    """
    Prefixes each key in orig_dict with prefix, stripping off the leading 'V_' 
    so you don?t end up with 'V_END_V_?'.
    """
    return { f"{prefix}{k[2:]}": v
             for k, v in orig_dict.items() }






def compute_fourier_coefficients2(
    df,
    f0,
    t0,
    n_harmonics: int = 1,          # kept for API compatibility; we compute the 1st harmonic
    n_periods: int = 4,
    voltage_cols=None,
    transcon_calc: bool = False,
    store_complex_parts: bool = True,  # NEW: also store 2*Re(C1) and 2*Im(C1)
):
    """
    Extended to also collect columns that begin with V(X...) into amp_voltages.
    """
    # --- find time column ---
    time_col = next((c for c in df.columns if c.lower() == 'time'), None)
    if time_col is None:
        raise KeyError("DataFrame must have a 'time' column.")

    # --- select voltage/current columns ---
    if voltage_cols is None:
        voltage_cols = [c for c in df.columns if c != time_col]
    else:
        missing = set(voltage_cols) - set(df.columns)
        if missing:
            raise KeyError(f"Columns not found: {missing}")

    # --- extract time vector once ---
    t_full = df[time_col].values
    if t_full.size < 2:
        raise ValueError("Need at least two samples.")
    dt = np.median(np.diff(t_full))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid or non-positive time step detected.")
    fs = 1.0 / dt

    # --- analysis window length (integer # of cycles) ---
    if f0 <= 0:
        raise ValueError("f0 must be positive.")
    T = 1.0 / f0
    N_cycle = int(round(T * fs))
    if N_cycle <= 0:
        raise ValueError("Sampling rate too low for the requested f0.")
    N = n_periods * N_cycle

    # starting index
    i0 = 0 if t0 <= t_full[0] else int(np.searchsorted(t_full, t0, side="left"))

    # --- outputs ---
    dc_gate_end, dc_gate, ac_gate = {}, {}, {}
    dc_ds_end, dc_ds, ac_ds = {}, {}, {}
    used_columns = []
    amp_voltages = {}  # NEW: for V(X...) voltages

    ac_isub = {} if transcon_calc else None
    dc_isub = {} if transcon_calc else None

    # NEW: explicit real/imag outputs (optional)
    ac_gate_re = {} if store_complex_parts else None
    ac_gate_im = {} if store_complex_parts else None
    ac_ds_re   = {} if store_complex_parts else None
    ac_ds_im   = {} if store_complex_parts else None
    ac_isub_re = {} if (store_complex_parts and transcon_calc) else None
    ac_isub_im = {} if (store_complex_parts and transcon_calc) else None

    for col in voltage_cols:
        used_columns.append(col)
        v = df[col].values

        # ensure we have enough samples
        if i0 + N > v.size:
            raise ValueError(f"Not enough samples in column {col} (need {N}, have {v.size - i0}).")

        seg = v[i0:i0 + N]
        t_seg = t_full[i0:i0 + N]

        # --- Goertzel-like coefficient at f0: C1 = (1/N) * sum(seg * e^{-j 2? f0 t}) ---
        expo = np.exp(-2j * np.pi * f0 * t_seg)
        C1 = np.dot(seg, expo) / N

        amp1 = 2.0 * C1.imag  # legacy: sine coefficient

        if store_complex_parts:
            amp_re = 2.0 * C1.real
            amp_im = 2.0 * C1.imag

        # --- handle ISUB(...) currents ---
        if col.startswith("ISUB"):
            if col.startswith("ISUB(") and col.endswith(")"):
                key = col[len("ISUB("):-1]
            else:
                key = col
            if transcon_calc:
                ac_isub[key] = amp1
                if store_complex_parts:
                    ac_isub_re[key] = amp_re
                    ac_isub_im[key] = amp_im
                dc_isub[key] = float(np.round(np.mean(seg), 6))
            continue

        # --- voltages ---
        last_val = float(np.round(v[-1], 3))

        if col.startswith('V(') and col.endswith(')'):
            vol_name = col[2:-1]
            mid_val = float(np.round(np.mean(seg), 6))

            if vol_name.startswith('VG'):
                dc_gate_end[vol_name] = last_val
                dc_gate[vol_name] = mid_val
                ac_gate[vol_name] = amp1
                if store_complex_parts:
                    ac_gate_re[vol_name] = amp_re
                    ac_gate_im[vol_name] = amp_im

            elif vol_name.startswith('V_'):
                dc_ds_end[vol_name] = last_val
                dc_ds[vol_name] = mid_val
                ac_ds[vol_name] = amp1
                if store_complex_parts:
                    ac_ds_re[vol_name] = amp_re
                    ac_ds_im[vol_name] = amp_im

            elif vol_name.startswith('X'):  # NEW: voltages like V(X...)
                amp_voltages[vol_name] = {
                    "dc_end": last_val,
                    "dc_mid": mid_val,
                    "ac": amp1,
                }
                if store_complex_parts:
                    amp_voltages[vol_name].update({
                        "ac_re": amp_re,
                        "ac_im": amp_im,
                    })

    results = {
        "dc_gate_end": dc_gate_end,
        "dc_gate": dc_gate,
        "ac_gate": ac_gate,
        "dc_ds_end": dc_ds_end,
        "dc_ds": dc_ds,
        "ac_ds": ac_ds,
        "used_columns": used_columns,
        "ac_isub": ac_isub,
        "dc_isub": dc_isub,
        "amp_voltages": amp_voltages,   # NEW
    }

    if store_complex_parts:
        results.update({
            "ac_gate_re": ac_gate_re, "ac_gate_im": ac_gate_im,
            "ac_ds_re":   ac_ds_re,   "ac_ds_im":   ac_ds_im,
            "ac_isub_re": ac_isub_re, "ac_isub_im": ac_isub_im,
        })

    return results





def compute_fourier_components_ac(
    df, f0, t0,
    n_harmonics=1,
    n_periods=4,
    voltage_cols=None
):
    """
    For each VG_* and V_* column, compute only the AC Fourier magnitudes
    via Goertzel-like correlation. Returns two dicts: gate harmonics and ds harmonics.

    Parameters
    ----------
    df : pandas.DataFrame
        Contains a 'time' column and voltage columns.
    f0 : float
        Fundamental frequency in Hz.
    t0 : float
        Start time for analysis window.
    n_harmonics : int
        Number of harmonics (1..n_harmonics) to compute.
    n_periods : int
        Number of fundamental cycles to include.
    voltage_cols : list of str, optional
        Specific columns to process; defaults to all non-time columns.

    Returns
    -------
    gate : dict
        AC magnitude list for VG_* columns: [|C1|, ..., |Ck|].
    ds : dict
        AC magnitude list for V_* columns: [|C1|, ..., |Ck|].
    """
    # find time column
    time_col = next((c for c in df.columns if c.lower() == 'time'), None)
    if time_col is None:
        raise KeyError("DataFrame must have a 'time' column.")

    # select voltage columns
    if voltage_cols is None:
        voltage_cols = [c for c in df.columns if c != time_col]
    else:
        missing = set(voltage_cols) - set(df.columns)
        if missing:
            raise KeyError(f"Columns not found: {missing}")

    # time array and sampling
    t = df[time_col].values
    if t.size < 2:
        raise ValueError("Need at least two samples.")
    dt = np.median(np.diff(t))
    fs = 1.0 / dt

    # analysis window length
    T = 1.0 / f0
    N_cycle = int(round(T * fs))
    N = n_periods * N_cycle
    i0 = 0 if t0 <= t[0] else np.searchsorted(t, t0)

    gate = {}
    ds = {}

    for col in voltage_cols:
        v = df[col].values
        if i0 + N > v.size:
            raise ValueError(f"Not enough samples in column {col}.")
        seg = v[i0:i0 + N]
        t_seg = t[i0:i0 + N]
    
        # compute only the first harmonic
        expo = np.exp(-2j * np.pi * 1 * f0 * t_seg)
        C1 = np.dot(seg, expo) / N
        C1_imag = -C1.imag
        amp1 = 2 * C1_imag
    
        if col.startswith('V('):
            vol_name = col[2:-1]
            if vol_name.startswith('VG'):
                gate[vol_name] = amp1     # single float
            elif vol_name.startswith('V_'):
                ds[vol_name] = amp1       # single float
    
    return gate, ds



def update_input_dict(input_dict, X):
    """
    Updates the values of input_dict with elements from array X.

    Parameters:
    input_dict (dict): Dictionary to be updated.
    X (list or array): Array of values used to update the dictionary.

    Returns:
    dict: Updated input_dict.
    """
    # Check if X has enough elements
    if len(X) < len(input_dict):
        raise ValueError("Array X does not contain enough elements to update all keys in the dictionary.")
    
    # Update the dictionary with values from X
    for i, key in enumerate(input_dict.keys()):
        input_dict[key] = X[i]
    
    return input_dict


def kill_processes(script_path):
    try:
        # Use subprocess to execute the shell script
        result = subprocess.run(["bash", script_path], check=True)
        print(f"Script executed successfully with return code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Script execution failed with return code: {e.returncode}")


def get_eldo_pids(eldo_identifier):
    try:
        # Use a shell command to run 'ps aux' piped to 'grep' with the identifier and exclude the grep command itself
        command = f"ps aux | grep '{eldo_identifier}' | grep -v grep"
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, shell=True)

        # Process the command output to extract PIDs
        processes = result.stdout.strip().split('\n')

        pids = []
        for process in processes:
            if process:  # Ensure the process string is not empty
                parts = process.split()
                pid = parts[1]  # PID is typically the second element in the output
                pids.append(pid)

        # Report the found PIDs
        if pids:
            print(f"Found {len(pids)} process(es) with PIDs: {', '.join(pids)}")
        else:
            print("No processes found with the specified identifier.")

        return pids

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def clear_aex_file(file_path):
    """
    Deletes all lines in the specified .aex file after printing its size.
    
    Parameters:
        file_path (str): Path to the .aex file.
    """
    # Get and print the file size in bytes
    file_size = os.path.getsize(file_path)
    print(f"File size before clearing: {file_size} bytes")
    
    # Open in 'w' mode to clear the file
    with open(file_path, 'w') as file:
        # Optionally, force a flush to disk.
        file.flush()
        os.fsync(file.fileno())
    
    # Give a short delay for the OS to update the file metadata.
    time.sleep(0.05)
    
    # Check the file size after clearing
    new_size = os.path.getsize(file_path)
    print(f"File size after clearing: {new_size} bytes")

def truncate__aex_file(file_path, keep_size=0):
    """
    Truncates the file to `keep_size` bytes from the beginning.
    By default, it truncates to 0 (fully clearing it) but leaves it valid.
    """
    with open(file_path, 'r+') as f:
        # Optionally read or process the current data
        # e.g. data = f.read()
        
        # Move pointer back to start of file
        f.seek(10)
        # Truncate the file to 'keep_size' bytes
        f.truncate()


def kill_process_PID(pids):
    try:
        if not pids:
            print("No eldo processes running")
            return
        
        for pid in pids:
            subprocess.run(['kill', pid])
            print(f"Killed process with a PID {pid}")      
            
    except Exception as e:
        print(f"Error occured {e}")
        
        
        
def start_eldo_simulation(sample_file, output_dir, m_thread, noascii, debug):
    """Starts the Eldo simulation subprocess in interactive mode, ensuring directory exists."""
    try:
        # Manually set the PATH to include the directory where Eldo is located
        os.environ['PATH'] += ':/cao/Softs/cadence/INNOVUS162/bin'
        os.environ['PATH'] += ':/cao/Softs/cadence/SPECTRE191/tools/bin'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Base command for starting Eldo
        eldo_command = ["eldo", sample_file, "-inter"]

        # Conditionally add multi-threading argument
        if m_thread:
            eldo_command += ["-mthread"]

        if noascii:
            eldo_command += ["-noascii"]


        # Specify the output directory
        eldo_command.append("-createoutpath")
        eldo_command.append(output_dir)

        return subprocess.Popen(
            eldo_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True
        )
    except Exception as e:
        print(f"Error starting Eldo simulation: {e}")
        return None



def start_eldo_simulation_drain(sample_file, output_dir, m_thread, noascii, debug):
    """Starts the Eldo simulation subprocess in interactive mode, ensuring directory exists."""
    try:
        # Manually set the PATH to include the directory where Eldo is located
        os.environ['PATH'] += ':/cao/Softs/cadence/INNOVUS162/bin'
        os.environ['PATH'] += ':/cao/Softs/cadence/SPECTRE191/tools/bin'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Base command for starting Eldo
        eldo_command = ["eldo", sample_file, "-inter"]

        # Conditionally add multi-threading argument
        if m_thread:
            eldo_command += ["-mthread"]

        if noascii:
            eldo_command += ["-noascii"]


        # Specify the output directory
        eldo_command.append("-createoutpath")
        eldo_command.append(output_dir)


        proc =  subprocess.Popen(
            eldo_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True
        )     
        
        q = queue.Queue()
        def _drain():
            for line in proc.stdout:
                q.put(line)
        t = threading.Thread(target=_drain, daemon=True)
        t.start()

        return proc, q

    except Exception as e:
        print(f"Error starting Eldo simulation: {e}")
        return None




def signal_handler(sig, frame, process):
    """Signal handler that sends 'QUIT' to the subprocess."""
    send_quit_command(process)
    sys.exit(0)



def send_quit_command(process):
    """Sends a 'QUIT' command to the subprocess."""
    print("Sending 'QUIT' command to the Eldo process...")
    process.stdin.write("QUIT\n")
    process.stdin.flush()
    process.terminate()
    process.wait()
    print("Eldo process terminated.")






def run_simulation_and_wait(eldo_process, simulation_type, q, debug, sim_time = 50e-6):
    if simulation_type == "FSST":
        run_eldo_simulation(eldo_process, debug)
        wait_for_eldos_completion_drain(eldo_process, q, debug)
                   
    elif simulation_type == "TRAN":
        run_trans_and_DC(eldo_process, sim_time, q, debug)
     
    elif simulation_type == "DC":
        run_eldo_simulation(eldo_process, debug)
        wait_for_eldos_completion_drain(eldo_process, q, debug)                   


def plot_tran_voltages(
    eldo_process,
    result_file,
    offset,
    simulation_type,
    debug,
    columns_to_plot=None  # <-- new parameter
):
    """
    Reads simulation results, updates voltage_dict_free, and optionally plots.

    Parameters
    ----------
    eldo_process
    result_file : str
    voltage_dict_free : dict
    offset : int
    simulation_type : {'FSST', 'TRAN'}
    n_of_node_voltages : int
    debug : bool
    columns_to_plot : list of str or None
        If provided and simulation_type=='TRAN', only these columns are plotted.
    """

    if simulation_type == "TRAN":
        # read last transient sweep
        df_last, col_names = open_and_read_txt_file(result_file, offset)

        # set AC-analysis params
        f0 = 1e6
        t0 = 15e-6

        # compute Fourier coefficients
        results = compute_fourier_coefficients2(
            df_last,
            f0,
            t0,
            n_harmonics=1,
            n_periods=4
        )
        dc_gate, ac_gate, dc_ds, ac_ds, used_columns = (
            results["dc_gate_end"],
            results["ac_gate"],
            results["dc_ds_end"],
            results["ac_ds"],
            results["used_columns"]
        )

        # plot only if user provided a list
        if columns_to_plot:
            plot_specified_columns(
                df_last,
                columns_to_plot,
                time_col=None
            )
import math
def plot_specified_columns(df_last, columns_to_plot, results, time_col=None):
    """
    Plot specified columns from df_last against a time axis, using multiple figures.
    Each figure contains at most 2 columns plotted together.
    For each plotted column, display:
      - real part of Fourier coefficient
      - imaginary part
      - phase shift (degrees)

    Handles:
      - ISUB(...) currents   -> uses ac_isub_re / ac_isub_im
      - V(...) voltages      -> uses ac_gate_re / ac_gate_im, ac_ds_re / ac_ds_im
      - V(X...) amp voltages -> uses results["amp_voltages"][<name>]["ac_re"] / ["ac_im"]
    """

    # detect or validate time column
    if time_col is None:
        matches = [c for c in df_last.columns if c.lower() == 'time']
        if not matches:
            raise KeyError("No time column found. Please specify time_col.")
        time_col = matches[0]

    # check that the requested columns exist
    missing = set(columns_to_plot) - set(df_last.columns)
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    # helper: normalize names
    def normalize_col_name(col: str) -> str:
        # ISUB(...) currents -> strip wrapper
        if col.startswith("ISUB(") and col.endswith(")"):
            return col[5:-1]
        # V(...) voltages -> strip wrapper and keep inside
        if col.startswith("V(") and col.endswith(")"):
            return col[2:-1]
        return col

    # split columns into chunks of size 2
    n_cols = len(columns_to_plot)
    n_figures = math.ceil(n_cols / 2)

    for i in range(n_figures):
        cols = columns_to_plot[i*2:(i+1)*2]
        plt.figure(figsize=(8, 5))
        time = df_last[time_col].values

        for col in cols:
            norm_col = normalize_col_name(col)
            re, im = None, None

            # Case 1: check standard Fourier dicts
            if "ac_gate_re" in results and norm_col in results["ac_gate_re"]:
                re = results["ac_gate_re"][norm_col]
                im = results["ac_gate_im"][norm_col]
            elif "ac_ds_re" in results and norm_col in results["ac_ds_re"]:
                re = results["ac_ds_re"][norm_col]
                im = results["ac_ds_im"][norm_col]
            elif "ac_isub_re" in results and norm_col in results["ac_isub_re"]:
                re = results["ac_isub_re"][norm_col]
                im = results["ac_isub_im"][norm_col]

            # Case 2: check amp_voltages (V(X...) case)
            elif "amp_voltages" in results and norm_col in results["amp_voltages"]:
                entry = results["amp_voltages"][norm_col]
                re = entry.get("ac_re")
                im = entry.get("ac_im")

            # Compose label
            if re is not None and im is not None:
                phase_rad = math.atan2(im, re)
                phase_deg = math.degrees(phase_rad)
                label = f"{col}\nRe={re:.3e}, Im={im:.3e}, Phase={phase_deg:.2f}°"
            else:
                label = f"{col} (Fourier N/A)"

            plt.plot(time, df_last[col], label=label)

        plt.xlabel(time_col)
        plt.ylabel("Value")
        plt.title(f"Channels {', '.join(cols)} over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()




def _group_columns_for_tran_plot(df_last):
    """Return (voltage_cols, current_cols) based on column names."""
    cols = list(df_last.columns)
    # time is excluded
    time_cols = [c for c in cols if c.lower() == 'time']
    time_col = time_cols[0] if time_cols else None

    voltage_cols = [c for c in cols if c != time_col and c.startswith('V(') and c.endswith(')')]
    current_cols = [c for c in cols if c != time_col and c.startswith('ISUB')]
    return voltage_cols, current_cols




def read_update(
    eldo_process,
    result_file,
    voltage_dict_free,
    offset,
    simulation_type,
    transcon_calc,
    debug,
    *,
    f0=None,
    t0=None,
    plot_target: str = None,          # NEW: what to plot in TRAN
    columns_to_plot: list = None):   
    
    
    current_dict = None
    dc_voltage_dict = None
    if simulation_type == "FSST":
        transcon_calc = True
        result_dict_output = parse_aex_file_from_end_offset(
                result_file, simulation_type, voltage_dict_free,
                transcon_calc, seek_position=offset)
    elif simulation_type == "TRAN":
        if f0 is None or t0 is None:
            raise ValueError("TRAN requires f0 and t0 as keyword args")

        df_last, _ = open_and_read_txt_file(result_file, offset)
        res = compute_fourier_coefficients2(
            df_last, f0, t0,
            n_harmonics=1, n_periods=1,
            transcon_calc=transcon_calc
        )
        # build result_dict
        result_dict = {
            'dc_gate_end':     res['dc_gate_end'],
            'dc_gate':     res['dc_gate'],
            'ac_gate':     res['ac_gate'],
            'dc_ds_end':       res['dc_ds_end'],
            'dc_ds':       res['dc_ds'],
            'ac_ds':       res['ac_ds'],
            'used_columns':res['used_columns'],
            'voltages':    voltage_dict_free,
            'ac_isub':   (res.get('ac_isub') or {}),
            'dc_isub':   (res.get('dc_isub') or {})
        }
        
        # ---------- NEW: optional plotting (TRAN only) ----------
        
        if plot_target and plot_target.lower() != "none":
            v_cols, i_cols = _group_columns_for_tran_plot(df_last)
            v_cols = ['V(V_IN_0_1)', 'V(V_OUT_0_1)', 'V(V_OUT_1_1)', 'V(V_IN_1_1)', 'V(XI011.NET05)', 'V(XI011.XI1.OUTPUT_CS_1)', 'V(XI011.XI1.OUTPUT_CS_2)']
            v_cols += ['V(XI011.XI2.INPUT_DIFFERENTIAL1)', 'V(XI011.XI2.INPUT_DIFFERENTIAL2)']
            # Plot voltages if v_cols is not None and not empty
            if v_cols:
                plot_specified_columns(df_last, v_cols, res)
        
            # Plot currents if i_cols is not None and not empty
            if i_cols:
                plot_specified_columns(df_last, i_cols, res)
        # update voltage_dict_free with AC drain-source
        for node, coeff in res['ac_ds'].items():
            if node in voltage_dict_free:
                voltage_dict_free[node] = coeff
        # send DC SET commands
        #THIS IS NOW DONE BY A SEPARATE FUNCTION
        # for node, val in res['dc_gate_end'].items():
        #     key = f"W_{node[3:]}"
        #     send_command_to_eldo(eldo_process, f"SET P({key})={round(val,3)}", debug)
        # ds_params = make_param_dict(res['dc_ds_end'], prefix="V_END_")
        # for key, val in ds_params.items():
        #     send_command_to_eldo(eldo_process, f"SET P({key})={val}", debug)
            
            
            # build and return a clean output dict
        result_dict_output = {
        'dc_ds': result_dict['dc_ds'],
        'dc_ds_end':       res['dc_ds_end'],
        'dc_gate_end': result_dict['dc_gate_end'],
        'ac_voltages':        voltage_dict_free,
        'ac_currents':     result_dict['ac_isub'],
        'dc_currents':     result_dict['dc_isub']}
        
    elif simulation_type == "DC":
        transcon_calc = False
        result_dict_output = parse_aex_file_from_end_offset(
                result_file, simulation_type, voltage_dict_free,
                transcon_calc, seek_position=offset)
        
        
    else:
        raise ValueError(f"Unknown simulation_type: {simulation_type!r}")
        
        

        
    return result_dict_output


def set_the_ic_voltages(eldo_process, dc_gate_end_dict, dc_ds_voltage_dict,debug):
    for node, val in dc_gate_end_dict.items():
        key = f"W_{node[3:]}"
        send_command_to_eldo(eldo_process, f"SET P({key})={round(val,6)}", debug)
    ds_params = make_param_dict(dc_ds_voltage_dict, prefix="V_END_")
    for key, val in ds_params.items():
        send_command_to_eldo(eldo_process, f"SET P({key})={val}", debug)




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




def send_command_to_eldo(process, command, debug):
    """Sends a command to the Eldo subprocess, ensuring it's still open."""
    if process.poll() is None:  # None means the process is still running
        if debug: print(f"Sending command: {command}")
        try:
            process.stdin.write(command + "\n")
            process.stdin.flush()
        except Exception as e:
            print(f"Error sending command: {e}")
    else:
        print("Cannot send command, subprocess has terminated.")
        
def set_eldo_simulation(process, mode, input_values, resistor_value_dict,  inudge_dict, debug):
    """Sets the simulation parameters for the Eldo process."""
    try:
        if mode == "free":
            for vol_source, vol in input_values.items():
                eldo_command = f"SET P ({vol_source}) = {vol}"
                send_command_to_eldo(process, eldo_command, debug)
            for inudge, curr in inudge_dict.items():
                eldo_command = f"SET P ({inudge}) = 0"
                send_command_to_eldo(process, eldo_command, debug)
        elif mode == "nudge":
            for inudge, curr in inudge_dict.items():
                eldo_command = f"SET P ({inudge}) = {curr}"
                send_command_to_eldo(process, eldo_command, debug)

        elif mode == "set_resistances":
            for res_key, res_value in resistor_value_dict.items():
                eldo_command = f"SET P ({res_key}) = {res_value}"
                send_command_to_eldo(process, eldo_command, debug)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

    
def wait_for_eldos_completion(process, debug):
    """
    Waits until the specified completion message is found in the process output
    and captures specific lines of interest.
    """
    completion_message = "Eldo interactive runs completed."
    lines_of_interest = []
    capture = False  # Flag to start capturing when DC analysis starts

    while True:
        line = process.stdout.readline().strip()
        if line:
            if debug: print(f"Reading output: {line}")

            # Start capturing after DC analysis line is found
            if "***>Current simulation completed" in line:
                capture = True
            
            # Capture lines if they contain specific keywords
            if capture and ("V_IN_" in line or "V_OUT_" in line):
                lines_of_interest.append(line)

            # Stop capturing after the DC analysis has definitely ended
            if "Eldo interactive runs completed." in line:
                capture = False

            # Check for the completion message
            if completion_message in line:
                if debug: print("Completion message detected.")
                break

    # # Optionally, write the captured lines to a file
    # with open('captured_voltages.txt', 'w') as file:
    #     for line in lines_of_interest:
    #         file.write(line + '\n')

    return lines_of_interest



def wait_for_eldos_completion_drain(process, q, debug):
    """
    Blocks until the Eldo completion message appears in the queue,
    capturing any V_IN_/V_OUT_ lines between the DC-analysis markers.
    """
    completion_message = "Eldo interactive runs completed."
    start_capture_marker = "***>Current simulation completed"
    lines_of_interest = []
    capturing = False

    while True:
        # Block until the next line is available from the reader thread
        raw = q.get()  
        line = raw.strip()

        if debug:
            print(f"Reading output: {line}")

        # Once we see the DC-analysis-complete marker, start collecting
        if start_capture_marker in line:
            capturing = True

        # If we're in capture mode and see a voltage line, stash it
        if capturing and ("V_IN_" in line or "V_OUT_" in line):
            lines_of_interest.append(line)

        # If we see the final completion message, break out
        if completion_message in line:
            if debug:
                print("Completion message detected.")
            break

    return lines_of_interest









def wait_for_eldos_completion_initialization(process, debug):
    """
    Waits until the specified completion message is found in the process output.
    """
    completion_message = ">eldo"

    while True:
        line = process.stdout.readline().strip()
        if line:
            if debug: print(f"Reading output: {line}")
            if completion_message in line:
             #   print("Completion message detected.")
                break

def wait_for_eldos_completion_old(process, debug):
    """
    Waits until the specified completion message is found in the process output.
    """
    completion_message = "Eldo interactive runs completed."

    while True:
        line = process.stdout.readline().strip()
        if line:
            if debug: print(f"Reading output: {line}")
            if completion_message in line:
             #   print("Completion message detected.")
                break


def extract_voltage_from_list(voltage_list, voltage_dict):
    for line in voltage_list:
        _node, eq, value, unit = line.split()
        node = _node.split('(')[1].split(')')[0]
        value = float(value)
        if node in voltage_dict.keys():
            voltage_dict[node] = value
        else:
            pass
    
    return voltage_dict

#new support functions
def extract_voltages(process, node_voltages, debug):
    for node in node_voltages.keys():
        eldo_command = f"PRINT V({node})"
        send_command_to_eldo(process, eldo_command, debug)
        line = read_eldo_output(process, node, debug)
        
        # Regular expression pattern to match the node voltage value
        pattern = rf"{node}\s+([0-9.eE+-]+)"
        match = re.search(pattern, line)
        
        if match:
            # Extracting the matched value from the capturing group
            new_voltage = match.group(1)
            node_voltages[node] = float(new_voltage)  # Update existing dictionary
        elif node in line:
            new_voltage = line.split("=")[1].strip()
            node_voltages[node] = float(new_voltage)  # Update existing dictionary
        else:
            print(f"Error retrieving voltage for {node}: {line}")
    
    return node_voltages  # Returning the updated dictionary is optional, but often useful.
 
def set_synapses(process, synapse_dict, debug):
    """Sets resistance values for the simulation."""
    for syn_key, syn_value in synapse_dict.items():
        eldo_command = f"SET P ({syn_key}) = {syn_value}"
        send_command_to_eldo(process, eldo_command, debug)
    

    


def initialize_synapses(eldo_process, resistive_layers, diode_connected_flash_params, simulation_type, debug):
    #If the simulation is FSST the synapse dict will contain the gate_voltages, which will be set in the beginning of the simulation
    #If the simulation is TRAN the synapse dict will also contain gate voltages, which will be used as an initial condition at the start of the simulation
    if diode_connected_flash_params:
        offset1layer = diode_connected_flash_params['offset1layer']
        offset2layer = diode_connected_flash_params['offset2layer']
        gain = diode_connected_flash_params['gain']
    if simulation_type == "FSST":
        for j, layer in enumerate(resistive_layers):
            layer.update_synapse_dict(diode_connected_flash_params)
            set_synapses(eldo_process, layer.synapse_dict, debug)
    if simulation_type == "TRAN":
        for j, layer in enumerate(resistive_layers):
                #so I need another dict, which will accumulate the deltaG updates and transform them into current pulses
            layer.update_synapse_dict(diode_connected_flash_params)
            set_synapses(eldo_process, layer.synapse_dict, debug)  
    elif simulation_type == "DC":
        for j, layer in enumerate(resistive_layers):
                #so I need another dict, which will accumulate the deltaG updates and transform them into current pulses
            layer.update_synapse_dict(diode_connected_flash_params)
            set_synapses(eldo_process, layer.synapse_dict, debug)  



def update_synapses(eldo_process, resistive_layers, diode_connected_flash_params, simulation_type, debug):
    """
    W - contains the transconductances
    deltaG - contains the changes in the transconductances (gradients)
    synapse_dict - contains the gate voltages 
    
    
    
    FSST: update W -> G -> push synapse_dict
    TRAN: update W -> G -> I -> push deltaI_dict
    """
    #If the simulation is FSST, I calculate the new synapse dict, which contains the new gate voltages to be applied
    #If the simulation is TRAN I also calculate the new gate voltages to be applied, but then I still need to calculate how much charge (or current) I need to supply to the
    #   gate to get these voltages
    if diode_connected_flash_params:
        offset1layer = diode_connected_flash_params['offset1layer']
        offset2layer = diode_connected_flash_params['offset2layer']
        gain = diode_connected_flash_params['gain']
    
    
    
    
    for layer in resistive_layers:
        layer.update_W(mode=None, clip=None)
        layer.update_synapse_dict(diode_connected_flash_params)

        if simulation_type == "TRAN":
            layer.update_deltaI_dict()
            payload = layer.synapse_dict
        elif simulation_type == "FSST":
            payload = layer.synapse_dict
        elif simulation_type == "DC":
            payload = layer.synapse_dict
        set_synapses(eldo_process, payload, debug)






def set_input_voltages(process, input_values, debug):
    """Sets simulation parameters in 'free' mode."""
    for vol_source, vol in input_values.items():
        eldo_command = f"SET P ({vol_source}) = {vol}"
        send_command_to_eldo(process, eldo_command, debug)

def disable_current_sources(process, inudge_dict, debug):
    """Disables current sources in simulation."""
    for inudge in inudge_dict.keys():
        if inudge.startswith("INUDGE"):
            eldo_command = f"SET P ({inudge}) = 0"
            send_command_to_eldo(process, eldo_command, debug)
        elif inudge.startswith("R"):
            eldo_command = f"SET P ({inudge}) = 9999999"
            send_command_to_eldo(process, eldo_command, debug)           
        else:
            continue
            
def set_currents_nudge_mode(process, inudge_dict, debug):
    """Sets current values in 'nudge' mode."""
    for inudge, curr in inudge_dict.items():
        eldo_command = f"SET P ({inudge}) = {curr}"
        send_command_to_eldo(process, eldo_command, debug)


def set_voltages_nudge_mode(process, inudge_dict, target, debug):
    """Sets current values in 'nudge' mode."""
    for inudge, curr in inudge_dict.items():
        
        eldo_command = f"SET P ({inudge}) = {curr}"
        send_command_to_eldo(process, eldo_command, debug)

        
def read_eldo_output(process, stop_here, debug):
    """Reads output from the Eldo subprocess until the prompt appears, storing only the second to last line."""
    last_line = None  # This will store the last line
    second_to_last_line = None  # This will store the second to last line
    
    while True:  # Use a loop to keep reading until the prompt is found
        line = process.stdout.readline().strip()
        if line:
            if debug: print(f"Reading output: {line}")
            if stop_here in line:  # Check for the prompt indicating ready for next command
                break  # Exit the loop when the prompt is detected
    
    return line
def reset_chi_file(process, debug):
    """Runs the Eldo simulation."""
    try:
        send_command_to_eldo(process, "RESET FILES", debug)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
def reset_extract_file(process, debug):
    """Runs the Eldo simulation."""
    try:
        send_command_to_eldo(process, "RESET EXTRACT", debug)
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 
        traceback.print_exc()
        
def run_eldo_simulation(process, debug):
    """Runs the Eldo simulation."""
    try:
        send_command_to_eldo(process, "RUN", debug)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        
        
def run_trans_and_DC(process, sim_time, q, debug):
    """Runs the Eldo simulation."""

    # --- First Simulation (TRANS) ---
    tran_line = f".TRAN 0.01u {sim_time} uic"
    try:
        #start_time = time.time()

        send_command_to_eldo(process, "RESET", debug)
        send_command_to_eldo(process, tran_line, debug)
        run_eldo_simulation(process, debug)
        wait_for_eldos_completion_drain(process, q, debug)


    except Exception as e:
        print(f"An unexpected error occurred during the first simulation: {e}")
        traceback.print_exc()
    
    # --- Second Simulation (DC) ---
    dc_line = f".TRAN 0.1u 0.1u"
    try:
        #start_time = time.time()

        send_command_to_eldo(process, "RESET", debug)
        send_command_to_eldo(process, dc_line, debug)
        run_eldo_simulation(process, debug)
        wait_for_eldos_completion_drain(process, q, debug)



    except Exception as e:
        print(f"An unexpected error occurred during the second simulation: {e}")
        traceback.print_exc()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
