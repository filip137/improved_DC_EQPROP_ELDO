#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import shutil
import os

"""
Created on Fri Jun 21 15:02:00 2024

@author: filip


eldo supporting functions
"""
def create_output_directory(output_dir_name):
    """
    Creates an output directory if it doesn't already exist.

    Parameters:
    output_dir_name (str): The name of the output directory to create.

    Returns:
    str: The path of the created or existing directory.
    """
    try:
        os.makedirs(output_dir_name, exist_ok=True)
        print(f"Directory '{output_dir_name}' created successfully.")
    except Exception as e:
        print(f"Error creating directory '{output_dir_name}': {e}")
    return output_dir_name



def delete_output_directory(output_dir_name):
    """
    Deletes the output directory and all its contents.

    Parameters:
    output_dir_name (str): The name of the output directory to delete.
    """
    try:
        shutil.rmtree(output_dir_name)
        print(f"Directory '{output_dir_name}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting directory '{output_dir_name}': {e}")

def extract_voltage_sources_with_literal_param(netlist):
    lines = netlist.split('\n')  # Split the netlist into lines
    source_connections = []
    
    # Scan for voltage sources
    for line in lines:
        if line.strip().startswith('V'):  # Identify voltage source lines
            parts = line.split()  # Split line into parts
            source_name = parts[0]
            node_pos = parts[1]
            node_neg = parts[2]
            dc_value = parts[-1]  # Assuming the DC value is part of the description
            if dc_value.startswith('DC'):
                dc_value = dc_value.split()[1]  # Extract the parameter name from the DC value
            source_connections.append((dc_value, node_pos, node_neg))
    
    return source_connections

def create_resistor_list_eldo(input_file_path, save_as_new=True):
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()
    resistors_list = []
    resistances = {}
    
    # Extract the resistance values from the .PARAM lines
    for line in lines:
        if line.strip().startswith('.PARAM'):
            params = line.split()
            for param in params:
                if 'RES' in param:
                    key, val = param.split('=')
                    resistances[key] = val.strip()
                    

    # Parse each line to extract resistor details from ELDO format
    for line in lines:
        if line.strip().startswith('R'):
            parts = line.split()
            name = parts[0]
            node1 = parts[1]
            node2 = parts[2]
            resistance_key = parts[3]
            if "RESL" in resistance_key:
                res_type = "load"
            else:
                res_type = "memristor"
            # Looking up the resistance value using the key
            resistance_value = resistances.get(resistance_key, "Unknown")

            # Creating the dictionary for the current resistor
            resistor_dict = {
                "name": name,  # Resistor label like R5, R4...
                "node voltages": (node1, node2),
                "resistance_key" : resistance_key,
                "node_voltage_ter1": None,
                "node_voltage_ter2": None,
                "current": "Unknown",  # Maybe useful for later
                "res_type": res_type,
                "resistance": resistance_value  # Actual resistance value assigned
            }
            resistors_list.append(resistor_dict)

    return resistors_list

def create_resistor_value_dict(resistors_list):
    resistor_value_dict = {}

    # Extract the resistance values and keys from the resistors_list
    for resistor in resistors_list:
        resistance_key = resistor.get('resistance_key')
        resistance_value = resistor.get('resistance')
        
        if resistance_key and resistance_value:
            resistor_value_dict[resistance_key] = resistance_value

    return resistor_value_dict

def resistor_voltage_array(resistors_list):
    
    
    dtype = [('resistor', 'U10'), ('voltage1', 'float32'), ('voltage2', 'float32')]
    structured_data = []
    
    for resistor in resistors_list:
        # Extract name and voltages, assume some default values if necessary
        name = resistor['resistance_key']
        voltage1 = resistor.get('node_voltage_ter1', 0.0)  # Default voltage 0.0 if not available
        voltage2 = resistor.get('node_voltage_ter2', 0.0)  # Default voltage 0.0 if not available
        
        # Append a tuple for each resistor to the list
        structured_data.append((name, voltage1, voltage2))
    
    # Create a structured array from the list of tuples
    resistor_array = np.array(structured_data, dtype=dtype)
    return resistor_array


def update_resistor_list(resistors_list, node_voltage_dict):
    # Iterate over each resistor in the list
    for resistor in resistors_list:
        # Extract the node voltages from each resistor
        node1, node2 = resistor['node voltages']

        # Fetch the voltage values from the node_voltage_dict
        # and update the resistor's node_voltage_values if the node exists in node_voltage_dict            
        voltage1 = node_voltage_dict.get(node1, "Unknown")  # Defaults to "Unknown" if node is not found
        voltage2 = node_voltage_dict.get(node2, "Unknown")  # Defaults to "Unknown" if node is not found
        if node1 == "0":
            voltage1 = 0
        if node2 == "0":
            voltage2 = 0
        
        # Update the node_voltage_values in the resistor dictionary
        resistor['node_voltage_ter1'] = voltage1
        resistor['node_voltage_ter2'] = voltage2

    # Optionally return the updated list if needed for further operations
    return resistors_list



def store_accumulated_resistances(resistance, epoch, file_name):
    """
    Store the accumulated resistances at the end of each epoch.
    
    Parameters:
    - resistance: The resistance value to be stored.
    - epoch: The current epoch number.
    - file_path: The path to the file where the resistance values are saved.
    """
    # Check if the directory exists, if not, create it
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    
    # Open the file and append the resistance value for the current epoch
    with open(file_name, 'a') as f:
        f.write(f"Epoch {epoch}: {resistance}\n")

def create_node_voltage_dict(resistors_list):
    node_voltage_dict = {}

    # Extract unique node voltages and initialize their values
    for resistor in resistors_list:
        for node_voltage in resistor['node voltages']:
            # Initialize each unique node voltage with a placeholder for its future voltage value
            # Exclude the key '0'
            if node_voltage != '0' and node_voltage not in node_voltage_dict:
                node_voltage_dict[node_voltage] = None  

    return node_voltage_dict

def initialize_res(resistor_value_dict, low_bound = 10, up_bound = 1000, uni_res = 100, mode1 = "random"):
    
    
    numb_of_res = len(resistor_value_dict)
    
    if mode1 == "random":
        res_values = np.random.uniform(low_bound, up_bound, numb_of_res)
        for i, key in enumerate(resistor_value_dict.keys()):
            resistor_value_dict[key] = res_values[i]
    if mode1 == "uniform":
        res_value = uni_res
        for key in resistor_value_dict.keys():
            resistor_value_dict[key] = res_value
    if mode1 == "custom":
        res_values = uni_res
        for i, key in enumerate(resistor_value_dict.keys()):
            resistor_value_dict[key] = res_values[i]
    
    

def loss_function(Y_vec, node_voltages, output_nodes):
    #note that Y_vec must have correspond to the results in the same order as the node_voltages or the output do
    losses = {}

    # Check if Y_vec is a list or numpy array and has the same length as outputs
    if not isinstance(Y_vec, (list, np.ndarray)):
        raise ValueError("Y_vec must be a list or numpy array.")
    if len(Y_vec) != len(output_nodes):
        raise ValueError("Y_vec and outputs must have the same length.")

    for i, output in enumerate(output_nodes):
        try:
            outputV = node_voltages[output]
            targetV = Y_vec[i]
            loss = outputV - targetV  # measured - target
            losses[output] = loss
        except KeyError:
            print(f"Error: No node named {output}")
            losses[output] = np.nan  # Use np.nan to handle errors but keep the array operations valid

    # Return the dictionary of losses
    return losses


def loss_function_xor(Y_vec, node_voltages, output_nodes):
    #note that Y_vec must have correspond to the results in the same order as the node_voltages or the output do
    losses = {}

    # Check if Y_vec is a list or numpy array and has the same length as outputs
    # if not isinstance(Y_vec, (list, np.ndarray)):
    #     raise ValueError("Y_vec must be a list or numpy array.")
    # if len(Y_vec) != len(output_nodes):
    #     raise ValueError("Y_vec and outputs must have the same length.")
        
    pos_output = output_nodes[0]
    pos_outputV = node_voltages[pos_output]
    neg_output = output_nodes[1]
    neg_outputV = node_voltages[neg_output]
    
    true_output = pos_outputV - neg_outputV
    
    if true_output > 0.5:
        pred_output = 1
    else:
        pred_output = 0
    
        
    
    pos_loss = Y_vec - true_output #measured - target
    neg_loss = -pos_loss
    losses[output_nodes[0]] = float(pos_loss)
    losses[output_nodes[1]] = float(neg_loss)
    # Return the dictionary of losses
    
    return losses


def loss_function_moon(Y_vec, node_voltages, output_nodes, boundary):
    #note that Y_vec must have correspond to the results in the same order as the node_voltages or the output do
    losses = {}

    # Check if Y_vec is a list or numpy array and has the same length as outputs
    # if not isinstance(Y_vec, (list, np.ndarray)):
    #     raise ValueError("Y_vec must be a list or numpy array.")
    # if len(Y_vec) != len(output_nodes):
    #     raise ValueError("Y_vec and outputs must have the same length.")
        
    pos_output = output_nodes[0]
    pos_outputV = node_voltages[pos_output]
    neg_output = output_nodes[1]
    neg_outputV = node_voltages[neg_output]
    
    true_output = pos_outputV - neg_outputV #this is okay

    if true_output > boundary:
        pred_output = 1
    else:
        pred_output = 0
    
        
    
    pos_loss = Y_vec - true_output #Here Y_vec will be 0 or 1
    neg_loss = -pos_loss # negative loss will be the minus of positive loss
    losses[output_nodes[1]] = float(pos_loss)
    losses[output_nodes[0]] = float(neg_loss)
    # Return the dictionary of losses
    
    return losses



def predicted_value(node_voltages, output_nodes, boundary):
    pos_output = output_nodes[0]
    pos_outputV = node_voltages[pos_output]
    neg_output = output_nodes[1]
    neg_outputV = node_voltages[neg_output]
    
    true_output = pos_outputV - neg_outputV

    if true_output > boundary:
        pred_output = 1
    else:
        pred_output = 0
        
    return pred_output


def diff_of_outputs(node_voltages, output_nodes):
    pos_output = output_nodes[0]
    pos_outputV = node_voltages[pos_output]

    neg_output = output_nodes[1]
    neg_outputV = node_voltages[neg_output]

    true_output = pos_outputV - neg_outputV


    return true_output






def calculate_accuracy(pred_outputs, true_outputs):
    correct_count = 0
    total = len(pred_outputs)  # Assuming both lists are the same length

    # Iterate through both lists and count matches
    for pred, true in zip(pred_outputs, true_outputs):
        if pred == true:
            correct_count += 1

    # Calculate accuracy
    accuracy = correct_count / total * 100  # Multiply by 100 to get percentage
    return accuracy



def voltage_values(node_voltages, output_nodes):
    pos_output = output_nodes[0]
    pos_outputV = node_voltages[pos_output]
    neg_output = output_nodes[1]
    neg_outputV = node_voltages[neg_output]
    
        
    return pos_outputV, neg_outputV




def create_inudge_dict(losses, node_to_inudge, beta, amp):
    inudge_dict = {}
    
    
    # Check if losses is None
    if losses is None:
    # Set all inudge values to 0
        for inudge in node_to_inudge.values():
            inudge_dict[inudge] = 0

    # Iterate through the inudge_map dictionary
    else:
        for output, inudge in node_to_inudge.items():
            try:
                loss = losses[output]
                inudge_dict[inudge] = beta * loss * amp
            except KeyError:
                print(f"Error: No node named {output}")
                inudge_dict[inudge] = np.nan  # Use np.nan to handle errors but keep the array operations valid

    # Return the inudge dictionary
    return inudge_dict

def update_inudge_dict(inudge_dict, losses, beta):    
    # Check if losses is None
    if losses is None:
    # Set all inudge values to 0
        for output, i_value in inudge_dict.items():
            inudge_dict[output] = 0

    # Iterate through the inudge_map dictionary
    else:
        for output, i_value in inudge_dict.items():
            try:
                loss = losses[output]
                inudge_dict[output] = beta * loss
            except KeyError:
                print(f"Error: No node named {output}")
                inudge_dict[output] = np.nan  # Use np.nan to handle errors but keep the array operations valid

    # Return the inudge dictionary
    return inudge_dict


def initialize_inudge_dict(node_to_inudge):
    inudge_dict = {}
    for output, inudge in node_to_inudge.items():
        inudge_dict[inudge] = 0

    # Return the inudge dictionary
    return inudge_dict


# def create_vnudge_dict(losses, node_to_vnudge, beta):
#     vnudge_dict = {}
    
    
#     # Check if losses is None
#     if losses is None:
#     # Set all inudge values to 0
#         for inudge in node_to_inudge.keys():
#             inudge_dict[inudge] = 0

#     # Iterate through the inudge_map dictionary
#     else:
#         for output, vnudge in node_to_vnudge.items():
#             try:
#                 loss = losses[output]
#                 vnudge_dict[ouput] = -beta * loss
#             except KeyError:
#                 print(f"Error: No node named {output}")
#                 vnudge_dict[vnudge] = np.nan  # Use np.nan to handle errors but keep the array operations valid

#     # Return the inudge dictionary
#     return vnudge_dict





def create_inudge_dict_const(losses, node_to_inudge, beta, inj_curr):
    inudge_dict = {}
    
    
    # Check if losses is None
    if losses is None:
    # Set all inudge values to 0
        for inudge in node_to_inudge.keys():
            inudge_dict[inudge] = 0

    # Iterate through the inudge_map dictionary
    else:
        for output, inudge in node_to_inudge.items():
            try:
                loss = losses[output]
                inudge_dict[inudge] = inj_curr * np.sign(loss)
            except KeyError:
                print(f"Error: No node named {output}")
                inudge_dict[inudge] = np.nan  # Use np.nan to handle errors but keep the array operations valid

    # Return the inudge dictionary
    return inudge_dict





# def update_inudge_values(node_to_inudge, outputs, losses, beta):
#     inudge_dict = {}

#     for node, loss in zip(outputs, losses):
#         inudge_key = node_to_inudge.get(node)
#         if inudge_key:
#             i_nudge = np.round(beta * loss, 8)
#             print(f"I nudge is: {i_nudge}")
#             inudge_dict[inudge_key] = i_nudge
    
#     return inudge_dict



#this currently doesn't work
def create_node_to_inudge_map(sample):
    node_to_inudge = {}
    with open(sample, 'r') as file:
        for line in file:
            if 'isource' in line:
                parts = line.split()
                # The node might be formatted like '(nodeX)', so we need to remove parentheses
                node = parts[2].strip('()')  # Strips off any parentheses
                inudge_key = next((part.split('=')[1] for part in parts if part.startswith('dc=')), None)
                if inudge_key:
                    node_to_inudge[node] = inudge_key
    return node_to_inudge

def create_voltage_source_to_value_dict(input_nodes_to_voltage_sources, voltage_source_values):
    voltage_source_to_value = {}
    
    for node, voltage_source in input_nodes_to_voltage_sources.items():
        if voltage_source in voltage_source_values:
            voltage_source_to_value[voltage_source] = voltage_source_values[voltage_source]
        else:
            print(f"Warning: Voltage source {voltage_source} not found in voltage_source_values dictionary.")
    
    return voltage_source_to_value

def calc_cond_update(voltage_matrix_f, voltage_matrix_n, gamma, beta, amp):
    resistor_names = voltage_matrix_f['resistor']
    deltaV_f = voltage_matrix_f['voltage1'] - voltage_matrix_f['voltage2']
    deltaV_n = voltage_matrix_n['voltage1'] - voltage_matrix_n['voltage2']
    cond_update_values = - gamma / beta * (deltaV_n ** 2 - deltaV_f ** 2) ##negative update value for cond

    cond_update = {}

    # Iterate through each element and update only if the resistor name starts with 'RES'
    for i in range(len(resistor_names)):
        if resistor_names[i].startswith('RESL'):
            cond_update[resistor_names[i]] = 0  # Set to 0 if the resistor name starts with 'RESL'
        elif resistor_names[i].startswith('R_') or resistor_names[i].startswith('fet'):
            if not np.isnan(cond_update_values[i]):
                cond_update[resistor_names[i]] = cond_update_values[i]
        else:
            cond_update[resistor_names[i]] = np.nan  # Use NaN or another placeholder to signify "unchanged" initially
            
            
    for res in cond_update.keys():
        if res.startswith("R_1"):
            cond_update[res] /= amp
            
    return cond_update









def accumulate_cond_update(accumulated_updates, cond_update):
    for key, value in cond_update.items():
        if key in accumulated_updates:
            accumulated_updates[key] += value  # Accumulate the update
        else:
            accumulated_updates[key] = value  # Initialize the key if it's missing
    return accumulated_updates

def update_resistor_value_dict(resistor_value_dict, cond_update):
    for key, resistance in resistor_value_dict.items():
        # Skip updating if resistance is zero or negative to avoid division by zero
        if resistance <= 0:
            #print(f"Warning: Resistance for {key} is non-positive, skipping update.")
            continue
        
        try:
            # Calculate conductance and updated conductance
            cond_value = 1 / resistance
            cond_upd = cond_update[key]
            new_cond = cond_value + cond_upd
            new_res = 1 / new_cond
            # Avoid division by zero or negative conductance
            # if new_cond <= 0:
            #     print(f"Warning: New conductance for {key} is non-positive, skipping update.")
            #     new_res = 10e7
            # else:
            #     new_res = 1 / new_cond
                
            # Ensure the new resistance is within a reasonable range
            if new_res < 0 or new_res > 10e4:
                new_res = 10e4
             #   print(f"Warning: New resistance for {key} is out of range, keeping the original resistance.")
                
            # Update the resistor value
            resistor_value_dict[key] = new_res
        
        except KeyError:
            print(f"Warning: No conductance update found for {key}, skipping update.")
    
    return resistor_value_dict



def accumulate_resistance_values(iteration_resistances, accumulated_resistances):
    for key, value in iteration_resistances.items():
        if key in accumulated_resistances:
            accumulated_resistances[key].append(value)
        else:
            accumulated_resistances[key] = [value]
            
def accumulate_inudge_values(i_nudge_dict, accumulated_inudge_values):
    for key, value in i_nudge_dict.items():
        if key in accumulated_inudge_values:
            accumulated_inudge_values[key].append(value)
        else:
            accumulated_inudge_values[key] = [value]
        

def calculate_single_sse(losses):
    """
    Calculate the single SSE value from a dictionary of losses.

    Parameters:
    losses (dict): A dictionary where keys are labels and values are the losses.

    Returns:
    float: The calculated SSE value.
    """
    sse = 0
    for loss in losses.values():
        sse += (loss ** 2)
    root_sse = np.sqrt(sse)
    return root_sse

