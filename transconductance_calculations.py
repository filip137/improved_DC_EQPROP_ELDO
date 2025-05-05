from support_layer import * 
import numpy as np
import matplotlib.pyplot as plt



#fets identifier should be something like 0_1_1
def calculate_transconductance(process, fets_identifier, aex_file, all_nodes, debug):
    #reset the simulation
    try:
        command = ".OPTION AEX"
        send_command_to_eldo(process, command, debug)
        command = "RESET EXTRACT"
        send_command_to_eldo(process, command, debug)
        fet1 = fets_identifier[0]
        fet2 = fets_identifier[1]
        fet1_name = "XM_" + fet1
        fet2_name = "XM_" + fet2
        #dc_command = f".DC"
        for node in all_nodes:
            part1 = (".EXTRACT DC")
            part2 = f" V({node})"
            full_command = part1 + part2
            send_command_to_eldo(process, full_command, debug)
        #send_command_to_eldo(process, dc_command, debug)
        command = f".EXTRACT DC ISUB({fet1_name}.D) ISUB({fet2_name}.D)"
        send_command_to_eldo(process, command, debug)
        weight1_name = "W_" + fet1
        weight2_name = "W_" + fet2
        #However the printfile command needs to be there even before
        
        #read_command = f".EXTRACT DC ISUB({fet1_name}.S ISUB{fet2_name}.S"
        #send_command_to_eldo(process, read_command, debug)
        weight_1 = np.linspace(-0.5, 0.5, 100)
        weight_2 = np.linspace(-0.5, 0.5, 100)
        synapse_dict = {weight1_name : 0,
                        weight2_name : 0}
        currents_list = []
        # Iterate over weights in parallel
        send_command_to_eldo(process, command, debug)
        for w1, w2 in zip(weight_1, weight_2):
            synapse_dict = {weight1_name: w1,
                            weight2_name: w2}
            set_resistances(process, synapse_dict, debug=True) 
            run_eldo_simulation(process, debug=True)
            wait_for_eldos_completion(process, debug=True)
            currents_list.append(parse_last_isub_values(aex_file))
        
            
        
            vm_list = []
    
        for node in all_nodes:
            part1 = (".EXTRACT FSST ")
            part2 = f"YVAL(V({node}), 1MEG)"
            full_command = part1 + part2
            send_command_to_eldo(process, full_command, debug)
        
        
        process_currents_list(currents_list, weight_1, weight_2)
        
        return currents_list
    except:
        print("You are probably trying to calculate the transconductance of the resistor")
        return

def process_currents_list(current_list, weight_1, weight_2):
    """
    Processes a list of current values, computes transconductance, and plots 
    transconductance versus weight1.
    
    Steps:
      1. Reshape current_list into an array with two columns.
      2. Take the element-wise square root of the current array.
      3. Multiply by a scaling factor to obtain transconductance.
      4. Plot transconductance versus weight1 for both sets of values.
    
    Parameters:
      current_list (list or array-like): A flat list of current values.
      weight_1 (array-like): 1D array of weight1 values (x-axis for the plot).
      weight_2 (array-like): 1D array of weight2 values (not used in this plot).
    
    Returns:
      np.ndarray: The transconductance array.
    """
    # Reshape the current list into an array with two columns.
    cur_arr = np.array(current_list).reshape(-1, 2)
    
    # Take the element-wise square root of the current array.
    # (Ensure that cur_arr contains non-negative values for real square roots.)
    cur_arr_sqrt = cur_arr
    
    # Compute transconductance by multiplying by the scaling factor.
    scal_fac = 1.52e-4
    transcond = np.sqrt(scal_fac * cur_arr_sqrt)
    
    # Plot transconductance versus weight1.
    plt.figure()
    plt.plot(weight_1, transcond[:, 0], label="Transconductance FET first layer")
    plt.plot(weight_1, transcond[:, 1], label="Transconductance FET second layer")
    plt.xlabel("Weight1")
    plt.ylabel("Transconductance")
    plt.title("Transconductance vs Weight1")
    plt.legend()
    plt.show()
    
    return transcond
    
    

def parse_last_isub_values(filename, seek_position=None):
    """
    Extracts the last two current values from the simulation output file,
    where each value is given on lines starting with "*ISUB(".
    
    Expected line format:
        *ISUB(XM_0_1_1.D) =  2.7168E-06
    
    Parameters:
        filename (str): Path to the simulation output file.
        seek_position (int, optional): A file position to seek to before reading.
        
    Returns:
        tuple: A tuple containing the two current values as floats.
    
    Raises:
        ValueError: If fewer than two valid ISUB lines are found.
    """
    # Open the file and read its content
    with open(filename, 'r') as file:
        if seek_position is not None:
            file.seek(seek_position)
        lines = file.readlines()
    
    # Collect the last two lines that start with "*ISUB("
    isub_lines = []
    for line in reversed(lines):
        if line.strip().startswith("*ISUB("):
            isub_lines.append(line)
            if len(isub_lines) >= 2:
                break

    if len(isub_lines) < 2:
        raise ValueError("Not enough ISUB lines found in the file.")
    
    # Reverse to restore original order (if needed)
    isub_lines = list(reversed(isub_lines))
    
    # Extract the current values from the two lines
    currents = []
    for line in isub_lines:
        # Expected format: "*ISUB(XM_0_1_1.D) =  2.7168E-06"
        parts = line.split("=")
        if len(parts) < 2:
            continue
        value_str = parts[1].strip()
        try:
            current_value = float(value_str)
            currents.append(current_value)
        except ValueError:
            raise ValueError(f"Could not convert '{value_str}' to float.")
    
    if len(currents) < 2:
        raise ValueError("Not enough valid ISUB current values found.")
    
    return currents[0], currents[1]



   
# def extract_last_current(print_file):
#     """
#     Extracts the last two numbers from the last non-comment line in the input text.
    
#     Parameters:
#         print_file (str): The path to the file containing the multi-line input.
    
#     Returns:
#         tuple: A tuple of two floats corresponding to the last two numbers.
#                Returns (None, None) if no suitable line is found.
#     """
#     # Open the file and read its content.
#     with open(print_file, 'r') as file:
#         content = file.read()
    
#     # Split the content into lines.
#     lines = content.splitlines()
    
#     # Iterate over lines in reverse order to get the last occurrence.
#     for line in reversed(lines):
#         stripped_line = line.strip()
#         # Skip empty lines and comment lines starting with '#'
#         if not stripped_line or stripped_line.startswith("#"):
#             continue
#         # Split the line into components.
#         parts = stripped_line.split()
#         # Check if there are at least two numbers.
#         if len(parts) >= 2:
#             try:
#                 # Convert the last two parts to floats.
#                 current1 = float(parts[-2])
#                 current2 = float(parts[-1])
#                 return current1, current2
#             except ValueError:
#                 # If conversion fails, skip this line.
#                 continue
#     # Return None if no valid line is found.
#     return None, None


