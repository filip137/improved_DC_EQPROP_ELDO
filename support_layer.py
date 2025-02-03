#############
## SUPPORT FUNCTIONS FOR THE LAYER CLASS APPROACH ##

import subprocess
import time
import re
import signal
import os
import sys
import datetime

def generate_filename(base_path, extension=".cir", identifier=None):
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
    datetime_stamp = now.strftime("%Y%m%d_%H%M%S")
    
    if extension is None:
        extension = ""  # If extension is None, use an empty string
    
    if identifier:
        path = f"{base_path}_{identifier}{extension}"
    else:
        path = f"{base_path}_{datetime_stamp}{extension}"
    
    return path


def create_filenames(network_config):
    """
    Creates a new subfolder in the output directory, generates circuit filenames,
    and returns the full paths for the .cir and .aex files.
    """
    # Step 1: Create a new subfolder
    subfolder_name = generate_filename("my_experiment", extension=None)
    full_subfolder_path = os.path.join(network_config["output_dir"], subfolder_name)
    os.makedirs(full_subfolder_path, exist_ok=True)
    
    # Step 2: Extract base name from sample_file
    base_sample_file = os.path.basename(network_config["sample_file"])
    base_sample_name, _ = os.path.splitext(base_sample_file)
    
    # Step 3: Generate filenames
    cir_filename = generate_filename(base_sample_name, extension=".cir")
    aex_filename = generate_filename(base_sample_name, extension=".aex")
    
    # Step 4: Construct full paths
    new_sample_file = os.path.join(full_subfolder_path, cir_filename)
    aex_file_path = os.path.join(full_subfolder_path, aex_filename)
    
    return full_subfolder_path, new_sample_file, aex_file_path




def extract_all_nodes_voltages(layers):
    all_node_voltages = []
    for layer in layers:
        if layer.type == 'resistive':
            in_node = layer.input_node_list
            out_node = layer.output_node_list
            all_node_voltages.extend(in_node)
            all_node_voltages.extend(out_node)
    
    return all_node_voltages

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



def parse_aex_file_no_end(filename, start_index, simulation_type="AC"):
    # Dictionary to store extracted data
    parsed_data = {}

    # Open the file and process line by line
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

            # Process the line based on the simulation type
            if simulation_type == "DC":
                if stripped_line.startswith("*V("):
                    parts = stripped_line.split()
                    # Extract node name and value
                    node_name = parts[0][3:-1].strip("'\"")
                    node_value = float(parts[2])
                    parsed_data[node_name] = node_value

            elif simulation_type == "AC":
                if stripped_line.startswith("*VR("):
                    parts = stripped_line.split()
                    node_name = parts[0][4:-1].strip("'\"")
                    node_value = float(parts[2])
                    parsed_data[node_name] = node_value

            elif simulation_type == "FSST":
                if stripped_line.startswith("*YVAL("):
                    parts = stripped_line.split()
                    # Example: "*YVAL(V(V_OUT_0_1),10MEG)"
                    signal_str = parts[0]
                    start_idx = signal_str.find('V(') + 2  # position after 'V('
                    end_idx = signal_str.find(')', start_idx)
                    node_name = signal_str[start_idx:end_idx].split(',')[0]
                    node_value = float(parts[-1])
                    parsed_data[node_name] = node_value

    return parsed_data

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
    Deletes all lines in the specified .aex file.
    
    Parameters:
        file_path (str): Path to the .aex file.
    """
    with open(file_path, 'w') as file:
        pass  # Opening in 'w' mode clears the file


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
            stderr=subprocess.STDOUT,
            text=True
        )
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
        
        
# def wait_for_eldos_completion(process, debug):
#     """
#     Waits until the specified completion message is found in the process output.
#     """
#     completion_message = "Eldo interactive runs completed."

#     while True:
#         line = process.stdout.readline().strip()
#         if line:
#             if debug: print(f"Reading output: {line}")
#             if completion_message in line:
#              #   print("Completion message detected.")
#                 break 
        

    
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

    # Optionally, write the captured lines to a file
    with open('captured_voltages.txt', 'w') as file:
        for line in lines_of_interest:
            file.write(line + '\n')

    if debug: print("Captured lines have been written to captured_voltages.txt.")
    return lines_of_interest

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
 
def set_resistances(process, resistor_value_dict, debug):
    """Sets resistance values for the simulation."""
    for res_key, res_value in resistor_value_dict.items():
        eldo_command = f"SET P ({res_key}) = {res_value}"
        send_command_to_eldo(process, eldo_command, debug)
    
 
def set_input_voltages(process, input_values, debug):
    """Sets simulation parameters in 'free' mode."""
    for vol_source, vol in input_values.items():
        eldo_command = f"SET P ({vol_source}) = {vol}"
        send_command_to_eldo(process, eldo_command, debug)

def disable_current_sources(process, inudge_dict, debug):
    """Disables current sources in simulation."""
    for inudge in inudge_dict.keys():
        eldo_command = f"SET P ({inudge}) = 0"
        send_command_to_eldo(process, eldo_command, debug)

def set_currents_nudge_mode(process, inudge_dict, debug):
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

def run_eldo_simulation(process, debug):
    """Runs the Eldo simulation."""
    try:
        send_command_to_eldo(process, "GO", debug)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")