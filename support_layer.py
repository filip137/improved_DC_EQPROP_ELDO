#############
## SUPPORT FUNCTIONS FOR THE LAYER CLASS APPROACH ##

import subprocess
import time
import re
import signal
import os
import sys





def parse_aex_file(filename, start_index, end_index):
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
            if stripped_line.startswith("*V("):
                # Extract node name and value using regex
                parts = stripped_line.split()
                node_name = parts[0][3:-1]
                node_value = float(parts[2])
                parsed_data[node_name] = node_value
            if current_index >= end_index:
                break

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