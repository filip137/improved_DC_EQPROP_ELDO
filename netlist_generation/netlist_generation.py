import sys
sys.path.insert(1, "/home/filip/simulations/improved_simulation_functions/electronics")

from electronics import amplifiers
from amplifiers import dicts
from electronics.sources.sources_dict import pmos_cs, self_biased_nmos_cs
from electronics.synapses.synapses_dict import synapses_dict
from layer_class import *
from simulation_parameters import SimulationParameters
from layers_initialization import initialize_network_layers
from datetime import datetime
from support_layer import extract_all_nodes_voltages



class netlist_builder():
    def __init__(self, layers, simulation_parameters):
        
        self.layers = layers
        self.neuron = simulation_parameters.neuron
        self.amplifier = simulation_parameters.amplifier
        self.freq = simulation_parameters.freq
        self.simulation_type = simulation_parameters.simulation_type
        self.network_parameters_for_netlist = simulation_parameters.network_parameters_for_netlist
        self.libraries = ".LIB /home/filip/CMOS130/corners.eldo \n"
        self.cs_bias = simulation_parameters.cs_bias
        
        self.bidir_dict = dicts[simulation_parameters.amplifier]
        
        
    def extract_subcircuits(self):
        subcircuit_lines = []
        if self.neuron == "amp_ss":
            lines = self.bidir_dict['SUBCIRCUIT']
            subcircuit_lines.append(lines)
        if self.cs_bias:  
            self_biased_lines = self_biased_nmos_cs['SUBCIRCUIT']
            pmos_cs_lines = pmos_cs['SUBCIRCUIT']
            subcircuit_lines.append(self_biased_lines)
            subcircuit_lines.append(pmos_cs_lines)
        return subcircuit_lines
        
        
    def extract_cmos_params(self):
        cmos_params = []
        if self.neuron == "amp_ss":
            lines = self.bidir_dict['PARAMS']
            cmos_params.append(lines)
        
        
        if self.cs_bias:  
            self_biased_lines = self_biased_nmos_cs['PARAMS']
            pmos_cs_lines = pmos_cs['PARAMS']
            cmos_params.append(self_biased_lines)
            cmos_params.append(pmos_cs_lines)
        cmos_params.append(synapses_dict["PARAMS"])
        return cmos_params
        
        
    def extract_connections(self, layers):
        extracted_connections = []
        for layer in layers:
            lines = layer.connections  # Assuming 'connections' is already a list
            extracted_connections.extend(lines)  # Use 'extend' instead of 'append' to add all elements of 'lines'
        return extracted_connections

    # def extract_subcircuits(layers):
    #     extracted_subcircuits = []
    #     for layer in layers:
    #         if layer.subcircuits != None:
    #             lines = layer.subcircuits  # Assuming 'connections' is already a list
    #             extracted_subcircuits.extend(lines)  # Use 'extend' instead of 'append' to add all elements of 'lines'
    #     return extracted_subcircuits

    def extract_layer_parameters(self, layers):
            # ampv = self.ampv
            # ampc = self.ampc
        layer_parameters = []
            # Add parameters from each layer with `.PARAM` prefix
        for layer in layers:
            try:
                if type(layer.parameters) == dict: 
                    for key, value in layer.parameters.items():
                        line = f".PARAM {key}={value}"
                        layer_parameters.append(line)
                elif type(layer.parameters) == list:
                    line = layer.parameters
                    layer_parameters.append(line)
                    
            except:
                #print(f"Layer {type(layer).__name__} has no parameters")
                continue
            # Add other parameters with `.PARAM` prefix
            # all_parameters.extend([f".PARAM {item}" for item in ampv])
            # all_parameters.extend([f".PARAM {item}" for item in ampc])
            
        return layer_parameters



    def extract_network_parameters(self):
            # ampc = self.ampc
        network_parameters = []
            # Add parameters from each layer with `.PARAM` prefix

        for key, value in self.network_parameters_for_netlist.items():
            line = f".PARAM {key}={value}"
            network_parameters.append(line)
            # Add other parameters with `.PARAM` prefix
            # all_parameters.extend([f".PARAM {item}" for item in ampv])
            # all_parameters.extend([f".PARAM {item}" for item in ampc])
            
        return network_parameters


    def build_netlist(self, file_name, transcon_calc, fet_identifiers = None, printfile = None):
            
            #needs frequency, neuron, file_name, all_nodes
            
             
            # freq = self.freq
            # neuron = self.neuron
            # file_name = self.new_sample_file
            # all_nodes = self.all_nodes
            # parameter_lines = self.extract_parameters()
            #parameter_lines.append(PARAMS["amp_ss1"])
            
            
            simulation_type = self.simulation_type 
            freq = self.freq
                
            #parameter_lines.extend(amp_parameter_lines)
            layers = self.layers
            layer_parameter_lines = self.extract_layer_parameters(layers) #these are the parameters from the layers
            network_parameter_lines = self.extract_network_parameters()
            cmos_parameters = self.extract_cmos_params()
            
            
            
            network_description = self.extract_connections(layers) #these are the connections which so far are generated in the layers
            subcircuits = self.extract_subcircuits()
            all_nodes = extract_all_nodes_voltages(layers)
            # Get current date and time
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Generate header with current date and time
            header = f"***\n" \
                     f"*** Generated for: eldoD\n" \
                     f"*** Generated on: {current_datetime}\n" \
                     f"*** Design library name: tests\n" \
                     f"*** Design cell name: 4moons\n" \
                     f"*** Design view name: schematic\n" \
                     f".GLOBAL\n"
                     
                     
            
            #not really sure how/if I can avoid doing this
            mid_sect = subcircuits

            counter = 0 
            
            if simulation_type == "FSST":
                vm_list = []
                for node in all_nodes:
                    if counter == 0:
                        vm_list.append(".EXTRACT FSST")
                    vm_node = f"YVAL(V({node}), {freq})"
                    vm_list.append(vm_node)
                    counter += 1
                    if counter == 5:
                        # Append the current line and reset for a new one
                        vm_list.append("\n")
                        counter = 0
               # Join the vm_list into a string, removing unnecessary spaces and ensuring formatting
                vm_string = " ".join(filter(None, vm_list)).replace(" \n.", "\n.")
                # vi_string = " ".join(filter(None, vi_list)).replace(" \n.", "\n.")
                
                
                if transcon_calc:
                    fet1 = fet_identifiers[0]
                    fet2 = fet_identifiers[1]
                    simulation_details = (
                    
                        

                    f".SST FUND1={freq} NHARM1=1\n"
                    ".DC\n"
                    f"{vm_string}\n" ##here also append vi string if it's needed
                    ".OPTION AEX\n"
                    ".OPTION NOASCII\n"
                    ".END\n"
                    )
                #f".AC LIST {freq}\n"
                else: 
                
                    simulation_details = (
                    
                        f".SST FUND1={freq} NHARM1=1\n"
                        f"{vm_string}\n" ##here also append vi string if it's needed
                        ".OPTION AEX\n"
                        ".OPTION NOASCII\n"
                        ".END\n"
                        )
    
            
            if simulation_type == "DC":
                vdc_list = []
                for node in all_nodes:
                    if counter == 0:
                        vdc_list.append(".EXTRACT DC")
                    vdc_node = f"V({node})"
                    vdc_list.append(vdc_node)
                    counter += 1
                    if counter == 5:
                        # Append the current line and reset for a new one
                        vdc_list.append("\n")
                        counter = 0
               # Join the vm_list into a string, removing unnecessary spaces and ensuring formatting
                vdc_string = " ".join(filter(None, vdc_list)).replace(" \n.", "\n.")
                # vi_string = " ".join(filter(None, vi_list)).replace(" \n.", "\n.")
                
                
                
                
                simulation_details = (
                    
                    f".DC\n"
                    f"{vdc_string}\n" ##here also append vi string if it's needed
                    ".OPTION AEX\n"
                    ".OPTION NOASCII\n"
                    ".END\n"
                )
                

                                 
            with open(file_name, 'w') as file:
                file.write(header)
                for line in layer_parameter_lines:
                    file.write(line + '\n')  # Ensure line is a string and add a newline
                for line in network_parameter_lines:
                    file.write(line + '\n')  # Ensure line is a string and add a newline
                cmos_parameters_lines = "".join(cmos_parameters)
                file.write(cmos_parameters_lines) 
                file.write("\n \n ***END OF THE PARAMETER SECTION \n \n \n")
                file.write(self.libraries)
                
                file.write("\n".join(mid_sect))
                
                file.write("\n \n ***END OF THE SUBCIRCUIT SECTION \n \n \n")

                #Include the needed libraries
                for section in network_description:
                    for line in section:
                        file.write(line)  # Write each line from the sections
                file.write(simulation_details)
            print(f"Network description and parameters have been saved to {file_name}.")


