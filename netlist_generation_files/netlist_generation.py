import sys
sys.path.insert(1, "/home/filip/simulations/improved_simulation_functions/electronics")

from electronics import amplifiers
from amplifiers import dicts
from electronics.sources.sources_dict import pmos_cs, self_biased_nmos_cs, improved_self_biased_nmos, improved_pmos_cs
from electronics.synapses.synapses_dict import synapses_dict
from electronics.discharge_elements.discharge_tran import discharge_tran
from .layer_class import BaseLayer, InputLayer, DenseLayer, NonLinearLayer, OutputLayer
from .simulation_parameters import SimulationParameters
from .layers_initialization import initialize_network_layers
from datetime import datetime
from support_layer import extract_all_nodes_voltages



class netlist_builder():
    def __init__(self, layers, simulation_parameters):
        
        self.layers = layers
        self.neuron = simulation_parameters.neuron
        self.amplifier = simulation_parameters.amplifier
        self.freq = simulation_parameters.freq
        self.simulation_time = simulation_parameters.simulation_time
        self.simulation_type = simulation_parameters.simulation_type
        #self.network_parameters_for_netlist = simulation_parameters.network_parameters_for_netlist
        self.libraries = ".LIB /home/filip/CMOS130/corners.eldo \n"
        self.cs_bias = simulation_parameters.cs_bias
        self.transient_params_for_netlist = simulation_parameters.transient_params_for_netlist
        self.fsst_params_for_netlist = simulation_parameters.fsst_params_for_netlist
        
        self.bidir_dict = dicts[simulation_parameters.amplifier]

        
        
    def extract_subcircuits(self):
        subcircuit_lines = []
        if self.neuron:
            lines = self.bidir_dict['SUBCIRCUIT']
            subcircuit_lines.append(lines)
        if self.cs_bias == "self_biased":  
            self_biased_lines = improved_self_biased_nmos['SUBCIRCUIT']
            #pmos_cs_lines = pmos_cs['SUBCIRCUIT']
            #the sources are actually the same only one is adapted to use in the transient analysis
            if self.simulation_type == "TRAN":
                pmos_cs_lines = improved_pmos_cs['SUBCIRCUIT']
            elif self.simulation_type == "FSST":
                pmos_cs_lines = pmos_cs['SUBCIRCUIT'] #
            subcircuit_lines.append(self_biased_lines)
            subcircuit_lines.append(pmos_cs_lines)
        if self.simulation_type == "TRAN":
                discharge_lines = discharge_tran['SUBCIRCUIT']
                subcircuit_lines.append(discharge_lines)
        return subcircuit_lines
        
       
    def extract_cmos_params(self):
        cmos_params = []
        if self.neuron:
            lines = self.bidir_dict['PARAMS']
            cmos_params.append(lines)
        
        
        if self.cs_bias:  
            self_biased_lines = improved_self_biased_nmos['PARAMS']
            if self.simulation_type == "TRAN":
                pmos_cs_lines = improved_pmos_cs['PARAMS']
            elif self.simulation_type == "FSST":
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
        """
        Pull out `layer.parameters`, which may be:
         - a dict
         - a list of dicts
         - nested lists of dicts/strings/None
         - None or other types (skipped)
        and return a flat list of strings like ".PARAM key=value".
        """
        layer_parameters = []
    
        def _process(entry):
            # Skip Nones
            if entry is None:
                return
    
            # Dict => .PARAM lines
            if isinstance(entry, dict):
                for k, v in entry.items():
                    layer_parameters.append(f".PARAM {k}={v}")
    
            # String => include directly (in case you ever store raw lines)
            elif isinstance(entry, str):
                layer_parameters.append(entry)
    
            # List => recurse
            elif isinstance(entry, list):
                for sub in entry:
                    _process(sub)
    
            # Everything else => ignore
            else:
                return
    
        for layer in layers:
            try:
                _process(layer.parameters)
            except Exception:
                # optionally log, but continue on errors
                continue
    
        return layer_parameters



    def extract_network_parameters(self):
            # ampc = self.ampc
        network_parameters = []
            # Add parameters from each layer with `.PARAM` prefix
        if self.simulation_type == "TRAN":
            for key, value in self.transient_params_for_netlist.items():
                line = f".PARAM {key}={value}"
                network_parameters.append(line)

        elif self.simulation_type == "FSST":
            for key, value in self.fsst_params_for_netlist.items():
                line = f".PARAM {key}={value}"
                network_parameters.append(line)
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
            simulation_time = self.simulation_time   
            #parameter_lines.extend(amp_parameter_lines)
            layers = self.layers
            layer_parameter_lines = self.extract_layer_parameters(layers) #these are the parameters from the layers
            network_parameter_lines = self.extract_network_parameters()
            cmos_parameters = self.extract_cmos_params()
            
            
            
            network_description = self.extract_connections(layers) #these are the connections which so far are generated in the layers
            subcircuits = self.extract_subcircuits()
            drain_source_nodes, gate_nodes = extract_all_nodes_voltages(layers) #drain and source nodes
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
                for node in drain_source_nodes:
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
                isub_string = ""
                vdc_string = ""
                idc_string = ""
                
                if transcon_calc:
                    fet1 = fet_identifiers[0]
                    fet2 = fet_identifiers[1]
                    fet3 = "1_3_2"
                    fet4 = "1_4_2"
                    fet5 = "0_2_1"
                    fet6 = "0_3_1"
                    fet7 = "0_4_1"
                    fet8 = "0_5_1"
                    isub_string = f".EXTRACT FSST YVAL(ISUB(XM_{fet1}.S), 1MEG) YVAL(ISUB(XM_{fet2}.S), 1MEG)"
                    isub2_string = f"YVAL(ISUB(XM_{fet5}.S), 1MEG) YVAL(ISUB(XM_{fet6}.S), 1MEG) YVAL(ISUB(XM_{fet7}.S), 1MEG) YVAL(ISUB(XM_{fet8}.S), 1MEG)"
                    isub_string += isub2_string
                    idc2_string = f"ISUB(XM_{fet5}.S) ISUB(XM_{fet6}.S) ISUB(XM_{fet7}.S) ISUB(XM_{fet8}.S)"
                    idc_string = f".EXTRACT FSST ISUB(XM_{fet1}.S) ISUB(XM_{fet2}.S) ISUB(XM_{fet3}.S) ISUB(XM_{fet4}.S)"
                    idc_string += idc2_string
                    measure_source_current = False
                    if measure_source_current:
                        idc_string += (f"ISUB(XSBCS011.INOUTPUT_SELF_BIASED_CS) ISUB(XPMOS_CS2.INOUTPUT_PMOS_CS)")
                    vdc_string = f".EXTRACT FSST V(V_IN_1_2) V(V_OUT_0_1) V(V_OUT_1_2)"

                simulation_details = (
                    
                        

                f".SST FUND1={freq} NHARM1=1\n"
                f"{vm_string}\n" ##here also append vi string if it's needed
                f"{isub_string}\n"
                f"{idc_string}\n"
                f"{vdc_string}\n"
                ".OPTION AEX\n"
                ".OPTION NOASCII\n"
                ".END\n"
                )

    
    
            #the idea is that each gate voltage is set by the uic command to the value of the last run
            #so I need both Inudge parameters as the Weights, which are esentially the gate voltages
            elif simulation_type == "TRAN":
                vm_list = []
                print_vg_values = True
                # if print_vg_values:
                #     drain_source_nodes.extend(gate_nodes)
                vg_ic_list = []   
                end_volt_list = []
                vds_list =[]
                vds_ic_list = []
                all_nodes = drain_source_nodes + gate_nodes
                for node in all_nodes:
                    if counter == 0:
                        printfile = printfile
                        vm_list.append(f".PRINTFILE TRAN FILE={printfile} START=0 STOP={self.simulation_time}")
                    vm_node = f"V({node})"
                    vm_list.append(vm_node)
                    counter += 1
                    if counter == 5:
                        # Append the current line and reset for a new one
                        vm_list.append("\n")
                        counter = 0
               # Join the vm_list into a string, removing unnecessary spaces and ensuring formatting
                vm_string = " ".join(filter(None, vm_list)).replace(" \n.", "\n.")
                counter = 0
                # vi_string = " ".join(filter(None, vi_list)).replace(" \n.", "\n.")

                counter_vds = 0
                for node in drain_source_nodes:
                    if counter_vds == 0:
                        vds_ic_list.append(".IC")
                    vds, identifier = node.split("_", 1)
                    volt = "V_END_" + identifier
                    param_line = f".PARAM {volt} = 0"
                    end_volt_list.append(param_line)
                    vds_ic_list.append(f"V({node})={volt}")
                    counter_vds += 1
                    if counter_vds == 5:
                        vds_ic_list.append("\n")
                        counter_vds = 0
                
                vds_string = " ".join(filter(None, vds_ic_list)).replace(" \n.", "\n.")               
                
                for node in gate_nodes:
                    if counter == 0:
                        vg_ic_list.append(".IC")
                    vg_node = node
                    vg, identifier = node.split("_", 1) # now I build the string for the weight again, this should probably be avoided
                    weight = "W_" + identifier
                    vg_ic_list.append(f"V({node})={weight}")
                    counter += 1
                    if counter == 5:
                        # Append the current line and reset for a new one
                        vg_ic_list.append("\n")
                        counter = 0
               # Join the vm_list into a string, removing unnecessary spaces and ensuring formatting
                vg_string = " ".join(filter(None, vg_ic_list)).replace(" \n.", "\n.")
                # vi_string = " ".join(filter(None, vi_list)).replace(" \n.", "\n.")                
                print_file_line = f".PRINTFILE TRAN FILE={printfile} START=0 STOP={self.simulation_time}"
                if transcon_calc:
                    fet1 = fet_identifiers[0]
                    fet2 = fet_identifiers[1]
                    pmos_cs = f"XPMOS_CS2.INOUTPUT_PMOS_CS"
                    isub_string = f"{print_file_line} ISUB(XSBCS011.INOUTPUT_SELF_BIASED_CS) ISUB(XM_{fet1}.S) ISUB(XM_{fet2}.S) ISUB({pmos_cs})"

                else:
                    isub_string = ""
                amp_string = f"{print_file_line} V(XI011.XI1.OUTPUT_CS_1) V(XI011.XI1.OUTPUT_CS_2) V(XI011.NET05)"
                inudge_string = "I(I_SOURCE2)"
                amp_string += " " + inudge_string
                amp_characterization = True
                

                simulation_details = (
                f".TRAN 0.1u {self.simulation_time} uic\n"
                f"{isub_string}\n"
                f"{amp_string}\n"
                f"{vm_string}\n"
                f"{vg_string}\n"
                f"{vds_string}\n"
                ".OPTION PRINTFILE_TIME_STEP=0.1u\n"
                ".OPTION NOASCII\n"
                ".END\n"
                )
                #f".AC LIST {freq}\n"

                    
                    
            elif simulation_type == "DC":
                vdc_list = []
                for node in drain_source_nodes:
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
                    if line is not None:

                        file.write(line + '\n')  # Ensure line is a string and add a newline
                for line in network_parameter_lines:
                    if line is not None:
                        file.write(line + '\n')  # Ensure line is a string and add a newline
                if simulation_type == 'TRAN':
                    for line in end_volt_list:
                        if line is not None:
                            file.write(line + '\n')
                cmos_parameters_lines = "".join(cmos_parameters)
                file.write(cmos_parameters_lines) 
                file.write("\n \n ***END OF THE PARAMETER SECTION \n \n \n")
                file.write(self.libraries)
                
                file.write("\n".join(mid_sect))
                
                file.write("\n \n ***END OF THE SUBCIRCUIT SECTION \n \n \n")

                #Include the needed libraries
                for section in network_description:
                    for line in section:
                        if line:
                            file.write(line)  # Write each line from the sections
                file.write(simulation_details)
            print(f"Network description and parameters have been saved to {file_name}.")


