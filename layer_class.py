import numpy as np


class BaseLayer:
    
    def __init__(self, n_of_inputs, n_of_outputs, which_layer, trainable=False):
        """
        This is the base layer that contains basic attributes that all classes must have

        units       : the number of components
        name        : the (unique) name of the layer
        layer_type  : a string to identify the layer
        trainable   : specifies whether the layer parameters should be trainable
        """
        self.n_of_inputs = n_of_inputs
        self.n_of_outputs = n_of_outputs
        self.which_layer = which_layer
        #self.name = name.lower()
        #self.layer_type = layer_type
        self.trainable = trainable
        self.input_node_template = "V_IN"
        self.output_node_template = "V_OUT"
        self.save_output_voltage = False
        self.save_power_params = False
        self.built = False
        self.type = None

        self.input_shape = None
        self.output_shape = None
        self.shape = None
        
        #self.parameters = []




#I create different kinds of layers and in each one I store the parameters in a dictionary which contains keys and values. I can then easily gather all the keys to generate the netlist

class InputLayer(BaseLayer):
    def __init__(self, n_of_nodes, freq, simulation_type, which_layer=0,trainable=False):
        # Initialize the parent class (BaseLayer) with the same number of inputs and outputs
        super().__init__(n_of_nodes, n_of_nodes, which_layer=which_layer, trainable=trainable)
        
        
        self.freq = freq
        
        self.simulation_type = simulation_type
        self.trainable = False
        
        self.inputs = {} ##this becomes a dict 
        # The number of inputs will now equal the number of outputs (n_of_nodes)
        self.input_node_list = self.generate_node_names()
        self.output_node_list = ['0'] * len(self.input_node_list)
        
        self.layer_parameters = self.generate_variables()
        self.voltage_sources = self.generate_sources()

        self.parameters = self.build_dict(self.layer_parameters)
        self.connections = self.build_connections()
        
        
        
    def generate_sources(self):
        voltage_sources = []
        for i in range(1, self.n_of_inputs +1):
            voltage_source = f"VSOURCE{i}"
            voltage_sources.append(voltage_source)
        return voltage_sources
    
    def generate_node_names(self):
        node_names = []
        for i in range(1,self.n_of_outputs+1):
            node = f"{self.input_node_template}_0_{i}"
            node_names.append(node)
        return node_names
    
    def generate_variables(self):
        vac_parameters = []
        for i in range(1, self.n_of_inputs + 1):
            if self.simulation_type == "FSST":
                v_ac = f"VAC{i}"
                self.inputs[v_ac] = 0
            if self.simulation_type == "DC":
                v_ac = f"VDC{i}"
                self.inputs[v_ac] = 0
            vac_parameters.append(v_ac)
        #vac_parameters.append("V_BIAS1")
        return vac_parameters
        
    def build_dict(self, layer_parameters):
        para_dict = {}
        for variable in layer_parameters:
            para_dict[variable] = 0
        return para_dict
    
    
    def build_connections(self):
        lines = []
        input_nodes = self.input_node_list
        vol_sources = self.layer_parameters
        freq = self.freq
        for i, (node, source) in enumerate(zip(input_nodes, vol_sources)):
            if self.simulation_type == "FSST":
                line = f"VSOURCE{i+1} {node} 0 DC VDC_BIAS1 AC 100m 0 SIN (VDC_BIAS1 {source} {self.freq})\n"
            elif self.simulation_type == "DC":
                line = f"VSOURCE{i+1} {node} 0 DC {source}\n"
            else:
                raise ValueError("Invalid simulation type")
            lines.append(line)
            #self.add_parameter(source) 
        return lines
        
        
class NonLinearLayer(BaseLayer):
    
    def __init__(self, n_of_nodes, neuron_type, cs_bias, non_lin, which_layer):
        # Initialize the parent class (BaseLayer) with the same number of inputs and outputs
        super().__init__(n_of_nodes, n_of_nodes, which_layer)

        # Call the methods to generate and assign attributes
        self.input_node_list = self.build_input_nodes()
        self.output_node_list = self.build_output_nodes()
        self.neuron_type = neuron_type
        self.trainable = False
        self.cs_bias = cs_bias
        self.non_lin = non_lin
        #Passing nonlin parameters as an input
        #self.parameters = self.initialize_nonlin_parameters(amp_parameters)
        #self.subcircuit = self.initialize_subcircuit(subcircuit_definitions)
        self.connections = self.build_connections()
        self.current_source = "perfect_curr_source" #"perfect_curr_source" or "self_biased"
        
    # def initialize_nonlin_parameters(self, amp_parameters):
    #     nonlin_parameters = amp_parameters
    #     if self.neuron_type == "amp_ss":
    #         parameters = nonlin_parameters["amp_ss"]
    #     elif self.neuron_type == "perfect_amp":
    #         parameters = nonlin_parameters["perfect_amp"]
    #     else:
    #         ValueError("Invalid neuron type")
    #     return parameters
    
    # def initialize_subcircuit(self):
    #     subcir_def = self.subcircuit_definitions
    #     if self.neuron_type == "amp_ss":
    #         lines = subcir_def["amp_ss"]
    #     elif self.neuron_type == "perfect_amp":
    #         lines = subcir_def["perfect_amp"]
    #     else:
    #         ValueError("Invalid subcircuit type")
    #     return lines
    
    def build_input_nodes(self):
        input_node_list = []
        if self.n_of_inputs != self.n_of_outputs:
            raise ValueError("The number of inputs does not match the number of outputs.")
        
        for i in range(1, self.n_of_inputs + 1):
            input_node = f"{self.output_node_template}_{self.which_layer}_{i}"
            input_node_list.append(input_node)
        
        return input_node_list
        
    
    def build_output_nodes(self):
        output_node_list = []
        if self.n_of_inputs != self.n_of_outputs:
            raise ValueError("The number of inputs does not match the number of outputs.")
        
        for i in range(1, self.n_of_outputs + 1):
            output_node = f"{self.input_node_template}_{self.which_layer + 1}_{i}"
            output_node_list.append(output_node)
        
        return output_node_list      
        
    
    #Initially set to None I can change it during the initialization
    def build_dict(self):
        para_dict = {}
        for i, variable in enumerate(self.parameters):
            para_dict[variable] = None
        return para_dict
            
    
    #def init_non_lin(self):
    
    
    def build_connections(self):
        lines = []
        layer = self.which_layer
        in_nodes = self.input_node_list
        out_nodes = self.output_node_list
        for i, (in_node, out_node) in enumerate(zip(in_nodes, out_nodes)):
            in_node_int = int(in_node.split("_")[-1])
            out_node_int = int(out_node.split("_")[-1])
            if self.neuron_type == "amp_ss":     
                #This already includes the current source biasing                               
                line1 = f"XI{layer}{in_node_int}{out_node_int} {in_node} {out_node} AMPLIFICATION_SS\n"
                if self.cs_bias == "self_biased":
                    line2 = f"XSBCS{layer}{in_node_int}{out_node_int} 0 {in_node} NMOS_SELF_BIASED_CS\n"
                elif self.cs_bias == "perfect_curr_source":
                    line2 = f"I{layer}{in_node_int}{out_node_int} {in_node} 0 layer1_bias_curr\n"
                else:
                    line2 = None
                
                line = [line1, line2]
                
                #This includes three terminal amplifier with the non-linearity
                if self.non_lin:             
                    line1 = f"XI{layer}{in_node_int}{out_node_int} {in_node} {out_node} {out_node}_CS AMPLIFICATION_SS\n"   
                    #NMOS NON-LINEARITY
                    drain = in_node
                    gate = out_node
                    source = 0
                    bulk = 0
                    #need to include a line to set the bulk and the source of the pmos non-linearity
                    tran_details_nmos = """EN5V0_BS3JU w=8e-06 l=0.5e-6 nfing=1 ncrsd=1 number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0 dmu_mdev=0 soa=1 lpe=0"""
                    identifier = f"{layer}_{in_node_int}_{out_node_int}"
                    fet = f"XNONLIN_NMOS_{identifier}"
                    tran_line_nmos = f"{fet} {drain} {gate} {source} {bulk} {tran_details_nmos}\n"
                    #line3 = f"XNONLIN{layer}{in_node_int}{out_node_int} {tran_line}\n"
                    
                    #PMOS NON-LINEARITY
                    drain = in_node
                    gate = out_node + "_CS"
                    source = in_node + "_PMOS_BIAS"
                    bulk = in_node + "_PMOS_BIAS"
                    tran_details_pmos = """EP5V0_BS3JU w=8e-06 l=0.5e-6 nfing=1 ncrsd=1 number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0 dmu_mdev=0 soa=1 lpe=0"""
                    #pmos_nonlin_bias = self.pmos_nonlin_bias
                    voltage_line = f"VSOURCE_PMOS_NONLIN{i+1} {source} 0 DC PMOS_NONLIN_BIAS\n"
                    identifier = f"{layer}_{in_node_int}_{out_node_int}"
                    fet = f"XNONLIN_PMOS_{identifier}"
                    tran_line_pmos = f"{fet} {drain} {gate} {source} {bulk} {tran_details_pmos}\n"                  
 
                    line = [line1, line2, tran_line_nmos, tran_line_pmos, voltage_line]
            elif self.neuron_type == "perfect_amp":
                line = f"XI{layer}{in_node_int}{out_node_int} {in_node} {out_node} NEURON\n"
            else:
                raise ValueError("Invalid neuron name")
            lines.append(line)
        return lines
          
 
class DenseLayer(BaseLayer):
    def __init__(self, n_of_inputs, n_of_outputs, synapse, bounds, gamma, initializer, which_layer):

        # Initialize the parent class (BaseLayer)
        super().__init__(n_of_inputs, n_of_outputs, which_layer)

        # Call the methods to generate and assign attributes
        #self.node_names = self.generate_node_names()

        
        
        
        self.input_node_list = self.build_input_nodes()
        self.output_node_list = self.build_output_nodes()

                
        self.synapse = synapse
        self.synapse_dict = self.build_synapse_dict() #R_0_1_1': None, 'R_0_1_2': None, 'R_0_1_3': None, 'R_0_1_4' this order
        self.parameters = self.synapse_dict


        self.connections = self.build_connections_new()
        
        
        self.trainable = "True"
        self.gamma = gamma
        self.initializer = initializer
    
        self.bounds = bounds
        
        self.input_free_voltages = self.initialize_voltages_in()
        self.output_free_voltages = self.initialize_voltages_out()

    
        self.input_nudge_voltages = self.initialize_voltages_in()
        self.output_nudge_voltages = self.initialize_voltages_out()        
 
    
        self.W = self.initialize_W()
        
        self.synapse_matrix = np.zeros(self.W.shape)
        self.deltaG = np.zeros(self.W.shape)
        self.gradient = np.zeros(self.W.shape)

            
    def initialize_voltages_in(self):
        all_nodes = self.input_node_list
        free_voltages = {}
        for node in all_nodes:
            free_voltages[node] = 0
        return free_voltages
    
    
    
        
    def initialize_voltages_out(self):
        all_nodes = self.output_node_list
        free_voltages = {}
        for node in all_nodes:
            free_voltages[node] = 0
        return free_voltages
    
    
    def build_input_nodes(self):
        input_node_list = []

        
        for i in range(1, self.n_of_inputs + 1):
            input_node = f"{self.input_node_template}_{self.which_layer}_{i}"
            input_node_list.append(input_node)
        
        return input_node_list
        
    
    def build_output_nodes(self):
        output_node_list = []

        
        for i in range(1, self.n_of_outputs + 1):
            output_node = f"{self.output_node_template}_{self.which_layer}_{i}"
            output_node_list.append(output_node)
        
        return output_node_list     
  
    # #I am worried that this will become very slow, so I will replace it by the matrix - I will still keep the resistor dict because it is useful to initialize the network
    def build_synapse_dict(self):
        
        synapse_dict = {}
        for input_node in self.input_node_list:
            first_index = int(input_node.split('_')[-1])
            for output_node in self.output_node_list:
                second_index = int(output_node.split('_')[-1])
                layer = self.which_layer
                if self.synapse == "resistor":
                    para = f"R_{layer}_{first_index}_{second_index}"
                elif self.synapse == "fet":
                    para = f"W_{layer}_{first_index}_{second_index}"
                if para not in synapse_dict:
                    synapse_dict[para] = 0.1
                else:
                    print(f"{para} already exists in the dictionary.")
        return synapse_dict
    
             
        
            
    def build_connections_new(self):
        lines = []
        for input_node_name in self.input_node_list:
            first_index = int(input_node_name.split('_')[-1])
            for output_node_name in self.output_node_list:
                second_index = int(output_node_name.split('_')[-1])
                layer = self.which_layer
                if self.synapse == "resistor":
                    res = f"R_{layer}_{first_index}_{second_index}"
                    line = f"R{layer}{first_index}{second_index} {input_node_name} {output_node_name} {res}\n"
                if self.synapse == "fet":
                    line = []
                    identifier = f"{layer}_{first_index}_{second_index}"
                    fet = f"XM_{identifier}"
                    v_dc = f"V_{identifier}"
                    weight = f"W_{identifier}"
                    if self.which_layer == 0:
                        drain = input_node_name
                        gate = f"VG_{identifier}"
                        source = output_node_name
                        bulk = source
                        tran_details = """EN5V0_BS3JU w=W_SYNAPSE l=L_SYNAPSE nfing=1 ncrsd=1 number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0 dmu_mdev=0 soa=1 lpe=0"""
                        tran_line = f"{fet} {drain} {gate} {source} {bulk} {tran_details}\n"
                        source_line = f"{v_dc} {gate} {drain} DC {weight}\n" #the first terminal is positive, the second one negative
                    #The transistors in the second layer are inverted
                    if self.which_layer == 1:
                        drain = output_node_name
                        gate = f"VG_{identifier}"
                        source = input_node_name
                        bulk = source
                        tran_details = """EN5V0_BS3JU w=W_SYNAPSE l=L_SYNAPSE nfing=1 ncrsd=1 number=1 srcefirst=1 ngcon=1 mismatch=1 po2act=-1 dvt_mdev=0 dmu_mdev=0 soa=1 lpe=0"""
                        tran_line = f"{fet} {drain} {gate} {source} {bulk} {tran_details}\n"
                        source_line = f"{v_dc} {gate} {drain} DC {weight}\n"#the first terminal is positive, the second one negative
                    line.extend([tran_line, source_line])
                lines.append(line)            
        return lines
    
    
    #Initialize the weight matrix
    def initialize_W(self):
        shape = (self.n_of_inputs, self.n_of_outputs)
        weight_matrix = self.initializer.initialize_weights(shape=shape)
        self.W = weight_matrix  # Element-wise inversion
        return self.W
    
    #Update the weight matrix
    def update_deltaG(self,free_vol_matrix_diff, nudge_vol_matrix_diff, batch_size, beta):
        gamma = self.gamma
        deltaG =  gamma/beta * (np.square(nudge_vol_matrix_diff) - np.square(free_vol_matrix_diff)) * 1/batch_size
        self.gradient = deltaG
        mode = "non_discrete"
        if mode == "discrete":
            step_size = 1e-2
            deltaG = np.sign(deltaG) * step_size
        self.deltaG += deltaG
        #clipped_W = np.clip(W, 10e-7, None)
        #clipped_W = np.clip(W, float(self.lower_cond_bound), float(self.upper_cond_bound))
    
    
    
    
    
    #Update the weight matrix
    def update_W(self, mode, clip, print_reduction = False):
        lower_cond_bound = self.bounds["min_conductance"]
        upper_cond_bound = self.bounds["max_conductance"]
        if mode == "clip_updates":
            step_size = clip
            
            # Compute the norm before clipping
            norm_before = np.linalg.norm(self.deltaG)
            
            # Apply clipping to the gradient update
            deltaG_clipped = np.clip(self.deltaG, -step_size, step_size)
            if print_reduction:
                # Compute the norm after clipping
                norm_after = np.linalg.norm(deltaG_clipped)
                
                # Optionally, compute the ratio or difference to see the effect of clipping
                reduction_ratio = norm_after / norm_before if norm_before > 0 else 0
                
                print("Norm before clipping: ", norm_before)
                print("Norm after clipping:  ", norm_after)
                print("Reduction ratio (after/before):", reduction_ratio)
            
            # Now you can proceed with the update using deltaG_clipped
            # self.W -= learning_rate * deltaG_clipped  (for example)
        
        self.W_old = self.W
        
        self.unclipped_W = self.W - self.deltaG
        clipped_W = np.clip(self.unclipped_W, lower_cond_bound, upper_cond_bound)
        
        self.W = clipped_W
        #self.resistor_matrix = 1/self.W
    
    def update__free_voltages(self, voltage_dict):
        """
        Updates the input_free_voltages and output_free_voltages attributes with values from voltage_dict.
    
        Args:
            voltage_dict: A dictionary containing voltage values to update.
        """
        # Update input_free_voltages
        for key, value in self.input_free_voltages.items():
            if key in voltage_dict:
                self.input_free_voltages[key] = voltage_dict[key]
    
        # Update output_free_voltages
        for key, value in self.output_free_voltages.items():
            if key in voltage_dict:
                self.output_free_voltages[key] = voltage_dict[key]
                
                
    def update__nudge_voltages(self, voltage_dict):
        """
        Updates the input_free_voltages and output_free_voltages attributes with values from voltage_dict.
    
        Args:
            voltage_dict: A dictionary containing voltage values to update.
        """
        # Update input_free_voltages
        for key, value in self.input_nudge_voltages.items():
            if key in voltage_dict:
                self.input_nudge_voltages[key] = voltage_dict[key]
    
        # Update output_free_voltages
        for key, value in self.output_nudge_voltages.items():
            if key in voltage_dict:
                self.output_nudge_voltages[key] = voltage_dict[key]
    
    
    
    
    def zero_grad(self):
        self.deltaG = np.zeros(self.deltaG.shape)
      
    
    def update_synapse_dict(self):
        #lower_cond_bound = self.bounds["min_conductance"]
        #upper_cond_bound = self.bounds["max_conductance"]
        if self.synapse == "resistor":
            self.synapse_matrix = 1/self.W
        elif self.synapse == "fet":
            #This is the max transconductance that a certain transistor can reach - it will probably depend layer by layer and my change throughout training
            #offset is the - of the minimum voltage
            offset = -0.5
            max_cond = 70e-6
            self.synapse_matrix = offset + self.W/max_cond
        else:
            ValueError("Wrong synapse")
            
        synapse_array = self.synapse_matrix.flatten(order = 'C')
        # Update the resistor_dict
        for i, (key, value) in enumerate(self.synapse_dict.items()):
            self.synapse_dict[key] = synapse_array[i]
        return self.synapse_dict
    
    
    
    # #Update the layer's resistor dictionary based on the layer's conductance matrix
    # def update_res_dict(self):
    #     #clipped_W = np.clip(self.W, self.lower_cond_bound, self.upper_cond_bound)
    #     resistor_matrix = 1/self.W
    #     # Step 2: Flatten the matrix into a 1D array
    #     resistor_array = resistor_matrix.flatten(order = 'C')
        
    #     # Check if the sizes match
    #     if len(resistor_array) != len(self.resistor_dict):
    #         print(f"Error: Size mismatch. Resistor array has {len(resistor_array)} elements, "
    #               f"but resistor_dict has {len(self.resistor_dict)} elements.")
    #         return  # Exit the function in case of a mismatch
        
    #     # Update the resistor_dict
    #     for i, (key, value) in enumerate(self.resistor_dict.items()):
    #         self.resistor_dict[key] = resistor_array[i]
    #     return self.resistor_dict
    
    
    
    # #Update the layer's voltage dictionary based on the layer's conductance matrix
    # def update_volt_dict(self, offset = -0.5):
    #     #clipped_W = np.clip(self.W, self.lower_cond_bound, self.upper_cond_bound)
        
    #     volt_matrix = offset + self.W/self.upper_cond_bound
    #     # Step 2: Flatten the matrix into a 1D array
    #     volt_array = volt_matrix.flatten(order = 'C')
        
    #     # Check if the sizes match
    #     if len(volt_array) != len(self.resistor_dict):
    #         print(f"Error: Size mismatch. Resistor array has {len(volt_array)} elements, "
    #               f"but resistor_dict has {len(self.resistor_dict)} elements.")
    #         return  # Exit the function in case of a mismatch
        
    #     # Update the resistor_dict
    #     for i, (key, value) in enumerate(self.resistor_dict.items()):
    #         self.resistor_dict[key] = volt_array[i]
    #     return self.resistor_dict    
    
    
    
    #Calculate the voltage diffrences
    def calc_vol_difference(self):
        f_input_volt_arr = np.array(list(self.input_free_voltages.values()))
        f_output_volt_arr = np.array(list(self.output_free_voltages.values()))

        n_input_volt_arr = np.array(list(self.input_nudge_voltages.values()))
        n_output_volt_arr = np.array(list(self.output_nudge_voltages.values()))        
        
        
        free_vol_matrix_diff = np.empty((len(f_input_volt_arr),len(f_output_volt_arr)))
        #So here the difference between the first input and the second output is stored in the element 1x2!
        for i in range(len(f_input_volt_arr)):
            for j in range(len(f_output_volt_arr)):
                free_vol_matrix_diff[i,j] = f_input_volt_arr[i] - f_output_volt_arr[j]
    
    
        nudge_vol_matrix_diff = np.empty((len(f_input_volt_arr),len(f_output_volt_arr)))
        
        for i in range(len(f_input_volt_arr)):
            for j in range(len(f_output_volt_arr)):
                nudge_vol_matrix_diff[i,j] = n_input_volt_arr[i] - n_output_volt_arr[j]
    
        
        return free_vol_matrix_diff, nudge_vol_matrix_diff
    
    # #Initializing the resistor conductance
    # def initialize_res(self):
    #     shape=(self.n_of_inputs, self.n_of_outputs)
    #     cond_array = self.initializer.initialize_weights(shape=shape).flatten()
    #     for i, (key, value) in enumerate(self.resistor_dict.items()):
    #         self.resistor_dict[key] = np.round(1/cond_array[i],2)
    #     self.parameters = self.resistor_dict
    
    
    # def update_resistances(self):
    #     layer = self.which_layer
    #     for key, value in self.resistor_dict.items():
    #         parts = key.split('_')
    #         input_index = parts[2]
    #         output_index = parts[3]
    #         input_node = f"V_IN_{layer}_{input_index}"
    #         output_node = f"V_OUT_{layer}_{output_index}"
    #         input_free_voltage = self.input_free_voltages[input_node]
            
            
            
    def run_update_process(self, batch_size, beta):
        # Step 1: Update voltage differences
        free_vol_matrix_diff, nudge_vol_matrix_diff = self.calc_vol_difference()
        
        # Step 2: Update weights
        self.update_deltaG(free_vol_matrix_diff, nudge_vol_matrix_diff, batch_size, beta)
        
    
    
class OutputLayer(BaseLayer):
    
    
    def __init__(self, n_of_nodes, freq, simulation_type, nudging_mode, cs_bias, which_layer):
        # Initialize the parent class (BaseLayer)
        super().__init__(n_of_nodes, n_of_nodes, which_layer)

        # Call the methods to generate and assign attributes

        self.freq = freq
        self.nudging_mode = nudging_mode
        self.trainable = False
        self.cs_bias = cs_bias

        self.simulation_type = simulation_type

        self.input_node_list = self.generate_node_names()
        self.output_node_list = ['0'] * len(self.input_node_list)

        self.sources = self.generate_sources_id()

        self.parameters = self.generate_sources_dict() #the important dict for the parameters
        self.connections = self.build_connections()
    
    def generate_sources_id(self):
        sources = []
        for i in range(1, self.n_of_inputs +1):
            if self.nudging_mode == "current":
                source = f"I_SOURCE{i}"
            elif self.nudging_mode == "voltage":
                source = f"V_SOURCE{i}"            
            sources.append(source)
        return sources
    
    def generate_node_names(self):
        node_names = []
        for i in range(1,self.n_of_outputs+1):
            last_layer = self.which_layer
            node = f"{self.output_node_template}_{last_layer}_{i}"
            #node_name = [f"{node} 0"]
            node_names.append(node)
        return node_names
    
    
    
    #I am again double doing it, it would be probably easier just to generate the connestions 
    
    def generate_sources_dict(self):        
        source_dict = {}
        rnudge_dict = {}
        for i in range(1, self.n_of_outputs +1):
            if self.nudging_mode == "current":
                idc = f"INUDGE_{i}"
                rdc = None
            elif self.nudging_mode == "voltage":
                rdc = f"RNUDGE_{i}"
                idc = f"VNUDGE_{i}"
            #inudge_parameters.append(idc)
            if idc not in source_dict:
                source_dict[idc] = 0
            if rdc is not None and rdc not in source_dict:
                rnudge_dict[rdc] = 99999999
                source_dict.update(rnudge_dict)
        return source_dict  
    
    def build_connections(self):
        lines= []
        for i, (source, node_name) in enumerate(zip(self.sources, self.input_node_list)):
            if self.nudging_mode == "current":
                if self.simulation_type == "FSST":
                    line = []
                    line_nudge = f"{source} {node_name} 0 DC 0 SIN (0 INUDGE_{i+1} {self.freq})\n"
                    line.extend([line_nudge])
                    if self.cs_bias == "self_biased":
                        line_cs = f"XPMOS_CS{i} 0 {node_name} PMOS_CS\n"
                    elif self.cs_bias == "perfect_curr_source":
                        line_cs = f"I_OUT_{i} 0 {node_name} layer2_bias_curr\n"
                    else: 
                        line_cs = None
                    line.extend([line_cs])
                elif self.simulation_type == "DC":
                    line = f"{source} {node_name} 0 DC INUDGE_{i+1}\n"
                    
            elif self.nudging_mode == "voltage":
                if self.simulation_type == "FSST":
                    line = []
                    line_r = f"RN_{i+1} {node_name}N {node_name} RNUDGE_{i+1}\n"
                    #this could be problematic as the DC point shifts
                    line_s = f"{source}N {node_name}N 0 DC 1.7 AC 100m 0 SIN (1.7 VNUDGE_{i+1} {self.freq})\n"
                    
                    line_cs = f"XPMOS_CS{i} 0 {node_name} PMOS_CS\n"
                    if self.cs_bias:
                        line.extend([line_r, line_s, line_cs])
                    else:
                        line.extend([line_r, line_s])
                elif self.simulation_type == "DC":
                    line = []
                    line_r = f"RN_{i+1} {node_name}N {node_name} RNUDGE_{i+1}\n"
                    line_s = f"{source} {node_name}N 0 DC VNUDGE_{i+1}\n"
                    line.extend([line_r, line_s])
            lines.append(line)
        return lines
    
    def update_parameters(self, values_array):
    # Ensure the values_array is the same length as the parameters dict
        if len(values_array) != len(self.parameters):
            raise ValueError("The size of the values array must match the size of the parameters dictionary")

    # Update each key in the parameters dict with the correspondi/.,m\';alue fom values_array
        for key, value in zip(self.parameters.keys(), values_array):
            self.parameters[key] = value
# class Synapse:
#     def __init__(self, trainable=False):
    
#         self.ID = ID
#         self.parameter = parameter
        
        
        
        
        