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
    def __init__(self, n_of_nodes, vdc_bias, freq, which_layer=0, trainable=False):
        # Initialize the parent class (BaseLayer) with the same number of inputs and outputs
        super().__init__(n_of_nodes, n_of_nodes, which_layer=which_layer, trainable=trainable)
        
        
        self.vdc_bias = vdc_bias
        self.freq = freq
        
        
        self.inputs = {}
        # The number of inputs will now equal the number of outputs (n_of_nodes)
        self.input_node_list = self.generate_node_names()
        self.output_node_list = ['0'] * len(self.input_node_list)
        
        self.vdc_parameters = self.generate_variables()
        self.voltage_sources = self.generate_sources()

        self.parameters = self.build_dict()
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
            v_ac = f"VAC{i}"
            self.inputs[v_ac] = 0
            vac_parameters.append(v_ac)
        #VDC bias at each voltage source
        #vac_parameters.append("VAC_BIAS")
        vac_parameters.append("VBIAS")
        return vac_parameters
        
    def build_dict(self):
        para_dict = {}
        for variable in self.vdc_parameters:
            para_dict[variable] = 0
        return para_dict
    
    def build_connections(self):
        lines = []
        input_nodes = self.input_node_list
        vol_sources = self.vdc_parameters
        for i, (node, source) in enumerate(zip(input_nodes, vol_sources)):
            line = f"VSOURCE{i+1} {node} 0 DC VAC_BIAS AC {source} SIN (0 {self.freq})\n"
            lines.append(line)
            #self.add_parameter(source) 
        return lines
        
class NonLinearLayer(BaseLayer):
    
    def __init__(self, n_of_nodes, which_layer, diode_dict):
        # Initialize the parent class (BaseLayer) with the same number of inputs and outputs
        super().__init__(n_of_nodes, n_of_nodes, which_layer)

        # Call the methods to generate and assign attributes
        self.input_node_list = self.build_input_nodes()
        self.output_node_list = self.build_output_nodes()

        self.parameters = diode_dict
        self.connections = self.build_connections()
        
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
            line = f"XI{layer}{in_node_int}{out_node_int} {in_node} {out_node} AMPLIFICATION_SS\n"
            lines.append(line)
        return lines
          
 
class DenseLayer(BaseLayer):
    def __init__(self, n_of_inputs, n_of_outputs, which_layer, synapse, lr, gamma, beta, initializer, lower_cond_bound, upper_cond_bound):

        # Initialize the parent class (BaseLayer)
        super().__init__(n_of_inputs, n_of_outputs, which_layer)

        # Call the methods to generate and assign attributes
        #self.node_names = self.generate_node_names()

        
        
        
        self.input_node_list = self.build_input_nodes()
        self.output_node_list = self.build_output_nodes()
        
        
        self.synapse_type = synapse
        self.resistor_dict = self.build_resistor_dict() #R_0_1_1': None, 'R_0_1_2': None, 'R_0_1_3': None, 'R_0_1_4' this order

        self.connections = self.build_connections_new()
        self.parameters = self.build_resistor_dict()
        
        self.type = "resistive"
        self.lr = lr
        self.initializer = initializer
        
        self.input_free_voltages = self.initialize_voltages_in()
        self.output_free_voltages = self.initialize_voltages_out()
 
        self.input_nudge_voltages = self.initialize_voltages_in()
        self.output_nudge_voltages = self.initialize_voltages_out()        
 
    
        self.W = self.initialize_W()
        self.gamma = gamma
        self.beta = beta
        
        
        self.lower_cond_bound = lower_cond_bound
        self.upper_cond_bound = upper_cond_bound
    
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
  
    #I am worried that this will become very slow, so I will replace it by the matrix - I will still keep the resistor dict because it is useful to initialize the network
    def build_resistor_dict(self):
        
        resistor_dict = {}
        synapse = "resistor"
        if synapse == "resistor":
            for input_node in self.input_node_list:
                first_index = int(input_node.split('_')[-1])
                for output_node in self.output_node_list:
                    second_index = int(output_node.split('_')[-1])
                    layer = self.which_layer
                    res = f"R_{layer}_{first_index}_{second_index}"
                    if res not in resistor_dict:
                        resistor_dict[res] = None
                    else:
                        print(f"{res} already exists in the dictionary.")
        return resistor_dict
    
    
    def build_connections1(self):
        lines = []
        for res in self.resistor_dict.keys():
            parts = res.split("_")
            last_part = parts[-1]
            input_node_int = int(last_part[0])
            output_node_int = int(last_part[1])
            layer = parts[-2]
            input_node_name = f"V_IN_{layer}_{input_node_int}{output_node_int}"
            output_node_name = f"V_OUT_{layer}_{input_node_int}{output_node_int}"
            line = f"R{layer}{input_node_int}{output_node_int} {input_node_name} {output_node_name} {res}"
            lines.append(line)            
        
            
    def build_connections_new(self):
        lines = []
        for input_node_name in self.input_node_list:
            first_index = int(input_node_name.split('_')[-1])
            for output_node_name in self.output_node_list:
                second_index = int(output_node_name.split('_')[-1])
                layer = self.which_layer
                res = f"R_{layer}_{first_index}_{second_index}"
                line = f"R{layer}{first_index}{second_index} {input_node_name} {output_node_name} {res}\n"
                lines.append(line)            
        return lines
    
    
    #Initialize the weight matrix
    def initialize_W(self):
        shape = (self.n_of_inputs, self.n_of_outputs)
        weight_matrix = self.initializer.initialize_weights(shape=shape)
        self.W = weight_matrix  # Element-wise inversion
        return self.W
    
    #Update the weight matrix
    def update_W(self, free_vol_matrix_diff, nudge_vol_matrix_diff):
        beta = self.beta
        gamma = self.gamma
        
        deltaG = gamma/beta * (np.square(nudge_vol_matrix_diff) - np.square(free_vol_matrix_diff)) * 1/self.lr
        W = self.W + deltaG
        #clipped_W = np.clip(W, 10e-7, None)
        lower_cond_bound = 0
        upper_cond_bound = 10
        clipped_W = np.clip(W, float(self.lower_cond_bound), float(self.upper_cond_bound))
        self.W = clipped_W
        return self.W
    
    
       
    
    #Update the layer's resistor dictionary based on the layer's conductance matrix
    def update_res_dict(self):
        clipped_W = np.clip(self.W, self.lower_cond_bound, self.upper_cond_bound)
        resistor_matrix = 1/self.W
        # Step 2: Flatten the matrix into a 1D array
        resistor_array = resistor_matrix.flatten(order = 'F')
        
        # Check if the sizes match
        if len(resistor_array) != len(self.resistor_dict):
            print(f"Error: Size mismatch. Resistor array has {len(resistor_array)} elements, "
                  f"but resistor_dict has {len(self.resistor_dict)} elements.")
            return  # Exit the function in case of a mismatch
        
        # Update the resistor_dict
        for i, (key, value) in enumerate(self.resistor_dict.items()):
            self.resistor_dict[key] = resistor_array[i]
        return self.resistor_dict
    
    
    
    
    
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
    
    #Initializing the resistor conductance
    def initialize_res(self):
        shape=(self.n_of_inputs, self.n_of_outputs)
        cond_array = self.initializer.initialize_weights(shape=shape).flatten()
        for i, (key, value) in enumerate(self.resistor_dict.items()):
            self.resistor_dict[key] = np.round(1/cond_array[i],2)
        self.parameters = self.resistor_dict
    
    
    def update_resistances(self):
        layer = self.which_layer
        for key, value in self.resistor_dict.items():
            parts = key.split('_')
            input_index = parts[2]
            output_index = parts[3]
            input_node = f"V_IN_{layer}_{input_index}"
            output_node = f"V_OUT_{layer}_{output_index}"
            input_free_voltage = self.input_free_voltages[input_node]
            
            
            
    def run_update_process(self):
        # Step 1: Update voltage differences
        free_vol_matrix_diff, nudge_vol_matrix_diff = self.calc_vol_difference()
        
        # Step 2: Update weights
        self.W = self.update_W(free_vol_matrix_diff, nudge_vol_matrix_diff)
        
    
        return self.W
    
class OutputLayer(BaseLayer):
    
    
    def __init__(self, n_of_nodes, freq, which_layer):
        # Initialize the parent class (BaseLayer)
        super().__init__(n_of_nodes, n_of_nodes, which_layer)

        # Call the methods to generate and assign attributes

        self.freq = freq


        self.input_node_list = self.generate_node_names()
        self.output_node_list = ['0'] * len(self.input_node_list)

        self.current_sources = self.generate_sources_id()

        self.parameters = self.generate_sources_dict() #the important dict for the parameters
        self.connections = self.build_connections()

    
    def generate_sources_id(self):
        current_sources = []
        for i in range(1, self.n_of_inputs +1):
            current_source = f"I_SOURCE{i}"
            current_sources.append(current_source)
        return current_sources
    
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
        inudge_dict = {}
        for i in range(1, self.n_of_outputs +1):
            idc = f"INUDGE_{i}"
            #inudge_parameters.append(idc)
            if idc not in inudge_dict:
                inudge_dict[idc] = 0
        return inudge_dict  
    
    def build_connections(self):
        lines= []
        for i, (current_source, node_name) in enumerate(zip(self.current_sources, self.input_node_list)):
            line = f"{current_source} {node_name} 0 DC 0 AC INUDGE_{i+1} SIN (0 {self.freq})\n"
            lines.append(line)
        return lines
    
    def update_parameters(self, values_array):
    # Ensure the values_array is the same length as the parameters dict
        if len(values_array) != len(self.parameters):
            raise ValueError("The size of the values array must match the size of the parameters dictionary")

    # Update each key in the parameters dict with the corresponding value from values_array
        for key, value in zip(self.parameters.keys(), values_array):
            self.parameters[key] = value
# class Synapse:
#     def __init__(self, trainable=False):
    
#         self.ID = ID
#         self.parameter = parameter
        
        
        
        
        