

class SimulationParameters:
    def __init__(self, scale_factor, bias, batch_size, beta, gamma_values):
        # Network initialization parameters
        self.simulation_type = "FSST"  # FSST OR DC
        self.network_size = [129, 64, 20]  # [input, hidden, output]
        self.freq = "1MEG"
        self.neuron = "amp_ss"  # amp_ss or perfect_amp
        self.amplifier = "ThreeTerminalBiDirAmp" # NewBiDirAmp or OldBiDirAmp or ThreeTerminalBiDirAmp
        if self.amplifier == "ThreeTerminalBiDirAmp":
            self.non_lin = True
        else:
            self.non_lin = False
        
        self.vdc_bias1 = 2.5  # Ensure this matches the netlist
        self.pmos_nonlin_bias = 2.4 #This is the biasing term for the non-linearity 
        
        self.layer1_bias_curr = 129 * 10e-6 #When I am using large networks that are difficult to bias with the nmos sources I am using DC sources with this bias current
        self.layer2_bias_curr = 80 * 10e-6 #The idea is that for each synapse that is connected to the neuron they should provide 10e-6 Amps
        
        
        
        
        
        self.synapse = "fet" # fet or resistor
        #[1e-6, 2e-7]
        self.gamma_values = gamma_values
        self.cs_bias = "perfect_curr_source" #"perfect_curr_source" or "self_biased" or False
        # Simulation hyper parameters

        self.beta = beta
        self.batch_size = batch_size
        self.nudging_mode = "current"
        self.loss_function = "MSE"
        self.bounds = {"min_conductance" : 1e-7,
                       "max_conductance" : 7e-5}

        # Output files
        self.sample_file = "/home/filip/simulations/aex_files/14_4"
        self.output_dir = "/home/filip/simulations/aex_files/14_4"
        self.trained_models_dir = "/home/filip/simulations/trained_models/res_with_non_lin_vary_beta_14_4"
        # Dataset parameters
        self.dataset = "moons"
        self.n_of_epochs = 40
        self.scale_factor = scale_factor
        self.noise = 0.1
        self.bias = bias
        self.num_samples = 2400

        
        self.initializer = {
            "initializer": {
                "init_type": "random_uniform",
                "params": {
                    "L": 7e-5,
                    "U": 1e-7
                },
                "seed" : 41
                }}

        

        # Layer parameters: a subset of network parameters (for example, layer sizes)
        self.layer_parameters = {
            "simulation_type" : self.simulation_type,
            "network_size": self.network_size,
            "gamma_values": self.gamma_values,
            "freq" : self.freq,
            "neuron" : self.neuron,
            "nudging_mode" : self.nudging_mode,
            "bounds" : self.bounds,
            "initializer" : self.initializer,
            "synapse" : self.synapse,
            "cs_bias" : self.cs_bias,
            "non_lin" : self.non_lin

        }

        #the additional network parameters needed to be defined when generating the netlist
        #This is passed to the netlist builder, which creates lines. PARAM VDC_BIAS1 = 2.3 etc
        #However, the circuit lines are still defined in the layer_class, so it is important that there parameters match the names that are defined in the layer_class. Say 
        #voltage_line = f"VSOURCE_PMOS{i+1} {source} 0 DC PMOS_NONLIN_BIAS\n" - adds a line but the names must be PMOS_NONLIN_BIAS
        self.network_parameters_for_netlist= {
            "VDC_BIAS1" : self.vdc_bias1,
            "PMOS_NONLIN_BIAS" : self.pmos_nonlin_bias,
            "layer1_bias_curr" : self.layer1_bias_curr,
            "layer2_bias_curr" : self.layer2_bias_curr,
            "FORM" : 0,
            "LOW_NOISE_OPTION" : 0
        }


