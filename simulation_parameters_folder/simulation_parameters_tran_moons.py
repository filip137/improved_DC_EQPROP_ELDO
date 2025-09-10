from datetime import datetime
import os

class SimulationParametersTran:
    def __init__(self, scale_factor, bias, batch_size, beta, gamma_values):
        
        
        mode = "TRAIN"
        self.load_weights = False

        
        date_str = datetime.now().strftime("%m%d")
        hour_str = datetime.now().strftime("%H%M%S")

        # Network initialization parameters
        self.simulation_type = "TRAN"  # FSST OR DC OR TRAN
        self.network_size = [5, 12, 4]  # [input, hidden, output]
        self.freq = "1MEG"
        self.neuron = "amp_ss"  # amp_ss or perfect_amp
        self.amplifier = "BiDirWithNonLin" # "BiDirWithNonLin" or "BiDirWithOutNonLin" or OldBiDirAmp or ThreeTerminalBiDirAmp
        if self.amplifier == "ThreeTerminalBiDirAmp":
            self.non_lin = True
        else:
            self.non_lin = False
        
  # This is essentially the reading voltage for the FSST analysis
        
        
        
        
        self.cs_bias = "perfect_curr_source" #"perfect_curr_source" or "self_biased" or False
        if self.cs_bias == "perfect_curr_source": 
            self.layer1_bias_curr = 3 * 5 * 5 * 1e-6 #When I am using large networks that are difficult to bias with the nmos sources I am using DC sources with this bias current
            self.layer2_bias_curr = 3 * 5 * 12 * 1e-6 #The idea is that for each synapse that is connected to the neuron they should provide 10e-6 Amps
        
 
        self.synapse = "fet" # fet or resistor
        #[1e-6, 2e-7]
        self.gamma_values = gamma_values
        # Simulation hyper parameters

        self.beta = beta
        self.batch_size = batch_size
        self.nudging_mode = "current"
        self.loss_function = "MSE"
        self.bounds = {"min_conductance" : 0.1e-5,
                       "max_conductance" : 7e-5}

        
        
        self.initializer = {
            "initializer": {
                "init_type": "random_uniform",
                "params": {
                    "L": 1.1e-5,
                    "U": 5.1e-5
                },
                "seed" : 40
                }}
        
        
        
        #this i suppose needs to be adjusted
        if self.simulation_type == "FSST":
            self.diode_connected_flash_params = {
                       "offset1layer" : -0.3,
                       "offset2layer" : -0.3,
                       "gain" : [1/7.5e-5, 1/7.5e-5]} #actually the inverse of deltagm/deltavgs
            
        elif self.simulation_type == "TRAN":
            self.diode_connected_flash_params = {
                       "offset1layer" : -1.25,
                       "offset2layer" : -1,
                       "gain" : [1/3e-5, 1/3e-5]} #gain layer1 and gainlayer2
            
        
        # base directories
        base_aex = "/home/filip/simulations/aex_files"
        if mode == "TRAIN":
            base_models = "/home/filip/simulations/trained_models"
        if mode == "TEST":
            base_models = "/home/filip/simulations/testing_plots"
        elif mode == "VALIDATION":
            base_models = "/home/filip/simulations/validation_plots"

        # Output files & folders with today?s date
        self.sample_file = f"{self.synapse}_{self.simulation_type}_netlist"
        self.output_dir         = os.path.join(base_aex, date_str)
        self.trained_models_dir = os.path.join(
            base_models,
            f"{self.synapse}_{self.simulation_type}_{date_str}"
        )
        # Dataset parameters
        self.dataset = "moons"
        self.n_of_epochs = 10
        self.scale_factor = scale_factor
        self.noise = 0.1
        self.bias = bias
        self.num_samples = 1000


        

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
        #voltage_linne = f"VSOURCE_PMOS{i+1} {source} 0 DC PMOS_NONLIN_BIAS\n" - adds a line but the names must be PMOS_NONLIN_BIAS
        
        self.start_read_time = 10e-6 #the pulse starts at start_read time then it reaches the max value after the 1u and it stays there
        self.end_read_time = 30e-6
        self.input_read_volt = 3.3 #until end_read_time
        self.output_read_volt = 1.9 #this is the pulse applied to the gate of the PMOS at the output
        self.simulation_time = 50e-6
        self.start_write_time = 1e-6 #for writing instead, the pulse is traingular 2ns long
        self.end_write_time  = 3e-6 #this is currently inactive 
        self.start_discharge_time = 32e-6
        self.end_discharge_time = 35-6
        self.syres = 100e12
        self.sycap = 10e-15
        self.rise_time = 1e-6

        self.transient_params_for_netlist = {
                "sycap": self.sycap,
                "syres" : self.syres,
                "input_read_volt" : self.input_read_volt,
                "output_read_volt" : self.output_read_volt,
                "start_read_time" : self.start_read_time,
                "end_read_time" : self.end_read_time,
                "start_write_time" : self.start_write_time,
                "end_write_time" : self.end_write_time,    
                "START_DISCHARGE_TIME" : self.start_discharge_time,
                "END_DISCHARGE_TIME" : self.end_discharge_time,
                "RISE_TIME" : self.rise_time,
                "layer1_bias_curr" : getattr(self, "layer1_bias_curr", 0),
                "layer2_bias_curr" : getattr(self, "layer2_bias_curr", 0),
                "FORM" : 0,
                "LOW_NOISE_OPTION" : 0
                }
        
        
        self.vdc_bias1 = 2.1 #essentially the input reading voltage for the FSST
        self.pmos_cs_v_bias  = 1.9 #essentially the output reading voltage for the FSST
        
        
        
        self.fsst_params_for_netlist = {"VDC_BIAS1" : self.vdc_bias1,
                    "PMOS_CS_V_BIAS" : getattr(self, "pmos_cs_v_bias", 0),
                    "layer1_bias_curr" : getattr(self, "layer1_bias_curr", 0),
                    "layer2_bias_curr" : getattr(self, "layer2_bias_curr", 0),
                    "FORM" : 0,
                    "LOW_NOISE_OPTION" : 0}
        
        
        # self.network_parameters_for_netlist= {
        #     "sycap": self.sycap,
        #     "syres" : self.syres,
        #     "VDC_BIAS1" : self.vdc_bias1,
        #     "start_read_time" : self.start_read_time,
        #     "input_read_volt" : self.input_read_volt,
        #     "end_read_time" : self.end_read_time,
        #     "input_read_volt" : self.input_read_volt,
        #     "end_read_time" : self.end_read_time,
        #     "START_DISCHARGE_TIME" : self.start_discharge_time,
        #     "END_DISCHARGE_TIME" : self.end_discharge_time,
        #     "output_read_volt" : self.output_read_volt,
        #     "simulation_time" : self.simulation_time,
        #     "start_write_time" : self.start_write_time,
        #     "end_write_time" : self.end_write_time,
        #     "layer1_bias_curr" : self.layer1_bias_curr,
        #     "layer2_bias_curr" : self.layer2_bias_curr,
        #     "RISE_TIME" : self.rise_time,
        #     "FORM" : 0,
        #     "LOW_NOISE_OPTION" : 0
        # }

        

            
        
        