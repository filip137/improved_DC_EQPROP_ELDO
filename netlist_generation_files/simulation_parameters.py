from datetime import datetime
import os

class SimulationParameters:
    def __init__(self, scale_factor, bias, batch_size, beta, gamma_values):
        
        date_str = datetime.now().strftime("%m%d")
        hour_str = datetime.now().strftime("%H%M%S")

        # Network initialization parameters
        self.simulation_type = "TRAN"  # FSST OR DC OR TRAN
        self.network_size = [5, 4, 4]  # [input, hidden, output]
        self.freq = "1MEG"
        self.neuron = "amp_ss"  # amp_ss or perfect_amp
        self.amplifier = "BiDirWithNonLin" # "BiDirWithNonLin" or OldBiDirAmp or ThreeTerminalBiDirAmp
        if self.amplifier == "ThreeTerminalBiDirAmp":
            self.non_lin = True
        else:
            self.non_lin = False
        
        self.vdc_bias1 = 2.5  # Ensure this matches the netlist
        self.pmos_nonlin_bias = 2.4 #This is the biasing term for the non-linearity 
        
        self.layer1_bias_curr = 5 * 10e-6 #When I am using large networks that are difficult to bias with the nmos sources I am using DC sources with this bias current
        self.layer2_bias_curr = 4 * 10e-6 #The idea is that for each synapse that is connected to the neuron they should provide 10e-6 Amps
        
        
        
        
        
        self.synapse = "fet" # fet or resistor
        #[1e-6, 2e-7]
        self.gamma_values = gamma_values
        self.cs_bias = "self_biased" #"perfect_curr_source" or "self_biased" or False
        # Simulation hyper parameters

        self.beta = beta
        self.batch_size = batch_size
        self.nudging_mode = "current"
        self.loss_function = "MSE"
        self.bounds = {"min_conductance" : 1e-7,
                       "max_conductance" : 5e-5}
        
        if self.simulation_type == "FSST":
            self.diode_connected_flash_params = {"full_voltage_range" : 0.8,
                       "offset1layer" : 5e-5,
                       "offset2layer" : 5e-5}
            
        elif self.simulation_type == "TRAN":
            self.diode_connected_flash_params = {"full_voltage_range" : 1.6,
                       "offset1layer" : -0.8,
                       "offset2layer" : -0.25}        
            
        
        # base directories
        base_aex = "/home/filip/simulations/aex_files"
        base_models = "/home/filip/simulations/trained_models"

        # Output files & folders with today?s date
        self.sample_file = f"{self.synapse}_{self.simulation_type}_netlist"
        self.output_dir         = os.path.join(base_aex, date_str)
        self.trained_models_dir = os.path.join(
            base_models,
            f"{self.synapse}_{self.simulation_type}_{date_str}"
        )
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
                    "L": 5e-5,
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
        #voltage_linne = f"VSOURCE_PMOS{i+1} {source} 0 DC PMOS_NONLIN_BIAS\n" - adds a line but the names must be PMOS_NONLIN_BIAS
        
        self.start_read_time = 10e-6 #the pulse starts at start_read time then it reaches the max value after the 1u and it stays there
        self.end_read_time = 30e-6
        self.input_read_volt = 3.3 #until end_read_time
        self.output_read_volt = 1.5 #this is the pulse applied to the gate of the PMOS at the output
        self.simulation_time = 50e-6
        self.start_write_time = 1e-6 #for writing instead, the pulse is traingular 2ns long
        self.end_write_time  = 3e-6 #this is currently inactive 
        self.start_discharge_time = 32e-6
        self.end_discharge_time = 35e-6
        self.syres = 1e12
        self.sycap = 10e-15
        self.rise_time = 1e-6

        self.transient_params = {
                "start_read_time" : self.start_read_time,
                "input_read_volt" : self.input_read_volt,
                "end_read_time" : self.end_read_time,
                "output_read_volt" : self.output_read_volt,
                "start_write_time" : self.start_write_time,
                "end_write_time" : self.end_write_time                
                }
            
        self.network_parameters_for_netlist= {
            "sycap": self.sycap,
            "syres" : self.syres,
            "VDC_BIAS1" : self.vdc_bias1,
            "start_read_time" : self.start_read_time,
            "input_read_volt" : self.input_read_volt,
            "end_read_time" : self.end_read_time,
            "input_read_volt" : self.input_read_volt,
            "end_read_time" : self.end_read_time,
            "START_DISCHARGE_TIME" : self.start_discharge_time,
            "END_DISCHARGE_TIME" : self.end_discharge_time,
            "output_read_volt" : self.output_read_volt,
            "simulation_time" : self.simulation_time,
            "start_write_time" : self.start_write_time,
            "end_write_time" : self.end_write_time,
            "PMOS_NONLIN_BIAS" : self.pmos_nonlin_bias,
            "layer1_bias_curr" : self.layer1_bias_curr,
            "layer2_bias_curr" : self.layer2_bias_curr,
            "RISE_TIME" : self.rise_time,
            "FORM" : 0,
            "LOW_NOISE_OPTION" : 0
        }

        

            
        
        