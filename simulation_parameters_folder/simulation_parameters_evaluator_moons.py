from datetime import datetime
import os
import json
import warnings

def load_config(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)

class SimulationParametersDCEvaluator:
    def __init__(self, config, model=None, amplifier=None, allow_exponential_default=False):
        

        date_str = datetime.now().strftime("%m%d")
        hour_str = datetime.now().strftime("%H%M%S")



        cfg = load_config(config)
        quadratic_params = None
        if "models" in cfg:
            if model is None:
                raise ValueError("model must be provided when config contains 'models'.")
            if model not in cfg["models"]:
                raise KeyError(f"model '{model}' not found in config.")
            model_cfg = cfg["models"][model]
            layer_shapes = model_cfg["layer_shapes"]
            self.amp = model_cfg["voltage_amp"]
            self.ampc = model_cfg["current_amp"]
            non_linearity = model_cfg["non_linearity"]
            exponential_params = model_cfg["exponential_diode_param"]
            quadratic_params = model_cfg["quadratic_diode_param"]
            self.input_gain = model_cfg.get("input_gain")
            self.input_target_std = model_cfg.get("input_target_std")
            weight_min = model_cfg.get("weight_min")
            weight_max = model_cfg.get("weight_max")
            amplifier_from_cfg = model_cfg.get("amplifier")
        else:
            layer_shapes = cfg["layer_shapes"]
            self.amp = cfg["voltage_amp"]
            self.ampc = cfg["current_amp"]
            non_linearity = cfg["non_linearity"]
            exponential_params = cfg["exponential_diode_param"]
            quadratic_params = cfg["quadratic_diode_param"]
            self.input_gain = cfg.get("input_gain")
            self.input_target_std = cfg.get("input_target_std")
            weight_min = cfg.get("weight_min")
            weight_max = cfg.get("weight_max")
            amplifier_from_cfg = cfg.get("amplifier")
        flat_shapes = []
        for layer in layer_shapes:
            flat_shape = 1
            for dim in layer:
                flat_shape *= dim
            flat_shapes.append(flat_shape)
        if isinstance(non_linearity, dict):
            non_linearity_type = non_linearity.get("type")
            exponential_params = non_linearity.get("exponential_params", exponential_params)
            quadratic_params = non_linearity.get("quadratic_params", quadratic_params)
        else:
            non_linearity_type = non_linearity
        self.non_linearity = non_linearity_type
        if non_linearity_type and "quadratic" in str(non_linearity_type):
            if quadratic_params is None:
                raise ValueError("quadratic_diode_param is required to set V_off.")
            if "v_off" in quadratic_params:
                self.vdiode1 = quadratic_params["v_off"]
            elif "V_off" in quadratic_params:
                self.vdiode1 = quadratic_params["V_off"]
            else:
                raise ValueError("quadratic_diode_param must include v_off.")
        else:
            if exponential_params is None:
                raise ValueError("exponential_diode_param is required to set V_off.")
            if "V_off" not in exponential_params:
                raise ValueError("exponential_diode_param must include V_off.")
            self.vdiode1 = exponential_params["V_off"]
        self.vdiode2 = -self.vdiode1
        #self.vdiode1 = 100
        #self.vdiode2 = -100

        # Network initialization parameters
        self.simulation_type = "DC"  # FSST OR DC OR TRAN
        self.network_size = flat_shapes  # [input, hidden, output]
        self.freq = None
        self.neuron = "perfect_amp"  # amp_ss or perfect_amp
        if self.neuron not in {"amp_ss", "perfect_amp"}:
            raise ValueError("neuron must be 'amp_ss' or 'perfect_amp'.")
        
        # Assign amplifier from metadata (or infer from non_linearity if not provided).
        if amplifier_from_cfg is None:
            if non_linearity_type == "single_diode_exponential":
                amplifier_from_cfg = "PerfectAmpSingleDiodes"
            elif non_linearity_type == "double_diode_exponential":
                amplifier_from_cfg = "PerfectAmpPerfectDiode"
            elif non_linearity_type and "quadratic" in str(non_linearity_type):
                amplifier_from_cfg = "PerfectAmpQuadraticDiode"
            elif non_linearity_type and "exponential" in str(non_linearity_type):
                amplifier_from_cfg = "PerfectAmpPerfectDiode"
            else:
                raise ValueError(
                    "Cannot infer amplifier: non_linearity must include "
                    "'quadratic' or 'exponential', or provide 'amplifier' in config."
                )
        if amplifier is not None and amplifier != amplifier_from_cfg:
            raise ValueError(
                f"amplifier argument '{amplifier}' does not match config amplifier "
                f"'{amplifier_from_cfg}'."
            )
        self.amplifier = amplifier_from_cfg  # allowed options below
        if self.amplifier not in {
            "BiDirWithNonLin",
            "BiDirWithOutNonLin",
            "OldBiDirAmp",
            "ThreeTerminalBiDirAmp",
            "PerfectAmpPerfectDiode",
            "PerfectAmpWithNonlin",
            "PerfectAmpQuadraticDiode",
            "PerfectAmpSingleDiodes"
        }:
            raise ValueError(
                "amplifier must be one of: "
                "'BiDirWithNonLin', 'BiDirWithOutNonLin', 'OldBiDirAmp', "
                "'ThreeTerminalBiDirAmp', 'PerfectAmpPerfectDiode', "
                "'PerfectAmpWithNonlin', 'PerfectAmpQuadraticDiode'."
            )
            
        if self.amplifier == "ThreeTerminalBiDirAmp":
            self.non_lin = True
        else:
            self.non_lin = False
        
  # This is essentially the reading voltage for the FSST analysis
        
        
        
        
        self.cs_bias = False #"perfect_curr_source" or "self_biased" or False
        if not (self.cs_bias is False or self.cs_bias in {"perfect_curr_source", "self_biased"}):
            raise ValueError("cs_bias must be False, 'perfect_curr_source', or 'self_biased'.")
            

        
 
        self.synapse = "resistor" # fet or resistor
        #[1e-6, 2e-7]
        # Simulation hyper parameters

        self.loss_function = "MSE"
        if weight_min is None or weight_max is None:
            warnings.warn(
                "weight_min/weight_max missing; defaulting to 1e-7/1e3.",
                RuntimeWarning,
            )
            if weight_min is None:
                weight_min = 1e-7
            if weight_max is None:
                weight_max = 1e3
        self.bounds = {
            "min_conductance": weight_min,
            "max_conductance": weight_max,
        }

        
        
        self.initializer = {
            "initializer": {
                "init_type": "random_uniform",
                "params": {
                    "L": 1.1e-5,
                    "U": 5.1e-5
                },
                "seed" : 40
                }}
        
        
        self.diode_connected_flash_params = None #gain layer1 and gainlayer2
            
        

        
        # base directories
        base_aex = "/home/filip/simulations/aex_files"

        base_models = "/home/filip/simulations/validation_plots"

        
        os.makedirs(base_models, exist_ok=True)  # ensure it exists

        # Output files & folders with today?s date
        self.sample_file = f"{self.synapse}_{self.simulation_type}_netlist"
        self.output_dir         = os.path.join(base_aex, date_str)
        self.trained_models_dir = os.path.join(
            base_models,
            f"{self.synapse}_{self.simulation_type}_{date_str}"
        )
        # Dataset parameters
        self.dataset = "moons_simulation"
        self.nudging_mode = "current"
    
        # Layer parameters: a subset of network parameters (for example, layer sizes)
        self.layer_parameters = {
            "simulation_type" : self.simulation_type,
            "network_size": self.network_size,
            "gamma_values": [0, 0],
            "freq" : self.freq,
            "neuron" : self.neuron,
            "nudging_mode" : self.nudging_mode,
            "bounds" : self.bounds,
            "initializer" : self.initializer,
            "synapse" : self.synapse,
            "cs_bias" : self.cs_bias,
            "non_lin" : self.non_lin,
            "non_linearity_type": non_linearity_type,
            "include_bias": False,

        }

        #the additional network parameters needed to be defined when generating the netlist
        #This is passed to the netlist builder, which creates lines. PARAM VDC_BIAS1 = 2.3 etc
        #However, the circuit lines are still defined in the layer_class, so it is important that there parameters match the names that are defined in the layer_class. Say 
        #voltage_linne = f"VSOURCE_PMOS{i+1} {source} 0 DC PMOS_NONLIN_BIAS\n" - adds a line but the names must be PMOS_NONLIN_BIAS

        

        #vdiode1 should be positive, vdiode2 should be negative
        self._params_for_netlist = {
            "AMP" : self.amp,
            "AMPC" : self.ampc,
            "VDIODE1" : self.vdiode1,
            "VDIODE2" : self.vdiode2

        }


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

        

            
        
        
