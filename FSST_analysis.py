from initializer import *
import datasets as ds
import numpy as np
import time

from netlist_generation_files import (
    BaseLayer,
    InputLayer,
    DenseLayer,
    NonLinearLayer,
    OutputLayer,
    netlist_builder,
    initialize_network_layers,
)

from save_and_load_functions import (
save_all, save_sim_parameters, 
best_epoch_from_dir, load_sim_parameters)

from simulation_parameters_folder import (
    SimulationParametersFSST,
    SimulationParametersTran)

from support_layer import *
from eldo_support_functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from loss_functions import * 
from sklearn.model_selection import train_test_split
from datetime import datetime
import signal
import subprocess
import torch
from torch.utils.data import DataLoader, TensorDataset
import json
from ac_plots import *
from transconductance_calculations import calculate_transconductance
import multiprocessing
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting (in some versions)
from matplotlib import cm  # For colormap support
import warnings
warnings.filterwarnings(
    "ignore",
    message="Failed to set pipe buffer size"
)





class NetworkAnalyzer:
    def __init__(
        self,
        layers,
        sim_params,
        input_files,
        all_nodes
    ):
        # Basic network and simulation parameters
        self.layers = layers
        self.simulation_details = sim_params
        self.nudging_mode = sim_params.nudging_mode
        self.freq = sim_params.freq
        self.simulation_type = sim_params.simulation_type

        # File paths
        (
            self.full_subfolder_path,
            self.new_sample_file,
            self.result_file
        ) = input_files

        # Dataset parameters
        self.scale_factor = sim_params.scale_factor
        self.bias = sim_params.bias
        
        if self.simulation_type == "FSST" or self.simulation_type == "TRAN":
            self.diode_connected_flash_params = sim_params.diode_connected_flash_params

        # Node list
        self.all_nodes = all_nodes




    def charactarize_synapses_tran(self, eldo_process, debug):

        input_layer = self.layers[0]
        output_layer = self.layers[-1]
        diode_connected_flash_params = self.diode_connected_flash_params

        simulation_type = self.simulation_type
        n_of_node_voltages = len(self.all_nodes)
        voltage_dict_free = dict.fromkeys(self.all_nodes[0], None)


        layers = self.layers
        resistive_layers = [layer for layer in layers if getattr(layer, 'trainable', None)]
        output_layer = resistive_layers[-1]
        prediction_list = []
        
        q = self.q

        result_file = self.result_file

        counter = 0
        
        
        gm1_list = []
        gm2_list = [] 
        
        dv1_list = []
        dv2_list = []
        
        dc1_list = []
        dc2_list = []
        
        
        offset = 0
        #initialize_synapses(eldo_process, resistive_layers, diode_connected_flash_params, simulation_type, debug)

      
        input_cmd1 = f"SET P(VAC1)=0.2"
        send_command_to_eldo(eldo_process, input_cmd1, debug)
        input_cmd1 = f"SET P(VAC2)=0"
        send_command_to_eldo(eldo_process, input_cmd1, debug)
        input_cmd1 = f"SET P(VAC3)=0"
        send_command_to_eldo(eldo_process, input_cmd1, debug)        
        #input_pmos_cs_cmd = f"SET P(PMOS_CS_W)=16u"
        #send_command_to_eldo(eldo_process, input_pmos_cs_cmd, debug)

        
        if os.path.exists(result_file):
            file_size = os.path.getsize(result_file)
            offset = file_size
            print(f"File size before draw grid: {file_size} bytes")
        
        
        ###HERE I RUN THE FIRST SIMULATION
          
        w1_temp = "W_0_1_1"
        w2_temp = "W_1_2_2"
            
        w1_val = 0
        w2_val = 0
            #this probably doesnt do anything
        cmd1 = f"SET P({w1_temp})={w1_val}" #
        cmd2 = f"SET P({w2_temp})={w2_val}"
        send_command_to_eldo(eldo_process, cmd1, debug)
        send_command_to_eldo(eldo_process, cmd2, debug)
            
            
        #run the first simulation
        run_simulation_and_wait(eldo_process, simulation_type, q, debug)  
        #command_pmos_cs = f"SET P(PMOS_CS_VDD_NEG)=0"
        
        f0, t0 = 1e6, 20e-6

        results = read_update(
                    eldo_process,
                    result_file,
                    voltage_dict_free,
                    offset,
                    simulation_type,
                    transcon_calc=True,
                    debug=debug,
                    f0=f0,
                    t0=t0
                )
        ##set the ic voltages here
            # compute voltages and currents
        dc_ds_end_voltages = results["dc_ds_end"] # DC DRAIN-SOURCE voltages]
        dc_gate_end_voltages = results["dc_gate_end"] # DC GATE VOLTAGES (at the end)
        set_the_ic_voltages(eldo_process, dc_gate_end_voltages, dc_ds_end_voltages, debug)
        

        
        flag11 = False
        flag21 = False 
        
        flag12 = False
        flag22 = False 
        max_number_of_pulses = 100
        
        
                    
        pulses2resetgm1 = pulses2maxgm1 = 1
        gm_reset1      = gm_max1      = 1
            
        pulses2resetgm2 = pulses2maxgm2 = 1
        gm_reset2      = gm_max2      = 1
            
        low_thresh  = 0.2e-5
        threshold_high = 6e-5
        
        synapse_char_dict = {"pulses2resetgm1" : None, "pulses2resetgm2" : None, "slopegm1" : None, "slopegm2" : None}
        synapse_char_list = []
        dc_ds_voltages_list = []
        dc_gate_end_voltages_list = []
        
        delta1 = -0.05
        delta2 = -0.05
        weight1 = np.linspace(-1.25, 0.75, max_number_of_pulses)
        weight2 = np.linspace(-1, 1, max_number_of_pulses)
        
        dc_source1_list = []
        dc_source2_list = []
    # main pulse loop
        for i in range(1, max_number_of_pulses):
            run_simulation_and_wait(eldo_process, simulation_type, q, debug=debug)
    
            # read results
            if simulation_type == "TRAN":
                f0, t0 = 1e6, 20e-6
                results = read_update(
                    eldo_process,
                    result_file,
                    voltage_dict_free,
                    offset,
                    simulation_type,
                    transcon_calc=True,
                    debug=debug,
                    f0=f0,
                    t0=t0
                )
    
            # update layers' free voltages
            for layer in resistive_layers:
                layer.update__free_voltages(voltage_dict_free)
    
            # compute voltages and currents
            ac_currents = results["ac_currents"]
            dc_currents = results["dc_currents"]
            voltages    = results["ac_voltages"] # AC voltage
            dc_ds_voltages = results["dc_ds"] # DC DRAIN-SOURCE voltages
            dc_ds_end_voltages = results["dc_ds_end"] # DC DRAIN-SOURCE voltages]
            dc_gate_voltages = results["dc_gate_end"] # DC GATE VOLTAGES (at the end)
            dc_gate_end_voltages_list.append(dc_gate_voltages.copy())
            
            #dc_source2_list.append(dc_currents['XPMOS_CS2.INOUTPUT_PMOS_CS'])
            #dc_source1_list.append(dc_currents['XSBCS011.INOUTPUT_SELF_BIASED_CS'])
            
            dv1 = voltages["V_IN_0_1"] - voltages["V_OUT_0_1"]
            dv2 = voltages["V_IN_1_2"] - voltages["V_OUT_1_2"]
    
            gm1 = ac_currents['XM_0_1_1.S'] / dv1
            gm2 = ac_currents['XM_1_2_2.S'] / dv2
            dc1 = dc_currents['XM_0_1_1.S']
            dc2 = dc_currents['XM_1_2_2.S']
    
            gm1_list.append(gm1)
            gm2_list.append(gm2)
            dc1_list.append(dc1)
            dc2_list.append(dc2)
            dc_ds_voltages_list.append(dc_ds_voltages.copy())
            
            
            
            #dc_gate_voltages['VG_0_1_1'] = weight1[i]
            #dc_gate_voltages['VG_1_2_2'] = weight2[i]
            #this probably doesnt do anything

            
            set_the_ic_voltages(eldo_process, dc_gate_voltages, dc_ds_end_voltages, debug)
            w1_temp = "W_0_1_1"
            w2_temp = "W_1_2_2"
            cmd1 = f"SET P({w1_temp})={weight1[i]}" #
            cmd2 = f"SET P({w2_temp})={weight2[i]}"
            
            send_command_to_eldo(eldo_process, cmd1, debug)
            send_command_to_eldo(eldo_process, cmd2, debug)
            
            #set_the_ic_voltages(eldo_process, dc_gate_voltages, dc_ds_end_voltages, debug)            
            
            # helper: avg of last 3 absolute gm values
            def avg_last(lst, N=3):
                abs_vals = [abs(x) for x in lst]
                window = abs_vals[-N:] if len(abs_vals) >= N else abs_vals
                return sum(window) / len(window)
    
            avg_gm1 = avg_last(gm1_list)
            avg_gm2 = avg_last(gm2_list)
    
            # threshold logic for gm1
            if abs(avg_gm1) < low_thresh and flag11 and i > 5:
                #send_command_to_eldo(eldo_process, f"SET P({i1_temp})=0", debug)
                delta1 = +0.05
                pulses2resetgm1 = i
                flag11 = False
            elif abs(avg_gm1) > threshold_high and flag12 and i > 5:
                #send_command_to_eldo(eldo_process, f"SET P({i1_temp})=0", debug)
                pulses2maxgm1 = i
                slopegm1 = gm1 / (pulses2maxgm1 - pulses2resetgm1)
                synapse_char_dict.update({
                    "pulses2resetgm1": pulses2resetgm1,
                    "slopegm1": slopegm1
                })
                flag12 = False
    
            # threshold logic for gm2
            if abs(avg_gm2) < low_thresh and flag21 and i > 15:
                #send_command_to_eldo(eldo_process, f"SET P({i2_temp})=1e-9", debug)
                delta2 = +0.05
                pulses2resetgm2 = i
                flag21 = False
            elif abs(avg_gm2) > threshold_high and flag22 and i > 15:
                #send_command_to_eldo(eldo_process, f"SET P({i2_temp})=1e-9", debug)
                pulses2maxgm2 = i
                slopegm2 = gm2 / (pulses2maxgm2 - pulses2resetgm2)
                synapse_char_dict.update({
                    "pulses2resetgm2": pulses2resetgm2,
                    "slopegm2": slopegm2
                })
                flag22 = False
    
            # exit when both saturated
            if abs(avg_gm1) > threshold_high and abs(avg_gm2) > threshold_high:
                synapse_char_list.append(synapse_char_dict)
                break
            
            
            
            
        self.plot_dc_voltage_list(dc_ds_voltages_list, keys=['V_IN_1_2', 'V_OUT_1_2'])
        self.plot_dc_voltage_list(dc_gate_end_voltages_list, keys=['VG_0_1_1', 'VG_1_2_2'])
        vg011_list = [d['VG_0_1_1'] for d in dc_gate_end_voltages_list]
        vg122_list = [d['VG_1_2_2'] for d in dc_gate_end_voltages_list]
        
        clipped_list_gm1 = [np.clip(gm1, -3e-4, +3e-4) for gm1 in gm1_list]
        clipped_list_gm2 = [np.clip(gm2, -3e-4, +3e-4) for gm2 in gm2_list]
        # final plots

        
        results1, slope_info1 = self.plot_gm_dc_full(
            gm_list=clipped_list_gm1, dc_list=dc1_list,
            gm_label='gm1', dc_label='dc1', x = vg011_list, 
            title=f'gm1 and dc1 for csw'
        )
        results2, slope_info2 = self.plot_gm_dc_full(
            gm_list=clipped_list_gm2, dc_list=dc2_list,
            gm_label='gm2', dc_label='dc2', x = vg122_list, 
            title=f'gm2 and dc2 for csw'
        )
        
        plot_two_dc_sources(dc_source1_list, dc_source2_list,
                        label1='Source 1', label2='Source 2',
                        x=None,
                        x_label='Index',
                        y_label='DC value (A)',
                        title=None)
    
        return synapse_char_dict

    def characterize_synapse_fsst(self, eldo_process, debug):
        
        measure_source_current = False
        
        
        input_layer = self.layers[0]
        output_layer = self.layers[-1]
        diode_connected_flash_params = self.diode_connected_flash_params

        simulation_type = self.simulation_type
        n_of_node_voltages = len(self.all_nodes)
        voltage_dict_free = dict.fromkeys(self.all_nodes[0], None)


        layers = self.layers
        resistive_layers = [layer for layer in layers if getattr(layer, 'trainable', None)]
        output_layer = resistive_layers[-1]
        prediction_list = []
        
        q = self.q

        result_file = self.result_file        
        
        
        offset = 0
        # apply the current source width
        cs_pmos_w = self.cs_pmos_w
      
        input_cmd1 = f"SET P(VAC1) = 0.2"
        send_command_to_eldo(eldo_process, input_cmd1, debug)
        input_cmd1 = f"SET P(VAC2) = 0"
        send_command_to_eldo(eldo_process, input_cmd1, debug)
        input_cmd1 = f"SET P(VAC3) = 0"
        send_command_to_eldo(eldo_process, input_cmd1, debug)        
        input_pmos_cs_cmd = f"SET P(PMOS_CS_W)=4u"
        send_command_to_eldo(eldo_process, input_pmos_cs_cmd, debug)
    
        initialize_synapses(eldo_process, resistive_layers, diode_connected_flash_params, simulation_type, debug)

        # initialization
        pulses2resetgm1 = pulses2maxgm1 = None
        pulses2resetgm2 = pulses2maxgm2 = None
        flag11 = True
        flag21 = True 
        
        flag12 = True
        flag22 = True 
        
        gm1_list = []
        gm2_list = []
        
        dc1_list = []
        dc2_list = []
        
        dv1_list = []
        dv2_list = []   
 
        weights = np.linspace(-1,0,201)
        w1_temp = "W_0_1_1"
        w2_temp = "W_1_2_2"
        #double_weights = [weights, weights]
        # main pulse loop
        low_thresh = 0.1e-5
        high_thresh = 6e-5
        dc_voltages_list = []
        synapse_char_dict = {}
        w_current1 = 0
        w_current2 = 0
        delta1 = -0.01/2
        delta2 = -0.01/2
        dc1_list = []
        dc2_list = []
        dc_source1_list = []
        dc_source2_list = []
        w_current1_list = []
        w_current2_list = []
        
        for i, weight in enumerate(weights):
            w_current1 += delta1
            w_current2 += delta2
            w_current1_list.append(w_current1)
            w_current2_list.append(w_current2)


            input_gm1_cmd= f"SET P({w1_temp})={w_current1}"
            input_gm2_cmd= f"SET P({w2_temp})={w_current2}"
            if i > 0:
                send_command_to_eldo(eldo_process, input_gm1_cmd, debug)
                send_command_to_eldo(eldo_process, input_gm2_cmd, debug)

            run_simulation_and_wait(eldo_process, simulation_type, q, debug=debug)

            results = read_update(
                    eldo_process,
                    result_file,
                    voltage_dict_free,
                    offset,
                    simulation_type,
                    transcon_calc=True,
                    debug=debug
                )
    
            # update layers' free voltages
            for layer in resistive_layers:
                layer.update__free_voltages(voltage_dict_free)
    
            # compute voltages and currents
            ac_currents = results["currents"]
            dc_currents = results["dc_currents"]
            voltages    = results["voltages"]
            dc_voltages = results["dc_voltages"]
            dc_voltages_list.append(dc_voltages.copy())
            
            
            dv1 = voltages["V_IN_0_1"] - voltages["V_OUT_0_1"]
            dv2 = voltages["V_IN_1_2"] - voltages["V_OUT_1_2"]
            dv1_list.append(dv1)
            dv2_list.append(dv2)
            dc1_list.append(dc_currents['XM_0_1_1.S'])
            dc2_list.append(dc_currents['XM_1_2_2.S'])
            
            if measure_source_current:
                dc_source2_list.append(dc_currents['XPMOS_CS2.INOUTPUT_PMOS_CS'])
                dc_source1_list.append(dc_currents['XSBCS011.INOUTPUT_SELF_BIASED_CS'])
            
            try:
                gm1 = ac_currents['XM_0_1_1.S'] / dv1
                gm2 = ac_currents['XM_1_2_2.S'] / dv2
            except ZeroDivisionError:
                gm1 = 0
                gm2 = 0
            gm1_list.append(gm1)
            gm2_list.append(gm2)
    
            # helper: avg of last 3 absolute gm values
            def avg_last(lst, N=3):
                abs_vals = [abs(x) for x in lst]
                window = abs_vals[-N:] if len(abs_vals) >= N else abs_vals
                return sum(window) / len(window)
    
            avg_gm1 = avg_last(gm1_list)
            avg_gm2 = avg_last(gm2_list)
    
            # threshold logic for gm1
            if abs(avg_gm1) < low_thresh and flag11 and i > 3:
                delta1 = -delta1
                pulses2resetgm1 = i
                w_reset1 = w_current1
                flag11 = False
            elif abs(avg_gm1) > high_thresh and flag12:
                pulses2maxgm1 = i
                w_max1 = w_current1
                slopegm1 = gm1 / (w_max1 - w_reset1)
                synapse_char_dict.update({
                    "pulses2resetgm1": pulses2resetgm1,
                    "w_reset1": w_reset1
                })
                flag12 = False
    
            # threshold logic for gm2
            if abs(avg_gm2) < low_thresh and flag21 and i > 3:
                delta2 = -delta2
                pulses2resetgm2 = i
                w_reset2 = w_current2
                flag21 = False
            elif abs(avg_gm2) > high_thresh and flag22:
                pulses2maxgm2 = i
                w_max2 = w_current2
                slopegm2 = gm2 / (w_max2 - w_reset2)
                synapse_char_dict.update({
                    "pulses2resetgm2": pulses2resetgm2,
                    "w_reset2": w_reset2
                })
                flag22 = False


        #dc1_list = None
        #dc2_list = None
        clipped_list_gm1 = [np.clip(gm1, -3e-4, +3e-4) for gm1 in gm1_list]
        clipped_list_gm2 = [np.clip(gm2, -3e-4, +3e-4) for gm2 in gm2_list]
        abs_list_gm1 = [abs(gm1) for gm1 in gm1_list]
        abs_list_gm2 = [abs(gm2) for gm2 in gm2_list]
        self.plot_dc_voltage_list(dc_voltages_list)
        # final plots
        results1, slope_info1 = self.plot_gm_dc_full(
            gm_list=clipped_list_gm1, dc_list=dc1_list,
            gm_label='gm1', dc_label='dc1', x = w_current1_list, 
            title=f'gm1 and dc1 for csw {cs_pmos_w}'
        )
        results2, slope_info2 = self.plot_gm_dc_full(
            gm_list=clipped_list_gm2, dc_list=dc2_list,
            gm_label='gm2', dc_label='dc2', x = w_current2_list, 
            title=f'gm2 and dc2 for csw {cs_pmos_w}'
        )
        
        
        if measure_source_current:
            plot_two_dc_sources(dc_source1_list, dc_source2_list,
                            label1='Source 1', label2='Source 2',
                            x=None,
                            x_label='Index',
                            y_label='DC value (A)',
                            title=None)
            
        threshold = 6e-5
    
        def find_threshold_crossing(gm_list, weights, threshold):
            """
            Find the first index and corresponding weight where |gm| exceeds threshold.
        
            Parameters
            ----------
            gm_list : sequence of float
                Transconductance values.
            weights : sequence of float
                Same?length sequence of weights (or x?axis values).
            threshold : float, optional
                The absolute?value cutoff to detect (default 6e-5).
        
            Returns
            -------
            (index, weight) or (None, None)
                The zero?based index and corresponding weight at which
                abs(gm_list[index]) >= threshold for the first time.
                If never exceeded, returns (None, None).
            """
            for i, g in enumerate(gm_list):
                if abs(g) >= threshold:
                    return i, weights[i]
            return None, None
        
        

    
        return 
        
            
        #def write_weights(self, eldo_process, debug):
        
        
                   

    def plot_transcond(self, eldo_process, debug):
        
        
        input_layer = self.layers[0]
        output_layer = self.layers[-1]
        diode_connected_flash_params = self.diode_connected_flash_params

        simulation_type = self.simulation_type
        n_of_node_voltages = len(self.all_nodes)
        voltage_dict_free = dict.fromkeys(self.all_nodes[0], None)


        layers = self.layers
        resistive_layers = [layer for layer in layers if getattr(layer, 'trainable', None)]
        output_layer = resistive_layers[-1]
        prediction_list = []
        
        q = self.q

        result_file = self.result_file

        counter = 0
        
        
        gm1_list = []

        gm2_list = [] 
        
        vout_list = []
        
        dc1_list = []
        dc2_list = []
        offset = 0
        initialize_synapses(eldo_process, resistive_layers, diode_connected_flash_params, simulation_type, debug)
        
        
        dv1_list = []
        dv2_list = []
        
        if simulation_type == "TRAN":
            w1 = np.linspace(0.8,1.3, 25)
            w2 = np.linspace(0.8, 1.3, 25)
            W_in = np.column_stack((w1, w2))  
            w1_temp = "W_0_1_1"
            w2_temp = "W_1_1_2"
            w3_temp = "W_1_2_2"
        elif simulation_type == "FSST":
            read_update(eldo_process, result_file, voltage_dict_free, offset, simulation_type, transcon_calc, debug)
        
        input_cmd1 = f"SET P(VAC1)=0"
        send_command_to_eldo(eldo_process, input_cmd1, debug)
        input_cmd1 = f"SET P(VAC2)=0.2"
        send_command_to_eldo(eldo_process, input_cmd1, debug)
        input_cmd1 = f"SET P(VAC3)=0.4"
        send_command_to_eldo(eldo_process, input_cmd1, debug)        
        input_pmos_cs_cmd = f"SET P(PMOS_CS_W)=10u"
        send_command_to_eldo(eldo_process, input_pmos_cs_cmd, debug)
        
        
        if os.path.exists(result_file):
            file_size = os.path.getsize(result_file)
            offset = file_size
            print(f"File size before draw grid: {file_size} bytes")
        
        
        w1_val = 0
        w2_val = 0
        w3_val = 0
        cmd1 = f"SET P({w1_temp})={w1_val}"
        cmd2 = f"SET P({w2_temp})={w2_val}"
        cmd3 = f"SET P({w3_temp})={w3_val}"
        send_command_to_eldo(eldo_process, cmd1, debug)
        send_command_to_eldo(eldo_process, cmd2, debug)
        send_command_to_eldo(eldo_process, cmd3, debug)
        i1_temp = "Iw_0_1_1"
        i2_temp = "Iw_1_1_2"              

        cmd1 = f"SET P({i1_temp})=-1e-9"
        cmd2 = f"SET P({i2_temp})=-1e-9"
        send_command_to_eldo(eldo_process, cmd1, debug)
        send_command_to_eldo(eldo_process, cmd2, debug)
        run_simulation_and_wait(eldo_process, simulation_type, q, debug)  
        command_pmos_cs = f"SET P(PMOS_CS_VDD_NEG)=0"
        
        flag1 = True
        flag2 = True 
        for i in range(1, 100):
            
            
            # cmd1 = f"SET P({w1_temp})={w1_val}"
            # cmd2 = f"SET P({w2_temp})={w2_val}"
            # send_command_to_eldo(eldo_process, cmd1, debug)
            # send_command_to_eldo(eldo_process, cmd2, debug)
                
                #move at the end
            run_simulation_and_wait(eldo_process, simulation_type, q, debug)  
                

        
            transcon_calc = True
            if simulation_type == "TRAN":
                f0 = 1e6
                t0 = 20e-6
                results_dict = read_update(eldo_process, result_file, voltage_dict_free, offset, simulation_type, transcon_calc, debug, f0 = f0, t0 = t0)
                # pick exactly which columns you want to see
                columns_to_plot = [
                    "V(VG_0_1_1)",
                    "V(VG_1_1_2)",
                    "V(V_OUT_0_1)",
                    "V(V_OUT_1_2)",
                    "V(V_IN_1_1)"
                ]
                
                # --- call the function with plotting ---
                # plot_tran_voltages(
                #     eldo_process=eldo_process,
                #     result_file=result_file,
                #     offset=offset,
                #     simulation_type=simulation_type,
                #     debug=debug,
                #     columns_to_plot=columns_to_plot
                # )
            elif simulation_type == "FSST":
                read_update(eldo_process, result_file, voltage_dict_free, offset, simulation_type, transcon_calc, debug)


            for layer in resistive_layers:
                layer.update__free_voltages(voltage_dict_free)
             
                
            dc_voltages = results_dict["dc_voltage_dict"]
            
            vout1 = dc_voltages['V_OUT_1_1']
            vout2 = dc_voltages['V_OUT_1_2']
            vout_list.append((vout1, vout2))
            
            
            ac_currents = results_dict["ac_currents"]
            voltages = results_dict["voltages"]
            dv1 = voltages["V_IN_0_1"] - voltages["V_OUT_0_1"]
            dv2 = voltages["V_IN_1_1"] - voltages["V_OUT_1_2"]
            dv1_list.append(dv1)
            dv2_list.append(dv2)
            
            dc_currents = results_dict["dc_currents"]
            dc1 = dc_currents['ISUB(XM_0_1_1.S)']
            dc1_list.append(dc1)
            dc2 = dc_currents['ISUB(XM_1_1_2.S)']
            dc2_list.append(dc2)
            
            gm1 = ac_currents['ISUB(XM_0_1_1.S)']/dv1
            gm2 = ac_currents['ISUB(XM_1_1_2.S)']/dv2



            gm1_list.append(gm1)
            gm2_list.append(gm2) 
            

# compute running average over last 3 (or fewer) values
            def avg_last(lst, N):
                window = lst[-N:] if len(lst) >= N else lst
                return sum(window) / len(window)
            
            avg_gm1 = avg_last(gm1_list, N=5)
            avg_gm2 = avg_last(gm2_list, N=5)
            # decide based on the averaged gm1
            if abs(avg_gm1) < 0.5e-5:
                cmd1 = f"SET P({i1_temp})=2e-9"
                send_command_to_eldo(eldo_process, cmd1, debug)
                
            elif abs(avg_gm1) > 7e-5 and flag1:
                cmd1 = f"SET P({i1_temp})=0"
                send_command_to_eldo(eldo_process, cmd1, debug)

            
            # decide based on the averaged gm2
            if abs(avg_gm2) < 0.5e-5:
                cmd2 = f"SET P({i2_temp})=2e-9"
                send_command_to_eldo(eldo_process, cmd2, debug)
            elif abs(avg_gm2) > 7e-5 and flag2:
                cmd2 = f"SET P({i2_temp})=0"
                send_command_to_eldo(eldo_process, cmd2, debug)
           
            if abs(avg_gm2) > 6e-5 and abs(avg_gm1) > 6e-5:
                break
            
            if i == 0:
                cmd1 = f"SET P({i1_temp})=-1e-9"
                cmd2 = f"SET P({i2_temp})=-1e-9"
                send_command_to_eldo(eldo_process, cmd1, debug)
                send_command_to_eldo(eldo_process, cmd2, debug)
            
            
            

        
        
        
        
        
        
        # end of loop: unpack into two series
        vout1_list = [pair[0] for pair in vout_list]
        vout2_list = [pair[1] for pair in vout_list]
        
        # title = 'Output Voltages vs. Iteration'
        # self.plot_output_voltages(vout1_list, title, vout2_list)


        # title = 'Voltage difference 1 vs. Iteration'
        # self.plot_output_voltages(dv1_list, title)

        # title = 'Current 1 vs. Iteration'
        # self.plot_output_voltages(dc1_list, title)



        
        # title = 'Voltage difference 2 vs. Iteration'
        # self.plot_output_voltages(dv2_list, title)

        # title = 'Current 2 vs. Iteration'
        # self.plot_output_voltages(dc2_list, title)


        # for gm1/dc1
        self.plot_gm_dc(
            gm_list=gm1_list,
            dc_list=dc1_list,
            gm_label='gm1',
            dc_label='dc1',
            title='gm1 and dc1 vs. Initial Voltage'
        )
        
        # for gm2/dc2
        self.plot_gm_dc(
            gm_list=gm2_list,
            dc_list=dc2_list,
            gm_label='gm2',
            dc_label='dc2',
            title='gm2 and dc2 vs. Initial Voltage'
)       
        
        
        


    def plot_output_voltages(self, vout1_list, title, vout2_list = None):
        """
        Plot V_OUT_1_1 and V_OUT_1_2 versus iteration on the same axes.
        """
        plt.figure()
        plt.plot(vout1_list, label='V_OUT_1_1')
        if vout2_list:
            plt.plot(vout2_list, label='V_OUT_1_2')
        plt.xlabel('Iteration'); plt.ylabel('Voltage (V)')
        plt.title(f'{title}'); 
        plt.tight_layout(); 
        plt.show()






    def print_dc_at_closest_gm(self, gm_list, dc_list, targets=(1e-5, 3e-5, 6e-5), start_index=10):
        """
        For each target in `targets`, find the closest value in gm_list only considering
        entries at index >= start_index, and print:
          - target gm
          - closest gm (from gm_list)
          - global index in the original array
          - dc_list value at that index
          - error = |closest_gm - target|
        """
        gm = abs(np.asarray(gm_list, dtype=float))
        dc = abs(np.asarray(dc_list, dtype=float))
        if gm.shape != dc.shape:
            raise ValueError("gm_list and dc_list must have the same shape")
        if start_index >= len(gm):
            raise ValueError("start_index is beyond the length of gm_list")
    
        header = f"{'Target':>10}  {'Closest gm':>14}  {'Index':>5}  {'dc value':>15}  {'Error':>10}"
        print(header)
        print("-" * len(header))
        for t in targets:
            sub = gm[start_index:]
            rel_idx = int(np.argmin(np.abs(sub - t)))
            idx = start_index + rel_idx
            closest = gm[idx]
            dc_val = dc[idx]
            error = abs(closest - t)
            print(f"{t:10.3e}  {closest:14.6e}  {idx:5d}  {dc_val:15.6e}  {error:10.2e}")

 

    def plot_gm_dc_full(self,
                   gm_list, dc_list,
                   gm_label, dc_label,
                   x=None,
                   x_label='Initial voltage',
                   title=None,
                   targets=(1e-5, 3e-5, 6e-5),
                   start_index=10):
        """
        Twin-axis plot of gm (left y-axis) and dc (right y-axis) versus x,
        annotating dc_list values at the closest gm to each target (searching only
        indices >= start_index), and showing average slope of gm vs sqrt(dc)
        and gm vs x (both least-squares and through-origin).
        Returns list of matched target info and slope info.
        """
        gm_arr = np.abs(np.asarray(gm_list, dtype=float))
        if dc_list is not None:
            dc_arr = np.abs(np.asarray(dc_list, dtype=float))
            if gm_arr.shape != dc_arr.shape:
                raise ValueError("gm_list and dc_list must have the same shape")
        else:
            dc_arr = None
    
        if x is None:
            x = np.arange(len(gm_arr))
        else:
            x = np.asarray(x, dtype=float)
            if x.shape != gm_arr.shape:
                raise ValueError("x and gm_list must have the same shape")
    
        fig, ax1 = plt.subplots()
        c_gm, c_dc = 'tab:blue', 'tab:red'
    
        # plot gm
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(f'{gm_label} (S)', color=c_gm)
        ax1.plot(x, gm_arr, color=c_gm, label=gm_label)
        ax1.tick_params(axis='y', labelcolor=c_gm)
    
        # plot dc if provided
        if dc_arr is not None:
            ax2 = ax1.twinx()
            ax2.set_ylabel(f'{dc_label} (A)', color=c_dc)
            ax2.plot(x, dc_arr, color=c_dc, linestyle='--', label=dc_label)
            ax2.tick_params(axis='y', labelcolor=c_dc)
        else:
            ax2 = None
    
        # find and annotate target points (only indices >= start_index)
        results = []
        for t in targets:
            if start_index >= len(gm_arr):
                raise ValueError("start_index is beyond length of gm_list")
            sub = gm_arr[start_index:]
            rel_idx = int(np.argmin(np.abs(sub - t)))
            idx = start_index + rel_idx
            closest = gm_arr[idx]
            dc_val = dc_arr[idx] if dc_arr is not None else None
            error = abs(closest - t)
            entry = {
                "target": t,
                "closest_gm": closest,
                "index": idx,
                "dc_value": dc_val,
                "error": error,
            }
            results.append(entry)
    
            # annotate on plot
            ax1.plot(x[idx], closest, marker='o', color=c_gm)
            ann_text = f"gm={closest:.2e}"
            if dc_val is not None:
                ax2.plot(x[idx], dc_val, marker='s', color=c_dc)
                ann_text += f"\ndc={dc_val:.2e}"
            ax1.annotate(ann_text,
                         xy=(x[idx], closest),
                         xytext=(5, 5),
                         textcoords='offset points',
                         fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.3", fc="w", alpha=0.7),
                         arrowprops=dict(arrowstyle="->", lw=0.5))
    
        # compute slope info
        slope_info = {}
    
        # gm vs sqrt(dc)
        if dc_arr is not None:
            sqrt_dc = np.sqrt(dc_arr)
            # least-squares linear fit: gm = m * sqrt(dc) + b
            coeffs = np.polyfit(sqrt_dc, gm_arr, 1)
            m_ls, b_ls = coeffs
            # through-origin slope
            denom = np.sum(sqrt_dc ** 2)
            if denom != 0:
                k_origin = np.sum(gm_arr * sqrt_dc) / denom
            else:
                k_origin = np.nan
    
            slope_info.update({
                "ls_slope_sqrt_dc": m_ls,
                "ls_intercept_sqrt_dc": b_ls,
                "origin_slope_sqrt_dc": k_origin
            })
    
            # inset for gm vs sqrt(dc)
            plot_gm_sqrtdc = False
            if plot_gm_sqrtdc:
                try:
                    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                    ax_inset1 = inset_axes(ax1, width="40%", height="35%", loc="upper left", borderpad=1)
                except ImportError:
                    ax_inset1 = fig.add_axes([0.55, 0.55, 0.35, 0.35])
        
                ax_inset1.scatter(sqrt_dc, gm_arr, s=10, label='data', alpha=0.7)
                x_fit = np.linspace(np.min(sqrt_dc), np.max(sqrt_dc), 100)
                ax_inset1.plot(x_fit, m_ls * x_fit + b_ls, linestyle='-', label='LS fit')
                ax_inset1.plot(x_fit, k_origin * x_fit, linestyle='--', label='Origin fit')
                ax_inset1.set_xlabel('sqrt(dc)')
                ax_inset1.set_ylabel('gm')
                ax_inset1.set_title('gm vs sqrt(dc)', fontsize=8)
                ax_inset1.tick_params(labelsize=7)
                ax_inset1.legend(fontsize=6, framealpha=0.7)
    
        # gm vs x
        # least-squares: gm = a * x + b
        coeffs_x = np.polyfit(x, gm_arr, 1)
        a_ls, b_x = coeffs_x
        denom_x = np.sum(x ** 2)
        if denom_x != 0:
            a_origin = np.sum(gm_arr * x) / denom_x
        else:
            a_origin = np.nan
    
        slope_info.update({
            "ls_slope_x": a_ls,
            "ls_intercept_x": b_x,
            "origin_slope_x": a_origin
        })
    
        # inset for gm vs x (place it to the right of the other if both exist)
        try:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            loc = "upper right" if dc_arr is not None else "upper left"
            ax_inset2 = inset_axes(ax1, width="40%", height="35%", loc=loc, borderpad=1)
        except ImportError:
            # fallback positioning: avoid overlap if both insets present
            if dc_arr is not None:
                ax_inset2 = fig.add_axes([0.55, 0.1, 0.35, 0.35])
            else:
                ax_inset2 = fig.add_axes([0.55, 0.55, 0.35, 0.35])
    
        ax_inset2.scatter(x, gm_arr, s=10, label='data', alpha=0.7)
        x_fit2 = np.linspace(np.min(x), np.max(x), 100)
        ax_inset2.plot(x_fit2, a_ls * x_fit2 + b_x, linestyle='-', label='LS fit')
        ax_inset2.plot(x_fit2, a_origin * x_fit2, linestyle='--', label='Origin fit')
        ax_inset2.set_xlabel(x_label)
        ax_inset2.set_ylabel('gm')
        ax_inset2.set_title('gm vs x', fontsize=8)
        ax_inset2.tick_params(labelsize=7)
        ax_inset2.legend(fontsize=6, framealpha=0.7)
    
        # Annotate slope summaries on main plot
        slope_text_lines = []
        if dc_arr is not None:
            slope_text_lines.append(
                f"gm vs sqrt(dc): LS slope={m_ls:.2e}, intercept={b_ls:.2e}; "
                f"origin={k_origin:.2e}"
            )
        slope_text_lines.append(
            f"gm vs x: LS slope={a_ls:.2e}, intercept={b_x:.2e}; origin={a_origin:.2e}"
        )
        slope_text = "\n".join(slope_text_lines)
        ax1.text(0.98, 0.02, slope_text,
                 transform=ax1.transAxes,
                 fontsize=8,
                 va='bottom', ha='right',
                 bbox=dict(boxstyle="round,pad=0.3", fc="w", alpha=0.6))
    
        if title:
            plt.title(title)
    
        fig.tight_layout()
        plt.show()
        return results, slope_info



   
    def plot_gm_dc(self, 
            gm_list, dc_list,
            gm_label, dc_label,
            x=None,
            x_label='Initial voltage',
            title=None
        ):
        """
        Twin?axis plot of gm (left y?axis) and dc (right y?axis) versus x.
    
        x         : array?like for x?axis (optional; will default to range(len(gm_list)))
        gm_list   : transconductance values
        dc_list   : DC current values (or None)
        gm_label  : label for gm curve
        dc_label  : label for dc curve
        x_label   : label for x?axis
        title     : plot title
        """
        fig, ax1 = plt.subplots()
        c_gm, c_dc = 'tab:blue', 'tab:red'
    
        # prepare x-axis
        if x is None:
            x = range(len(gm_list))
    
        # plot gm
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(f'{gm_label} (S)', color=c_gm)
        ax1.plot(x, gm_list, color=c_gm, label=gm_label)
        ax1.tick_params(axis='y', labelcolor=c_gm)
    
        # plot dc if provided
        if dc_list is not None:
            ax2 = ax1.twinx()
            ax2.set_ylabel(f'{dc_label} (A)', color=c_dc)
            ax2.plot(x, dc_list, color=c_dc, linestyle='--', label=dc_label)
            ax2.tick_params(axis='y', labelcolor=c_dc)
    
       # ax1.set_ylim(-6e-5, 6e-5)  # limit left y?axis
        if title:
            plt.title(title)
    
        fig.tight_layout()
        plt.show()



    def free_test_FSST(self, eldo_process, X_grid, input_function, mode, h5_path, debug):
        
        


        # Here I first need to write weights
        debug = True

        input_layer = self.layers[0]
        output_layer = self.layers[-1]
        input_dict = input_layer.inputs
        input_keys_list = list(input_dict.keys())

        simulation_type = self.simulation_type
        diode_connected_flash_params = self.diode_connected_flash_params
        q = self.q

        n_of_node_voltages = len(self.all_nodes)
        voltage_dict_free = dict.fromkeys(self.all_nodes[0], None)

        layers = self.layers
        resistive_layers = [
            layer for layer in layers if getattr(layer, 'trainable', None)
        ]

        # write the weights
        best_idx, best_accuracy, wm_evo1_at_best, wm_evo2_at_best = \
            best_epoch_from_dir(h5_path)
        W1 = wm_evo1_at_best
        W2 = wm_evo2_at_best

        resistive_layers[0].W = W1
        resistive_layers[1].W = W2

        output_layer = resistive_layers[-1]
        binary_list = []
        prediction_list = []

        result_file = self.result_file
        counter = 0

        input_amp_voltages = []
        output_amp_voltages = []

        run_simulation_and_wait(
            eldo_process, simulation_type, q, debug
        )


        try:
            offset = os.path.getsize(result_file)
        except FileNotFoundError:
            print("need to run the first simulation")
            offset = 0

        results = read_update(
            eldo_process,
            result_file,
            voltage_dict_free,
            offset,
            simulation_type,
            transcon_calc=True,
            debug=debug
        )


        plot_voltage = False
        plot_gm = True
        X_in = input_function(X_grid)
        for X in X_in:
            for i, key in enumerate(input_keys_list):
                if i < len(X):
                    input_dict[key] = - X[i]
                else:
                    input_dict[key] = 0
                
            set_input_voltages(eldo_process, input_dict, debug)
                #move at the end
            run_simulation_and_wait(eldo_process, simulation_type, q, debug)  

                           #for plotting there's a function read_update_and_plot
                           
            results = read_update(
                eldo_process,
                result_file,
                voltage_dict_free,
                offset,
                simulation_type,
                transcon_calc=True,
                debug=debug)
                           ##set the ic voltages here
                               # compute voltages and currents
            
            # compute voltages and currents
            ac_currents = results["ac_currents"]
            voltage_dict_free    = results["ac_voltages"]
                
            dv1 = voltage_dict_free["V_IN_0_1"] - voltage_dict_free["V_OUT_0_1"]
            dv2 = voltage_dict_free["V_IN_1_2"] - voltage_dict_free["V_OUT_1_2"]
                
                
            gm1 = ac_currents['XM_0_1_1.S'] / dv1
            gm2 = ac_currents['XM_1_2_2.S'] / dv2
               
                
                #dv1_list.append(dv1)
                #dv2_list.append(dv2)
                
                
            #voltage_dict_free = read_update_and_plot(eldo_process, result_file, voltage_dict_free, offset, simulation_type, n_of_node_voltages, debug)


            for layer in resistive_layers:
                layer.update__free_voltages(voltage_dict_free)
                
                    
            input_amp_voltages.append(list(resistive_layers[0].output_free_voltages.values()))
            output_amp_voltages.append(list(resistive_layers[1].input_free_voltages.values()))
            output_layer.update__free_voltages(voltage_dict_free)
            
            outputs = output_layer.output_free_voltages #the outputs are just the outputs of the last layer
            output_values = list(outputs.values())
            
            if plot_voltage:
                prediction = [output_values[0] - output_values[1], output_values[2] - output_values[3]] 
            elif plot_gm:
                prediction = [gm1, gm2] 
                prediction_clipped = [np.clip(abs(gm), 0, 10e-5) for gm in prediction]
                
            prediction_list.append(prediction_clipped)
            
        #accuracy = np.mean(np.equal(binary_array, Y_in)) * 100
        #print(f"Accuracy: {accuracy:.2f}%")
        
        return prediction_list, input_amp_voltages, output_amp_voltages 









    def free_test_tran(self, eldo_process, X_grid, input_function, mode, h5_path, debug):
        
        


        # Here I first need to write weights
        debug = True

        input_layer = self.layers[0]
        output_layer = self.layers[-1]
        input_dict = input_layer.inputs
        input_keys_list = list(input_dict.keys())

        simulation_type = self.simulation_type
        diode_connected_flash_params = self.diode_connected_flash_params
        q = self.q

        n_of_node_voltages = len(self.all_nodes)
        voltage_dict_free = dict.fromkeys(self.all_nodes[0], None)

        layers = self.layers
        resistive_layers = [
            layer for layer in layers if getattr(layer, 'trainable', None)
        ]

        # write the weights
        best_idx, best_accuracy, wm_evo1_at_best, wm_evo2_at_best = \
            best_epoch_from_dir(h5_path)
        W1 = wm_evo1_at_best
        W2 = wm_evo2_at_best

        resistive_layers[0].W = W1
        resistive_layers[1].W = W2

        output_layer = resistive_layers[-1]
        binary_list = []
        prediction_list = []

        result_file = self.result_file
        counter = 0

        input_amp_voltages = []
        output_amp_voltages = []

        run_simulation_and_wait(
            eldo_process, simulation_type, q, debug
        )

        f0, t0 = 1e6, 20e-6

        try:
            offset = os.path.getsize(result_file)
        except FileNotFoundError:
            print("need to run the first simulation")
            offset = 0

        results = read_update(
            eldo_process,
            result_file,
            voltage_dict_free,
            offset,
            simulation_type,
            transcon_calc=True,
            debug=debug,
            f0=f0,
            t0=t0
        )

        # set the ic voltages here
        dc_ds_end_voltages = results["dc_ds_end"]
        dc_gate_end_voltages = results["dc_gate_end"]
        set_the_ic_voltages(
            eldo_process,
            dc_gate_end_voltages,
            dc_ds_end_voltages,
            debug
        )


        update_synapses(
            eldo_process,
            resistive_layers,
            diode_connected_flash_params,
            simulation_type,
            debug
        )



        plot_voltage = False
        plot_gm = True
        X_in = input_function(X_grid)
        for X in X_in:
            for i, key in enumerate(input_keys_list):
                if i < len(X):
                    input_dict[key] = - X[i]
                else:
                    input_dict[key] = 0
                
            set_input_voltages(eldo_process, input_dict, debug)
                #move at the end
            run_simulation_and_wait(eldo_process, simulation_type, q, debug)  

                           #for plotting there's a function read_update_and_plot
                           
            results = read_update(
                eldo_process,
                result_file,
                voltage_dict_free,
                offset,
                simulation_type,
                transcon_calc=True,
                debug=debug,
                f0=f0,
                t0=t0)
                           ##set the ic voltages here
                               # compute voltages and currents
            dc_ds_end_voltages = results["dc_ds_end"] # DC DRAIN-SOURCE voltages]
            dc_gate_end_voltages = results["dc_gate_end"] # DC GATE VOLTAGES (at the end)
            set_the_ic_voltages(eldo_process, dc_gate_end_voltages, dc_ds_end_voltages, debug)
            
            
            # compute voltages and currents
            ac_currents = results["ac_currents"]
            voltage_dict_free    = results["ac_voltages"]
                
            dc_ds_end_voltages = results["dc_ds_end"] # DC DRAIN-SOURCE voltages]
            dc_gate_voltages = results["dc_gate_end"] # DC GATE VOLTAGES (at the end)
                
            dv1 = voltage_dict_free["V_IN_0_1"] - voltage_dict_free["V_OUT_0_1"]
            dv2 = voltage_dict_free["V_IN_1_2"] - voltage_dict_free["V_OUT_1_2"]
                
                
            gm1 = ac_currents['XM_0_1_1.S'] / dv1
            gm2 = ac_currents['XM_1_2_2.S'] / dv2
               
                
            if simulation_type == "FSST":
                results = read_update(
                        eldo_process,
                        result_file,
                        voltage_dict_free,
                        offset,
                        simulation_type,
                        transcon_calc=True,
                        debug=debug
                    )
                ac_currents = results["currents"]
                voltage_dict_free    = results["voltages"]
                
                
                dv1 = voltage_dict_free["V_IN_0_1"] - voltage_dict_free["V_OUT_0_1"]
                dv2 = voltage_dict_free["V_IN_1_2"] - voltage_dict_free["V_OUT_1_2"]
                #dv1_list.append(dv1)
                #dv2_list.append(dv2)
                
                
                gm1 = ac_currents['XM_0_1_1.S'] / dv1
                gm2 = ac_currents['XM_1_2_2.S'] / dv2
        
                
            #voltage_dict_free = read_update_and_plot(eldo_process, result_file, voltage_dict_free, offset, simulation_type, n_of_node_voltages, debug)


            for layer in resistive_layers:
                layer.update__free_voltages(voltage_dict_free)
                
                    
            input_amp_voltages.append(list(resistive_layers[0].output_free_voltages.values()))
            output_amp_voltages.append(list(resistive_layers[1].input_free_voltages.values()))
            output_layer.update__free_voltages(voltage_dict_free)
            
            outputs = output_layer.output_free_voltages #the outputs are just the outputs of the last layer
            output_values = list(outputs.values())
            
            if plot_voltage:
                prediction = [output_values[0] - output_values[1], output_values[2] - output_values[3]] 
            elif plot_gm:
                prediction = [gm1, gm2] 
                prediction_clipped = [np.clip(abs(gm), 0, 10e-5) for gm in prediction]
                
            prediction_list.append(prediction_clipped)
            
        #accuracy = np.mean(np.equal(binary_array, Y_in)) * 100
        #print(f"Accuracy: {accuracy:.2f}%")
        
        return prediction_list, input_amp_voltages, output_amp_voltages        
        
    def analyze_and_plot(self, eldo_process, input_function, plot_directory, debug):
      # Start a new figure for this iteration
      plt.figure()
      
      # Define bounds of the domain
      min1, max1 = -0.4, 0.4
      min2, max2 = -0.4, 0.4
      num_points = 10
  
      x1grid = np.linspace(min1, max1, num_points)
      x2grid = np.linspace(min2, max2, num_points)
  
      # Create a meshgrid from the grid points
      xx, yy = np.meshgrid(x1grid, x2grid)
      X_grid = np.c_[xx.ravel(), yy.ravel()]
  
    
      
  
      # Make predictions for the grid
      draw_grid_flag = True
      h5_path = "/home/filip/simulations/testing_plots/fet_FSST_0802/FSST_114247_262613/plots/metrics_data.h5"
      if self.simulation_type == "TRAN":
          outputs, input_amp_voltages, output_amp_voltages = self.free_test_new(eldo_process, X_grid, input_function, mode, h5_path, debug)
      elif self.simulation_type == "FSST":
          outputs, input_amp_voltages, output_amp_voltages = self.free_test_new(eldo_process, X_grid, input_function, mode, h5_path, debug)

      epoch = 0
      #self.draw_boundary(xx, yy, outputs, epoch, plot_directory)
      self.draw_boundary_3d(xx, yy, outputs, epoch, plot_directory, title = self.simulation_type)
      #self.draw_boundary_3d(xx, yy, input_amp_voltages, epoch, plot_directory, title = "INPUT AMPS")
      #self.draw_boundary_3d(xx, yy, output_amp_voltages, epoch, plot_directory, title = "OUTPUT AMPS")






    def plot_dc_voltage_list(self, history, keys=None):
        """
        Plot voltage history from a list of dicts, each containing the same keys.
        
        Parameters:
        - history: list of dicts mapping signal names to numeric values.
        - keys:    list of signal names to plot; if None, plot all signals.
        """
        if not history:
            raise ValueError("History list is empty")
        
        # figure setup
        plt.figure()
        
        # determine which signals to plot
        all_signals = list(history[0].keys())
        if keys is None:
            signals_to_plot = all_signals
        else:
            # ensure every requested key actually exists
            missing = [k for k in keys if k not in all_signals]
            if missing:
                raise KeyError(f"Requested keys not found in history: {missing}")
            signals_to_plot = keys
    
        # plot each requested signal
        for sig in signals_to_plot:
            values = [entry[sig] for entry in history]
            plt.plot(range(len(values)), values, label=sig)
        
        # decorations
        plt.xlabel("Iteration")
        plt.ylabel("Voltage")
        plt.title("Voltage History")
        plt.legend()
        plt.show()

    
    def draw_boundary_3d(self, xx, yy, y_predictions, epoch, plot_directory, title):
        plt.ion()   # ensure interactive mode is on
        preds = np.array(y_predictions)
        n_outputs = 2
    
        for idx in range(n_outputs):
            zz = preds[:, idx].reshape(xx.shape)
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(xx, yy, zz,
                                   rstride=1, cstride=1,
                                   cmap='viridis', edgecolor='none')
            fig.colorbar(surf, ax=ax, pad=0.1, label='Prediction value')
    
            ax.set_title(f"3D Prediction Surface for {title} (epoch {epoch}, output {idx}, )")
            ax.set_xlabel('Input 1')
            ax.set_ylabel('Input 2')
            ax.set_zlabel('Output')
    
            plt.show()   # each window is now interactive


    def draw_boundary(self, xx, yy, y_predictions, epoch, plot_directory):
        """
        Draws filled-contour maps of continuous predictions over the grid (xx, yy).
        Handles y_predictions as a list of lists: shape (n_points, n_outputs).

        Parameters
        ----------
        xx, yy : 2D arrays
            Meshgrid arrays over your feature space.
        y_predictions : list of lists or 2D array
            Each element is a list/array of length n_outputs; total length = xx.size.
        epoch : int
            Current training epoch (used in filenames/title).
        plot_directory : str
            Directory where data and plot PNGs will be saved.
        """
        # convert to NumPy array of shape (n_points, n_outputs)
        preds = np.array(y_predictions)
        # number of separate outputs to plot
        n_outputs = preds.shape[1]

        # ensure output directory exists
        os.makedirs(plot_directory, exist_ok=True)

        # reshape and plot each output dimension separately
        for idx in range(n_outputs):
            zz = preds[:, idx].reshape(xx.shape)

            plt.figure()
            contour = plt.contourf(xx, yy, zz, levels=100, cmap='viridis')
            cbar = plt.colorbar(contour)
            cbar.set_label('Prediction value', rotation=270, labelpad=15)

            plt.title(f"Model output over grid after epoch {epoch} (output {idx})")
            plt.xlabel('Input 1')
            plt.ylabel('Input 2')

            # save the underlying data for this output
            data_path = os.path.join(
                plot_directory, f"contour_data_epoch{epoch}_output{idx}.npz"
            )
            np.savez(data_path, xx=xx, yy=yy, zz=zz, epoch=epoch, output_index=idx)

            # save the figure
            save_path = os.path.join(
                plot_directory, f"decision_map_epoch{epoch}_output{idx}.png"
            )
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()



    def calc_gradients(self, eldo_process, debug):
        
        
        input_layer = self.layers[0]
        output_layer = self.layers[-1]

        simulation_type = self.simulation_type
        n_of_node_voltages = len(self.all_nodes)
        voltage_dict_free = dict.fromkeys(self.all_nodes[0], None)


        layers = self.layers
        resistive_layers = [layer for layer in layers if getattr(layer, 'trainable', None)]
        output_layer = resistive_layers[-1]
        prediction_list = []
        
        
        diode_connected_flash_params = None
        q = self.q

        result_file = self.result_file

        counter = 0
        
        
        offset = 0
        initialize_synapses(eldo_process, resistive_layers, diode_connected_flash_params, simulation_type, debug)

      
        input_cmd1 = f"SET P(VAC1)=20"
        send_command_to_eldo(eldo_process, input_cmd1, debug)
        input_cmd1 = f"SET P(VAC2)=0"
        send_command_to_eldo(eldo_process, input_cmd1, debug)
        input_cmd1 = f"SET P(VAC3)=-15"
        send_command_to_eldo(eldo_process, input_cmd1, debug)        

        input_cmd1 = f"SET P(AMP)=3"
        send_command_to_eldo(eldo_process, input_cmd1, debug)
        input_cmd1 = f"SET P(AMPC)=1"
        send_command_to_eldo(eldo_process, input_cmd1, debug)  


        input_cmd1 = f"SET P(VDIODE2)=-0.2"
        send_command_to_eldo(eldo_process, input_cmd1, debug)
        input_cmd1 = f"SET P(VDIODE1)=0.2"
        send_command_to_eldo(eldo_process, input_cmd1, debug)  


        input_cmd1 = f"SET P(R_1_3_2)=30000"
        send_command_to_eldo(eldo_process, input_cmd1, debug)
        input_cmd1 = f"SET P(R_1_1_3)=10000"
        send_command_to_eldo(eldo_process, input_cmd1, debug)  

        
        if os.path.exists(result_file):
            file_size = os.path.getsize(result_file)
            offset = file_size
            print(f"File size before draw grid: {file_size} bytes")
            
            
        #run the first simulation
        run_simulation_and_wait(eldo_process, simulation_type, q, debug)  
            

        results = read_update(
                        eldo_process,
                        result_file,
                        voltage_dict_free,
                        offset,
                        simulation_type,
                        transcon_calc=True,
                        debug=debug,
                    )
        voltages_free = results["voltages"].copy()
        
        
        output1f = voltages_free["V_OUT_1_1"]
        output2f = voltages_free["V_OUT_1_2"]
        
        delta_arr = np.logspace(-4,-1,1000)
        loss_grads = []
        res_nudge_list = []
        
        
        res_template = "R_1_1_1"
        og_res = resistive_layers[1].synapse_dict[res_template]

        
        for delta in delta_arr:
            
            
            
            if os.path.exists(result_file):
                file_size = os.path.getsize(result_file)
                offset = file_size
            
            
            cond = 1/resistive_layers[1].synapse_dict[res_template]
            nudge = delta * cond
            new_cond = cond + nudge
            new_res = 1/new_cond
            
            
            input_cmd1 = f"SET P({res_template})={new_res}"
            send_command_to_eldo(eldo_process, input_cmd1, debug)
                
                
            run_simulation_and_wait(eldo_process, simulation_type, q, debug)  



            results = read_update(
                eldo_process,
                result_file,
                voltage_dict_free,
                offset,
                simulation_type,
                transcon_calc=False,
                debug=debug)
            
            voltage_dict_weight_nudged = results["voltages"].copy()
            output1n = voltage_dict_weight_nudged["V_OUT_1_1"]
            output2n = voltage_dict_weight_nudged["V_OUT_1_2"]



            new_cond_neg = cond - nudge  
            new_res_neg = 1/new_cond_neg
            input_cmd_neg = f"SET P({res_template})={new_res_neg}"
            send_command_to_eldo(eldo_process, input_cmd_neg, debug)
                
                
            run_simulation_and_wait(eldo_process, simulation_type, q, debug)  
            #stored_voltages.append(list(voltage_dict_weight_nudged.values()))
            
            results = read_update(
                eldo_process,
                result_file,
                voltage_dict_free,
                offset,
                simulation_type,
                transcon_calc=False,
                debug=debug)
            
            voltage_dict_weight_nudged_neg = results["voltages"].copy()
            
            
            output1n_neg = voltage_dict_weight_nudged_neg["V_OUT_1_1"]
            output2n_neg = voltage_dict_weight_nudged_neg["V_OUT_1_2"]
            
            res_nudge_list.append([output1n/output1f])
                
            weight_grad1 = 1/2 *(output1n ** 2 - output1n_neg ** 2)/(2*nudge)
            weight_grad2 = 1/2 *(output2n ** 2 - output2n_neg ** 2)/(2*nudge)
            
            loss_grad = [weight_grad1, weight_grad2]
            loss_grads.append(loss_grad)



            input_cmd1 = f"SET P({res_template})={og_res}"
            send_command_to_eldo(eldo_process, input_cmd1, debug)
        
            loss_grads_arr = np.array(loss_grads)
            
            
            
        beta_arr = np.logspace(-9, -4, 1000)
        voltage_nudge_list = []
        grad1_list = []
        
        output_layer = layers[-1]
        inudge_dict = output_layer.parameters
        inudge_keys_list = list(inudge_dict.keys())
        
        
        target = 0
        
        in_node_layer0 = "V_IN_0_1"
        out_node_layer0 = "V_OUT_0_1"
        in_node_layer1 = "V_IN_1_1"
        out_node_layer1 = "V_OUT_1_1"
        
        for beta in beta_arr:
 
            #set_resistances(eldo_process, resistor_value_dict, debug = False)
            nudge_current1 = - beta *  output1f
            #nudge_current2 = beta *  output2f
            
            
            
            def calc_losses_set_currents(inudge_dict, outputs, target, beta, mode):
                sample_losses, currents = loss_fn(outputs, target, beta_r, mode)
                inudge_keys_list = list(inudge_dict.keys())
                for k, key in enumerate(inudge_keys_list):
                    inj_currents =  currents.flatten() 
                    inudge_dict[key] = inj_currents[k]    
                        
                set_currents_nudge_mode(eldo_process, inudge_dict, debug)
                return sample_losses
            
            
            
            inudge_dict = {"INUDGE_1" : nudge_current1, "INUDGE_2" : 0}
            set_currents_nudge_mode(eldo_process, inudge_dict, debug = True)
            
            
            
            file_size = os.path.getsize(result_file)
            offset = file_size
            
            
            run_simulation_and_wait(eldo_process, simulation_type, q, debug)  

            results = read_update(
                eldo_process,
                result_file,
                voltage_dict_free,
                offset,
                simulation_type,
                transcon_calc=False,
                debug=debug)
            
            voltage_dict_vol_nudged = results["voltages"].copy()
            
            
            voltage_output_f = voltages_free[out_node_layer1]
            voltage_output_n = voltage_dict_vol_nudged[out_node_layer1]            
            
            voltage1f = voltages_free[in_node_layer1]
            voltage2f = voltages_free[out_node_layer1]
            diff_F = voltage1f - voltage2f
            
            voltage1n = voltage_dict_vol_nudged[in_node_layer1]
            voltage2n = voltage_dict_vol_nudged[out_node_layer1]
            diff_N = voltage1n - voltage2n
            
            
            voltage_nudge = voltage_output_n/voltage_output_f
            voltage_nudge_list.append(voltage_nudge)
            
            grad1 = (1/(2*beta)) * (diff_N ** 2 - diff_F ** 2)
            grad1_list.append(grad1)
            

        def plot_weight_nudge_gradients(delta_arr, loss_grads, central_grads=None):
            """
            loss_grads: array-like shape (N, 2) giving [grad1, grad2] for each delta (one-sided)
            delta_arr: same length array of delta multipliers used (fractional conductance nudges)
            central_grads: optional array-like shape (N, 2) giving central-difference grad estimates
            """
            loss_grads_arr = np.asarray(loss_grads, dtype=float)
            grad1 = loss_grads_arr[:, 0]
            grad2 = loss_grads_arr[:, 1]
        
            fig, ax = plt.subplots()
            ax.plot(delta_arr, grad1, label="weight_grad1 (one-sided)", marker='o')
            ax.plot(delta_arr, grad2, label="weight_grad2 (one-sided)", marker='o')
            if central_grads is not None:
                central_grads_arr = np.asarray(central_grads, dtype=float)
                ax.plot(delta_arr, central_grads_arr[:, 0], '--', label="weight_grad1 (central)", marker='x')
                ax.plot(delta_arr, central_grads_arr[:, 1], '--', label="weight_grad2 (central)", marker='x')
            ax.set_xscale("log")
            ax.set_xlabel("delta (fractional conductance perturbation)") 
            ax.set_ylabel("Gradient estimate") 
            ax.set_title("Weight-nudge gradient vs delta")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.show()
        
        def plot_voltage_nudge_gradient(beta_arr, grad1_list):
            """
            grad1_list: array-like of EqProp-style gradient estimates for each beta
            beta_arr: same length array of beta values used
            """
            grad1_arr = np.asarray(grad1_list, dtype=float)
        
            fig, ax = plt.subplots()
            ax.plot(beta_arr, grad1_arr, label="EqProp-style grad", marker='o')
            ax.set_xscale("log")
            ax.set_xlabel("beta (nudging strength)")
            ax.set_ylabel("Gradient estimate")
            ax.set_title("Voltage-nudge gradient vs beta")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.show()
    
        plot_weight_nudge_gradients(delta_arr, loss_grads)
        plot_voltage_nudge_gradient(beta_arr, grad1_list)
        
        
        
        # assume loss_grads is a list of [weight_grad1, weight_grad2] pairs
        gr1 = np.asarray([i[0] for i in loss_grads], dtype=float)
        
        # take last 200 (or fewer if not enough)
        tail = gr1 if len(gr1) < 200 else gr1[-200:]
        mean_weight_nudged = tail.mean()
        mean_current_nudged = np.asarray(grad1_list[-200:]).mean()
        ratio = mean_weight_nudged/mean_current_nudged
        print(f"Ratio between weight nudged and current nudged gradient {ratio}")
        
        return loss_grads, grad1_list
    
    def free_test_new(self, eldo_process, X_grid, input_function, mode, h5_path, debug=False):
        """
        Unified free-test method for both FSST and TRAN simulations.
        Splits common logic into helper functions.
        """
    
        debug = True
        # 1. Prepare layers, weights, and inputs
        input_layer, resistive_layers, output_layer = self._prepare_layers()
        best_idx, best_acc, W1, W2 = best_epoch_from_dir(h5_path)

        # 2. Initial simulation to get offset and DC operating point
        result_file = self.result_file
        simulation_type = self.simulation_type
        q = self.q
        
        
        diode_connected_flash_params = self.diode_connected_flash_params
        self._write_weights(resistive_layers, W1, W2)
        update_synapses(eldo_process, resistive_layers, diode_connected_flash_params, simulation_type, debug)
        offset = 0
        voltage_dict_free = dict.fromkeys(self.all_nodes[0], None) 
    
        if simulation_type == "TRAN":
            f0, t0 = 1e6, 20e-6
            run_simulation_and_wait(eldo_process, simulation_type, q, debug)
            self._setup_tran(eldo_process, result_file,
                                            offset, voltage_dict_free,
                                            simulation_type, q,
                                            f0, t0, debug)

    # 3. Run sweeps
        return self._sweep_inputs(
            eldo_process,
            X_grid,
            input_function,
            resistive_layers,
            output_layer,
            simulation_type,
            result_file,
            offset,
            voltage_dict_free,
            debug
        )

    def _prepare_layers(self):
        input_layer = self.layers[0]
        resistive_layers = [l for l in self.layers if getattr(l, 'trainable', None)]
        output_layer = resistive_layers[-1]
        return input_layer, resistive_layers, output_layer
    
    def _write_weights(self, layers, W1, W2):
        layers[0].W = W1
        layers[1].W = W2
    
            # now compute column?sums
        col_sum1 = np.sum(W1, axis=0)
        col_sum2 = np.sum(W2, axis=0)
        
        # display them
        print("W1 column sums:", col_sum1)
        print("W2 column sums:", col_sum2)
        
        return col_sum1, col_sum2
            
    def _initial_run(self, eldo_process, sim_type, q, result_file, debug):
        # Kick off first sim and read results
        run_simulation_and_wait(eldo_process, sim_type, q, debug)
        offset = 0
        free_voltage_dict = dict.fromkeys(self.all_nodes[0], None)
        results = read_update(
            eldo_process,
            result_file,
            free_voltage_dict,
            offset,
            sim_type,
            transcon_calc=True,
            debug=debug
        )
        
        return offset, free_voltage_dict
    
    def _setup_tran(self, eldo_process, result_file, offset,
                    voltage_dict, sim_type, q, f0, t0, debug):
        # Perform TRAN-specific DC and bias setup
        results = read_update(
            eldo_process,
            result_file,
            voltage_dict,
            offset,
            sim_type,
            transcon_calc=True,
            debug=debug,
            f0=f0,
            t0=t0
        )
        dc_ds = results["dc_ds_end"]
        dc_gate = results["dc_gate_end"]
        set_the_ic_voltages(eldo_process, dc_gate, dc_ds, debug)

        return voltage_dict
    
    def _sweep_inputs(
        self,
        eldo_process,
        X_grid,
        input_function,
        resistive_layers,
        output_layer,
        sim_type,
        result_file,
        offset,
        voltage_dict,
        debug
    ):
        input_dict = self.layers[0].inputs
        keys = list(input_dict.keys())
        predictions, vin_list, vout_list = [], [], []
        X_in = input_function(X_grid)
    
        for X in X_in:
            # set input vector
            for i, k in enumerate(keys):
                input_dict[k] = -X[i] if i < len(X) else 0
            set_input_voltages(eldo_process, input_dict, debug)
    
            # run sim and read AC
            try:
                offset = os.path.getsize(result_file)
            except FileNotFoundError:
                offset = 0
            run_simulation_and_wait(eldo_process, sim_type, self.q, debug)
            results = read_update(
                eldo_process,
                result_file,
                voltage_dict,
                offset,
                sim_type,
                transcon_calc=True,
                debug=debug,
                **({'f0': 1e6, 't0': 20e-6} if sim_type == 'TRAN' else {})
            )
    
            if sim_type == 'TRAN':
                dc_ds = results["dc_ds_end"]
                dc_gate = results["dc_gate_end"]
                set_the_ic_voltages(eldo_process, dc_gate, dc_ds, debug)

    
    
            ac_currents = results.get('ac_currents', results.get('currents'))
            voltage_dict = results.get('ac_voltages', results.get('voltages'))
    
            # compute gms
            dv1 = voltage_dict['V_IN_0_1'] - voltage_dict['V_OUT_0_1']
            dv2 = voltage_dict['V_IN_1_2'] - voltage_dict['V_OUT_1_2']
            gm1 = ac_currents['XM_0_1_1.S'] / dv1
            gm2 = ac_currents['XM_1_2_2.S'] / dv2
            clipped = [np.clip(abs(g), 0, 1e-4) for g in (gm1, gm2)]
    
            # propagate voltages
            for layer in resistive_layers:
                layer.update__free_voltages(voltage_dict)
            output_layer.update__free_voltages(voltage_dict)
    
            vin_list.append(list(resistive_layers[0].output_free_voltages.values()))
            vout_list.append(list(resistive_layers[1].input_free_voltages.values()))
            predictions.append(clipped)
    
        return predictions, vin_list, vout_list
    
    
    def _setup_tran(
       self,
       eldo_process,
       result_file,
       offset,
       voltage_dict,
       sim_type,
       q,
       f0,
       t0,
       debug
   ):
       # Perform TRAN-specific DC and bias setup
       results = read_update(
           eldo_process,
           result_file,
           voltage_dict,
           offset,
           sim_type,
           transcon_calc=True,
           debug=debug,
           f0=f0,
           t0=t0
       )
       # Extract DC operating point and set initial conditions
       dc_ds = results["dc_ds_end"]
       dc_gate = results["dc_gate_end"]
       set_the_ic_voltages(eldo_process, dc_gate, dc_ds, debug)

       return voltage_dict
def main_setup(sim_params):
    """
    Core setup for netlist build and directory initialization.
    Returns: layers, sample_file, result_file, out_dir, plot_dir, all_nodes
    """

    layers = initialize_network_layers(sim_params) #here also the weight matrices are initialized
    builder = netlist_builder(layers, sim_params)
   
   
   #Here I need to generate a new file name
   ########
    simulation_type = sim_params.simulation_type
    
    transcon_calc = True
    fet_identifiers = ["0_1_1", "1_2_2"]
    process_id = None
    
    
    input_files = create_filenames(sim_params.output_dir, sim_params.sample_file, simulation_type, process_id)
    full_subfolder_path, new_sample_file, result_file_path = input_files
   #printfile = os.path.join(full_subfolder_path, "PRINTFILE.TXT")


    all_nodes = extract_all_nodes_voltages(layers)
    builder.build_netlist(new_sample_file, transcon_calc, fet_identifiers, result_file_path)

   
    net = NetworkAnalyzer(layers, sim_params, input_files, all_nodes)
    net.cs_pmos_w = 11e-6
    
    
    all_nodes = extract_all_nodes_voltages(layers)


    # Setup output directories
    base_dir = sim_params.trained_models_dir
    ts = datetime.now().strftime("%H%M%S")
    folder = (
        f"{simulation_type}_{ts}_{process_id}"
        if process_id else
        f"{simulation_type}_{ts}"
    )
    out_dir = os.path.join(base_dir, folder)
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)



    m_thread = True
    noascii =  True
    debug = False
    eldo_process, q = start_eldo_simulation_drain(new_sample_file, full_subfolder_path, m_thread, noascii, debug)
    setattr(net, 'q', q)    
    def pos_neg_inputs(X, scale_factor = 1):
        # Scale positive and negative inputs
        X_pos = X
        X_neg = -X
        bias = np.ones(X.shape) * 0.3
        # Combine inputs
        X_in = np.hstack((X_pos, X_neg, bias))
        
        return X_in
    
    
    
    
    #net.calc_gradients(eldo_process, debug)
    #if simulation_type == "TRAN":
     #   net.charactarize_synapses_tran(eldo_process, debug)
        
    # elif simulation_type == "FSST":
    #     net.characterize_synapse_fsst(eldo_process, debug)

    #net.plot_transcond(eldo_process, debug)

    net.analyze_and_plot(eldo_process, pos_neg_inputs, plot_dir, debug)
    ###here I run the analyyer
    
    
    
    

if __name__ == "__main__":
    gamma_value =  [1e-8, 5e-9]
    batch_size = 2
    beta = 5e-5
    scale_factor = 0.1
    bias = 0.3
    simulation_type = "TRAN"
    mode = "TESTING"
    if simulation_type == "TRAN":
        sim_params = SimulationParametersTran(scale_factor, bias, batch_size, beta, gamma_value)
    elif simulation_type == "FSST":
        sim_params = SimulationParametersFSST(scale_factor, bias, batch_size, beta, gamma_value)

    main_setup(sim_params)
