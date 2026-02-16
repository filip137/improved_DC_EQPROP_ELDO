#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 18:46:56 2025

@author: filip
"""

import numpy as np
import os
import time
import traceback
from netlist_generation_files import (
    BaseLayer,
    InputLayer,
    DenseLayer,
    NonLinearLayer,
    OutputLayer,
    netlist_builder,
    initialize_network_layers,
)
from simulation_parameters_folder import SimulationParametersDCEvaluator
from support_layer import *
from eldo_support_functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from save_and_load_functions import save_all, save_sim_parameters, best_epoch_from_dir
from datetime import datetime
from loss_functions import MSE
import logging
import logging.handlers
import signal
import subprocess
import torch
from torch.utils.data import DataLoader, TensorDataset
import json
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting (in some versions)
from matplotlib import cm  # For colormap support
import warnings
from pathlib import Path
import shutil
import argparse
warnings.filterwarnings(
    "ignore",
    message="Failed to set pipe buffer size"
)
#Initialize a complete neural network and build a netlist


def load_npz_params(npz_path):
    npz_path = Path(npz_path).expanduser()
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def apply_weights_to_layers(layers, weights_file_path, include_bias=False):
    if weights_file_path is None:
        return
    params = load_npz_params(weights_file_path)
    resistive_layers = [layer for layer in layers if getattr(layer, "trainable", None)]
    if include_bias:
        if any(f"bias_{idx}" in params for idx in range(len(resistive_layers))):
            weights = []
            biases = []
            for idx in range(len(resistive_layers)):
                w_key = f"param_{idx}"
                b_key = f"bias_{idx}"
                if w_key not in params:
                    raise KeyError(
                        f"Missing expected weight {w_key} in {weights_file_path}"
                    )
                if b_key not in params:
                    raise KeyError(
                        f"Missing expected bias {b_key} in {weights_file_path}"
                    )
                weights.append(params[w_key])
                biases.append(params[b_key])
            stacked = stack_weights_with_bias(weights, biases)
            for layer, w in zip(resistive_layers, stacked):
                layer.W = w
                layer.update_synapse_dict(diode_connected_flash_params=None)
            return
        if {"param_0", "param_1", "param_2"}.issubset(params):
            w0 = np.asarray(params["param_0"]).reshape(-1, params["param_0"].shape[-1])
            b0 = np.asarray(params["param_2"]).reshape(1, -1)
            if w0.shape[1] != b0.shape[1]:
                raise ValueError(
                    f"bias length {b0.shape[1]} does not match "
                    f"param_0 out_features {w0.shape[1]}"
                )
            stacked = np.vstack([w0, b0])
            if len(resistive_layers) < 2:
                raise ValueError("Expected at least 2 trainable layers.")
            resistive_layers[0].W = stacked
            resistive_layers[0].update_synapse_dict(diode_connected_flash_params=None)
            resistive_layers[1].W = params["param_1"]
            resistive_layers[1].update_synapse_dict(diode_connected_flash_params=None)
            return
        raise KeyError(
            "include_bias is True but no recognized bias parameters were found."
        )
    for idx, layer in enumerate(resistive_layers):
        param_key = f"param_{idx}"
        if param_key not in params:
            raise KeyError(
                f"Missing expected weight {param_key} in {weights_file_path}"
            )
        layer.W = params[param_key]
        layer.update_synapse_dict(diode_connected_flash_params=None)

def create_pca_analysis_bundle(
    pca_inputs,
    config_path,
    weights_path,
    spice_outputs,
    base_dir="/home/filip/paper_maxicao_simulations/train_and_validate/pca_analysis",
    timestamp=None,
):
    """
    Create a dated PCA analysis bundle with inputs/config/weights/spice outputs.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp is None else str(timestamp)
    out_dir = Path(base_dir).expanduser() / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    pca_inputs_path = out_dir / "pca_inputs.npz"
    if isinstance(pca_inputs, (str, Path)):
        shutil.copy2(Path(pca_inputs).expanduser(), pca_inputs_path)
    else:
        np.savez(pca_inputs_path, **pca_inputs)

    shutil.copy2(Path(config_path).expanduser(), out_dir / "config.json")
    shutil.copy2(Path(weights_path).expanduser(), out_dir / "weights.npz")

    spice_outputs_path = out_dir / "spice_outputs.npz"
    if isinstance(spice_outputs, (str, Path)):
        shutil.copy2(Path(spice_outputs).expanduser(), spice_outputs_path)
    else:
        np.savez(spice_outputs_path, **spice_outputs)

    return out_dir

def create_mnist_analysis_bundle(
    mnist_labels,
    config_path,
    weights_path,
    spice_outputs,
    base_dir="/home/filip/paper_maxicao_simulations/train_and_validate/mnist_analysis",
    timestamp=None,
):
    """
    Create a dated MNIST analysis bundle with inputs/labels/config/weights/spice outputs.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp is None else str(timestamp)
    out_dir = Path(base_dir).expanduser() / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    mnist_labels_path = out_dir / "mnist_labels.npz"
    np.savez(mnist_labels_path, y=mnist_labels)

    shutil.copy2(Path(config_path).expanduser(), out_dir / "config.json")
    shutil.copy2(Path(weights_path).expanduser(), out_dir / "weights.npz")

    spice_outputs_path = out_dir / "spice_outputs.npz"
    np.savez(spice_outputs_path, **spice_outputs)

    return out_dir

def sweep_current_amp_folders(
    root_dir,
    model=None,
    amplifier="PerfectAmpPerfectDiode",
    include_bias=False,
    allow_exponential_default=True,
    inputs_name="linspace_states.npz",
    weights_name="model.npz",
):
    root_path = Path(root_dir).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"Missing sweep root: {root_path}")

    config_paths = []
    for cfg_path in root_path.rglob("run_metadata.json"):
        if cfg_path.parent == root_path:
            continue
        if "current_amp_" not in str(cfg_path.parent):
            continue
        config_paths.append(cfg_path)

    if not config_paths:
        raise FileNotFoundError(f"No run_metadata.json found under {root_path}")

    run_items = []
    results = []
    for cfg_path in sorted(config_paths):
        run_dir = cfg_path.parent
        inputs_path = run_dir / inputs_name
        if not inputs_path.exists():
            alt_inputs = run_dir / "linspace_inputs.npz"
            if alt_inputs.exists():
                inputs_path = alt_inputs
            else:
                results.append(
                    {
                        "run_dir": str(run_dir),
                        "status": "missing_inputs",
                        "config_path": str(cfg_path),
                    }
                )
                continue

        weights_path = run_dir / weights_name
        if not weights_path.exists():
            root_weights = root_path / weights_name
            if root_weights.exists():
                weights_path = root_weights
            else:
                results.append(
                    {
                        "run_dir": str(run_dir),
                        "status": "missing_weights",
                        "config_path": str(cfg_path),
                    }
                )
                continue

        run_items.append(
            {
                "run_dir": run_dir,
                "inputs_path": inputs_path,
                "weights_path": weights_path,
                "config_path": cfg_path,
            }
        )

    for item in run_items:
        run_dir = item["run_dir"]
        inputs_path = item["inputs_path"]
        weights_path = item["weights_path"]
        cfg_path = item["config_path"]
        try:
            sim_params = SimulationParametersDCEvaluator(
                str(cfg_path),
                model=model,
                amplifier=amplifier,
                allow_exponential_default=allow_exponential_default,
            )
            set_include_bias(sim_params, include_bias)

            spice_dump_path = main(
                sim_params,
                str(inputs_path),
                str(weights_path),
                config_path=str(cfg_path),
                validate_mnist=False,
                validate_moons=True,
            )
        except Exception as exc:
            results.append(
                {
                    "run_dir": str(run_dir),
                    "status": "error",
                    "config_path": str(cfg_path),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            continue

        if spice_dump_path is not None:
            spice_dump_path = Path(spice_dump_path)
            if spice_dump_path.parent != run_dir:
                dst = run_dir / spice_dump_path.name
                shutil.copy2(spice_dump_path, dst)
                spice_dump_path = dst

        results.append(
            {
                "run_dir": str(run_dir),
                "status": "ok",
                "config_path": str(cfg_path),
                "spice_dump_path": str(spice_dump_path) if spice_dump_path else None,
            }
        )

    return results

def set_include_bias(sim_params, include_bias):
    current = sim_params.layer_parameters.get("include_bias", False)
    if include_bias and not current:
        sim_params.network_size[0] += 1
    elif not include_bias and current:
        sim_params.network_size[0] -= 1
    if sim_params.network_size[0] <= 0:
        raise ValueError("network_size[0] is invalid after include_bias update.")
    sim_params.layer_parameters["network_size"] = sim_params.network_size
    sim_params.layer_parameters["include_bias"] = include_bias

def stack_weights_with_bias(weights, biases):
    """
    Stack bias vectors as the last row of each weight matrix.

    Parameters
    ----------
    weights : list[np.ndarray]
        Weight matrices shaped [in_features, out_features].
    biases : list[np.ndarray]
        Bias vectors shaped [out_features] or [1, out_features].

    Returns
    -------
    list[np.ndarray]
        Weight matrices with bias appended as the last row.
    """
    if len(weights) != len(biases):
        raise ValueError("weights and biases must have the same length.")

    stacked = []
    for idx, (w, b) in enumerate(zip(weights, biases)):
        w = np.asarray(w)
        b = np.asarray(b).reshape(1, -1)
        if w.ndim != 2:
            raise ValueError(f"weights[{idx}] must be 2D, got shape {w.shape}")
        if b.shape[1] != w.shape[1]:
            raise ValueError(
                f"biases[{idx}] length {b.shape[1]} does not match "
                f"weights[{idx}] out_features {w.shape[1]}"
            )
        stacked.append(np.vstack([w, b]))
    return stacked

def normalize_scale_with_neg_channel(dataset, target_std, input_gain, eps=1e-12):
    """
    Normalize dataset to zero mean/unit std, scale to target_std, apply input_gain,
    and append a negative channel (inputs * -1).

    Parameters
    ----------
    dataset : np.ndarray
        Input data shaped [N, D].
    target_std : float
        Desired standard deviation after scaling.
    input_gain : float
        Gain applied after scaling.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Transformed data shaped [N, 2*D] with positive inputs followed by negative.
    """
    data = np.asarray(dataset, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"dataset must be 2D [N, D], got shape {data.shape}")

    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    normalized = (data - mean) / (std + eps)
    scaled = normalized * float(target_std) * float(input_gain)
    neg = -scaled
    return np.concatenate([scaled, neg], axis=1)

def save_spice_layer_outputs(
    inputs_path,
    layer_voltage_snapshots,
    layer_node_groups,
    layer2_spice,
):
    inputs_path_obj = Path(inputs_path).expanduser()
    stem = inputs_path_obj.stem
    if stem.endswith("cd"):
        stem = stem[:-3]  # remove the trailing 'cd'
    spice_dump_path = inputs_path_obj.with_name(stem + "_spice_layers.npz")
    save_data = {
        f"Layer_{idx + 1}": np.array(layer_voltage_snapshots[idx])
        for idx in range(len(layer_voltage_snapshots))
    }
    for idx, nodes in enumerate(layer_node_groups):
        save_data[f"Layer_Node_Order_{idx + 1}"] = np.array(nodes)
    save_data["Predictions"] = layer2_spice
    np.savez(spice_dump_path, **save_data)
    return spice_dump_path



class MyNetwork:
    def __init__(self, layers, sim_params, input_files, all_nodes):
        # Unpack configuration details
        
        self.layers = layers
        simulation_details = sim_params
        self.loss_fn = MSE(0.0)
        
        
        # Unpack network details.
        self.simulation_type = sim_params.simulation_type
        full_subfolder_path, self.new_sample_file, self.result_file = input_files



        #dataset_details = config.get("dataset_details", {})
        #self.bias = dataset_details["bias"]
        
        self.batch_size = 1
        self.scale_factor = 1
        
        self.all_nodes = all_nodes
        
        
    def free_test_record_layers(
        self,
        eldo_process,
        X_grid,
        weights_file_path,
        debug,
    ):
        X_in = X_grid
        def _voltage_values(layer, attr):
            return list(getattr(layer, attr).values())
        def _layer_node_groups(layers):
            groups = []
            for layer in layers:
                if not getattr(layer, "trainable", False):
                    continue
                node_list = getattr(layer, "output_node_list", None)
                if not node_list:
                    continue
                valid_nodes = [node for node in node_list if node != "0"]
                if valid_nodes:
                    groups.append(valid_nodes)
            return groups
        
        ###Here I first need to write weights
        
        
        input_layer = self.layers[0]
        output_layer = self.layers[-1]
        input_dict = input_layer.inputs
        input_keys_list = list(input_dict.keys())

        q = self.q
        simulation_type = self.simulation_type
        n_of_node_voltages = len(self.all_nodes)
        if isinstance(self.all_nodes[0], list):
            flatten_list = [n for sub in self.all_nodes for n in sub]
        
        voltage_dict_free = dict.fromkeys(flatten_list, None)
        
        
        layers = self.layers
        resistive_layers = [layer for layer in layers if getattr(layer, 'trainable', None)]
        
        output_layer = resistive_layers[-1]
        prediction_list = []

        result_file = self.result_file

        layer_node_groups = _layer_node_groups(layers)
        layer_voltage_snapshots = [[] for _ in layer_node_groups]
        q = self.q
      
        for X in X_in:
            for i, key in enumerate(input_keys_list):
                input_dict[key] = X[i]  # Directly assign the value from X to the corresponding key
                
            set_input_voltages(eldo_process, input_dict, debug)


                ##### 
            try:
                offset = os.path.getsize(result_file)
            except FileNotFoundError:
                if debug:
                    print("need to run the first simulation")
                offset = 0

            run_simulation_and_wait(eldo_process, simulation_type, q, debug)  

                #for plotting there's a function read_update_and_plot
                
            results = read_update(
                    eldo_process,
                    result_file,
                    voltage_dict_free,
                    offset,
                    simulation_type,
                    transcon_calc=False,
                    debug=debug
                    )
                
            voltage_dict_free = results["voltages"]

            for layer in resistive_layers:
                layer.update__free_voltages(voltage_dict_free)
                    
            for idx, nodes in enumerate(layer_node_groups):
                layer_voltage_snapshots[idx].append(
                    [voltage_dict_free[node] for node in nodes]
                )
            
            prediction_list.append(_voltage_values(output_layer, "output_free_voltages"))


        return layer_voltage_snapshots, prediction_list, layer_node_groups

    def free_test_predict_only(
        self,
        eldo_process,
        X_grid,
        weights_file_path,
        debug,
        labels,
    ):
        X_in = X_grid
        def _voltage_values(layer, attr):
            return list(getattr(layer, attr).values())
        
        ###Here I first need to write weights
        
        
        input_layer = self.layers[0]
        output_layer = self.layers[-1]
        input_dict = input_layer.inputs
        input_keys_list = list(input_dict.keys())

        q = self.q
        simulation_type = self.simulation_type
        n_of_node_voltages = len(self.all_nodes)
        if isinstance(self.all_nodes[0], list):
            flatten_list = [n for sub in self.all_nodes for n in sub]
        
        voltage_dict_free = dict.fromkeys(flatten_list, None)
        
        
        layers = self.layers
        resistive_layers = [layer for layer in layers if getattr(layer, 'trainable', None)]
        
        output_layer = resistive_layers[-1]
        prediction_list = []

        result_file = self.result_file

        q = self.q

        if labels is None:
            raise ValueError("labels must be provided for predict-only mode.")

        accuracy_list = []
        correct = 0
        for sample_idx, X in enumerate(X_in):
            for i, key in enumerate(input_keys_list):
                input_dict[key] = X[i]  # Directly assign the value from X to the corresponding key
                
            set_input_voltages(eldo_process, input_dict, debug)


                ##### 
            try:
                offset = os.path.getsize(result_file)
            except FileNotFoundError:
                if debug:
                    print("need to run the first simulation")
                offset = 0

            run_simulation_and_wait(eldo_process, simulation_type, q, debug)  

                #for plotting there's a function read_update_and_plot
                
            results = read_update(
                    eldo_process,
                    result_file,
                    voltage_dict_free,
                    offset,
                    simulation_type,
                    transcon_calc=False,
                    debug=debug
                    )
                
            voltage_dict_free = results["voltages"]

            for layer in resistive_layers:
                layer.update__free_voltages(voltage_dict_free)
            
            outputs = np.array(_voltage_values(output_layer, "output_free_voltages"))
            if outputs.size % 2 != 0:
                raise ValueError("Output layer size must be even for even/odd scoring.")
            y_even = outputs[0::2]
            y_odd = outputs[1::2]
            scores = y_even - y_odd
            prediction = int(np.argmax(scores))
            prediction_list.append(prediction)
            correct += int(prediction == int(labels[sample_idx]))
            accuracy = correct / (sample_idx + 1)
            accuracy_list.append(accuracy)
            print(
                f"Sample {sample_idx + 1}: accuracy={accuracy:.6f}",
                end="\r",
                flush=True,
            )


        print()
        return prediction_list, accuracy_list


def main(sim_params, inputs_path, weights_file_path, config_path=None, validate_mnist=False, validate_moons=False):

    
    #sim_params = SimulationParameters()
    debug = False
    layers = initialize_network_layers(sim_params) #here also the weight matrices are initialized
    builder = netlist_builder(layers, sim_params)
    if debug:
        print(f"Using amplifier: {sim_params.amplifier}")

    # Load trained weights before building the netlist so params in the file match.
    include_bias = sim_params.layer_parameters.get("include_bias", False)
    apply_weights_to_layers(layers, weights_file_path, include_bias=include_bias)

    
    
    #Here I need to generate a new file name
    ########
    simulation_type = sim_params.simulation_type
    
    input_files = create_filenames(sim_params.output_dir, sim_params.sample_file, simulation_type)
    full_subfolder_path, new_sample_file, result_file_path = input_files


    # Setup output directories
    base_dir = sim_params.trained_models_dir
    ts = datetime.now().strftime("%H%M%S")
    folder = (
        f"{simulation_type}_{ts}"
    )
    
    out_dir = os.path.join(base_dir, folder)
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    data_path = os.path.join(plot_dir, 'metrics_data.h5')
    config_file_path = os.path.join(out_dir, "config") 


    

    all_nodes = extract_all_nodes_voltages(layers)
    transcon_calc = None
    fet_identifiers = None
    builder.build_netlist(
        new_sample_file,
        transcon_calc,
        fet_identifiers,
        result_file_path,
    )

    
    net = MyNetwork(layers, sim_params, input_files, all_nodes)


    #pids = get_eldo_pids(eldo_identifier = 'eldo_64.exe')
    m_thread = True
    noascii =  True
    debug = False
    eldo_process, q = start_eldo_simulation_drain(new_sample_file, full_subfolder_path, m_thread, noascii, debug)
    setattr(net, 'q', q)
    time.sleep(1)

    
    
    if validate_mnist:
        from torchvision.datasets import MNIST
        mnist_root = getattr(
            sim_params,
            "mnist_root",
            os.path.expanduser("~/.torch/datasets")
        )
        input_gain = getattr(sim_params, "input_gain", None)
        target_std = getattr(sim_params, "input_target_std", None)
        if input_gain is None:
            raise ValueError("input_gain is missing from config for MNIST validation.")
        if target_std is None:
            target_std = 0.3
        include_bias = sim_params.layer_parameters.get("include_bias", False)
        mnist_test = MNIST(
            root=mnist_root,
            train=False,
            download=True,
            transform=None
        )
        mnist_flat = (
            mnist_test.data.float().reshape(len(mnist_test), -1).numpy()
            / 255.0
        )
        mnist_labels = mnist_test.targets.numpy()
        X_inputs = normalize_scale_with_neg_channel(
            mnist_flat,
            target_std=target_std,
            input_gain=input_gain
        )
        if include_bias:
            bias_col = np.zeros((X_inputs.shape[0], 1), dtype=X_inputs.dtype)
            X_inputs = np.hstack([X_inputs, bias_col])
        X_grid = X_inputs
    else:
        voltages = load_npz_params(inputs_path)
        if validate_moons:
            layer_keys = [k for k in voltages.keys() if k.startswith("Layer_")]
            if not layer_keys:
                raise KeyError("No Layer_* keys found in inputs_path.")
            def _layer_idx(key):
                try:
                    return int(key.split("_", 1)[1])
                except (IndexError, ValueError):
                    return float("inf")
            first_key = min(layer_keys, key=_layer_idx)
            X_grid = voltages[first_key]
        else:
            X_grid = voltages['Layer_0']
        layer1_cd = voltages.get('Layer_1')
        layer2_cd = voltages.get('Layer_2')
        if sim_params.layer_parameters.get("include_bias", False):
            bias_col = np.zeros((X_grid.shape[0], 1), dtype=X_grid.dtype)
            X_grid = np.hstack([X_grid, bias_col])

    if validate_mnist:
        prediction_list, accuracy_list = net.free_test_predict_only(
            eldo_process,
            X_grid,
            weights_file_path,
            debug,
            labels=mnist_labels,
        )
        if config_path is not None:
            create_mnist_analysis_bundle(
                mnist_labels=mnist_labels,
                config_path=config_path,
                weights_path=weights_file_path,
                spice_outputs={
                    "predictions": np.array(prediction_list),
                    "accuracy": np.array(accuracy_list),
                },
            )
    else:
        layer_voltage_snapshots, prediction_list, layer_node_groups = net.free_test_record_layers(
            eldo_process,
            X_grid,
            weights_file_path,
            debug,
        )
    layer2_spice = np.array(prediction_list)
    spice_dump_path = None
    if not validate_mnist:
        spice_dump_path = save_spice_layer_outputs(
            inputs_path,
            layer_voltage_snapshots,
            layer_node_groups,
            layer2_spice,
        )
        if config_path is not None:
            create_pca_analysis_bundle(
                pca_inputs=inputs_path,
                config_path=config_path,
                weights_path=weights_file_path,
                spice_outputs=spice_dump_path,
            )

    

    
        #delete_file_with_chi_extension(new_sample_file)
        
        
        
        
        

    if validate_mnist:
        return accuracy_list[-1] if accuracy_list else 0.0
    return spice_dump_path
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate CD results with SPICE.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--mnist", action="store_true", help="Run MNIST validation.")
    mode_group.add_argument("--pca", action="store_true", help="Run PCA sweep validation.")
    mode_group.add_argument("--moons", action="store_true", help="Run moons validation.")
    parser.add_argument("--weights", default=None, help="Path to weights .npz")
    parser.add_argument("--inputs", default=None, help="Path to inputs .npz")
    parser.add_argument("--config", default=None, help="Path to config JSON")
    parser.add_argument("--model", default=None, help="Model name inside config")
    parser.add_argument(
        "--current-amp-sweep",
        action="store_true",
        help="Sweep current_amp folders under a root and run DC validation.",
    )
    parser.add_argument("--include-bias", action="store_true", default=False)
    parser.add_argument(
        "--amplifier",
        default=None,
        help="Amplifier name (e.g. PerfectAmpPerfectDiode, PerfectAmpQuadraticDiode).",
    )

    parser.add_argument(
        "--sweep-root",
        default="/home/filip/paper_maxicao_simulations/moons_results/current_amp_sweep",
        help="Root folder containing current_amp_* sweep results.",
    )
    args = parser.parse_args()

    base_mnist = "/home/filip/paper_maxicao_simulations/mnist_simulations/2h"
    base_moons = (
        "/home/filip/paper_maxicao_simulations/moons_results/exponential/hidden_3/mismatched amplification/hidden_3/4_0.1/hidden_3/20260126-183311_double_diode_exponential_linspace"
    )

    if args.current_amp_sweep:
        sweep_current_amp_folders(
            args.sweep_root,
            model=args.model,
            amplifier=args.amplifier,
        )
    elif args.moons or (not args.mnist and not args.pca):
        validate_mnist = False
        inputs_path = args.inputs or f"{base_moons}/linspace_states.npz"
        weight_path = args.weights or f"{base_moons}/model.npz"
        config = args.config or f"{base_moons}/run_metadata.json"
        sim_params = SimulationParametersDCEvaluator(
            config,
            model=args.model,
            amplifier=args.amplifier,
            allow_exponential_default=True,
        )
        set_include_bias(sim_params, False)
        sim_params._params_for_netlist["AMP"] = sim_params.amp
        sim_params._params_for_netlist["AMPC"] = sim_params.ampc
        spice_dump_path = main(
            sim_params,
            inputs_path,
            weight_path,
            config_path=config,
            validate_mnist=validate_mnist,
            validate_moons=True,
        )
    elif args.pca:
        validate_mnist = False
        inputs_path = args.inputs or f"{base_mnist}/pca_sweep_states_flat_2h_exp_flat.npz"
        weight_path = args.weights or f"{base_mnist}/model2h_flat.npz"
        config = args.config or f"{base_mnist}/metadata2h.json"
        sim_params = SimulationParametersDCEvaluator(
            config,
            model=args.model,
            amplifier=args.amplifier,
        )
        if args.include_bias is not None:
            set_include_bias(sim_params, args.include_bias)
        main(sim_params, inputs_path, weight_path, config_path=config, validate_mnist=validate_mnist)
    else:
        validate_mnist = True
        inputs_path = args.inputs or f"{base_mnist}/pca_sweep_states_flat_2h_exp_flat.npz"
        weight_path = args.weights or f"{base_mnist}/model2h_flat.npz"
        config = args.config or f"{base_mnist}/metadata2h.json"
        sim_params = SimulationParametersDCEvaluator(
            config,
            model=args.model,
            amplifier=args.amplifier,
        )
        if args.include_bias is not None:
            set_include_bias(sim_params, args.include_bias)
        accuracy = main(
            sim_params,
            inputs_path,
            weight_path,
            config_path=config,
            validate_mnist=validate_mnist,
        )
        print(f"Final accuracy: {accuracy:.6f}")
