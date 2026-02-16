import numpy as np
import os
import json
#import h5py

from typing import Optional, Sequence, Any, Dict, Union, Tuple
from netlist_generation_files import SimulationParameters

def load_metrics(h5_path: str) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as f:
        out = {}
        for name in f:
            data = f[name][()]
            if isinstance(data, np.ndarray) and data.dtype == object:
                # unpickle fallback elements
                import pickle
                data = [pickle.loads(x) for x in data.tolist()]
                out[name] = data
            else:
                out[name] = data
        return out


def _load_npz_array(path: str, key: str):
    data = np.load(path, allow_pickle=True)
    if key not in data:
        raise KeyError(f"{key} not in {path}")
    return data[key]

def best_epoch_from_dir(metrics_dir: str):
    """
    Given a directory containing accuracy.npz, big_loss.npz, and weights.npz,
    find the epoch with the highest accuracy and return:
        best_idx, best_accuracy, weight_matrix_evolution_1_at_best,
        weight_matrix_evolution_2_at_best
    """
    if not os.path.isdir(metrics_dir):
        raise NotADirectoryError(f"{metrics_dir} is not a directory")

    acc_path = os.path.join(metrics_dir, "accuracy.npz")
    weights_path = os.path.join(metrics_dir, "weights.npz")

    if not os.path.isfile(acc_path):
        raise FileNotFoundError(f"{acc_path} missing")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"{weights_path} missing")

    accuracy_list = _load_npz_array(acc_path, "accuracy_list")
    # collapse if multidimensional: average over trailing dims
    acc_arr = np.array(accuracy_list, dtype=float)
    if acc_arr.size == 0:
        raise ValueError("accuracy_list is empty")
    if acc_arr.ndim > 1:
        acc_per_epoch = np.nanmean(acc_arr, axis=tuple(range(1, acc_arr.ndim)))
    else:
        acc_per_epoch = acc_arr

    # pick best epoch (first max), ignoring NaNs
    try:
        best_idx = int(np.nanargmax(acc_per_epoch))
    except ValueError:
        raise ValueError("accuracy_list contains only NaNs; cannot pick best epoch")
    best_accuracy = float(acc_per_epoch[best_idx])

    # load weight evolution
    evo1 = _load_npz_array(weights_path, "weight_matrix_evolution_1")
    evo2 = _load_npz_array(weights_path, "weight_matrix_evolution_2")

    def extract_epoch(evo, idx):
        # handle object arrays or plain arrays
        arr = np.array(evo, copy=False)
        if arr.dtype == object:
            return arr[idx]
        return arr[idx]  # assumes first axis is epoch

    wm_evo1_best = extract_epoch(evo1, best_idx)
    
    wm_evo2_best = extract_epoch(evo2, best_idx)

    return best_idx, best_accuracy, wm_evo1_best, wm_evo2_best


def _unwrap_object_array(obj):
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        try:
            return obj.tolist()
        except Exception:  # fallback: leave as-is
            pass
    return obj





def _make_array(lst: Sequence[Any]) -> np.ndarray:
    """
    Helper to convert a list of matrices or scalars into a numpy array.
    Tries stacking; if that fails, falls back to an object array (requires allow_pickle=True when loading).
    """
    try:
        return np.stack(lst)
    except Exception:
        obj = np.empty(len(lst), dtype=object)
        obj[:] = lst
        return obj


def save_all(
    out_dir: str,
    results: Dict[str, Any],
    weight_matrices_1: Sequence[Any],
    weight_matrices_2: Sequence[Any],
    accuracy_list: Optional[Sequence[Union[float, Any]]] = None,
    big_loss_list: Optional[Sequence[Union[float, Any]]] = None,
) -> Dict[str, str]:
    """
    Save weights and their evolutions to weights.npz, and always write accuracy.npz and big_loss.npz.
    If accuracy_list or big_loss_list is None, saves an empty array in the respective file.

    Returns a dict with keys 'weights', 'accuracy', 'big_loss' mapping to the saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ===== weights =====
    weights_file_path = os.path.join(out_dir, "weights.npz")
    wm1 = results["weight_matrix_1"]
    wm2 = results["weight_matrix_2"]
    evolution1 = _make_array(weight_matrices_1)
    evolution2 = _make_array(weight_matrices_2)

    np.savez_compressed(
        weights_file_path,
        weight_matrix_1=wm1,
        weight_matrix_2=wm2,
        weight_matrix_evolution_1=evolution1,
        weight_matrix_evolution_2=evolution2,
    )

    # ===== accuracy ===== (always save, empty if None)
    accuracy_path = os.path.join(out_dir, "accuracy.npz")
    if accuracy_list is not None:
        acc_array = _make_array(accuracy_list)
    else:
        acc_array = np.empty((0,), dtype=float)
    np.savez_compressed(accuracy_path, accuracy_list=acc_array)

    # ===== big loss ===== (always save, empty if None)
    big_loss_path = os.path.join(out_dir, "big_loss.npz")
    if big_loss_list is not None:
        loss_array = _make_array(big_loss_list)
    else:
        loss_array = np.empty((0,), dtype=float)
    np.savez_compressed(big_loss_path, big_loss_list=loss_array)

    return {
        "weights": weights_file_path,
        "accuracy": accuracy_path,
        "big_loss": big_loss_path,
    }

def load_metrics(path):
    with h5py.File(path, "r") as f:
        accuracy_list = f["accuracy_list"][()]
        losses = f["losses"][()]

        def load_evolution(name):
            node = f[name]
            if isinstance(node, h5py.Dataset):
                return list(node)  # stacked array -> list of arrays per epoch if desired
            else:  # group with per-epoch datasets
                # sorted by key to preserve order
                return [node[k][()] for k in sorted(node.keys(), key=lambda x: int(x))]

        w_evo1 = load_evolution("weight_matrix_evolution_1")
        w_evo2 = load_evolution("weight_matrix_evolution_2")

    return {
        "accuracy_list": accuracy_list,
        "losses": losses,
        "weight_matrix_evolution_1": w_evo1,
        "weight_matrix_evolution_2": w_evo2,
    }


def save_sim_parameters(sim_params, file_path):
    """
    Save the simulation parameters to a JSON file.

    Args:
        sim_params (SimulationParameters): An instance of SimulationParameters.
        file_path (str): The file path where the JSON data will be saved.
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(sim_params.__dict__, file, indent=4)
        print(f"Simulation parameters successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving simulation parameters: {e}")

from dataclasses import dataclass, fields
from typing import Any, Iterator, List, Dict
from collections.abc import Mapping

# A little mix-in that turns any dataclass
# into a dict-like Mapping
class DataClassMapping(Mapping):
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        # yield each field name in declaration order
        for f in fields(self):
            yield f.name

    def __len__(self) -> int:
        return len(fields(self))


@dataclass
class Bounds(DataClassMapping):
    min_conductance: float
    max_conductance: float


@dataclass
class InitializerInner(DataClassMapping):
    init_type: str
    params: Dict[str, float]
    seed: int


@dataclass
class Initializer(DataClassMapping):
    initializer: InitializerInner


@dataclass
class DiodeConnectedFlashParams(DataClassMapping):
    offset1layer: float
    offset2layer: float
    gain: List[float]


@dataclass
class TransientParamsForNetlist(DataClassMapping):
    sycap: float
    syres: float
    input_read_volt: float
    output_read_volt: float
    start_read_time: float
    end_read_time: float
    start_write_time: float
    end_write_time: float
    START_DISCHARGE_TIME: float
    END_DISCHARGE_TIME: float
    RISE_TIME: float
    FORM: int
    LOW_NOISE_OPTION: int


@dataclass
class FSSTParamsForNetlist(DataClassMapping):
    VDC_BIAS1: float
    PMOS_CS_V_BIAS: float
    layer1_bias_curr: float
    layer2_bias_curr: float
    FORM: int
    LOW_NOISE_OPTION: int

from dacite import from_dict, Config
from netlist_generation_files.simulation_parameters import SimulationParameters
def load_simulation_parameters_new(path: str) -> SimulationParameters:
    with open(path, "r") as f:
        cfg = json.load(f)
    return from_dict(
        data_class=SimulationParameters,
        data=cfg,
        config=Config(cast=[float, int])  # allow ints?floats where needed
    )

def load_sim_parameters(file_path: str) -> SimulationParameters:
    """
    Load SimulationParameters from a JSON file previously written by save_sim_parameters.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Simulation parameters file not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    # Expected keys depend on SimulationParameters signature; adjust if yours differs
    expected_keys = {
        "scale_factor",
        "bias",
        "batch_size",
        "beta",
        "gamma_values"
    }
    missing = expected_keys - data.keys()
    if missing:
        raise ValueError(f"SimulationParameters JSON missing fields: {missing}")

    return SimulationParameters(
        data["scale_factor"],
        data["bias"],
        data["batch_size"],
        data["beta"],
        data["gamma_values"],
    )


from typing import Union
from simulation_parameters_folder import SimulationParametersFSST
from pathlib import Path

# ---- MAIN LOADER ----
def load_fsst_simparams(config_path: Union[str, Path]) -> SimulationParametersFSST:
    """
    Load a SimulationParametersFSST from a JSON config and overwrite attributes
    on the created instance with values from the file (including nested dicts).

    Required keys in JSON for constructor:
        - scale_factor, bias, batch_size, beta, gamma_values, load_weights, h5_file
    Everything else is applied post-init via attribute overwrite.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        data = json.load(f)

    required = ["scale_factor", "bias", "batch_size", "beta",
                "gamma_values", "load_weights", "h5_file"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Config missing required keys for constructor: {missing}")

    # 1) Construct with the minimal required args
    sp = SimulationParametersFSST(
        scale_factor = data["scale_factor"],
        bias         = data["bias"],
        batch_size   = data["batch_size"],
        beta         = data["beta"],
        gamma_values = data["gamma_values"],
        output_scale = data["output_scale"],
        load_weights = data["load_weights"],
        h5_file      = data["h5_file"],
    )

    # 2) Overwrite ALL other keys from JSON onto the instance
    #    (including nested dicts like layer_parameters, transient_params_for_netlist, etc.)
    for k, v in data.items():
        # Skip the ones the ctor already set; we'll still set them again to ensure exact match
        setattr(sp, k, v)

    # 3) Optional: keep derived convenience dicts consistent if the JSON didn't provide them
    # If your JSON already has these dicts, the setattr above overwrote them; if not, keep sp's originals.
    # (Nothing to do here unless you want to force a recompute)

    # 4) Optional sanity checks (uncomment if you want hard validation)
    # _validate_fsst(sp)

    return sp

