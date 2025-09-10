# plot_vs_evolution.py
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from scipy.optimize import fsolve

# --------- YOUR PATH & DIODE/FET MAPPING PARAMS (edit if needed) ----------
METRICS_DIR = "/home/filip/simulations/trained_models/fet_FSST_0816/FSST_105547_33956/plots/metrics_data"
METRICS_DIR = "/home/filip/simulations/trained_models/fet_FSST_0816/FSST_105547_33955/plots/metrics_data"
METRICS_DIR = "/home/filip/simulations/trained_models/fet_FSST_0819/FSST_100914_212222/plots/metrics_data"
# Map weights -> gate voltages: Vg = offset + W * gain[layer]
DIODE_PARAMS = {
    "offset1layer": -0.4,
    "offset2layer": -0.4,
    "gain": [1/8.5e-5, 1/8.5e-5],   # per-layer gains
}
# --------------------------------------------------------------------------

# -------- device model (your function, with I_source = N * 5e-6) --------
def solve_source_and_gm(
    Vg_list,
    k=8.9e-5/2,
    I_per_device=5e-6,   # each device draws this much; total I = N * I_per_device
    Vth=0.537,
    Vs_guess=None,
    tol=1e-12
):
    Vg = np.array(Vg_list, dtype=float) + 2.1
    N = Vg.size
    I_source = N * I_per_device

    if Vs_guess is None:
        Vs_guess = (np.min(Vg) - Vth) / 2.0

    def residual(Vs):
        Ids = k * np.maximum(Vg - Vs - Vth, 0.0)**2
        return np.sum(Ids) - I_source

    Vs_solution, = fsolve(residual, x0=Vs_guess, xtol=tol)
    Vov = np.maximum(Vg - Vs_solution - Vth, 0.0)
    gm_list = 2.0 * k * Vov
    return float(Vs_solution), np.asarray(gm_list, dtype=float)

# -------------------- helpers --------------------
def _load_npz_array(path: str, key: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        if key not in data:
            raise KeyError(f"Key '{key}' not in {path}. keys={list(data.keys())}")
        return data[key]

def _as_epoch_list(obj) -> List[np.ndarray]:
    arr = np.array(obj, copy=False)
    if arr.dtype == object:
        return [np.array(x, copy=False) for x in arr.tolist()]
    if arr.ndim == 0:
        return [np.atleast_1d(arr)]
    return [arr[i] for i in range(arr.shape[0])]

def _compute_vs_for_columns(Vg_matrix: np.ndarray) -> np.ndarray:
    """Run solve_source_and_gm for each COLUMN of Vg_matrix; returns Vs per column."""
    Vg = np.asarray(Vg_matrix, dtype=float)
    n_in, n_out = Vg.shape
    Vs_cols = np.empty(n_out, dtype=float)
    for j in range(n_out):
        Vs_j, _ = solve_source_and_gm(Vg[:, j])
        Vs_cols[j] = Vs_j
    return Vs_cols

def _pad_to_rectangular(series: List[np.ndarray]) -> np.ndarray:
    """[epochs, max_cols], pad with NaN for missing columns."""
    E = len(series)
    max_cols = max((len(x) for x in series), default=0)
    A = np.full((E, max_cols), np.nan, dtype=float)
    for i, row in enumerate(series):
        A[i, :len(row)] = row
    return A

def plot_vs_evolution(vs_list: List[np.ndarray], title: str, save_path: str = None):
    """Plot per-column Vs over epochs + mean ± std band."""
    vs_arr = _pad_to_rectangular(vs_list)   # shape [epochs, n_cols]
    epochs = np.arange(vs_arr.shape[0])

    plt.figure()
    for j in range(vs_arr.shape[1]):          # each column (output) trajectory
        plt.plot(epochs, vs_arr[:, j])
    mu = np.nanmean(vs_arr, axis=1)
    sigma = np.nanstd(vs_arr, axis=1)
    plt.plot(epochs, mu, linewidth=2)
    plt.fill_between(epochs, mu - sigma, mu + sigma, alpha=0.2)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Vs (V)")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()

# ---------- weights -> gate voltages using your mapping ----------
def _weights_to_gate_voltages(W1_list: List[np.ndarray], W2_list: List[np.ndarray]) -> (List[np.ndarray], List[np.ndarray]):
    off1 = float(DIODE_PARAMS["offset1layer"])
    off2 = float(DIODE_PARAMS["offset2layer"])
    g0, g1 = [float(x) for x in DIODE_PARAMS["gain"]]
    Vg1_list = [off1 + g0 * W for W in W1_list]
    Vg2_list = [off2 + g1 * W for W in W2_list]
    return Vg1_list, Vg2_list

# -------------------- main workflow --------------------
def main():
    metrics_dir = METRICS_DIR
    gv_npz = os.path.join(metrics_dir, "gate_voltages.npz")
    weights_npz = os.path.join(metrics_dir, "weights.npz")


    if os.path.isfile(weights_npz):
            # Build gate voltages from weights using your mapping
        try:
            evo1 = _load_npz_array(weights_npz, "weight_matrix_evolution_1")
            evo2 = _load_npz_array(weights_npz, "weight_matrix_evolution_2")
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load weight evolutions from {weights_npz}: {e}"
            )
        W1_list, W2_list = _as_epoch_list(evo1), _as_epoch_list(evo2)
        gv1_list, gv2_list = _weights_to_gate_voltages(W1_list, W2_list)

            # (optional) save for reuse next time
        np.savez_compressed(
                gv_npz,
                gate_voltages_layer1_evolution=np.array(gv1_list, dtype=object),
                gate_voltages_layer2_evolution=np.array(gv2_list, dtype=object),
            )
    else:
            # Nothing found?help the user debug
        files = sorted(os.listdir(metrics_dir)) if os.path.isdir(metrics_dir) else []
        raise FileNotFoundError(
            f"None of vs_gm_from_gate_voltages.npz, gate_voltages.npz, or weights.npz "
            f"were found in:\n{metrics_dir}\n\nContents:\n{files}"
        )

        # Compute Vs from gate voltages
    vs1 = [ _compute_vs_for_columns(M) for M in gv1_list ]
    vs2 = [ _compute_vs_for_columns(M) for M in gv2_list ]

        # (optional) save Vs for quick future plotting

    # Plot & save
    plot_vs_evolution(vs1, "Vs evolution ? Layer 1", save_path=os.path.join(metrics_dir, "vs_evolution_layer1.png"))
    plot_vs_evolution(vs2, "Vs evolution ? Layer 2", save_path=os.path.join(metrics_dir, "vs_evolution_layer2.png"))

if __name__ == "__main__":
    main()
