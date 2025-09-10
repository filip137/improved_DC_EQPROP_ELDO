# new_multiprocessing_limited.py
"""
Multiprocessing runner with controlled concurrency ? importable API.

Key improvements:
- Exposes run(...) so you can call from Python (REPL, notebook, or other scripts).
- All previous globals become function parameters with sensible defaults.
- Preserves your logging, concurrency caps, and ELDO semaphore throttling.
- Returns (successes, errors) instead of only printing.

Usage example (Python/REPL):
    from new_multiprocessing_limited import run
    successes, errors = run(
        simulation_type="FSST",
        dataset="moons",
        load_weights=False,
        h5_path=None,
        gamma_value_list=[[-6e-9,0], [0,0], [6e-9,0]],
        scale_factor_list=[0.4],
        sim_max_workers=None,  # None -> auto compute
        eldo_max=None,         # None -> default half of workers
        verbose_progress=True
    )
"""

import os
import traceback
import multiprocessing
import logging
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from typing import List, Tuple, Optional, Any

from save_and_load_functions import load_fsst_simparams

# ----------------- Hard thread limits (set BEFORE heavy imports) -----------------
def _set_thread_env_limits():
    # Keep 1 thread per worker for common math libs
    env_defaults = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",  # Apple Accelerate
        "BLIS_NUM_THREADS": "1",
        "TORCH_NUM_THREADS": "1",
        "KMP_WARNINGS": "0",
        "KMP_AFFINITY": "disabled",
    }
    for k, v in env_defaults.items():
        os.environ.setdefault(k, v)

_set_thread_env_limits()  # parent sets env so children inherit

# Optional progress bar (only used if available and verbose_progress=True)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

# Explicit spawn context (reliable with threads/subprocesses)
MP_CTX = multiprocessing.get_context("spawn")
try:
    multiprocessing.set_start_method("spawn", force=False)
except RuntimeError:
    # already set by the interpreter or another lib
    pass

# Import after env limits to avoid pre-spawn thread storms
from neural_network_unified import train


# ----------------- Defaults you had, now encapsulated -----------------
_DEFAULTS = {
    "TRAN": {
        "LOG_DIR": "/home/filip/simulations/logging_folder_trans",
        "DEFAULT_H5": "/home/filip/simulations/testing_plots/fet_FSST_0802/FSST_114247_262613/plots/metrics_data.h5",
    },
    "FSST": {
        "LOG_DIR": "/home/filip/simulations/logging_folder_FSST",
        # a few of your historical paths; last one wins unless overridden by h5_path/config_path
        "DEFAULT_H5s": [
            "/home/filip/simulations/validation_plots/fet_FSST_0805/FSST_110129_228827/plots/metrics_data.h5",
            "/home/filip/simulations/trained_models/fet_FSST_0816/FSST_085744_11546/plots/metrics_data",
            "/home/filip/simulations/trained_models/resistor_FSST_0816/FSST_155236_81520/plots/metrics_data",
            "/home/filip/simulations/trained_models/fet_FSST_0816/FSST_181418_145068/plots/metrics_data",
        ],
        "DEFAULT_CONFIG": "/home/filip/simulations/trained_models/fet_FSST_0816/FSST_181418_145068/config",
    },
    "DC": {
        "LOG_DIR": "/home/filip/simulations/logging_folder_DC",
        "DEFAULT_H5": "/home/filip/simulations/trained_models/resistor_DC_0815/DC_170919_136152/plots/metrics_data.h5",
    },
}


def _pick_defaults(simulation_type: str):
    st = simulation_type.upper()
    if st not in _DEFAULTS:
        raise ValueError(f"Unknown SIMULATION_TYPE: {simulation_type}")
    return _DEFAULTS[st]


def _configure_main_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, "main.log"), mode="a", encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s [PID %(process)d] %(levelname)s %(name)s: %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.propagate = False
    return logger


def _get_worker_logger(log_dir: str, sim_params):
    pid = os.getpid()
    sim_id = repr(sim_params)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in sim_id)[:50]
    name = f"worker-{pid}-{safe}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fname = os.path.join(log_dir, f"{name}.log")
        fh = logging.FileHandler(fname, mode="a", encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s [PID %(process)d] %(levelname)s %(name)s: %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.propagate = False
    return logger


# ----------------- Worker init: signals + thread caps + shared ELDO semaphore ----
_ELDO_SEMA = None  # set by initializer
_LOG_DIR_FOR_WORKER = None  # filled by initializer for proper per-worker logging


def _init_worker_signals_threads_and_sema(sema=None, log_dir=None):
    # Ignore Ctrl-C in workers; main coordinates shutdown
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Re-assert thread env caps inside workers (paranoia)
    _set_thread_env_limits()
    # Optional: cap PyTorch interop if used inside train()
    try:
        import torch  # noqa
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    except Exception:
        pass
    # Install shared semaphore + log dir
    global _ELDO_SEMA, _LOG_DIR_FOR_WORKER
    _ELDO_SEMA = sema
    _LOG_DIR_FOR_WORKER = log_dir


def _worker(sim_params):
    logger = _get_worker_logger(_LOG_DIR_FOR_WORKER, sim_params)
    pid = os.getpid()
    logger.info(
        f"START simulation: {sim_params} (pid={pid}) | "
        f"load_weights={getattr(sim_params,'load_weights',None)} | "
        f"h5_file={getattr(sim_params,'h5_file',None)}"
    )
    try:
        if _ELDO_SEMA is not None:
            _ELDO_SEMA.acquire()
        try:
            result = train(sim_params, process_id=pid, logger=logger)
        finally:
            if _ELDO_SEMA is not None:
                _ELDO_SEMA.release()
        logger.info(f"SUCCESS simulation: {sim_params} (pid={pid}) -> {result}")
        return sim_params, result, None
    except Exception:
        tb = traceback.format_exc()
        logger.exception(f"EXCEPTION in simulation: {sim_params} (pid={pid})")
        return sim_params, None, tb


# ----------------- Utilities -----------------
def _terminate_active_children():
    # Best-effort cleanup of any still-alive Python child processes
    for p in multiprocessing.active_children():
        try:
            p.terminate()
        except Exception:
            pass
    for p in multiprocessing.active_children():
        try:
            p.join(timeout=5)
        except Exception:
            pass


def _safe_shutdown(executor, futures, *, wait, cancel_pending=True):
    """
    Shutdown compatible with Python < 3.9 (no cancel_futures kw).
    Cancels pending futures, then calls shutdown(wait=...).
    """
    if cancel_pending:
        for f in futures:
            f.cancel()  # only cancels if not yet running
    try:
        executor.shutdown(wait=wait, cancel_futures=True)  # type: ignore[arg-type]
    except TypeError:
        executor.shutdown(wait=wait)


def _getenv_int(name, default):
    val = os.getenv(name)
    if val is None:
        return default
    try:
        n = int(val)
        return n if n >= 0 else default
    except ValueError:
        return default


def _compute_max_workers(env_override: Optional[int] = None):
    # Env override
    if env_override and env_override >= 1:
        return env_override
    env_n = _getenv_int("SIM_MAX_WORKERS", 0)
    if env_n >= 1:
        return env_n
    # Auto: conservative for small CPUs; ~half cores for big machines
    cpu = os.cpu_count() or 1
    if cpu >= 24:
        return max(1, cpu // 2)   # e.g., 32 cores -> 16 workers
    return min(4, max(1, cpu // 3))


def _pick_simulation_parameters_class(simulation_type: str, dataset:str):
    st = simulation_type.upper()
    if st == "TRAN":
        from simulation_parameters_folder import SimulationParametersTran as SimulationParameters
    elif st == "FSST":
        if dataset == "moons":
            from simulation_parameters_folder import SimulationParametersFSST as SimulationParameters
        elif dataset == "iris":
            from simulation_parameters_folder import SimulationParametersFSST_iris as SimulationParameters
        else:
            # Fall back to the common FSST class if user provides a custom dataset
            from simulation_parameters_folder import SimulationParametersFSST as SimulationParameters
    elif st == "DC":
        if dataset == "moons":
            from simulation_parameters_folder import SimulationParametersDC as SimulationParameters
        elif dataset == "iris":
            from simulation_parameters_folder import SimulationParametersDC_iris as SimulationParameters
    else:
        raise ValueError(f"Unknown SIMULATION_TYPE: {simulation_type}")
    return SimulationParameters


import datetime

def _build_param_grid(
    SimulationParameters,
    simulation_type: str,
    gamma_value_list: List[List[float]],
    scale_factor_list: List[float],
    batch_size: int,
    beta: float,
    bias: float,
    load_weights: bool,
    h5_path_to_use: Optional[str],
    load_config_path: bool,
    config_path: Optional[str],
    amplifier: str,
    beta_list: List[float],
    output_scale_list: List[float],
    base_aex: str = "/home/filip/simulations/aex",
    base_models: str = "/home/filip/simulations/trained_models"
    ):
    sim_params_list = []

    # Recompute date string once per batch
    date_str = datetime.datetime.now().strftime("%m%d")

    if load_weights and (h5_path_to_use is not None) and (not os.path.isdir(h5_path_to_use)) and (not os.path.isfile(h5_path_to_use)):
        raise FileNotFoundError(f"H5 weights path not found: {h5_path_to_use}")

    if load_config_path:
        if not config_path:
            raise FileNotFoundError("load_config_path=True but config_path is not provided.")
        base_sp = load_fsst_simparams(config_path)
        for gamma_values in gamma_value_list:
            sp = copy.deepcopy(base_sp)
            sp.gamma_values = gamma_values
            sp.batch_size = batch_size
            sp.load_weights = load_weights
            sp.beta = beta
            sp.bias = bias
            sp.h5_file = h5_path_to_use
            try:
                sp.simulation_type = simulation_type
            except Exception:
                pass
            sp.amplifier = amplifier

                # --- RE-INIT paths ---
            sp.sample_file = f"{sp.synapse}_{sp.simulation_type}_netlist"
            sp.output_dir = os.path.join(base_aex, date_str)
            sp.trained_models_dir = os.path.join(
                    base_models,
                    f"{sp.synapse}_{sp.simulation_type}_{date_str}"
                )

            sim_params_list.append(sp)
        return sim_params_list

    for gamma_values in gamma_value_list:
        for scale_factor in scale_factor_list:
            for beta in beta_list:
                for output_scale in output_scale_list:
                    sp = SimulationParameters(
                        scale_factor, bias, batch_size, beta, gamma_values, output_scale, load_weights, h5_path_to_use
                    )
                    sp.amplifier = amplifier
                    try:
                        sp.simulation_type = simulation_type
                    except Exception:
                        pass
        
                    # --- RE-INIT paths ---
                    sp.sample_file = f"{sp.synapse}_{sp.simulation_type}_netlist"
                    sp.output_dir = os.path.join(base_aex, date_str)
                    sp.trained_models_dir = os.path.join(
                        base_models,
                        f"{sp.synapse}_{sp.simulation_type}_{date_str}"
                    )
        
                    sim_params_list.append(sp)

    return sim_params_list


# ----------------- Public API -----------------
def run(
    *,
    simulation_type: str = "DC",       # "TRAN", "FSST", or "DC"
    dataset: str = "moons",    # 
    load_weights: bool = False,
    load_config_path: bool = False,
    h5_path: Optional[str] = "/home/filip/simulations/trained_models/resistor_FSST_0821/FSST_145936_126767/plots/metrics_data",       # str or None
    config_path: Optional[str] = "/home/filip/simulations/trained_models/resistor_FSST_0821/FSST_145936_126767/config",   # FSST config dir if load_config_path=True
    gamma_value_list: Optional[List[List[float]]] = None,
    scale_factor_list: Optional[List[float]] = None,
    batch_size: int = 5,
    beta: float = 2e-6,
    bias: float = 0.3,
    amplifier: str = "PerfectAmp",
    sim_max_workers: Optional[int] = None,   # None -> auto; else fixed
    eldo_max: Optional[int] = 24,          # None -> default half of workers; 0 -> disable semaphore
    log_dir: Optional[str] = None,           # None -> defaults by sim type
    verbose_progress: bool = True,
) -> Tuple[list, list]:
    """
    Run a batch of simulations with controlled concurrency.

    Returns:
        (successes, errors)
        successes: list of (sim_params, result)
        errors:    list of (sim_params, traceback_str)
    """
    # Defaults & sanity
    defaults = _pick_defaults(simulation_type)
    if log_dir is None:
        log_dir = defaults["LOG_DIR"]

    # Determine default H5
    if h5_path is None:
        if simulation_type.upper() == "FSST":
            h5_candidates = defaults.get("DEFAULT_H5s", [])
            h5_path_to_use = h5_candidates[-1] if h5_candidates else None
        else:
            h5_path_to_use = defaults.get("DEFAULT_H5", None)
    else:
        h5_path_to_use = h5_path

    if simulation_type.upper() == "FSST" and config_path is None:
        config_path = defaults.get("DEFAULT_CONFIG", None)

    # Parameter lists
    #gamma_value_list = [[-1e-7, 0], [1e-7, 0]]
    gamma_value_list = [[0.5e-6, 5e-7]]
#
    #gamma_value_list = [[-1e-7, 0], [-3e-7, 0], [-5e-7, 0]]

    scale_factor_list = [0.3, 0.4, 0.5]
    beta_list = [1e-6]
    output_scale_list = [2, 4, 6, 8]
    # Logging
    main_logger = _configure_main_logger(log_dir)

    # Worker counts
    max_workers = _compute_max_workers(sim_max_workers)
    default_eldo = max(1, max_workers // 2)
    if eldo_max is None:
        eldo_max = _getenv_int("ELDO_MAX", default_eldo)

    # Shared semaphore
    eldo_sema = None
    if eldo_max > 0:
        eldo_sema = MP_CTX.BoundedSemaphore(eldo_max)

    # SimulationParameters selection
    SimulationParameters = _pick_simulation_parameters_class(simulation_type, dataset)

    # Build grid
    sim_params_list = _build_param_grid(
        SimulationParameters,
        simulation_type,
        gamma_value_list,
        scale_factor_list,
        batch_size,
        beta,
        bias,
        load_weights,
        h5_path_to_use,
        load_config_path,
        config_path,
        amplifier,
        beta_list,
        output_scale_list
    )

    # Header logs/prints
    main_logger.info(
        f"Starting parameter search with {len(sim_params_list)} jobs | "
        f"max_workers={max_workers} | ELDO_MAX={eldo_max}"
    )
    print("Start method:", multiprocessing.get_start_method())
    print("Max workers:", max_workers)
    print("ELDO_MAX:", eldo_max)

    # Submit & run
    successes, errors = [], []
    executor = None
    futures = []
    try:
        executor = ProcessPoolExecutor(
            mp_context=MP_CTX,
            initializer=_init_worker_signals_threads_and_sema,
            initargs=(eldo_sema, log_dir),
            max_workers=max_workers,
        )

        for sim in sim_params_list:
            gv = getattr(sim, "gamma_values", None)
            main_logger.info(f"Submitting job with gamma_values={gv}")
            futures.append(executor.submit(_worker, sim))

        future_to_param = dict(zip(futures, sim_params_list))

        iterator = as_completed(futures)
        if verbose_progress and tqdm is not None:
            iterator = tqdm(iterator, total=len(futures), desc="Running sims")

        for future in iterator:
            sim = future_to_param[future]
            try:
                sim_ret, result, err = future.result()
            except Exception as e:
                main_logger.exception(f"Future for {sim} raised unexpectedly: {e}")
                errors.append((sim, f"Future exception: {e}"))
                continue

            if err:
                errors.append((sim_ret, err))
            else:
                successes.append((sim_ret, result))
                gv = getattr(sim, "gamma_values", None)
                main_logger.info(f"Completed job gamma_values={gv} -> {result}")

    except KeyboardInterrupt:
        main_logger.warning("Interrupted by user. Cancelling futures and terminating workers...")
        if executor is not None:
            _safe_shutdown(executor, futures, wait=False, cancel_pending=True)
        _terminate_active_children()
        raise
    finally:
        if executor is not None:
            _safe_shutdown(executor, futures, wait=True, cancel_pending=True)
        _terminate_active_children()

    # Console summary (non-fatal)
    if errors:
        print("\nErrors encountered in some simulations:")
        for sim, err in errors:
            print(f"\nSimulation {sim} failed with traceback:")
            print(err)
    else:
        print("\nNo errors encountered.")

    print("\nSuccessful results:")
    for sim, res in successes:
        gv = getattr(sim, "gamma_values", None)
        print(f"{sim} (gamma_values={gv}) -> {res}")

    main_logger.info("Parameter search complete")
    return successes, errors


# (Optional) keep CLI behavior too
if __name__ == "__main__":
    run()

