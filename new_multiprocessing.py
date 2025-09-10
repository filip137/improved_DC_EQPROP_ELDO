# new_multiprocessing.py
import os
import traceback
import multiprocessing
import logging
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from save_and_load_functions import load_fsst_simparams

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

# ----------------- Choose your settings here -----------------
SIMULATION_TYPE = "FSST"     # "TRAN", "FSST", or "DC"
LOAD_WEIGHTS    = True    # True to load weights file
load_config_path = True
H5_PATH         = None     # override path to metrics_data.h5 (None -> per-mode default below)
dataset = "iris"
# -------------------------------------------------------------

# Explicit spawn context (more reliable with threads/subprocesses)
MP_CTX = multiprocessing.get_context("spawn")
try:
    multiprocessing.set_start_method("spawn", force=False)
except RuntimeError:
    pass  # already set

from neural_network_unified import train

# Pick params class, log dir, and default h5 per mode
if SIMULATION_TYPE == "TRAN":
    from simulation_parameters_folder import SimulationParametersTran as SimulationParameters
    LOG_DIR = "/home/filip/simulations/logging_folder_trans"
    DEFAULT_H5 = "/home/filip/simulations/testing_plots/fet_FSST_0802/FSST_114247_262613/plots/metrics_data.h5"
elif SIMULATION_TYPE == "FSST":
    if dataset == "moons":
        from simulation_parameters_folder import SimulationParametersFSST as SimulationParameters
    elif dataset == "iris":
        from simulation_parameters_folder import SimulationParametersFSST_iris as SimulationParameters
        
    LOG_DIR = "/home/filip/simulations/logging_folder_FSST"
    DEFAULT_H5 = "/home/filip/simulations/validation_plots/fet_FSST_0805/FSST_110129_228827/plots/metrics_data.h5" ##IRIS
    DEFAULT_H5 = "/home/filip/simulations/trained_models/fet_FSST_0816/FSST_085744_11546/plots/metrics_data" ## 5x16x4
    
    DEFAULT_H5 = "/home/filip/simulations/trained_models/resistor_FSST_0821/FSST_145936_126767/plots/metrics_data" ##5x12x4 RES, 0.3 scale factor
    CONFIG_PATH = '/home/filip/simulations/trained_models/resistor_FSST_0821/FSST_145936_126767/config'
    
    # DEFAULT_H5 = "/home/filip/simulations/trained_models/fet_FSST_0816/FSST_181418_145068/plots/metrics_data" ##5x12x4 FET, self-biased
    # CONFIG_PATH = "/home/filip/simulations/trained_models/fet_FSST_0816/FSST_181418_145068/config"
    
    # DEFAULT_H5 = "/home/filip/simulations/trained_models/resistor_FSST_0817/FSST_215720_96626/plots/metrics_data" ##5x12x4 res, self-biased
    # CONFIG_PATH = "/home/filip/simulations/trained_models/resistor_FSST_0817/FSST_215720_96626/config" ##5x12x4 res, self-biased
    
    # DEFAULT_H5 = "/home/filip/simulations/trained_models/fet_FSST_0816/FSST_105547_33955/plots/metrics_data"
    # CONFIG_PATH = "/home/filip/simulations/trained_models/fet_FSST_0816/FSST_105547_33955/config"
elif SIMULATION_TYPE == "DC":
    from simulation_parameters_folder import SimulationParametersDC as SimulationParameters
    LOG_DIR = "/home/filip/simulations/logging_folder_DC"
    DEFAULT_H5 = "/home/filip/simulations/trained_models/resistor_DC_0815/DC_170919_136152/plots/metrics_data.h5"
else:
    raise ValueError(f"Unknown SIMULATION_TYPE: {SIMULATION_TYPE}")

os.makedirs(LOG_DIR, exist_ok=True)

# Final H5 path to pass into SimulationParameters(..., load_weights, h5_file)
H5_PATH_TO_USE = H5_PATH or DEFAULT_H5


def configure_main_logger():
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(LOG_DIR, "main.log"), mode="a", encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s [PID %(process)d] %(levelname)s %(name)s: %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.propagate = False
    return logger


def get_worker_logger(sim_params):
    pid = os.getpid()
    sim_id = repr(sim_params)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in sim_id)[:50]
    name = f"worker-{pid}-{safe}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fname = os.path.join(LOG_DIR, f"{name}.log")
        fh = logging.FileHandler(fname, mode="a", encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s [PID %(process)d] %(levelname)s %(name)s: %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.propagate = False
    return logger


def build_param_grid():
    sim_params_list = []
    gamma_value_list = [[-3e-9, 0], [3e-9, 0], [-3e-8, 0], [3e-8, 0]]
    #gamma_value_list = [[6e-9, 6e-10], [3e-8, 3e-9], [10e-9, 10e-10], [-6e-9, 6e-10], [-3e-8, 3e-9], [-10e-9, 10e-10]]
    

    scale_factor_list = [0.4]
    batch_size = 2
    beta = 5e-5
    bias = 0.3

    if LOAD_WEIGHTS and not os.path.isdir(H5_PATH_TO_USE):
        raise FileNotFoundError(f"H5 weights file not found: {H5_PATH_TO_USE}")
    
    if load_config_path:
        base_sp = load_fsst_simparams(CONFIG_PATH)
        for gamma_values in gamma_value_list:
            for scale_factor in scale_factor_list:
                sp = copy.deepcopy(base_sp)
                # overwrite only what you want to sweep
                sp.gamma_values = gamma_values
                sp.scale_factor = scale_factor
                sp.amplifier = "BiDirWithNonLinCAP"

        
                # (optional) keep these aligned with your script?s choices
                sp.batch_size = batch_size
                sp.beta = beta
                sp.bias = bias
        
                sim_params_list.append(sp)
        return sim_params_list

    
    
    for gamma_values in gamma_value_list:
        for scale_factor in scale_factor_list:
            # Your constructor: (..., load_weights, h5_file)
            sp = SimulationParameters(
                scale_factor, bias, batch_size, beta, gamma_values, LOAD_WEIGHTS, H5_PATH_TO_USE
            )
            # Ensure the type is set if not set by the class
            try:
                sp.simulation_type = SIMULATION_TYPE
            except Exception:
                pass
            sim_params_list.append(sp)
    return sim_params_list


# Workers ignore Ctrl-C; the main process coordinates cancel/cleanup
def _init_worker_signals():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def worker(sim_params):
    logger = get_worker_logger(sim_params)
    pid = os.getpid()
    logger.info(
        f"START simulation: {sim_params} (pid={pid}) | "
        f"load_weights={getattr(sim_params,'load_weights',None)} | "
        f"h5_file={getattr(sim_params,'h5_file',None)}"
    )
    try:
        result = train(sim_params, process_id=pid, logger=logger)
        logger.info(f"SUCCESS simulation: {sim_params} (pid={pid}) -> {result}")
        return sim_params, result, None
    except Exception:
        tb = traceback.format_exc()
        logger.exception(f"EXCEPTION in simulation: {sim_params} (pid={pid})")
        return sim_params, None, tb


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
        # Python 3.9+ has cancel_futures
        executor.shutdown(wait=wait, cancel_futures=True)  # type: ignore[arg-type]
    except TypeError:
        # Python 3.8 and earlier
        executor.shutdown(wait=wait)


def main():
    sim_params_list = build_param_grid()
    successes, errors = [], []

    main_logger = configure_main_logger()
    main_logger.info(f"Starting parameter search with {len(sim_params_list)} jobs")
    print("Start method:", multiprocessing.get_start_method())

    executor = None
    futures = []
    try:
        executor = ProcessPoolExecutor(
            mp_context=MP_CTX,
            initializer=_init_worker_signals,
            max_workers=os.cpu_count() or 1,
        )
        # Keep explicit list of futures for compatibility shutdown
        futures = [executor.submit(worker, sim) for sim in sim_params_list]
        future_to_param = dict(zip(futures, sim_params_list))

        iterator = as_completed(futures)
        if tqdm is not None:
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

    if errors:
        print("\nErrors encountered in some simulations:")
        for sim, err in errors:
            print(f"\nSimulation {sim} failed with traceback:")
            print(err)
    else:
        print("\nNo errors encountered.")

    print("\nSuccessful results:")
    for sim, res in successes:
        print(f"{sim} -> {res}")

    main_logger.info("Parameter search complete")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

