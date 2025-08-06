import os
import traceback
import multiprocessing
import logging
from neural_network_4_output_version_fsst import train
from concurrent.futures import ProcessPoolExecutor, as_completed

# Optional progress bar; if you have tqdm installed it will show progress.
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

from netlist_generation_files import SimulationParameters

# Force spawn to avoid fork-inherited thread/lock deadlocks (safe on all platforms).
try:
    multiprocessing.set_start_method("spawn", force=False)
except RuntimeError:
    pass  # already set


def setup_worker_logger():
    logger = logging.getLogger(f"worker-{os.getpid()}")
    if not logger.handlers:
        fh = logging.FileHandler(f"worker_{os.getpid()}.log")
        fmt = logging.Formatter("%(asctime)s [PID %(process)d] %(levelname)s: %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
    return logger

def worker(sim_params):
    pid = os.getpid()
    logger = setup_worker_logger()
    logger.info(f"worker start sim={sim_params} pid={pid}")
    try:
        result = train(sim_params, pid)
        logger.info(f"worker success sim={sim_params}")
        return sim_params, result, None
    except Exception:
        tb = traceback.format_exc()
        logger.exception(f"worker exception for {sim_params}")
        return sim_params, None, tb


def build_param_grid():
    mode = "TESTING"
    sim_params_list = []
    gamma_value_list = [[3e-6, 1e-9]]
    batch_size = 2
    beta = 5e-5
    scale_factor_list = [0.4] 
    bias = 0.3

    for gamma_value in gamma_value_list:
        for scale_factor in scale_factor_list:
            sim_params_list.append(
                SimulationParameters(scale_factor, bias, batch_size, beta, gamma_value, mode)
            )
    return sim_params_list


def main():
    sim_params_list = build_param_grid()
    successes = []
    errors = []

    print("Start method:", multiprocessing.get_start_method())

    with ProcessPoolExecutor() as executor:
        future_to_param = {executor.submit(worker, sim): sim for sim in sim_params_list}
        iterator = as_completed(future_to_param)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(future_to_param), desc="Running sims")
        for future in iterator:
            sim, result, err = future.result()
            if err:
                errors.append((sim, err))
            else:
                successes.append((sim, result))

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


if __name__ == "__main__":
    main()
