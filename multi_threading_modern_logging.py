import argparse
import os
import traceback
import multiprocessing
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed



# Optional progress bar; if you have tqdm installed it will show progress.
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

# Assume you can edit this module to accept a logger; otherwise wrap below.

# Force spawn (avoids fork-related thread/lock issues)
try:
    multiprocessing.set_start_method("spawn", force=False)
except RuntimeError:
    pass  # already set


simulation_type = "DC"  # or "FSST", "AC", etc.


from neural_network_unified import train 


if simulation_type == "TRAN":
    # from neural_network_4_output_version_tran import train
    from simulation_parameters_folder import SimulationParametersTran as SimulationParameters
    LOG_DIR = "/home/filip/simulations/logging_folder_trans"
    os.makedirs(LOG_DIR, exist_ok=True)
elif simulation_type == "FSST":
    # from neural_network_4_output_version_fsst import train
    from simulation_parameters_folder import SimulationParametersFSST as SimulationParameters
    LOG_DIR = "/home/filip/simulations/logging_folder_FSST"
    os.makedirs(LOG_DIR, exist_ok=True)
elif simulation_type == "DC":
    # from neural_network_4_output_version_DC import train
    from simulation_parameters_folder import SimulationParametersDC as SimulationParameters
    LOG_DIR = "/home/filip/simulations/logging_folder_DC"
    os.makedirs(LOG_DIR, exist_ok=True)
else:
    raise ValueError(f"Unknown simulation_type: {simulation_type}")





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
    gamma_value_list = [[6e-10, 25e-11], [3e-9, 25e-10], [10e-10, 25e-11]]
    batch_size = 2
    beta = 5e-5
    scale_factor_list = [0.4, 0.5]
    bias = 0.3
    load_weights = False
    h5_file = None
    for gamma_values in gamma_value_list:
        for scale_factor in scale_factor_list:
            sim_params_list.append(
                SimulationParameters(scale_factor, bias, batch_size, beta, gamma_values, load_weights, h5_file)
            )
    return sim_params_list



def worker(sim_params):
    logger = get_worker_logger(sim_params)
    pid = os.getpid()
    logger.info(f"START simulation: {sim_params} (pid={pid})")
    try:
        result = train(sim_params, process_id=pid, logger=logger)
        logger.info(f"SUCCESS simulation: {sim_params} (pid={pid}) -> {result}")
        return sim_params, result, None
    except Exception:
        tb = traceback.format_exc()
        logger.exception(f"EXCEPTION in simulation: {sim_params} (pid={pid})")
        return sim_params, None, tb


def main():
    sim_params_list = build_param_grid()
    successes = []
    errors = []

    main_logger = configure_main_logger()
    main_logger.info(f"Starting parameter search with {len(sim_params_list)} jobs")
    print("Start method:", multiprocessing.get_start_method())

    with ProcessPoolExecutor() as executor:
        future_to_param = {executor.submit(worker, sim): sim for sim in sim_params_list}
        iterator = as_completed(future_to_param)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(future_to_param), desc="Running sims")
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
